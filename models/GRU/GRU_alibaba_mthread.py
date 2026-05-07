import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.GRU.GRU import GRU
from data_provider.TS_Dataset import TS_Dataset


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==========================================
# 1. 加载配置文件
# ==========================================
def load_config(config_path):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将配置转换为字典格式，并添加多进程相关的参数
    args = {
        "data_path": config.get('data_path', './data/Alibaba/cpu_util_percent_selected_machines_100_1min_no_time_stamp.parquet'),
        "save_dir": '../../experiments/gru_single_machine_60_6',
        "max_workers": 10,  # 多进程并发数
        "pred_len": config.get('out_len', 6),
        "context_len": config.get('in_len', 60),
        "stride": config.get('stride', 1),
        "epochs": config.get('epochs', 100),
        "batch_size": config.get('batch_size', 64),
        "learning_rate": config.get('learning_rate', 0.001),
        "hidden_size": config.get('hidden_size', 64),
        "num_layers": config.get('num_layers', 2),
        "T_embed_dim": config.get('T_embed_dim', 8),
        "timeenc": config.get('timeenc', 0),
        "norm_type": config.get('norm_type', 'ZScore'),
        "if_col_norm": config.get('if_col_norm', True),
        "train_val_test_ratio": config.get('train_val_test_ratio', [0.7, 0.1, 0.2]),
        "freq": config.get('freq', '1min'),
        "in_var": config.get('in_var', [0]),
        "out_var": config.get('out_var', [0])
    }
    return args

# 默认配置文件路径（相对于项目根目录）
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'conf', 'baseline', 'Alibaba_time_machine', 'GRU.yaml'
)

# 加载配置（在模块级别加载，以便在 main 函数中使用）
if len(sys.argv) > 1:
    config_path = sys.argv[1]
    if not os.path.isabs(config_path):
        # 如果是相对路径，则相对于项目根目录
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            config_path
        )
else:
    config_path = DEFAULT_CONFIG_PATH

# 检查配置文件是否存在
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

ARGS = load_config(config_path)

def calc_metrics(pred, truth):
    """计算 MSE 和 MAE"""
    mse = np.mean((pred - truth) ** 2)
    mae = np.mean(np.abs(pred - truth))
    return mse, mae

# ==========================================
# 3. Worker 函数 (单个机器的训练流程)
# ==========================================
def train_evaluate_worker(m_idx, machine_data, timestamps, args, train_len, test_start_idx, max_idx):
    """
    运行在子进程中的函数：负责单台机器的 训练 -> 保存 -> 预测 -> 评估
    """
    # 每个子进程需要重新设置一些环境，避免冲突
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # 创建临时 logger
    worker_logger = logging.getLogger(f"worker_{m_idx}")
    worker_logger.setLevel(logging.INFO)
    
    temp_data_file = None
    try:
        # ----------------------------------------
        # A. 为当前机器创建临时数据文件
        # ----------------------------------------
        temp_dir = Path(args['save_dir']) / "temp_data"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_data_file = temp_dir / f"machine_{m_idx}_data.parquet"
        
        # 创建单机数据 DataFrame
        machine_df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_util': machine_data
        })
        machine_df.to_parquet(temp_data_file, engine='pyarrow', index=False)
        
        # ----------------------------------------
        # B. 使用 TS_Dataset 加载训练数据
        # ----------------------------------------
        train_dataset = TS_Dataset(
            logger=worker_logger,
            data_path=str(temp_data_file),
            train_val_test_ratio=args['train_val_test_ratio'],
            in_len=args['context_len'],
            out_len=args['pred_len'],
            mode='train',
            norm_type=args['norm_type'],
            if_col_norm=args['if_col_norm'],
            timeenc=args['timeenc'],
            freq=args['freq'],
            stride=args['stride'],
            in_var=args['in_var'],
            out_var=args['out_var']
        )
        
        # 获取 scaler（用于后续反标准化）
        scaler = train_dataset.normalizer
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        # ----------------------------------------
        # D. 创建模型
        # ----------------------------------------
        class Args:
            def __init__(self, args_dict):
                for k, v in args_dict.items():
                    setattr(self, k, v)
        
        model_args = Args({
            'hidden_size': args['hidden_size'],
            'num_layers': args['num_layers'],
            'timeenc': args['timeenc'],
            'T_embed_dim': args['T_embed_dim'],
            'out_len': args['pred_len']
        })
        
        model = GRU(model_args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
        criterion = nn.MSELoss()
        
        # ----------------------------------------
        # E. 训练模型
        # ----------------------------------------
        model.train()
        for epoch in range(args['epochs']):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                if args['timeenc'] > -1:
                    # TS_Dataset 返回: (x, x_time, y, y_time, job_time)
                    x, x_time, y, y_time, job_time = batch
                    x = x.to(device).float()  # (B, in_len, 1)
                    x_time = x_time.to(device).long()  # (B, in_len, 4) - 注意：TS_Dataset 返回的 day 是 1-31，需要减1
                    y = y.to(device).float()  # (B, out_len, 1)
                    
                    # TS_Dataset 返回的 day 是 1-31，但 GRU embedding 需要 0-31
                    # 所以需要将 day 减1，并确保在有效范围内
                    if x_time.shape[-1] >= 1:
                        x_time_adj = x_time.clone()
                        # day: 1-31 -> 0-30 (确保在 0-31 范围内)
                        x_time_adj[:, :, 0] = torch.clamp(x_time[:, :, 0] - 1, 0, 31)
                        pred = model((x, x_time_adj, None))
                    else:
                        pred = model((x, x_time, None))
                else:
                    x, y = batch
                    x = x.to(device).float()  # (B, in_len, 1)
                    y = y.to(device).float()  # (B, out_len, 1)
                    pred = model(x)
                
                loss = criterion(pred.squeeze(-1), y.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # ----------------------------------------
        # F. 保存模型
        # ----------------------------------------
        machine_save_dir = Path(args['save_dir']) / f"machine_{m_idx}"
        machine_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean if hasattr(scaler, 'mean') else None,
            'scaler_std': scaler.std if hasattr(scaler, 'std') else None,
        }, machine_save_dir / "model.pt")
        
        # ----------------------------------------
        # G. 使用 TS_Dataset 加载测试数据进行预测
        # ----------------------------------------
        test_dataset = TS_Dataset(
            logger=worker_logger,
            data_path=str(temp_data_file),
            train_val_test_ratio=args['train_val_test_ratio'],
            in_len=args['context_len'],
            out_len=args['pred_len'],
            mode='test',
            norm_type=args['norm_type'],
            if_col_norm=args['if_col_norm'],
            timeenc=args['timeenc'],
            freq=args['freq'],
            stride=args['stride'],
            in_var=args['in_var'],
            out_var=args['out_var']
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0
        )
        
        ground_truths = []
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if args['timeenc'] > -1:
                    x, x_time, y, y_time, job_time = batch
                    x = x.to(device).float()  # (B, in_len, 1)
                    x_time = x_time.to(device).long()  # (B, in_len, 4)
                    y = y.to(device).float()  # (B, out_len, 1)
                    
                    # 调整 day 索引：1-31 -> 0-30，并确保在有效范围内
                    if x_time.shape[-1] >= 1:
                        x_time_adj = x_time.clone()
                        # day: 1-31 -> 0-30 (确保在 0-31 范围内)
                        x_time_adj[:, :, 0] = torch.clamp(x_time[:, :, 0] - 1, 0, 31)
                        pred = model((x, x_time_adj, None))
                    else:
                        pred = model((x, x_time, None))
                else:
                    x, y = batch
                    x = x.to(device).float()
                    y = y.to(device).float()
                    pred = model(x)
                
                # 反标准化预测结果和真实值
                pred_np = pred.squeeze().cpu().numpy()  # (out_len,)
                y_np = y.squeeze().cpu().numpy()  # (out_len,)
                
                # 反标准化
                if args['if_col_norm']:
                    # inver_transform_col 期望形状为 (B, L, C) 或类似
                    pred_denorm = scaler.inver_transform_col(pred_np.reshape(1, -1, 1)).squeeze()
                    y_denorm = scaler.inver_transform_col(y_np.reshape(1, -1, 1)).squeeze()
                else:
                    pred_denorm = scaler.inver_transform(pred_np.reshape(-1, 1)).flatten()
                    y_denorm = scaler.inver_transform(y_np.reshape(-1, 1)).flatten()
                
                predictions.append(pred_denorm)
                ground_truths.append(y_denorm)
        
        if not predictions:
            return m_idx, 0, 0, 0  # 无数据返回
        
        # ----------------------------------------
        # H. 评估
        # ----------------------------------------
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        m_mse = 0.0
        m_mae = 0.0
        
        for pred, gt in zip(predictions, ground_truths):
            mse_val, mae_val = calc_metrics(pred, gt)
            m_mse += mse_val
            m_mae += mae_val
        
        count_this = len(predictions)
        
        # 返回结果：(机器ID, 总MSE, 总MAE, 样本数)
        return m_idx, m_mse, m_mae, count_this
        
    except Exception as e:
        # 捕获子进程异常，防止主进程挂起
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return m_idx, float('nan'), str(error_msg), 0
    finally:
        # 清理临时文件
        if temp_data_file and temp_data_file.exists():
            try:
                temp_data_file.unlink()
            except:
                pass

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # --- 配置日志 ---
    save_dir_path = Path(ARGS['save_dir'])
    save_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = save_dir_path / "train_all_machines_multiprocess.log"
    if log_path.exists(): os.remove(log_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()
    logger.info(f"配置参数: {ARGS}")
    logger.info(f"模式: Multi-Processing (并发数: {ARGS['max_workers']})")

    # --- 数据加载 ---
    logger.info("正在加载数据...")
    # 处理数据路径：如果是相对路径，则相对于项目根目录
    data_path = ARGS['data_path']
    if not os.path.isabs(data_path):
        # 相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(project_root, data_path.lstrip('./'))
    
    logger.info(f"数据路径: {data_path}")
    try:
        data = pd.read_parquet(data_path, engine='pyarrow')
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        logger.error(f"尝试的数据路径: {data_path}")
        sys.exit(1)

    # 提取时间戳和数据
    timestamps = data['timestamp'].values
    values = data.drop(columns=['timestamp']).values.T.astype(np.float32)

    num_machines, total_len = values.shape
    train_len = int(total_len * 0.8)
    test_start_idx = train_len
    max_idx = total_len - ARGS['pred_len']
    
    logger.info(f"数据加载完成。机器数: {num_machines}, 样本长度: {total_len}")

    # --- 准备全局统计变量 ---
    total_mse = 0.0
    total_mae = 0.0 
    total_samples_count = 0

    logger.info("开始并行训练...")

    with ProcessPoolExecutor(max_workers=ARGS['max_workers']) as executor:
        futures = []
        for m_idx in range(num_machines):
            machine_data = values[m_idx, :]
            # 提交任务
            futures.append(
                executor.submit(
                    train_evaluate_worker, 
                    m_idx, machine_data, timestamps, ARGS, train_len, test_start_idx, max_idx
                )
            )
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=num_machines, desc="Processing Machines"):
            m_idx, m_mse, m_mae, count = future.result()
            
            if isinstance(m_mae, str): # 发生错误的情况
                logger.error(f"Machine {m_idx:02d} Failed: {m_mae}")
                continue
                
            if count > 0:
                # 累加结果
                total_mse += m_mse
                total_mae += m_mae
                total_samples_count += count
                
                # 计算单机平均值用于日志
                cur_rmse = np.sqrt(m_mse / count)
                cur_mae_avg = m_mae / count
                
                logger.info(f"Machine {m_idx:02d} Finished | RMSE: {cur_rmse:.4f} | MAE: {cur_mae_avg:.4f}")
            else:
                logger.warning(f"Machine {m_idx:02d} No samples predicted.")

    # --- 全局最终结果 ---
    if total_samples_count > 0:
        final_rmse = np.sqrt(total_mse / total_samples_count)
        final_mae = total_mae / total_samples_count
        
        res_str = (
            "\n" + "="*50 + "\n"
            " FINAL RESULTS (Multi-Processing Mode)\n" + 
            "="*50 + "\n"
            f"Total Samples Processed : {total_samples_count}\n"
            f"RMSE (Mean Squared)     : {final_rmse:.4f}\n"
            f"MAE  (Mean Absolute)    : {final_mae:.4f}\n" +
            "="*50
        )
        logger.info(res_str)
    else:
        logger.error("未生成任何预测结果。")

if __name__ == "__main__":
    # Windows/MacOS 下 multiprocessing 必须放在 main 块中
    # 使用方式: python GRU_alibaba_mthread.py [config_path]
    # 如果不提供 config_path，则使用默认路径
    print(f"使用配置文件: {config_path}")
    print(f"配置参数: {ARGS}")
    main()
