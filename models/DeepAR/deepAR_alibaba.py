import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import warnings
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator

# ==========================================
# 0. 警告屏蔽与环境设置
# ==========================================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==========================================
# 1. 配置参数
# ==========================================
ARGS = {
    "data_path": '../../data/Alibaba/cpu_util_percent_selected_machines_100_1min_no_time_stamp.parquet',
    "save_dir": '../../experiments/deepar_single_machine_60_30', 
    
    # 时间设置
    "freq": "1min",
    "pred_len": 30,          
    "context_len": 60,      
    "stride": 1,            
    
    # 训练设置
    "epochs": 20,
    "batches_per_epoch": 100, 
    "num_samples": 100       
}

# --- 配置日志 ---
def setup_logging(save_dir):
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    log_path = save_dir_path / "train_all_machines.log"
    if log_path.exists():
        os.remove(log_path)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

logger = setup_logging(ARGS['save_dir'])
logger.info(f"配置参数: {ARGS}")
logger.info("模式: Single Machine (每台机器独立训练模型)")

# ==========================================
# 2. 数据加载
# ==========================================
logger.info("正在加载数据...")
try:
    data = pd.read_parquet(ARGS['data_path'], engine='pyarrow')
except Exception as e:
    logger.error(f"数据加载失败: {e}")
    sys.exit(1)

start_time = pd.Period(data['timestamp'].iloc[0], freq=ARGS['freq'])
values = data.drop(columns=['timestamp']).values.T.astype(np.float32)

num_machines, total_len = values.shape
train_len = int(total_len * 0.8)
test_start_idx = train_len
max_idx = total_len - ARGS['pred_len']

logger.info(f"数据加载完成。机器数: {num_machines}, 时间长度: {total_len}")

# ==========================================
# 3. 循环：单机训练 -> 单机预测 -> 单机评估
# ==========================================
total_mse = 0.0
total_mae = 0.0 
total_crps = 0.0
total_samples_count = 0

# 辅助函数
def calc_metrics(samples, truth):
    pred_mean = np.mean(samples, axis=0)
    mse = np.mean((pred_mean - truth) ** 2)
    mae = np.mean(np.abs(pred_mean - truth))
    
    term1 = np.mean(np.abs(samples - truth[None, :]), axis=0)
    term2 = np.mean(np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1)) * 0.5
    crps = np.mean(term1 - term2)
    
    return mse, mae, crps

logger.info("开始逐台机器处理...")

# 外层循环：遍历每一台机器
for m_idx in range(num_machines):
    machine_data = values[m_idx, :]
    
    # ----------------------------------------
    # A. 构造该机器的训练集
    # ----------------------------------------
    train_ds = ListDataset(
        [{"start": start_time, "target": machine_data[:train_len]}],
        freq=ARGS['freq']
    )
    
    # ----------------------------------------
    # B. 训练该机器的专属模型
    # ----------------------------------------
    # 定义该机器的模型保存路径 (用于存放临时 checkpoint 和最终模型)
    machine_save_dir = Path(ARGS['save_dir']) / f"machine_{m_idx}"
    machine_save_dir.mkdir(parents=True, exist_ok=True)
    
    estimator = DeepAREstimator(
        freq=ARGS['freq'],
        prediction_length=ARGS['pred_len'],
        context_length=ARGS['context_len'],
        num_batches_per_epoch=ARGS['batches_per_epoch'],
        trainer_kwargs={
            "max_epochs": ARGS['epochs'],
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 5,
            "enable_progress_bar": False, 
            "logger": False,
            "enable_model_summary": False,
            # 【重要】删除了 "enable_checkpointing": False
            # 【新增】指定 root_dir，防止在当前目录生成一堆 lightning_logs
            "default_root_dir": str(machine_save_dir) 
        }
    )
    
    # 训练
    predictor = estimator.train(train_ds)
    
    # ----------------------------------------
    # C. 保存最终模型 (序列化)
    # ----------------------------------------
    # 注意：predictor.serialize 会把模型保存到这个文件夹下
    # 如果文件夹里有其他东西（比如 lightning_logs），shutil.rmtree 会删掉它们
    # 为了安全，我们在 serialize 前只清理旧的序列化文件，不删 lightning logs
    
    # 重新确保目录存在
    machine_save_dir.mkdir(parents=True, exist_ok=True)
    predictor.serialize(machine_save_dir)

    # ----------------------------------------
    # D. 准备回测数据 (串行构造)
    # ----------------------------------------
    input_list = []
    ground_truths = []
    
    for t in range(test_start_idx, max_idx, ARGS['stride']):
        input_list.append({
            "start": start_time,
            "target": machine_data[:t]
        })
        ground_truths.append(machine_data[t : t + ARGS['pred_len']])
    
    if not input_list:
        continue

    # ----------------------------------------
    # E. 预测与评估
    # ----------------------------------------
    test_ds = ListDataset(input_list, freq=ARGS['freq'])
    forecast_it = predictor.predict(test_ds, num_samples=ARGS['num_samples'])
    forecasts = list(forecast_it)
    
    m_mse = 0.0
    m_mae = 0.0
    m_crps = 0.0
    
    for forecast, gt in zip(forecasts, ground_truths):
        mse_val, mae_val, crps_val = calc_metrics(forecast.samples, gt)
        m_mse += mse_val
        m_mae += mae_val
        m_crps += crps_val
        
    # 累加本台机器的指标到全局
    count_this = len(forecasts)
    total_mse += m_mse
    total_mae += m_mae
    total_crps += m_crps
    total_samples_count += count_this
    
    # 计算当前这台机器的平均指标
    cur_rmse = np.sqrt(m_mse / count_this)
    cur_mae = m_mae / count_this
    cur_crps = m_crps / count_this
    
    logger.info(f"Machine {m_idx:02d} | RMSE: {cur_rmse:.4f} | MAE: {cur_mae:.4f} | CRPS: {cur_crps:.4f}")

# ==========================================
# 4. 全局最终结果
# ==========================================
if total_samples_count > 0:
    final_rmse = np.sqrt(total_mse / total_samples_count)
    final_mae = total_mae / total_samples_count
    final_crps = total_crps / total_samples_count
    
    res_str = (
        "\n" + "="*50 + "\n"
        " FINAL RESULTS (Single Machine Training Mode)\n" + 
        "="*50 + "\n"
        f"Total Samples Processed : {total_samples_count}\n"
        f"RMSE (Mean Squared)     : {final_rmse:.4f}\n"
        f"MAE  (Mean Absolute)    : {final_mae:.4f}\n"
        f"CRPS (Probabilistic)    : {final_crps:.4f}\n" +
        "="*50
    )
    logger.info(res_str)
else:
    logger.error("未生成任何预测结果。")