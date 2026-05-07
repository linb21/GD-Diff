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
from concurrent.futures import ProcessPoolExecutor, as_completed


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==========================================
# 1. 配置参数
# ==========================================
ARGS = {
    "data_path": '../../data/Alibaba/cpu_util_percent_selected_machines_100_1min_no_time_stamp.parquet',
    "save_dir": '../../experiments/deepar_single_machine_60_6',
    "max_workers": 10, 
    "freq": "1min",
    "pred_len": 6,          
    "context_len": 60,      
    "stride": 1,            
    "epochs": 20,
    "batches_per_epoch": 100, 
    "num_samples": 100       
}

def calc_metrics(samples, truth):
    pred_mean = np.mean(samples, axis=0)
    mse = np.mean((pred_mean - truth) ** 2)
    mae = np.mean(np.abs(pred_mean - truth))
    
    term1 = np.mean(np.abs(samples - truth[None, :]), axis=0)
    term2 = np.mean(np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1)) * 0.5
    crps = np.mean(term1 - term2)
    return mse, mae, crps

# ==========================================
# 2. Worker 函数 (单个机器的训练流程)
# ==========================================
def train_evaluate_worker(m_idx, machine_data, args, start_time, train_len, test_start_idx, max_idx):
    """
    运行在子进程中的函数：负责单台机器的 训练 -> 保存 -> 预测 -> 评估
    """
    # 每个子进程需要重新设置一些环境，避免冲突
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    
    # ----------------------------------------
    # A. 构造该机器的训练集
    # ----------------------------------------
    train_ds = ListDataset(
        [{"start": start_time, "target": machine_data[:train_len]}],
        freq=args['freq']
    )
    
    # ----------------------------------------
    # B. 训练该机器的专属模型
    # ----------------------------------------
    machine_save_dir = Path(args['save_dir']) / f"machine_{m_idx}"
    machine_save_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    estimator = DeepAREstimator(
        freq=args['freq'],
        prediction_length=args['pred_len'],
        context_length=args['context_len'],
        num_batches_per_epoch=args['batches_per_epoch'],
        trainer_kwargs={
            "max_epochs": args['epochs'],
            "accelerator": accelerator,
            "devices": 1,  
            "enable_progress_bar": False, 
            "logger": False,
            "enable_model_summary": False,
            "default_root_dir": str(machine_save_dir) 
        }
    )
    
    try:
        predictor = estimator.train(train_ds)
        
        # ----------------------------------------
        # C. 保存模型
        # ----------------------------------------
        machine_save_dir.mkdir(parents=True, exist_ok=True)
        predictor.serialize(machine_save_dir)

        # ----------------------------------------
        # D. 准备回测数据
        # ----------------------------------------
        input_list = []
        ground_truths = []
        
        for t in range(test_start_idx, max_idx, args['stride']):
            input_list.append({
                "start": start_time,
                "target": machine_data[:t]
            })
            ground_truths.append(machine_data[t : t + args['pred_len']])
        
        if not input_list:
            return m_idx, 0, 0, 0, 0 # 无数据返回

        # ----------------------------------------
        # E. 预测与评估
        # ----------------------------------------
        test_ds = ListDataset(input_list, freq=args['freq'])
        forecast_it = predictor.predict(test_ds, num_samples=args['num_samples'])
        forecasts = list(forecast_it)
        
        m_mse = 0.0
        m_mae = 0.0
        m_crps = 0.0
        
        for forecast, gt in zip(forecasts, ground_truths):
            mse_val, mae_val, crps_val = calc_metrics(forecast.samples, gt)
            m_mse += mse_val
            m_mae += mae_val
            m_crps += crps_val
            
        count_this = len(forecasts)
        
        # 返回结果：(机器ID, 总MSE, 总MAE, 总CRPS, 样本数)
        return m_idx, m_mse, m_mae, m_crps, count_this

    except Exception as e:
        # 捕获子进程异常，防止主进程挂起
        return m_idx, float('nan'), str(e), 0, 0

# ==========================================
# 3. 主程序
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
    
    logger.info(f"数据加载完成。机器数: {num_machines}, 样本长度: {total_len}")

    # --- 准备全局统计变量 ---
    total_mse = 0.0
    total_mae = 0.0 
    total_crps = 0.0
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
                    m_idx, machine_data, ARGS, start_time, train_len, test_start_idx, max_idx
                )
            )
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=num_machines, desc="Processing Machines"):
            m_idx, m_mse, m_mae, m_crps, count = future.result()
            
            if isinstance(m_mae, str): # 发生错误的情况
                logger.error(f"Machine {m_idx:02d} Failed: {m_mae}")
                continue
                
            if count > 0:
                # 累加结果
                total_mse += m_mse
                total_mae += m_mae
                total_crps += m_crps
                total_samples_count += count
                
                # 计算单机平均值用于日志
                cur_rmse = np.sqrt(m_mse / count)
                cur_mae_avg = m_mae / count
                cur_crps_avg = m_crps / count
                
                logger.info(f"Machine {m_idx:02d} Finished | RMSE: {cur_rmse:.4f} | MAE: {cur_mae_avg:.4f}")
            else:
                logger.warning(f"Machine {m_idx:02d} No samples predicted.")

    # --- 全局最终结果 ---
    if total_samples_count > 0:
        final_rmse = np.sqrt(total_mse / total_samples_count)
        final_mae = total_mae / total_samples_count
        final_crps = total_crps / total_samples_count
        
        res_str = (
            "\n" + "="*50 + "\n"
            " FINAL RESULTS (Multi-Processing Mode)\n" + 
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

if __name__ == "__main__":
    # Windows/MacOS 下 multiprocessing 必须放在 main 块中
    main()