# GD-Diff: Decomposing Temporal and Job-Induced Dynamics for Probabilistic Computing Workload Forecasting via Graph-Conditioned Dual-Branch Diffusion

基于图增强扩散模型的多变量时间序列预测框架，融合时空图结构信息进行不确定性建模。

## 项目结构

```
├── main.py                    # 入口文件
├── conf/
│   ├── config.yaml            # 主配置（指定默认baseline）
│   └── baseline/              # 各模型+数据集的YAML配置
│       ├── Alibaba_time_machine/
│       ├── GoogleCluster_time_machine/
│       └── ...
├── models/                    # 模型实现
│   ├── CSDI_sg/               # GD-Diff（核心模型）
│   ├── CSDI/                  # CSDI基线
│   ├── D3U/                   # D3U
│   ├── TMDM/                  # TMDM
│   ├── TimeMixer/             # TimeMixer
│   ├── timefilter/            # TimeFilter
│   └── ...
├── Trainer/                   # 训练器
│   ├── CSDI_trainer.py        # 扩散模型训练器
│   ├── MTS_trainer.py         # 多变量时间序列训练器
│   └── UTS_trainer.py         # 单变量时间序列训练器
├── data_provider/             # 数据加载
│   ├── csdi_dataloader.py     # CSDI专用数据加载器
│   └── TS_Dataset.py          # 通用时间序列数据集
└── utils/
    ├── metrics.py             # 评估指标
    └── scaler.py              # 归一化工具
```

## 快速开始

### 环境配置

```bash
conda env create -n py310 python=3.10
pip install -r requirements.txt
```

### 训练与推理

使用Hydra配置管理，通过命令行override参数：

```bash
# 基本用法：使用默认配置训练
python main.py

# 指定配置文件和参数
python main.py \
  baseline=Alibaba_time_machine/CSDI_sg.yaml \
  baseline.training_config.device=cuda:0 \
  baseline.training_config.seed=2024 \
  baseline.seed=2024 \
  baseline.data_config.pred_len=6

# 仅推理（指定已训练好的模型路径）
python main.py \
  baseline=Alibaba_time_machine/CSDI_sg.yaml \
  baseline.training_config.mode=pred \
  baseline.training_config.model_path=./experiments/csdi_sg/Alibaba/xxx/best_model_0.pth
```

### 关键参数说明

#### 训练参数 (`training_config`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `device` | GPU设备 | `cuda:0` |
| `seed` | 随机种子 | `2024` |
| `epochs` | 最大训练轮数 | `100` |
| `patience` | 早停耐心值 | `20` |
| `batch_size` | 批大小 | `4` |
| `lr` | 学习率 | `1e-3` |
| `p1`, `p2` | 学习率衰减milestone | `15`, `18` |
| `valid_nsample` | 验证时采样数 | `3` |

#### 数据参数 (`data_config`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `past_len` | 历史窗口长度 | `60` |
| `pred_len` | 预测长度 | `6` 或 `30` |
| `is_non_stationary` | 是否进行非平稳处理 | `True` |
| `data_path` | 数据文件路径 | - |
| `graph_dir` | 时间图文件目录 | - |
| `graph_path` | 静态图文件路径 | - |

#### 模型参数 (`model_config`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nsample` | 推理采样数 | `100` |
| `diffusion.layers` | 去噪网络层数 | `4` |
| `diffusion.channels` | 通道数 | `64` |
| `diffusion.num_steps` | 扩散步数 | `50` |
| `graph.graph_type` | 图类型: `time`/`sg`/`None` | `time` |
| `graph.graph_architecture` | 图卷积: `gategcn`/`avwgcn` | `gategcn` |
| `graph.top_k` | 图邻域Top-K | `10` |
| `graph.embed_dim` | 节点嵌入维度 | `16` |
| `graph.order_version` | 残差块融合版本 | `7` |

#### Hydra Override语法

- `key=value`: 覆盖已有参数
- `+key=value`: 添加新参数（若已存在则报错）
- `++key=value`: 添加或覆盖参数

## 数据格式

### 输入数据

支持 `.parquet`、`.csv`、`.pkl` 格式，要求：
- 包含 `timestamp` 列作为索引
- 其余列为各变量的时间序列观测值

### 图数据

- **静态图**: 单个 `.npy` 文件，形状 `(N, N)` 的邻接矩阵
- **时间图**: 目录下多个 `.npy` 文件，每个文件对应一个时间步的邻接矩阵，按文件名中的数字排序

## 输出说明

### results.pkl

多变量预测结果DataFrame，列名格式：

| 列名 | 含义 |
|------|------|
| `timestamp` | 时间戳 |
| `Real_i` | 第i个变量的真实值 |
| `Prediction_i` | 第i个变量的中位数预测值 |
| `Sample_x_i` | 第x次采样中第i个变量的预测值 |

其中 `x` 为采样编号（0到nsample-1），`i` 为变量编号（0到K-1）。

### raw_results_nsample{N}.pkl

原始推理结果列表，包含：
- `all_generated_samples`: `(B, nsample, K, L)` 所有采样
- `all_target`: `(B, L, K)` 真实值
- 归一化参数等

## 多GPU并行实验

使用tmux管理多GPU实验：

```bash
# 示例：6个GPU并行跑不同seed和pred_len
tmux new-session -d -s exp
tmux new-window -t exp -n "GPU0"
tmux new-window -t exp -n "GPU1"

tmux send-keys -t exp:GPU0 "python main.py baseline=... baseline.training_config.device=cuda:0 ..." Enter
tmux send-keys -t exp:GPU1 "python main.py baseline=... baseline.training_config.device=cuda:1 ..." Enter

# 查看
tmux attach -t exp
```

## 评估指标

支持以下指标：
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **SMAPE** (Symmetric MAPE)
- **R2** (决定系数)
- **Pearson** (皮尔逊相关系数)
- **Spearman** (斯皮尔曼相关系数)

推理时使用采样中位数（50%分位数）作为点预测值。
