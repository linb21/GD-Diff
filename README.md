# GD-Diff: Decomposing Temporal and Job-Induced Dynamics for Probabilistic Computing Workload Forecasting via Graph-Conditioned Dual-Branch Diffusion

A graph-enhanced diffusion framework for multivariate time-series forecasting, which incorporates spatio-temporal graph structures for uncertainty-aware workload prediction.

## Project Structure

├── main.py # Entry file
├── conf/
│ ├── config.yaml # Main configuration file specifying the default baseline
│ └── baseline/ # YAML configurations for different models and datasets
│ ├── Alibaba_time_machine/
│ ├── GoogleCluster_time_machine/
│ └── ...
├── models/ # Model implementations
│ ├── CSDI_sg/ # GD-Diff: the proposed model
│ ├── CSDI/ # CSDI baseline
│ ├── D3U/ # D3U
│ ├── TMDM/ # TMDM
│ ├── TimeMixer/ # TimeMixer
│ ├── timefilter/ # TimeFilter
│ └── ...
├── Trainer/ # Trainers
│ ├── CSDI_trainer.py # Trainer for diffusion-based models
│ ├── MTS_trainer.py # Trainer for multivariate time-series models
│ └── UTS_trainer.py # Trainer for univariate time-series models
├── data_provider/ # Data loading modules
│ ├── csdi_dataloader.py # Data loader for CSDI-style diffusion models
│ └── TS_Dataset.py # General time-series dataset
└── utils/
├── metrics.py # Evaluation metrics
└── scaler.py # Normalization utilities

## Quick Start

### Environment Setup

```bash
conda env create -n py310 python=3.10
pip install -r requirements.txt
```

### Training and Inference

This project uses Hydra for configuration management. Parameters can be overridden from the command line.

```bash
# Basic usage: train with the default configuration
python main.py

# Specify the configuration file and override parameters
python main.py \
  baseline=Alibaba_time_machine/CSDI_sg.yaml \
  baseline.training_config.device=cuda:0 \
  baseline.training_config.seed=2024 \
  baseline.seed=2024 \
  baseline.data_config.pred_len=6

# Inference only: specify the path to a trained model
python main.py \
  baseline=Alibaba_time_machine/CSDI_sg.yaml \
  baseline.training_config.mode=pred \
  baseline.training_config.model_path=./experiments/csdi_sg/Alibaba/xxx/best_model_0.pth
```

## Key Parameters

### Training Parameters (`training_config`)

| Parameter       | Description                          | Default Value |
| --------------- | ------------------------------------ | ------------- |
| `device`        | GPU device                           | `cuda:0`      |
| `seed`          | Random seed                          | `2024`        |
| `epochs`        | Maximum number of training epochs    | `100`         |
| `patience`      | Early-stopping patience              | `20`          |
| `batch_size`    | Batch size                           | `4`           |
| `lr`            | Learning rate                        | `1e-3`        |
| `p1`, `p2`      | Learning-rate decay milestones       | `15`, `18`    |
| `valid_nsample` | Number of samples used in validation | `3`           |

### Data Parameters (`data_config`)

| Parameter           | Description                                | Default Value |
| ------------------- | ------------------------------------------ | ------------- |
| `past_len`          | Historical look-back window length         | `60`          |
| `pred_len`          | Forecasting horizon                        | `6` or `30`   |
| `is_non_stationary` | Whether to apply non-stationary processing | `True`        |
| `data_path`         | Path to the time-series data file          | -             |
| `graph_dir`         | Directory of time-varying graph files      | -             |
| `graph_path`        | Path to the static graph file              | -             |

### Model Parameters (`model_config`)

| Parameter                  | Description                                    | Default Value |
| -------------------------- | ---------------------------------------------- | ------------- |
| `nsample`                  | Number of samples during inference             | `100`         |
| `diffusion.layers`         | Number of layers in the denoising network      | `4`           |
| `diffusion.channels`       | Number of hidden channels                      | `64`          |
| `diffusion.num_steps`      | Number of diffusion steps                      | `50`          |
| `graph.graph_type`         | Graph type: `time` / `sg` / `None`             | `time`        |
| `graph.graph_architecture` | Graph convolution module: `gategcn` / `avwgcn` | `gategcn`     |
| `graph.top_k`              | Top-K neighbors retained in the graph          | `10`          |
| `graph.embed_dim`          | Node embedding dimension                       | `16`          |
| `graph.order_version`      | Fusion version used in the residual block      | `7`           |

### Hydra Override Syntax

- `key=value`: override an existing parameter.
- `+key=value`: add a new parameter. An error will be raised if the key already exists.
- `++key=value`: add a new parameter or override an existing one.

## Data Format

### Input Data

The framework supports `.parquet`, `.csv`, and `.pkl` files. The input data should satisfy the following requirements:

- It contains a `timestamp` column as the time index.
- The remaining columns correspond to the observed time series of different variables.

### Graph Data

- **Static graph**: a single `.npy` file with shape `(N, N)`, representing the adjacency matrix.
- **Time-varying graph**: a directory containing multiple `.npy` files, where each file corresponds to the adjacency matrix at one time step. Files are sorted according to the numeric indices in their filenames.

## Output Description

### `results.pkl`

A DataFrame containing multivariate forecasting results. The column names follow the format below:

| Column Name    | Description                                       |
| -------------- | ------------------------------------------------- |
| `timestamp`    | Timestamp                                         |
| `Real_i`       | Ground-truth value of the `i`-th variable         |
| `Prediction_i` | Median prediction of the `i`-th variable          |
| `Sample_x_i`   | Prediction of the `i`-th variable from sample `x` |

Here, `x` denotes the sampling index, ranging from `0` to `nsample - 1`, and `i` denotes the variable index.

### `raw_results_nsample{N}.pkl`

The raw inference result list, including:

- `all_generated_samples`: all generated samples with shape `(B, nsample, K, L)`;
- `all_target`: ground-truth values with shape `(B, L, K)`;
- normalization parameters and other auxiliary information.

## Evaluation Metrics

The framework supports the following evaluation metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **R2**: Coefficient of determination
- **Pearson**: Pearson correlation coefficient
- **Spearman**: Spearman rank correlation coefficient

During inference, the median of the generated samples, namely the 50th percentile, is used as the point prediction.

```

```
