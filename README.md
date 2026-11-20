# GD-Diff

This repository contains the simplified GD-Diff implementation for probabilistic multivariate time-series forecasting.

## Project Structure

```text
.
|-- main.py
|-- conf/
|   |-- config.yaml
|   `-- baseline/
|       |-- Alibaba_time_machine/gd_diff.yaml
|       `-- GoogleCluster_time_machine/gd_diff.yaml
|-- data_provider/
|   `-- csdi_dataloader.py
|-- models/
|   `-- gd_diff/
|-- utils/
|   `-- metrics.py
`-- requirements
```

## Data

GD-Diff reads one CPU workload table and a sequence of job co-occurrence graphs for each cluster. The raw trace sources are [Alibaba Cluster Trace 2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018) and [Google ClusterData2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) (cell a). The configurations expect the processed files below. The repository currently contains code and configuration files only for these inputs.

| Dataset | Machines | Workload timestamps | Time between rows | Timestamp for graph 0 |
| --- | ---: | ---: | --- | --- |
| Alibaba | 100 | 11,520 | 1 minute | - |
| Google | 100 | 8,930 | 5 minutes | 2019-05-01 00:05:00 |

Place the processed files under `data/` at the repository root. Each configuration reads its workload table from `data_config.data_path` and its dynamic graphs from `data_config.graph_dir`.

```text
data/
  Alibaba/
    cpu_util_percent_selected_machines_100_1min_no_time_stamp.parquet
    minute_graph/step4_node100_spread_normalized/
      adjacency_matrix_day_0.npy
      adjacency_matrix_day_1.npy
      ...
  GoogleCluster/
    cpus_avg_100machines_5min.parquet
    interval_graph/
      adjacency_matrix_interval_000000.npy
      adjacency_matrix_interval_000001.npy
      ...
```

Each Parquet table needs a `timestamp` column and one numeric CPU workload column per machine. Keep timestamps in increasing order at the interval shown above, without timezone information. The supplied configurations split rows chronologically into 70% training, 10% validation and 20% testing, then construct windows within each split.

Each NPY file holds a 100 x 100 adjacency matrix. Its integer suffix counts intervals from graph 0. For example, Alibaba's `adjacency_matrix_day_2.npy` corresponds to 00:02 on January 1. Use consecutive indices starting at zero that cover the full workload table. The loader reads graphs in numeric suffix order and zeroes the graph inputs for the forecast horizon.

Graph row and column i must identify the same machine as workload column i after excluding `timestamp`. When preparing a subset, look up each workload machine ID in the original graph's node list and select both graph axes in workload column order. Keep that ordered list alongside the processed inputs. 

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default configuration:

```bash
python main.py
```

`main.py` reads the default GD-Diff configuration from `conf/config.yaml`.
