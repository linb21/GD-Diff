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
