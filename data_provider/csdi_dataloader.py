import glob
import os
from datetime import datetime
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Diffusion_Dataset(Dataset):
    def __init__(self, config, data, mode):
        data_config = config["data_config"]
        model_config = config["model_config"]
        graph_config = model_config.get("graph", {})

        if data_config["data_mode"] != 2:
            raise ValueError("Only data_mode=2 is supported in this simplified GD-Diff repo.")

        self.pred_length = data_config["pred_len"] * data_config["unit_len"]
        self.past_length = data_config["past_len"] * data_config["unit_len"]
        self.seq_length = self.past_length + self.pred_length
        self.columns = list(data_config["columns"])
        self.target_cols = list(data_config["target_cols"])
        self.target_dim = len(self.target_cols)
        self.training_step = data_config["training_step"]

        self.graph_type = graph_config.get("graph_type", "None")
        self.graph_freq = graph_config.get("graph_freq", "minute")
        self.graph_path = data_config.get("graph_path")
        self.graph_dir = data_config.get("graph_dir")
        self.real_time_emb_dim = model_config["diffusion"].get("real_time_emb_dim", 0)

        total_len = len(data)
        train_ratio, val_ratio, _ = data_config["train_val_test_ratio"]
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)

        if mode == "train":
            frame = data.iloc[:train_len]
            step = self.training_step
        elif mode == "valid":
            frame = data.iloc[train_len : train_len + val_len]
            step = data_config["unit_len"]
        elif mode == "test":
            frame = data.iloc[train_len + val_len :]
            step = 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.index = frame.index
        self.main_data = frame.to_numpy(dtype=np.float32, copy=True)
        self.mask_data = np.ones_like(self.main_data, dtype=np.float32)

        last_start = len(self.main_data) - self.seq_length + 1
        if last_start <= 0:
            raise ValueError(f"Not enough data for mode={mode}: got {len(self.main_data)} rows.")
        self.use_index = np.arange(0, last_start, step)

        reference_time = graph_config.get("reference_time")
        self.graph_reference_time = (
            datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S") if reference_time else None
        )
        self.graph_interval = graph_config.get("graph_interval", 60)
        self.graph = self._load_graph()

    def _load_graph(self):
        if self.graph_type == "static":
            if not self.graph_path:
                raise ValueError("graph_path is required when graph_type=static.")
            return torch.from_numpy(np.load(self.graph_path)).float()

        if self.graph_type == "time":
            if not self.graph_dir:
                raise ValueError("graph_dir is required when graph_type=time.")
            graph_pattern = os.path.join(self.graph_dir, "adjacency_matrix_*.npy")
            graph_files = sorted(
                glob.glob(graph_pattern),
                key=lambda p: int(re.search(r'(\d+)\.npy$', os.path.basename(p)).group(1)),
            )
            if not graph_files:
                raise ValueError(f"No graph files found under {self.graph_dir}.")
            return [torch.from_numpy(np.load(path)).float() for path in graph_files]

        return None

    def _slice_graph(self, time_index):
        if self.graph_type == "static":
            return self.graph

        if self.graph_type != "time":
            return None

        if self.graph_freq != "minute":
            raise ValueError("Only graph_freq=minute is supported in this simplified GD-Diff repo.")
        if self.graph_reference_time is None:
            raise ValueError("reference_time is required when graph_type=time.")

        graph_list = []
        for ts in time_index:
            graph_idx = int((ts.to_pydatetime() - self.graph_reference_time).total_seconds() // self.graph_interval)
            graph_list.append(self.graph[graph_idx].clone())

        for idx in range(self.seq_length - self.pred_length, self.seq_length):
            graph_list[idx].zero_()

        return torch.stack(graph_list, dim=0)

    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        seq_slice = slice(index, index + self.seq_length)

        observed_data = self.main_data[seq_slice].copy()
        observed_mask = self.mask_data[seq_slice].copy()
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_length :, : self.target_dim] = 0.0

        sample = {
            "observed_data": np.ascontiguousarray(observed_data),
            "observed_mask": np.ascontiguousarray(observed_mask),
            "gt_mask": np.ascontiguousarray(gt_mask),
            "timepoints": np.arange(self.seq_length, dtype=np.float32),
            "feature_id": np.arange(len(self.columns), dtype=np.float32),
        }

        graph = self._slice_graph(self.index[seq_slice])
        if graph is not None:
            sample["graph"] = graph

        if self.real_time_emb_dim > 0:
            time_index = self.index[seq_slice]
            sample["day"] = np.asarray([ts.day for ts in time_index], dtype=np.int32)
            sample["hour"] = np.asarray([ts.hour for ts in time_index], dtype=np.int32)
            sample["minute"] = np.asarray([ts.minute for ts in time_index], dtype=np.int32)

        return sample

    def __len__(self):
        return len(self.use_index)
