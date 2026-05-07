import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_provider.csdi_dataloader import Diffusion_Dataset
from models.gd_diff.csdi_model_architecture import Forecasting
from utils.metrics import (
    MAE_np,
    MAE_torch,
    MAPE_np,
    MAPE_torch,
    PEARSON_np,
    PEARSON_torch,
    R2_np,
    R2_torch,
    RMSE_np,
    RMSE_torch,
    SMAPE_np,
    SMAPE_torch,
    SPEARMAN_np,
    SPEARMAN_torch,
)


class gd_diff:
    def __init__(self, configs):
        self.config = dict(configs) if not isinstance(configs, dict) else configs
        self.config.setdefault("root", "./")
        self.config.setdefault("training_config", self.config)
        self.config.setdefault("data_config", self.config)
        self.config.setdefault("model_config", self.config)

        if "all" in self.config["data_config"]["target_cols"]:
            self.config = self.preprocess_data(self.config)

        self._set_seed(self.config["training_config"]["seed"])

        self.model = None
        self.mean = None
        self.std = None
        self.test_index = None
        self.now_epoch_no = 0

        self.target_dim = len(self.config["data_config"]["columns"])
        self.real_target_dim = len(self.config["data_config"]["target_cols"])
        self.device = self.config["training_config"]["device"]
        self.past_len = self.config["data_config"]["past_len"] * self.config["data_config"]["unit_len"]
        self.is_non_stationary = self.config["data_config"]["is_non_stationary"]
        self.non_stationary_type = self.config["data_config"]["non_stationary_type"]
        self.dataloader = {"train": None, "valid": None, "test": None}

        roll_run = self.config["training_config"].get("roll_run", 0)
        if roll_run != 0:
            raise ValueError("This simplified GD-Diff repo only supports roll_run=0.")

        self.foldername = self._init_output_dir()

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_output_dir(self):
        save_root = self.config["training_config"]["save_path"]
        data_path = self.config["data_config"]["data_path"]
        dataset_name = os.path.basename(os.path.dirname(data_path)) or "dataset"
        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H-%M-%S")
        foldername = os.path.join(save_root, dataset_name, current_time)
        os.makedirs(foldername, exist_ok=True)

        config_path = self.config.get("config_path")
        if config_path:
            source_config = os.path.join(self.config["root"], config_path)
            if os.path.exists(source_config):
                shutil.copy(source_config, os.path.join(foldername, os.path.basename(source_config)))

        print(f"model folder: {foldername}")
        return foldername

    def init_model(self):
        self.model = Forecasting(self.config, self.device, self.target_dim).to(self.device)
        total_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def preprocess_data(self, configs):
        raw_dataset = self._read_data_file(configs["data_config"]["data_path"])
        raw_dataset.set_index("timestamp", inplace=True)
        raw_dataset.index = pd.to_datetime(raw_dataset.index)

        configs["data_config"]["target_cols"] = raw_dataset.columns.tolist()
        configs["data_config"]["columns"] = raw_dataset.columns.tolist()
        configs["data_config"]["feature_cols_mask2D"] = []
        configs["data_config"]["feature_cols_mask1D"] = []
        configs["data_config"]["feature_cols_da"] = []
        return configs

    def _read_data_file(self, file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            return pd.read_pickle(file_path)
        raise ValueError(f"Unsupported file format: {file_path}")

    def process_data(self):
        if self.config["data_config"]["data_mode"] != 2:
            raise ValueError("This simplified GD-Diff repo only supports data_mode=2.")

        raw_dataset = self._read_data_file(self.config["data_config"]["data_path"])
        raw_dataset.set_index("timestamp", inplace=True)
        raw_dataset.index = pd.to_datetime(raw_dataset.index)

        ordered_columns = (
            self.config["data_config"]["target_cols"]
            + self.config["data_config"]["feature_cols_mask2D"]
            + self.config["data_config"]["feature_cols_mask1D"]
            + self.config["data_config"]["feature_cols_da"]
        )
        processed_dataset = raw_dataset.reindex(ordered_columns, axis=1)

        total_len = len(processed_dataset)
        train_ratio, val_ratio, _ = self.config["data_config"]["train_val_test_ratio"]
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        train_dataset = processed_dataset.iloc[:train_len]
        self.test_index = processed_dataset.iloc[train_len + val_len :].index

        processed_dataset = self.normalize(train_dataset, processed_dataset)
        self.generate_dataloader(processed_dataset)

    def normalize(self, train_dataset, processed_dataset):
        if self.config["training_config"]["model_path"] == "":
            col_mean = np.mean(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std = np.std(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std[col_std == 0] = 1
            self.mean = col_mean
            self.std = col_std
        elif self.mean is None or self.std is None:
            raise ValueError("Mean and std are not available before loading a checkpoint.")

        return (processed_dataset - self.mean) / self.std

    def generate_dataloader(self, processed_dataset):
        batch_size = self.config["training_config"]["batch_size"]
        self.dataloader["train"] = DataLoader(
            Diffusion_Dataset(self.config, processed_dataset, "train"),
            batch_size=batch_size,
            shuffle=True,
        )
        self.dataloader["valid"] = DataLoader(
            Diffusion_Dataset(self.config, processed_dataset, "valid"),
            batch_size=batch_size,
            shuffle=False,
        )
        self.dataloader["test"] = DataLoader(
            Diffusion_Dataset(self.config, processed_dataset, "test"),
            batch_size=batch_size,
            shuffle=False,
        )

    def _normalize_batch_for_non_stationary(self, batch):
        if not self.is_non_stationary:
            return None, None

        observed_data = batch["observed_data"]
        means = torch.mean(observed_data[:, : self.past_len, :], dim=1, keepdim=True).detach()
        normalized = observed_data - means

        if self.non_stationary_type == 0:
            stdev = torch.ones_like(means)
        else:
            stdev = torch.sqrt(
                torch.var(observed_data[:, : self.past_len, :], dim=1, keepdim=True, unbiased=False)
            ).detach()
            stdev = torch.where(stdev == 0, torch.ones_like(stdev), stdev)

        batch["observed_data"] = normalized / stdev
        return means, stdev

    def _restore_batch_scale(self, samples, targets, means, stdev):
        if means is None or stdev is None:
            return samples, targets
        means = means.to(samples.device)
        stdev = stdev.to(samples.device)
        samples = samples * stdev.unsqueeze(1) + means.unsqueeze(1)
        targets = targets * stdev + means
        return samples, targets

    def train(self):
        self.init_model()
        self.process_data()

        cfg = self.config["training_config"]
        optimizer = Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(cfg["p1"]), int(cfg["p2"])],
            gamma=cfg["lr_gamma"],
        )

        output_path = os.path.join(self.foldername, "model_0.pth")
        best_path = os.path.join(self.foldername, "best_model_0.pth")
        writer = SummaryWriter(log_dir=os.path.join(self.foldername, "tensorboard"))

        train_losses = []
        valid_losses = []
        best_valid_loss = float("inf")
        patience = cfg["patience"]
        patience_counter = 0

        for epoch_no in range(cfg["epochs"]):
            self.now_epoch_no = epoch_no
            self.model.train()
            epoch_loss = 0.0

            with tqdm(self.dataloader["train"], mininterval=5.0, maxinterval=50.0) as progress:
                for batch_no, train_batch in enumerate(progress, start=1):
                    optimizer.zero_grad()
                    self._normalize_batch_for_non_stationary(train_batch)
                    loss = self.model(train_batch, is_train=1)
                    if torch.isnan(loss):
                        raise ValueError(f"NaN loss detected at epoch {epoch_no}, batch {batch_no}.")

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    global_step = epoch_no * len(self.dataloader["train"]) + batch_no
                    writer.add_scalar("train/batch_loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    progress.set_postfix(loss=loss.item(), avg_epoch_loss=epoch_loss / batch_no, epoch=epoch_no)

            scheduler.step()
            avg_train_loss = epoch_loss / max(len(self.dataloader["train"]), 1)
            train_losses.append(avg_train_loss)
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch_no)

            valid_loss = self.validate()
            valid_losses.append(valid_loss)
            writer.add_scalar("valid/loss", valid_loss, epoch_no)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                self.save_model(self.model, best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch_no}.")
                    break

        writer.close()
        self.save_model(self.model, output_path)

        with open(os.path.join(self.foldername, "losses_0.pkl"), "wb") as file_obj:
            pickle.dump({"train_losses": train_losses, "valid_losses": valid_losses}, file_obj)

        self.load_model(best_path)
        self.test(test=1)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for valid_batch in self.dataloader["valid"]:
                self._normalize_batch_for_non_stationary(valid_batch)
                loss = self.model(valid_batch, is_train=0)
                total_loss += loss.item()

        return total_loss / max(len(self.dataloader["valid"]), 1)

    def predict(self, dataloader, nsample):
        self.model.eval()
        all_target = []
        all_evalpoint = []
        all_observed_point = []
        all_observed_time = []
        all_generated_samples = []

        with torch.no_grad():
            with tqdm(dataloader, mininterval=5.0, maxinterval=50.0) as progress:
                for batch_no, test_batch in enumerate(progress, start=1):
                    means, stdev = self._normalize_batch_for_non_stationary(test_batch)
                    samples, targets, eval_points, observed_points, observed_time = self.model.evaluate(
                        test_batch,
                        nsample,
                    )
                    samples = samples.permute(0, 1, 3, 2)
                    targets = targets.permute(0, 2, 1)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                    samples, targets = self._restore_batch_scale(samples, targets, means, stdev)

                    all_target.append(targets.cpu())
                    all_evalpoint.append(eval_points.cpu())
                    all_observed_point.append(observed_points.cpu())
                    all_observed_time.append(observed_time.cpu())
                    all_generated_samples.append(samples.cpu())
                    progress.set_postfix(batch_no=batch_no)

        return (
            torch.cat(all_target, dim=0),
            torch.cat(all_evalpoint, dim=0),
            torch.cat(all_observed_point, dim=0),
            torch.cat(all_observed_time, dim=0),
            torch.cat(all_generated_samples, dim=0),
        )

    def test(self, test=0):
        if test == 0:
            self.load_model()
        else:
            self.load_model(self.config["training_config"]["model_path"])

        self.process_data()
        output = self.predict(self.dataloader["test"], self.config["model_config"]["nsample"])
        metrics = self._compute_metrics(output, target_dim=self.real_target_dim)
        self.output_save(output, target_dim=self.real_target_dim)

        print(f"Test MAE: {metrics['mae']}")
        print(f"Test RMSE: {metrics['rmse']}")
        print(f"Test MAPE: {metrics['mape']}")
        print(f"Test R2: {metrics['r2']}")
        print(f"Test Spearman: {metrics['spearman']}")
        print(f"Test Pearson: {metrics['pearson']}")
        print(f"Test SMAPE: {metrics['smape']}")
        return output

    def _compute_metrics(self, output, target_dim):
        return {
            "mae": self.evaluate(MAE_torch, *output, self.mean, self.std, target_dim=target_dim),
            "rmse": self.evaluate(RMSE_torch, *output, self.mean, self.std, target_dim=target_dim),
            "mape": self.evaluate(MAPE_torch, *output, self.mean, self.std, target_dim=target_dim),
            "r2": self.evaluate(R2_torch, *output, self.mean, self.std, target_dim=target_dim),
            "spearman": self.evaluate(SPEARMAN_torch, *output, self.mean, self.std, target_dim=target_dim),
            "pearson": self.evaluate(PEARSON_torch, *output, self.mean, self.std, target_dim=target_dim),
            "smape": self.evaluate(SMAPE_torch, *output, self.mean, self.std, target_dim=target_dim),
        }

    def evaluate(self, metric_func, *data, target_dim=10):
        all_target, _, _, _, all_generated_samples = data[:5]
        mean = data[5] if len(data) > 5 else self.mean
        std = data[6] if len(data) > 6 else self.std

        targets = all_target * std + mean
        samples = all_generated_samples * std + mean

        pred_length = int(self.config["data_config"]["unit_len"] * self.config["data_config"]["pred_len"])
        if target_dim == 1:
            targets = targets[:, -pred_length:, 0]
            samples = samples[:, :, -pred_length:, 0]
        else:
            targets = targets[:, -pred_length:, :]
            samples = samples[:, :, -pred_length:, :]

        pred_values = torch.quantile(samples, 0.5, dim=1) if samples.shape[1] > 1 else samples.squeeze(1)

        if metric_func in [MAE_torch, RMSE_torch, MAPE_torch, R2_torch, SPEARMAN_torch, PEARSON_torch, SMAPE_torch]:
            return metric_func(pred_values.contiguous(), targets.contiguous())
        if metric_func in [MAE_np, RMSE_np, MAPE_np, R2_np, SPEARMAN_np, PEARSON_np, SMAPE_np]:
            return metric_func(pred_values.cpu().numpy(), targets.cpu().numpy())
        return metric_func(*data)

    def output_save(self, output, target_dim=10):
        all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples = output
        if self.config["training_config"]["inverse"]:
            targets = all_target * self.std + self.mean
            samples = all_generated_samples * self.std + self.mean
        else:
            targets = all_target
            samples = all_generated_samples

        pred_len = int(self.config["data_config"]["unit_len"] * self.config["data_config"]["pred_len"])
        targets = targets[:, -pred_len:, :]
        samples = samples[:, :, -pred_len:, :]
        targets = targets.reshape(-1, target_dim).numpy()

        nsample = self.config["model_config"]["nsample"]
        quantile_mid = torch.quantile(samples, 0.5, dim=1).reshape(-1, target_dim).numpy()
        samples = np.transpose(samples, (0, 2, 1, 3)).reshape(-1, nsample, target_dim)
        samples_concat = np.concatenate((quantile_mid[:, np.newaxis, :], samples), axis=1)
        samples_concat = samples_concat.reshape(-1, (nsample + 1) * target_dim)

        datas = np.concatenate((targets, samples_concat), axis=1)
        cols = [f"Real_{i}" for i in range(target_dim)] + [f"Prediction_{i}" for i in range(target_dim)]
        cols += [f"Sample_{sample_idx}_{dim_idx}" for sample_idx in range(nsample) for dim_idx in range(target_dim)]

        output_pd = pd.DataFrame(datas, columns=cols)
        output_pd = pd.concat((pd.DataFrame(self.test_index, columns=["timestamp"]), output_pd), axis=1)
        with open(os.path.join(self.foldername, "results.pkl"), "wb") as file_obj:
            pickle.dump(output_pd, file_obj)

        with open(
            os.path.join(self.foldername, f"raw_results_nsample{self.config['model_config']['nsample']}.pkl"),
            "wb",
        ) as file_obj:
            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    self.std,
                    self.mean,
                ],
                file_obj,
            )

    def save_model(self, model, save_path):
        torch.save({"model": model.state_dict(), "mean": self.mean, "std": self.std}, save_path)

    def load_model(self, best_model_path=None):
        model_path = best_model_path or self.config["training_config"]["model_path"]
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Invalid model path: {model_path}")

        self.init_model()
        params = torch.load(model_path, map_location=self.device, weights_only=False)
        self.mean = params["mean"]
        self.std = params["std"]
        self.model.load_state_dict(params["model"])
        self.model.to(self.device)
