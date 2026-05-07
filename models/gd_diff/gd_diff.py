#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import pickle
import numpy as np
import torch
import pytz
import shutil
import os
from utils.metrics import MAE_np, RMSE_np, MAPE_np, SPEARMAN_np, PEARSON_np,R2_np, SMAPE_np
from utils.metrics import MAE_torch, RMSE_torch, MAPE_torch, R2_torch, SPEARMAN_torch, PEARSON_torch, SMAPE_torch
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from models.gd_diff.csdi_model_architecture import Forecasting
from data_provider.csdi_dataloader import Diffusion_Dataset
from utils.metrics import MAE_torch, RMSE_torch, MAPE_torch, MAE_np, RMSE_np, MAPE_np


def time_shift(time_str, shift):
    time0 = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time0 = time0 + timedelta(days=shift)
    return time0.strftime("%Y-%m-%d %H:%M:%S")

class gd_diff:
    def __init__(self, configs) -> None:
        if not isinstance(configs, dict):
            configs = dict(configs)
        
        if 'root' not in configs:
            configs['root'] = './'
        if 'config_path' not in configs:
            configs['config_path'] = 'conf/baseline/csdi_UTS.yaml'
        
        if 'training_config' not in configs:
            configs['training_config'] = configs
        if 'data_config' not in configs:
            configs['data_config'] = configs
        if 'model_config' not in configs:
            configs['model_config'] = configs

        
        if 'all' in configs['data_config']['target_cols']:
            configs = self.preprocess_data(configs)
        self.config = configs
        seed = configs['training_config']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = None

        self.mean = None
        self.std = None

        self.foldername = None

        self.target_dim = len(configs['data_config']["columns"])
        self.real_target_dim=len(configs['data_config']["target_cols"])
        self.device = configs['training_config']["device"]

        self.dataloader = {"train": None, "valid": None, "test": None}

        self.data_info = []
        self.norm_info = []
        self.data_process = []

        self.roll_run = configs['training_config']["roll_run"]
        self.roll_step = configs['training_config']["roll_step"]

        self.test_index = None

        self.is_non_stationary = configs['data_config']['is_non_stationary']
        self.non_stationary_type = configs['data_config']['non_stationary_type']
        self.past_len = configs['data_config']["past_len"] * configs['data_config']['unit_len']
        
        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H:%M:%S")
        foldername = (
            self.config['training_config']["save_path"]
            + self.config['data_config']["data_path"].split("/")[-2] + "/"
            + current_time
            + "/"
        )
        print("model folder:", foldername)
        os.makedirs(foldername, exist_ok=True)
        self.foldername = foldername

        def copy_file_to_folder(file_path, folder_name):
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(folder_name, file_name)
            shutil.copy(file_path, destination_path)

        copy_file_to_folder(os.path.join(configs['root'], configs['config_path']), self.foldername)

    def init_model(self) -> None:
        self.model = Forecasting(self.config, self.device, self.target_dim).to(self.config['training_config']["device"])

        def print_model_info(model):
            total_params = 0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    total_params += num_params
            print(f"Total Trainable Parameters: {total_params}")

        print_model_info(self.model)
    def preprocess_data(self, configs) -> None:
        file_path = configs['data_config']["data_path"]
        if file_path.endswith('.csv'):
            raw_dataset = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            raw_dataset = pd.read_parquet(file_path)
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            raw_dataset = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        raw_dataset.set_index('timestamp',inplace=True)
        raw_dataset.index = pd.to_datetime(raw_dataset.index)
        if 'all' in configs['data_config']['target_cols']:
            configs['data_config']['target_cols'] = raw_dataset.columns.tolist()
            configs['data_config']['columns'] = raw_dataset.columns.tolist()
            configs['target_dim'] = len(configs['data_config']['target_cols'])
            configs['real_target_dim'] = len(configs['data_config']['target_cols'])
            configs['data_config']['feature_cols_mask2D'] = []
            configs['data_config']['feature_cols_mask1D'] = []
            configs['data_config']['feature_cols_da'] = []
        return configs
    def process_data(self, rolls_step=0) -> None:
        file_path = self.config['data_config']["data_path"]
        if file_path.endswith('.csv'):
            raw_dataset = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            raw_dataset = pd.read_parquet(file_path)
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            raw_dataset = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        raw_dataset.set_index('timestamp',inplace=True)
        raw_dataset.index = pd.to_datetime(raw_dataset.index)

        raw_dataset = raw_dataset.reindex(self.config['data_config']['target_cols']+
                                        self.config['data_config']['feature_cols_mask2D']+
                                        self.config['data_config']['feature_cols_mask1D']+
                                        self.config['data_config']['feature_cols_da'], axis=1)
  
        config_temp = self.config

        config_temp['data_config']["data_len"]["shift"] = (
            self.config['data_config']["data_len"]["shift"] - rolls_step
        )
        config_temp['data_config']["data_time"]["train"]["start"] = time_shift(
            self.config['data_config']["data_time"]["train"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["train"]["end"] = time_shift(
            self.config['data_config']["data_time"]["train"]["end"], rolls_step
        )
        config_temp['data_config']["data_time"]["valid"]["start"] = time_shift(
            self.config['data_config']["data_time"]["valid"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["valid"]["end"] = time_shift(
            self.config['data_config']["data_time"]["valid"]["end"], rolls_step
        )
        config_temp['data_config']["data_time"]["test"]["start"] = time_shift(
            self.config['data_config']["data_time"]["test"]["start"], rolls_step
        )
        config_temp['data_config']["data_time"]["test"]["end"] = time_shift(
            self.config['data_config']["data_time"]["test"]["end"], rolls_step
        )

        unit_len = config_temp['data_config']["unit_len"]
        if config_temp['data_config']["data_mode"] == 0:
            end_time = config_temp['data_config']["data_len"]["end"]
            processed_dataset = raw_dataset.loc[:end_time]
            if config_temp['data_config']["data_len"]["train"] == -1:
                if config_temp['data_config']["data_len"]["shift"] != 0:
                    processed_dataset = processed_dataset.iloc[
                        : -config_temp['data_config']["data_len"]["shift"] * unit_len
                    ]
            else:
                total_len = (
                    config_temp['data_config']["data_len"]["train"]
                    + config_temp['data_config']["data_len"]["valid"]
                    + config_temp['data_config']["data_len"]["test"]
                    + config_temp['data_config']["data_len"]["shift"]
                ) * unit_len

                shift = config_temp['data_config']["data_len"]["shift"]
                if shift != 0:
                    processed_dataset = processed_dataset.iloc[
                        -total_len : -shift * unit_len
                    ]
                else:
                    processed_dataset = processed_dataset.iloc[-total_len:]

                train_temp_len = (
                    total_len - config_temp['data_config']["data_len"]["train"] * unit_len
                )
                train_dataset = processed_dataset.iloc[-total_len:-train_temp_len]

            test_start = (
                len(processed_dataset)
                - (
                    config_temp['data_config']["data_len"]["test"] +
                    config_temp['data_config']["data_len"]["shift"]
                )
                * unit_len
            )
            test_end = (
                len(processed_dataset)
                - (
                    config_temp['data_config']["data_len"]["shift"]
                )
                * unit_len
            )
            self.test_index = processed_dataset.iloc[test_start:test_end].index
        elif config_temp['data_config']["data_mode"] == 1:
            times = [
                config_temp['data_config']["data_time"]["train"]["start"],
                config_temp['data_config']["data_time"]["train"]["end"],
                config_temp['data_config']["data_time"]["valid"]["start"],
                config_temp['data_config']["data_time"]["valid"]["end"],
                config_temp['data_config']["data_time"]["test"]["start"],
                config_temp['data_config']["data_time"]["test"]["end"],
            ]
            times_datetime = [
                datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in times
            ]
            earliest_time = min(times_datetime)
            latest_time = max(times_datetime)
            processed_dataset = raw_dataset.loc[earliest_time:latest_time]

            train_dataset = processed_dataset.loc[
                config_temp['data_config']["data_time"]["train"][
                    "start"
                ] : config_temp['data_config']["data_time"]["train"]["end"]
            ]

            self.test_index = processed_dataset.loc[
                config_temp['data_config']["data_time"]["test"][
                    "start"
                ] : config_temp['data_config']["data_time"]["test"]["end"]
            ].index
        elif config_temp['data_config']["data_mode"] == 2:
            processed_dataset = raw_dataset.copy()
            total_len = len(processed_dataset)
            train_ratio, val_ratio, test_ratio = config_temp['data_config']["train_val_test_ratio"]
            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len

            train_dataset = processed_dataset.iloc[:train_len]
            self.test_index = processed_dataset.iloc[train_len+val_len:].index
        elif config_temp['data_config']["data_mode"] == 3:
            train_dataset = processed_dataset.iloc[-config_temp['data_config']["data_len"]["train"]*unit_len:]
            self.test_index = processed_dataset.iloc[-config_temp['data_config']["data_len"]["train"]*unit_len-config_temp['data_config']["data_len"]["valid"]*unit_len-config_temp['data_config']["data_len"]["test"]*unit_len:].index
        processed_dataset = self.normalize(train_dataset, processed_dataset)

        self.generate_dataloader(processed_dataset, config_temp)

    def normalize(self, train_dataset, processed_dataset) -> None:
        self.data_process.append("Normalization")

        if self.config['training_config']["model_path"] == "":
            col_mean = np.mean(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std = np.std(train_dataset.to_numpy(), axis=0, keepdims=True)
            col_std[col_std == 0] = 1

            self.model.mean = col_mean
            self.model.std = col_std

        else:
            col_mean = self.model.mean
            col_std = self.model.std

        self.mean = self.model.mean
        self.std = self.model.std

        return (processed_dataset - col_mean) / col_std

    def generate_dataloader(self, processed_dataset, config):
        train_dataset = Diffusion_Dataset(config, processed_dataset, "train")
        valid_dataset = Diffusion_Dataset(config, processed_dataset, "valid")
        test_dataset = Diffusion_Dataset(config, processed_dataset, "test")

        self.dataloader["train"] = DataLoader(
            train_dataset,
            batch_size=config['training_config']["batch_size"],
            shuffle=True,
        )
        self.dataloader["valid"] = DataLoader(
            valid_dataset,
            batch_size=config['training_config']["batch_size"],
            shuffle=False,
        )
        self.dataloader["test"] = DataLoader(
            test_dataset, batch_size=config['training_config']["batch_size"], shuffle=False
        )

    def train(self) -> None:
        valid_epoch_interval = 1

        for k in range(self.roll_run + 1):
            print(f"-----------第{k}次训练和推理-------------")

            self.init_model()
            self.process_data(k)

            config_temp = self.config['training_config']
            optimizer = Adam(
                self.model.parameters(),
                lr=config_temp["lr"],
                weight_decay=1e-6,
            )
            p1 = int(config_temp["p1"])
            p2 = int(config_temp["p2"])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=config_temp["lr_gamma"]
            )

            if self.foldername != "":
                output_path = self.foldername + f"model_{k}.pth"
                best_path = (
                    self.foldername + f"best_model_{k}.pth"
                )
                tensorboard_dir = os.path.join(self.foldername, f"tensorboard_run_{k}")
                os.makedirs(tensorboard_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"TensorBoard logs saved to: {tensorboard_dir}")
                print(f"View with: tensorboard --logdir={tensorboard_dir}")

            train_losses = []
            valid_losses = []

            best_valid_loss = 1e10
            patience = config_temp["patience"]
            patience_counter = 0
            for epoch_no in range(config_temp["epochs"]):
                self.now_epoch_no = epoch_no
                avg_loss = 0
                self.model.train()

                with tqdm(
                    self.dataloader["train"], mininterval=5.0, maxinterval=50.0
                ) as it:
                    for batch_no, train_batch in enumerate(it, start=1):
                        optimizer.zero_grad()
                        if self.is_non_stationary:
                            observed_data = train_batch['observed_data']  # [batch_size, seq_len, feature_dim]
                            means = torch.mean(observed_data[:, :self.past_len, :], dim=1, keepdim=True).detach()  # [batch_size, 1, feature_dim]
                            observed_data_normalized = observed_data - means
                            
                            if self.non_stationary_type == 0:
                                stdev = torch.ones_like(means)
                            else:
                                stdev = torch.sqrt(torch.var(observed_data[:, :self.past_len, :], dim=1, keepdim=True, unbiased=False)).detach()
                                stdev = torch.where(stdev == 0, torch.ones_like(stdev), stdev)
                            
                            observed_data_normalized = observed_data_normalized / stdev
                            
                            train_batch['observed_data'] = observed_data_normalized                       
                        loss = self.model(train_batch)
                        
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss detected at epoch {epoch_no}, batch {batch_no}")
                            continue
                        
                        loss.backward()
                        
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        avg_loss += loss.item()
                        optimizer.step()
                        
                        if self.foldername != "":
                            global_step = epoch_no * len(self.dataloader["train"]) + batch_no
                            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                        
                        it.set_postfix(
                            ordered_dict={
                                "loss": loss.item(),
                                "avg_epoch_loss": avg_loss / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

                    lr_scheduler.step()

                avg_train_loss = avg_loss / batch_no
                train_losses.append(avg_train_loss)
                
                if self.foldername != "":
                    writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch_no)
                    writer.add_scalar('Train/LearningRate_Epoch', optimizer.param_groups[0]['lr'], epoch_no)

                if (
                    self.dataloader["valid"] is not None
                    and (epoch_no + 1) % valid_epoch_interval == 0
                ):
                    output = self.predict(
                        self.dataloader["valid"],
                        self.config['training_config']["valid_nsample"],
                    )
                    self.valid_test(output,target_dim=self.real_target_dim)
                    (
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        all_generated_samples,
                    ) = output
                    
                    mae = self.evaluate(
                        MAE_torch,
                        all_target, all_evalpoint, all_observed_point, 
                        all_observed_time, all_generated_samples,
                        self.mean, self.std, target_dim=self.real_target_dim
                    )
                    
                    rmse = self.evaluate(
                        RMSE_torch,
                        all_target, all_evalpoint, all_observed_point, 
                        all_observed_time, all_generated_samples,
                        self.mean, self.std, target_dim=self.real_target_dim
                    )

                    if self.foldername != "":
                        writer.add_scalar('Valid/MAE', mae, epoch_no)
                        writer.add_scalar('Valid/RMSE', rmse, epoch_no)

                    if best_valid_loss > mae:
                        best_valid_loss = mae
                        patience_counter = 0
                        print(
                            "\n best loss is updated to ",
                            mae,
                            "at",
                            epoch_no,
                        )
                        if self.foldername != "":
                            self.save_model(self.model, best_path)
                            writer.add_scalar('Valid/BestMAE', best_valid_loss, epoch_no)
                    else:
                        patience_counter += 1

                    if self.foldername != "":
                        writer.add_scalar('Valid/MAE', mae, epoch_no)
                        writer.add_scalar('Valid/RMSE', rmse, epoch_no)

                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch_no}, patience={patience} exhausted")
                        break

                valid_losses.append(mae)
            self.load_model(best_path)
            output = self.predict(self.dataloader["test"], self.config['model_config']["nsample"])
            self.output_save(output, target_dim=self.real_target_dim)
            (
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
            ) = output

            mae = self.evaluate(
                MAE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )

            rmse = self.evaluate(
                RMSE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )

            mape = self.evaluate(
                MAPE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )
            r2 = self.evaluate(
                R2_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )
            spearman = self.evaluate(
                SPEARMAN_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )
            pearson = self.evaluate(
                PEARSON_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )
            smape = self.evaluate(
                SMAPE_torch,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                all_generated_samples,
                self.mean,
                self.std,
                target_dim=self.real_target_dim
            )
            print(f"第{k}次测试集MAE：", mae)
            print(f"第{k}次测试集RMSE：", rmse)
            print(f"第{k}次测试集MAPE：", mape)
            print(f"第{k}次测试集R2：", r2)
            print(f"第{k}次测试集Spearman：", spearman)
            print(f"第{k}次测试集Pearson：", pearson)
            print(f"第{k}次测试集SMAPE：", smape)
            
            if self.foldername != "":
                writer.add_scalar('Test/MAE', mae, k)
                writer.add_scalar('Test/RMSE', rmse, k)
                writer.add_scalar('Test/MAPE', mape, k)
                writer.add_scalar('Test/R2', r2, k)
                writer.add_scalar('Test/Spearman', spearman, k)
                writer.add_scalar('Test/Pearson', pearson, k)
                writer.add_scalar('Test/SMAPE', smape, k)
                
                for epoch_idx, train_loss in enumerate(train_losses):
                    writer.add_scalar('Loss/Train', train_loss, epoch_idx)
                for epoch_idx, valid_loss in enumerate(valid_losses):
                    writer.add_scalar('Loss/Valid', valid_loss, epoch_idx)
                
                if len(train_losses) == len(valid_losses):
                    for epoch_idx, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
                        writer.add_scalars('Loss/Train_Valid_Comparison', {
                            'Train': train_loss,
                            'Valid': valid_loss
                        }, epoch_idx)
                
                writer.close()
                print(f"TensorBoard logs closed for run {k}")
            
            self.save_model(self.model, output_path)

            losses_path = self.foldername + f"losses_{k}.pkl"
            losses_data = {"train_losses": train_losses, "valid_losses": valid_losses}
            with open(losses_path, "wb") as f:
                pickle.dump(losses_data, f)

    def predict(self, dataloader, nsample):

        with torch.no_grad():
            self.model.eval()

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(dataloader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):

                    if self.is_non_stationary:
                        observed_data = test_batch['observed_data']  # [batch_size, seq_len, feature_dim]
                        means = torch.mean(observed_data[:, :self.past_len, :], dim=1, keepdim=True).detach()
                        observed_data_normalized = observed_data - means
                        
                        if self.non_stationary_type == 0:
                            stdev = torch.ones_like(means)
                        else:
                            stdev = torch.sqrt(torch.var(observed_data[:, :self.past_len, :], dim=1, keepdim=True, unbiased=False)).detach()
                            stdev = torch.where(stdev == 0, torch.ones_like(stdev), stdev)
                        
                        observed_data_normalized = observed_data_normalized / stdev
                        
                        test_batch['observed_data'] = observed_data_normalized


                    output = self.model.evaluate(test_batch, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = (
                        output
                    )
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    if self.is_non_stationary:
                        means = means.to(samples.device)
                        stdev = stdev.to(samples.device)
                        samples = samples * stdev.unsqueeze(1) + means.unsqueeze(1)
                        c_target = c_target * stdev + means

                    all_target.append(c_target.cpu())
                    all_evalpoint.append(eval_points.cpu())
                    all_observed_point.append(observed_points.cpu())
                    all_observed_time.append(observed_time.cpu())
                    all_generated_samples.append(samples.cpu())

                    it.set_postfix(
                        ordered_dict={
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        return (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        )
    def test(self,test=0):
        if test==0:
            self.load_model()
        else:
            self.load_model(self.config['training_config']['model_path'])
        self.process_data(0)
        dataloader=self.dataloader["test"]
        nsample=self.config['model_config']["nsample"]
        with torch.no_grad():
            self.model.eval()

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(dataloader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    if self.is_non_stationary:
                        observed_data = test_batch['observed_data']  # [batch_size, seq_len, feature_dim]
                        means = torch.mean(observed_data[:, :self.past_len, :], dim=1, keepdim=True).detach()
                        observed_data_normalized = observed_data - means
                        
                        if self.non_stationary_type == 0:
                            stdev = torch.ones_like(means)
                        else:
                            stdev = torch.sqrt(torch.var(observed_data[:, :self.past_len, :], dim=1, keepdim=True, unbiased=False)).detach()
                            stdev = torch.where(stdev == 0, torch.ones_like(stdev), stdev)
                        
                        observed_data_normalized = observed_data_normalized / stdev
                        
                        test_batch['observed_data'] = observed_data_normalized
                    output = self.model.evaluate(test_batch, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = (
                        output
                    )
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                    if self.is_non_stationary:
                        means = means.to(samples.device)
                        stdev = stdev.to(samples.device)
                        samples = samples * stdev.unsqueeze(1) + means.unsqueeze(1)
                        c_target = c_target * stdev + means
                    all_target.append(c_target.cpu())
                    all_evalpoint.append(eval_points.cpu())
                    all_observed_point.append(observed_points.cpu())
                    all_observed_time.append(observed_time.cpu())
                    all_generated_samples.append(samples.cpu())

                    it.set_postfix(
                        ordered_dict={
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)
        output=(all_target,all_evalpoint,all_observed_point,all_observed_time,all_generated_samples)
        self.output_save(output,target_dim=self.real_target_dim)
        return (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        )
    def evaluate(self, metric_func, *data, target_dim=10):
        all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples = data[:5]
        
        mean = data[5] if len(data) > 5 else self.mean
        std = data[6] if len(data) > 6 else self.std
        
        targets = all_target * std + mean
        samples = all_generated_samples * std + mean
        
        pred_length = int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len'])
        if target_dim == 1:
            targets = targets[:, -pred_length:, 0]
            samples = samples[:, :, -pred_length:, 0]
        else:
            targets = targets[:, -pred_length:, :]  # (B, pred_len, K)
            samples = samples[:, :, -pred_length:, :]  # (B, nsample, pred_len, K)
        
        if samples.shape[1] > 1:
            pred_values = torch.quantile(samples, 0.5, dim=1)  # (B, pred_len, K)
        else:
            pred_values = samples.squeeze(1)  # (B, pred_len, K)
        
        targets = targets.contiguous()
        pred_values = pred_values.contiguous()
        
        if metric_func in [MAE_torch, RMSE_torch, MAPE_torch, R2_torch, SPEARMAN_torch, PEARSON_torch, SMAPE_torch]:
            return metric_func(pred_values, targets)
        elif metric_func in [MAE_np, RMSE_np, MAPE_np, R2_np, SPEARMAN_np, PEARSON_np, SMAPE_np]:
            return metric_func(pred_values.cpu().numpy(), targets.cpu().numpy())
        else:
            return metric_func(*data)

    def valid_test(self, output, target_dim=10):
        (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        ) = output
        targets = all_target * self.std + self.mean
        samples = all_generated_samples * self.std + self.mean

        pred_len = int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len'])

        if target_dim == 1:
            targets = targets[:, -pred_len:, 0]
            targets = targets.reshape(-1).unsqueeze(1).numpy()
            samples = samples[:, :, -pred_len:, 0]
        else:
            targets = targets[:, -pred_len:, :]
            targets = targets.reshape(-1, target_dim).numpy()
            samples = samples[:, :, -pred_len:, :]

        qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles = []
        if target_dim == 1:
            for q in qlist:
                quantiles.append(torch.quantile(samples, q, dim=1).numpy())
            samples = np.transpose(samples, (0, 2, 1))
            samples = samples.reshape(-1, self.config['training_config']["valid_nsample"])
            samples = np.concatenate(
                (quantiles[4].reshape(-1)[:, np.newaxis], samples), axis=1
            )
            pred_values = samples[:, 0]
            mae = MAE_np(pred_values, targets.flatten())
            rmse = RMSE_np(pred_values, targets.flatten())
            r2 = R2_np(pred_values, targets.flatten())
            spearman = SPEARMAN_np(pred_values, targets.flatten())
            pearson = PEARSON_np(pred_values, targets.flatten())
            print(f"验证集去归一化MAE: {mae}, RMSE: {rmse}")

            datas = np.concatenate((targets, samples), axis=1)
            cols = ["Real", "Prediction"] + [
                f"Sample_{x}" for x in range(self.config['training_config']["valid_nsample"])
            ]
        else:
            quantiles = []
            for q in qlist:
                quantiles.append(torch.quantile(samples, q, dim=1).numpy())  # (B, pred_len, K)
            median_pred = quantiles[4].reshape(-1, target_dim)
            # (B, nsample, pred_len, K) -> (B, pred_len, nsample, K)
            samples = np.transpose(samples, (0, 2, 1, 3))
            # (B*pred_len, nsample, K)
            samples = samples.reshape(-1, self.config['training_config']["valid_nsample"], target_dim)
            samples_concat = np.concatenate((median_pred, samples[:, :, 0]), axis=1) if samples.shape[2] == 1 else np.concatenate((median_pred, samples.reshape(-1, self.config['training_config']["valid_nsample"]*target_dim)), axis=1)
            pred_values = median_pred  # (B*pred_len, K)
            mae = MAE_np(pred_values, targets)
            rmse = RMSE_np(pred_values, targets)
            r2 = R2_np(pred_values, targets)
            spearman = SPEARMAN_np(pred_values, targets)
            pearson = PEARSON_np(pred_values, targets)
            print(f"验证集去归一化MAE: {mae}, RMSE: {rmse}")

            datas = np.concatenate((targets, median_pred), axis=1)
            cols = ["Real_" + str(i) for i in range(target_dim)] + ["Prediction_" + str(i) for i in range(target_dim)]

        output_pd = pd.DataFrame(datas, columns=cols)
        times = pd.DataFrame(range(len(targets)), columns=["timestamp"])
        output_pd = pd.concat((times, output_pd), axis=1)
        with open(self.foldername + f"valid_results_{self.now_epoch_no}.pkl", 'wb') as file:
            pickle.dump(output_pd, file)
    def output_save(self, output, target_dim=10) -> None:
        (
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            all_generated_samples,
        ) = output
        if self.config['training_config']['inverse']:
            targets = all_target * self.std + self.mean
            samples = all_generated_samples * self.std + self.mean
        else:
            targets = all_target
            samples = all_generated_samples

        if target_dim == 1:
            targets = targets[:, -int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len']):, 0]
            targets = targets.reshape(-1).unsqueeze(1).numpy()
            samples = samples[:, :, -int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len']):, 0]
        else:
            targets = targets[:, -int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len']):, :]
            targets = targets.reshape(-1, target_dim).numpy()
            samples = samples[:, :, -int(self.config['data_config']["unit_len"] * self.config['data_config']['pred_len']):, :]
            # samples shape: (B, nsample, L, K)

        qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles = []
        if target_dim == 1:
            for q in qlist:
                quantiles.append(torch.quantile(samples, q, dim=1).numpy())
            samples = np.transpose(samples, (0, 2, 1))
            samples = samples.reshape(-1, self.config['model_config']["nsample"])
            samples = np.concatenate(
                (quantiles[4].reshape(-1)[:, np.newaxis], samples), axis=1
            )
            pred_values = samples[:, 0]
            mae = MAE_np(pred_values, targets.flatten())
            rmse = RMSE_np(pred_values, targets.flatten())
            smape = SMAPE_np(pred_values, targets.flatten())
            r2 = R2_np(pred_values, targets.flatten())
            spearman = SPEARMAN_np(pred_values, targets.flatten())
            pearson = PEARSON_np(pred_values, targets.flatten())
            print(f"测试集MAE: {mae}, RMSE: {rmse}")

            datas = np.concatenate((targets, samples), axis=1)
            cols = ["Real", "Prediction"] + [
                f"Sample_{x}" for x in range(self.config['model_config']["nsample"])
            ]
        else:
            nsample = self.config['model_config']["nsample"]
            K = samples.shape[-1]
            for q in qlist:
                quantiles.append(torch.quantile(samples, q, dim=1).numpy())
            samples = np.transpose(samples, (0, 2, 1, 3))  # (B, L, nsample, K)
            samples = samples.reshape(-1, nsample, K)  # (B*L, nsample, K)
            quantile_mid = quantiles[4].reshape(-1, K)  # (B*L, K)
            samples_np = samples  # (B*L, nsample, K)
            samples_concat = np.concatenate((quantile_mid[:, np.newaxis, :], samples_np), axis=1)  # (B*L, nsample+1, K)
            samples_concat = samples_concat.reshape(-1, (nsample + 1) * K)
            pred_values = quantile_mid  # (B*L, K)
            mae = MAE_np(pred_values, targets)
            rmse = RMSE_np(pred_values, targets)
            smape = SMAPE_np(pred_values, targets)
            r2 = R2_np(pred_values, targets)
            spearman = SPEARMAN_np(pred_values, targets)
            pearson = PEARSON_np(pred_values, targets)
            print(f"测试集MAE: {mae}, RMSE: {rmse}")

            datas = np.concatenate((targets, samples_concat), axis=1)
            cols = [f"Real_{i}" for i in range(K)] + \
                   [f"Prediction_{i}" for i in range(K)] + \
                   [f"Sample_{x}_{i}" for x in range(nsample) for i in range(K)]

        output_pd = pd.DataFrame(datas, columns=cols)
        times = pd.DataFrame(self.test_index, columns=["timestamp"])
        output_pd = pd.concat((times, output_pd), axis=1)
        with open(self.foldername + "results.pkl", 'wb') as file:
            pickle.dump(output_pd, file)

        with open(
            self.foldername
            + "raw_results_nsample"
            + str(self.config['model_config']["nsample"])
            + ".pkl",
            "wb",
        ) as f:
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
                f,
            )

    def save_model(self, model, save_path) -> None:
        mean = self.mean
        std = self.std
        param = {
            'model': model.state_dict(),
            'mean': mean,
            'std': std
        }
        torch.save(param, save_path)

    def load_model(self, best_model_path=None) -> None:
        model_path = self.config['training_config']["model_path"]
        
        if best_model_path is not None:
            model_path = best_model_path
            print(f"训练结束，加载最优模型: {model_path}")
        elif model_path is None or not os.path.exists(model_path):
            raise ValueError(f"模型路径无效: {model_path}")
        self.model = Forecasting(self.config, self.device, self.target_dim).to(self.config['training_config']["device"])
        param = torch.load(model_path, map_location=self.config['training_config']["device"], weights_only=False)
        self.model.mean = param['mean']
        self.model.std = param['std']
        self.model.load_state_dict(param['model'])
        self.model.to(self.config['training_config']["device"])



