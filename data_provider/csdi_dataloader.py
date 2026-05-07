#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : diffusion_dataset
# @Time          : 2024-11-20 11:07:32
# @Author        : ZhangHonglin
# @Email         : honglin-24@mails.tsinghua.edu.cn
# @description   : 
"""

import numpy as np
import pandas as pd
import os
import glob

from torch.utils.data import Dataset
from datetime import datetime, timedelta
from itertools import combinations
import torch
import torch.nn as nn

class Diffusion_Dataset(Dataset):
    def __init__(self, config, data, mode, job_data=None, job_data_info=None,nodetype_to_idx=None):
        unit_len = config['data_config']["unit_len"]
        self.pred_length = config['data_config']["pred_len"]*unit_len
        self.past_length = config['data_config']["past_len"] * unit_len
        self.target_dim=config['data_config']["target_cols"]
        self.seq_length = self.past_length + self.pred_length
        self.kill_dim = len(config['data_config']["feature_cols_mask2D"])
        self.kill_dim1D = len(config['data_config']["feature_cols_mask1D"])
        self.dim = len(config['data_config']["columns"])
        self.job_aware = config['data_config'].get('job_aware', 0)
        self.job_attention = config['model_config'].get('job_attention', 0)
        self.graph_path = config['data_config'].get('graph_path', None)
        self.graph_dir = config['data_config'].get('graph_dir', None)
        self.graph_type = config['model_config'].get('graph',{}).get('graph_type', 'None')
        self.graph_freq = config['model_config'].get('graph',{}).get('graph_freq', '1d')
        self.real_time_emb=config['model_config']['diffusion'].get('real_time_emb_dim', 0)
        if self.job_attention:
            self.job_data_id = job_data.columns
            self.job_data = job_data
        self.job_version = config['data_config'].get('job_version', 0)
        if self.job_aware:
            self.job_data_info = job_data_info
            self.job_data_info['numcores']=self.job_data_info['numcores']/48
            self.job_data=job_data
            self.nodetype_to_idx = nodetype_to_idx
        
        # 预计算job_id到(numcores, nodetype_idx)的映射，用于加速
            self.job_id_to_info = {}
            if self.job_aware and self.job_data_info is not None:
                for _, row in self.job_data_info.iterrows():
                    job_id = str(row['id'])
                    numcores = row['numcores']
                    nodetype = row['nodetypes']
                    nodetype_tuple = tuple(sorted([nodetype]))  # 单个nodetype也转为tuple
                    nodetype_idx = self.nodetype_to_idx.get(nodetype_tuple, 0)
                    self.job_id_to_info[job_id] = (numcores, nodetype_idx, nodetype)

        if config['data_config']["data_mode"] == 1:
            if mode == "train" or mode == "examine":
                start = config['data_config']["data_time"]["train"]["start"]
                end = config['data_config']["data_time"]["train"]["end"]
            else:
                # 直接获取past_length * unit_len长度的历史数据
                start_idx = data.index.get_loc(config['data_config']["data_time"][mode]["start"])
                start_idx = max(0, start_idx - self.past_length )
                start = data.index[start_idx].strftime("%Y-%m-%d %H:%M:%S")
                end = config['data_config']["data_time"][mode]["end"]

        if mode == "train" or mode == "examine":
            if config['data_config']["data_len"]["train"] == -1:
                start_len = len(data)
                end_len = (
                    config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
            else:
                start_len = (
                    config['data_config']["data_len"]["train"]
                    + config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
                end_len = start_len - config['data_config']["data_len"]["train"] * unit_len
        elif mode == "valid":
            start_len = (
                config['data_config']["data_len"]["valid"]
                + config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["valid"] * unit_len
                - self.past_length
            )
        else:
            start_len = (
                config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["test"] * unit_len
                - self.past_length
            )

        if config['data_config']["data_mode"] == 0:
            if end_len == 0:
                self.main_data = data.iloc[-start_len:]
            else:
                self.main_data = data.iloc[-start_len:-end_len]
            self.job_data = self.job_data.iloc[-start_len:-end_len]
            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 1:
            self.main_data = data.loc[start:end]
            if self.job_attention:
                self.job_data = self.job_data.loc[start:end]
            else:
                self.job_data = None
            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 2:
            # 使用比例划分训练集、验证集和测试集
            total_len = len(data)
            train_ratio, val_ratio, test_ratio = config['data_config']["train_val_test_ratio"]
            
            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len  # 剩余部分作为测试集
            
            if mode == "train":
                self.main_data = data.iloc[:train_len]

                if self.job_aware or self.job_attention:
                    self.job_data = self.job_data.iloc[:train_len]
            elif mode == "valid":
                self.main_data = data.iloc[train_len:train_len+val_len]
                if self.job_aware or self.job_attention:
                    self.job_data = self.job_data.iloc[train_len:train_len+val_len]
            elif mode == "test":
                self.main_data = data.iloc[train_len+val_len:]
                if self.job_aware or self.job_attention:
                    self.job_data = self.job_data.iloc[train_len+val_len:]
            else:
                self.main_data = data
                if self.job_aware or self.job_attention:
                    self.job_data = self.job_data
            step = config['data_config']["training_step"]
        else:
            self.main_data = data.iloc[-start_len:]
            if self.job_aware or self.job_attention:
                self.job_data = self.job_data.iloc[-start_len:]
            step = config['data_config']["training_step"]

        # 此处认为所有的数据都被观测到了，因此mask都为1
        self.data_index = self.main_data.index
        self.mask_data = pd.DataFrame(np.ones_like(self.main_data.values)).to_numpy()
        self.main_data = self.main_data.to_numpy()
        if self.job_attention:
            self.job_data=self.job_data.to_numpy()
        if self.job_aware:
            self.job_data_id=self.job_data.columns
            self.job_data = self.job_data.to_numpy()
        if mode == "train":
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            self.use_index = np.arange(0, total - self.seq_length + step, step)
        elif  mode=='examine' or mode == 'valid':
        # elif mode == 'examine' or mode == 'test':
            step = unit_len
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            # self.use_index = np.arange(0, total - self.seq_length + step, step)
            self.use_index = np.arange(0, total- self.seq_length+1, unit_len)
            # self.use_index = np.arange(0, total- self.seq_length+1, 1)
        else:
            # unit_len=12
            index_end = len(self.main_data) - self.seq_length + 1
            # self.use_index = np.arange(0, index_end, unit_len)
            # self.use_index = np.arange(0, index_end, self.pred_length)
            self.use_index = np.arange(0, index_end, 1)
        # 日内数据掩码
        # 此处要对齐，数据结构为电价+日内数据+日前数据
        self.kill_mask = np.ones([self.seq_length, self.dim])
        self.kill_mask[- 2*self.pred_length :, 1 : (1 + self.kill_dim)] = 0
        self.kill_mask[- self.pred_length :, (1 + self.kill_dim) : (1 + self.kill_dim+self.kill_dim1D)] = 0
        if self.job_attention:
            self.kill_mask_job = np.ones([self.seq_length, self.job_data.shape[1]])
            self.kill_mask_job[- self.pred_length :, :] = 0.0

        # Configurable graph reference time and interval (for non-Alibaba datasets)
        graph_cfg = config['model_config'].get('graph', {})
        ref_str = graph_cfg.get('reference_time', '2018-01-01 00:00:00')
        self.graph_reference_time = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
        self.graph_interval = graph_cfg.get('graph_interval', 60)  # seconds

        # 静态图加载
        if self.graph_type == 'static':
            self.graph = np.load(self.graph_path)
            self.graph = torch.from_numpy(self.graph).float()
        elif self.graph_type == 'time':
            graph_pattern = os.path.join(self.graph_dir, 'adjacency_matrix_*.npy')
            graph_files = sorted(glob.glob(graph_pattern))
            self.graph=[torch.from_numpy(np.load(graph_file)).float() for graph_file in graph_files]
            print(f"成功加载 {len(self.graph)} 个时间图文件")

    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]  # 坐标转化
        target_mask = self.mask_data[
            index : index + self.seq_length
        ].copy()  # 目标掩码，告诉模型哪里是要预测的电价真值
        if self.target_dim == 1:
            target_mask[-self.pred_length :, 0] = 0.0
        else:
            target_mask[-self.pred_length :, :] = 0.0

        # 确保数据是连续的numpy数组，并且是float32类型
        observed_data = np.ascontiguousarray(self.main_data[index : index + self.seq_length] * self.kill_mask).astype(np.float32)
        observed_mask = np.ascontiguousarray(self.mask_data[index : index + self.seq_length]).astype(np.float32)
        gt_mask = np.ascontiguousarray(target_mask).astype(np.float32)
        timepoints = np.ascontiguousarray(np.arange(self.seq_length) * 1.0).astype(np.float32)
        feature_id = np.ascontiguousarray(np.arange(self.main_data.shape[1]) * 1.0).astype(np.float32)
        
        s = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "feature_id": feature_id,
        }
        if self.job_attention:
            observed_job_data = np.ascontiguousarray(self.job_data[index : index + self.seq_length]).astype(np.float32)
            observed_job_data = observed_job_data*self.kill_mask_job
            s['observed_job_data'] = observed_job_data
        if self.job_aware:
            observed_job_data = np.ascontiguousarray(self.job_data[index : index + self.pred_length+self.past_length]).astype(np.float32)
            
            if self.job_version == 2 or self.job_version == 3:
                # version 2/3: 分时间步处理，使用优化的批量编码方法
                numcores_array, nodetype_idx_array = self._encode_job_batch_fast(observed_job_data)
                s["total_numcores"] = numcores_array
                s["nodetype_combo_idx"] = nodetype_idx_array
            else:
                # version 0/1: 一次性获取整个时间段内的job
                total_numcores, nodetype_combo_idx = self._encode_job_at_timestep(observed_job_data)
                s["total_numcores"] = np.array([total_numcores], dtype=np.float32)
                s["nodetype_combo_idx"] = np.array([nodetype_combo_idx], dtype=np.int64)

        if self.graph_type == 'static':
            s["graph"] = self.graph
        elif self.graph_type == 'time':
            # 对于time类型的图，返回所有图的list
            graph_index = self.data_index[index : index + self.seq_length]
            if self.graph_freq == 'hour':
                reference_time = self.graph_reference_time
                hour_delta = int((graph_index[-self.pred_length] - reference_time).total_seconds() // 3600)
                s["graph"] = self.graph[hour_delta]
            elif self.graph_freq == 'minute':
                reference_time = self.graph_reference_time
                minute_deltas = [int((ts - reference_time).total_seconds() // self.graph_interval) for ts in graph_index]
                graph_list = [self.graph[minute_delta] for minute_delta in minute_deltas]
                # 将最后pred_length张图置为0
                graph_shape = graph_list[0].shape
                for i in range(-self.pred_length, 0):
                    graph_list[i] = torch.zeros_like(graph_list[i])
                s["graph"] = torch.stack(graph_list, dim=0)
            elif self.graph_freq == 'day':
                s["graph"] = self.graph[graph_index[0].day]
        if self.real_time_emb!=0:
            time_index = self.data_index[index : index + self.seq_length]
            # 提取day、hour、minute
            s["day"] = np.ascontiguousarray([ts.day for ts in time_index]).astype(np.int32)
            s["hour"] = np.ascontiguousarray([ts.hour for ts in time_index]).astype(np.int32)
            s["minute"] = np.ascontiguousarray([ts.minute for ts in time_index]).astype(np.int32)
        return s

    def __len__(self):
        return len(self.use_index)
    
    def _encode_job_at_timestep(self, job_data_slice):
        """
        对单个时间步或时间段的job数据进行编码
        
        Args:
            job_data_slice: 1D array for single timestep or 2D array for time range
            
        Returns:
            tuple: (numcores_sum, nodetype_combo_idx)
        """
        # 找出正在进行的job
        if job_data_slice.ndim == 1:
            # 单个时间步
            not_nan_cols = ~np.isnan(job_data_slice)
        else:
            # 多个时间步
            not_nan_cols = ~np.all(np.isnan(job_data_slice), axis=0)
        
        valid_cols_indices = np.where(not_nan_cols)[0]
        valid_job_ids = self.job_data_id[valid_cols_indices]
        
        # 初始化默认值
        numcores_sum = 0
        nodetype_combo_idx = 0  # 空集对应索引0
        
        # 从job_data_info中获取对应id的nodetypes和numcores
        if self.job_data_info is not None and len(valid_job_ids) > 0:
            valid_job_ids_list = valid_job_ids.tolist()
            job_info_subset = self.job_data_info[self.job_data_info['id'].astype(str).isin(valid_job_ids_list)]
            numcores_sum = job_info_subset['numcores'].sum()
            unique_nodetypes = sorted(job_info_subset['nodetypes'].unique().tolist())
            
            # 将unique_nodetypes转换为索引
            nodetype_tuple = tuple(unique_nodetypes)
            if nodetype_tuple in self.nodetype_to_idx:
                nodetype_combo_idx = self.nodetype_to_idx[nodetype_tuple]
            else:
                # 如果出现未见过的组合，默认使用0（空集）
                nodetype_combo_idx = 0
        
        return numcores_sum, nodetype_combo_idx
    
    def _encode_job_batch_fast(self, observed_job_data):
        """
        快速批量编码所有时间步的job数据（优化版本）
        使用预计算的字典，避免重复的Pandas查询
        
        Args:
            observed_job_data: (T, num_jobs) 2D array
            
        Returns:
            numcores_array: (T,) array
            nodetype_idx_array: (T,) array
        """
        T = observed_job_data.shape[0]
        numcores_array = np.zeros(T, dtype=np.float32)
        nodetype_idx_array = np.zeros(T, dtype=np.int64)
        
        # 对每个时间步进行处理（使用字典查询，比Pandas快很多）
        for t in range(T):
            # 找出当前时间步非NaN的job（正在运行的job）
            not_nan_mask = ~np.isnan(observed_job_data[t, :])
            if not np.any(not_nan_mask):
                continue  # 如果没有运行的job，保持默认值0
            
            valid_job_ids = self.job_data_id[not_nan_mask]
            
            # 使用预计算的字典查询（避免Pandas操作）
            total_numcores = 0
            nodetypes_set = set()
            
            for job_id in valid_job_ids:
                job_id_str = str(job_id)
                if job_id_str in self.job_id_to_info:
                    numcores, _, nodetype = self.job_id_to_info[job_id_str]
                    total_numcores += numcores
                    nodetypes_set.add(nodetype)
            
            # 编码nodetype组合
            if nodetypes_set:
                nodetype_tuple = tuple(sorted(nodetypes_set))
                nodetype_idx_array[t] = self.nodetype_to_idx.get(nodetype_tuple, 0)
                numcores_array[t] = total_numcores
        
        return numcores_array, nodetype_idx_array
    
    def get_nodetype_vocab_size(self):
        """返回nodetype组合的总数（用于创建embedding层）"""
        if self.nodetype_combinations is not None:
            return len(self.nodetype_combinations)
        return 0
    
    def get_nodetype_combinations(self):
        """返回所有nodetype组合列表"""
        return self.nodetype_combinations if self.nodetype_combinations is not None else []


class Diffusion_Dataset_MultiScale(Dataset):
    def __init__(self, config, data, mode):
        unit_len = config['data_config']["unit_len"]
        self.pred_length = config['data_config']["pred_len"]*unit_len
        self.past_length = config['data_config']["past_len"] * unit_len
        self.target_dim=config['data_config']["target_cols"]
        self.seq_length = self.past_length + self.pred_length
        self.kill_dim = len(config['data_config']["feature_cols_mask2D"])
        self.kill_dim1D = len(config['data_config']["feature_cols_mask1D"])
        self.dim = len(config['data_config']["columns"])
        
        # 读取降采样参数
        downsample_points_raw = config['model_config'].get("downsample_points", [])
        self.downsample_points = [int(dp) for dp in downsample_points_raw if int(dp) > 1]
        self.downsample_method = config['model_config'].get("downsample_method", "mean")
        self.use_downsample=len(self.downsample_points) > 0
        
        if config['data_config']["data_mode"] == 1:
            if mode == "train" or mode == "examine":
                start = config['data_config']["data_time"]["train"]["start"]
                end = config['data_config']["data_time"]["train"]["end"]
            else:
                # 直接获取past_length * unit_len长度的历史数据
                start_idx = data.index.get_loc(config['data_config']["data_time"][mode]["start"])
                start_idx = max(0, start_idx - self.past_length )
                start = data.index[start_idx].strftime("%Y-%m-%d %H:%M:%S")
                end = config['data_config']["data_time"][mode]["end"]

        if mode == "train" or mode == "examine":
            if config['data_config']["data_len"]["train"] == -1:
                start_len = len(data)
                end_len = (
                    config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
            else:
                start_len = (
                    config['data_config']["data_len"]["train"]
                    + config['data_config']["data_len"]["valid"]
                    + config['data_config']["data_len"]["test"]
                    + config['data_config']["data_len"]["shift"]
                ) * unit_len
                end_len = start_len - config['data_config']["data_len"]["train"] * unit_len
        elif mode == "valid":
            start_len = (
                config['data_config']["data_len"]["valid"]
                + config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["valid"] * unit_len
                - self.past_length
            )
        else:
            start_len = (
                config['data_config']["data_len"]["test"]
                + config['data_config']["data_len"]["shift"]
            ) * unit_len + self.past_length
            end_len = (
                start_len
                - config['data_config']["data_len"]["test"] * unit_len
                - self.past_length
            )

        if config['data_config']["data_mode"] == 0:
            if end_len == 0:
                self.main_data = data.iloc[-start_len:]
            else:
                self.main_data = data.iloc[-start_len:-end_len]

            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 1:
            self.main_data = data.loc[start:end]

            step = config['data_config']["training_step"]
        elif config['data_config']["data_mode"] == 2:
            # 使用比例划分训练集、验证集和测试集
            total_len = len(data)
            train_ratio, val_ratio, test_ratio = config['data_config']["train_val_test_ratio"]
            
            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len  # 剩余部分作为测试集
            
            if mode == "train":
                self.main_data = data.iloc[:train_len]

            elif mode == "valid":
                self.main_data = data.iloc[train_len:train_len+val_len]

            elif mode == "test":
                self.main_data = data.iloc[train_len+val_len:]
            else:
                self.main_data = data
            step = config['data_config']["training_step"]
        else:
            self.main_data = data.iloc[-start_len:]
            step = config['data_config']["training_step"]

        # 此处认为所有的数据都被观测到了，因此mask都为1
        self.mask_data = pd.DataFrame(np.ones_like(self.main_data.values)).to_numpy()
        self.main_data = self.main_data.to_numpy()
        
        if mode == "train":
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            self.use_index = np.arange(0, total - self.seq_length + step, step)
        elif  mode=='examine' or mode == 'valid':
        # elif mode == 'examine' or mode == 'test':
            step = unit_len
            total = len(self.main_data)
            index_end = len(self.main_data) - self.seq_length + 1
            # self.use_index = np.arange(0, total - self.seq_length + step, step)
            self.use_index = np.arange(0, total- self.seq_length+1, unit_len)
            # self.use_index = np.arange(0, total- self.seq_length+1, 1)
        else:
            # unit_len=12
            index_end = len(self.main_data) - self.seq_length + 1
            # self.use_index = np.arange(0, index_end, unit_len)
            self.use_index = np.arange(0, index_end, 1)
        # 日内数据掩码
        # 此处要对齐，数据结构为电价+日内数据+日前数据
        self.kill_mask = np.ones([self.seq_length, self.dim])
        self.kill_mask[- 2*self.pred_length :, 1 : (1 + self.kill_dim)] = 0
        self.kill_mask[- self.pred_length :, (1 + self.kill_dim) : (1 + self.kill_dim+self.kill_dim1D)] = 0
    
    def _downsample_timeseries(self, data, downsample_points, method='mean'):
        """
        对时间序列数据进行降采样（类方法版本）
        将若干点聚合为一点，然后将该聚合值填充回这些点，保证长度不变
        
        Args:
            data: numpy数组，形状为 (time, feature)
            downsample_points: int，每多少个点聚合为一点
            method: str，降采样方法，可选 'mean'(平均), 'max'(最大), 'min'(最小)
        
        Returns:
            downsampled_data: 降采样后的数据，形状为 (time, feature)（长度不变）
        """
        if downsample_points <= 1:
            return data
        
        data_np = np.asarray(data)
        time_len, feature_dim = data_np.shape
        
        # 计算聚合后的点数
        num_groups = (time_len + downsample_points - 1) // downsample_points
        
        # 对每组点进行聚合
        aggregated_list = []
        for i in range(num_groups):
            start_idx = i * downsample_points
            end_idx = min(start_idx + downsample_points, time_len)
            window_data = data_np[start_idx:end_idx]  # (window_size, feature)
            
            if method == 'mean':
                aggregated_value = np.mean(window_data, axis=0)
            elif method == 'max':
                aggregated_value = np.max(window_data, axis=0)
            elif method == 'min':
                aggregated_value = np.min(window_data, axis=0)
            else:
                raise ValueError(f"不支持的降采样方法: {method}，可选: 'mean', 'max', 'min'")
            
            aggregated_list.append(aggregated_value)
        
        # 将聚合后的值填充回原来的位置，保证长度一致
        downsampled_data_list = []
        for i in range(num_groups):
            start_idx = i * downsample_points
            end_idx = min(start_idx + downsample_points, time_len)
            # 将该组的聚合值填充到该组的所有位置
            for j in range(start_idx, end_idx):
                downsampled_data_list.append(aggregated_list[i])
        
        # 如果最后一个组不足downsample_points个点，用最后一个聚合值填充
        while len(downsampled_data_list) < time_len:
            downsampled_data_list.append(aggregated_list[-1])
        
        downsampled_data_np = np.stack(downsampled_data_list, axis=0)  # (time, feature)
        
        return downsampled_data_np.astype(data_np.dtype)
    
    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]  # 坐标转化
        target_mask = self.mask_data[
            index : index + self.seq_length
        ].copy()  # 目标掩码，告诉模型哪里是要预测的电价真值
        if self.target_dim == 1:
            target_mask[-self.pred_length :, 0] = 0.0
        else:
            target_mask[-self.pred_length :, :] = 0.0

        # 确保数据是连续的numpy数组，并且是float32类型
        observed_data = np.ascontiguousarray(self.main_data[index : index + self.seq_length] * self.kill_mask).astype(np.float32)
        observed_mask = np.ascontiguousarray(self.mask_data[index : index + self.seq_length]).astype(np.float32)
        gt_mask = np.ascontiguousarray(target_mask).astype(np.float32)
        timepoints = np.ascontiguousarray(np.arange(self.seq_length) * 1.0).astype(np.float32)
        feature_id = np.ascontiguousarray(np.arange(self.main_data.shape[1]) * 1.0).astype(np.float32)

        s = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "feature_id": feature_id,
        }
        
        # 在get_item中进行多尺度降采样
        if self.use_downsample:
            # 获取当前序列的原始数据
            current_seq_data = self.main_data[index : index + self.seq_length]
            
            for dp in self.downsample_points:
                # 对当前序列进行降采样
                observed_data_downsampled = self._downsample_timeseries(
                    current_seq_data, dp, self.downsample_method
                )
                # 应用kill_mask
                observed_data_downsampled = np.ascontiguousarray(
                    observed_data_downsampled * self.kill_mask
                ).astype(np.float32)
                timepoints_downsampled = np.ascontiguousarray(
                    np.arange(self.seq_length) * 1.0
                ).astype(np.float32)
                
                # 使用尺度作为key添加到字典中
                s[f"observed_data_downsampled_{dp}"] = observed_data_downsampled
                s[f"timepoints_downsampled_{dp}"] = timepoints_downsampled
        
        return s
    def __len__(self):
        return len(self.use_index)

def time_shift(time_str, shift):
    time0 = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time0 = time0 + timedelta(days=shift)
    return time0.strftime("%Y-%m-%d %H:%M:%S")
