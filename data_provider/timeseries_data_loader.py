import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.FEDformer.units.timefeatures import time_features
from models.FEDformer.units.augmentation import run_augmentation_single
import pickle as pkl
import warnings


warnings.filterwarnings('ignore')



# 每隔1点滚动推理96点
class Dataset_common(Dataset):
    def __init__(self, args, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [past_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.past_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.past_len = size[0]*self.args.unit_len
            self.label_len = int(size[1]*self.args.unit_len)
            self.pred_len = size[2]*self.args.unit_len

        self.train_len=self.args.data_len['train']*self.args.unit_len
        self.valid_len=self.args.data_len['valid']*self.args.unit_len
        self.test_len=self.args.data_len['test']*self.args.unit_len
        # 此处认为所有的数据都被观测到了，因此mask都为1

        self.kill_dim = len(self.args.feature_cols_mask2D)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target_cols = args.target_cols #目标列
        self.feature_cols_da=args.feature_cols_da #日内特征
        self.feature_cols_mask1D=args.feature_cols_mask1D #日前特征
        self.feature_cols_mask2D=args.feature_cols_mask2D #日前特征
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 这里可能是pkl文件或者parquet文件
        if self.data_path.endswith('.pkl'):
            with open(self.data_path, 'rb') as file:
                df_raw = pkl.load(file)
        elif self.data_path.endswith('.parquet'):
            df_raw = pd.read_parquet(self.data_path)
        else:
            df_raw = pd.read_csv(self.data_path)
        # with open(self.data_path,'rb') as file:
        #     df_raw = pkl.load(file)
        
        #TimeXer 的目标列在最后一列
        df_raw.set_index('timestamp',inplace=True)
        df_raw.index = pd.to_datetime(df_raw.index)
        if 'all' in self.target_cols:
            self.target_cols = df_raw.columns.tolist()
            self.feature_cols_mask2D = []
            self.feature_cols_mask1D = []
            self.feature_cols_da = []
        df_raw=df_raw.reindex(columns=self.feature_cols_mask2D+self.feature_cols_mask1D+self.feature_cols_da+self.target_cols)
        df_raw=df_raw.reset_index()
    
        if self.args.data_mode==0:
            df_raw=df_raw.reset_index()
            df_raw=df_raw.loc[:self.args.data_len['end']]
            border1s = [0, self.train_len-self.past_len, self.train_len+self.valid_len-self.past_len] 
            border2s = [self.train_len, self.train_len+self.valid_len, self.train_len+self.valid_len+self.test_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            #计算可以被滚动的起始点index
            sample_interval = [self.pred_len//self.pred_len,self.pred_len//self.pred_len,self.pred_len//self.pred_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    

            #数据集划分
            border1s = [-self.past_len-self.test_len-self.valid_len-self.train_len, -self.past_len-self.test_len-self.valid_len, -self.test_len-self.past_len] 
            border2s = [-self.test_len-self.valid_len, -self.test_len, None]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]  
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==1:
            df_raw=df_raw.set_index('timestamp')
            # df_raw.index.name='date'
            train_row_start = df_raw.index.get_loc(self.args.data_time.train.start)
            train_row_end = df_raw.index.get_loc(self.args.data_time.train.end)
            valid_row_start = df_raw.index.get_loc(self.args.data_time.valid.start)
            valid_row_end = df_raw.index.get_loc(self.args.data_time.valid.end)
            test_row_start = df_raw.index.get_loc(self.args.data_time.test.start)
            test_row_end = df_raw.index.get_loc(self.args.data_time.test.end)
            df_raw=df_raw.reset_index()
            border1s = [train_row_start, valid_row_start-self.past_len, test_row_start-self.past_len]
            border2s = [train_row_end+1, valid_row_end+1, test_row_end +1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
 
            sample_interval = [self.args.unit_len//self.args.unit_len,self.args.unit_len,self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)    
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            self.origin_data=np.array(df_data[border1:border2]) #原始的数据，用于给schrodinger提供未经tranform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==2:
            # 按比例划分数据集
            # df_raw = df_raw.reset_index()
            total_data_len = len(df_raw)
            
            # 根据比例计算各数据集的长度
            train_ratio, valid_ratio, test_ratio = self.args.train_val_test_ratio
            train_len_ratio = int(total_data_len * train_ratio)
            valid_len_ratio = int(total_data_len * valid_ratio)
            test_len_ratio = total_data_len - train_len_ratio - valid_len_ratio  # 确保总和等于总长度
            
            # 计算边界
            border1s = [0, train_len_ratio-self.past_len, train_len_ratio+valid_len_ratio-self.past_len]
            border2s = [train_len_ratio, train_len_ratio+valid_len_ratio, train_len_ratio+valid_len_ratio+test_len_ratio]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            # 计算可以被滚动的起始点index
            sample_interval = [self.args.unit_len//self.args.unit_len, self.args.unit_len, self.args.unit_len][self.set_type]
            self.sample_index = np.arange(0, border2-border1-self.past_len-self.pred_len+1, sample_interval)
            
            # 数据集划分
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[self.target_cols]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['timestamp'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            
            self.origin_data = np.array(df_data[border1:border2])  # 原始的数据，用于给schrodinger提供未经transform处理的原始
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp = data_stamp
        elif self.args.data_mode==3:

            split_names = ['train', 'val', 'test']
            split_name = split_names[self.set_type]

 
            if split_name == 'train':
                samples_path = getattr(self.args, 'train_data_path', None)
                index_path = getattr(self.args, 'train_index_path', None)
            elif split_name == 'val':
                samples_path = getattr(self.args, 'val_data_path', None)
                index_path = getattr(self.args, 'val_index_path', None)
            elif split_name == 'test':
                samples_path = getattr(self.args, 'test_data_path', None)
                index_path = getattr(self.args, 'test_index_path', None)
            
            if samples_path is None or index_path is None:
                raise ValueError(f"data_mode==3 需要设置 {split_name}_data_path 和 {split_name}_index_path 参数")

            samples = np.load(samples_path)  # (N, T, F)
            index_samples = np.load(index_path)  # (N, T) of timestamps strings

            if samples.ndim != 3:
                raise ValueError(f"samples 期望为3维 (N, T, F)，实际为 {samples.shape}")

            num_samples, total_len, feature_dim = samples.shape
            if total_len != (self.past_len + self.pred_len):
                raise ValueError(f"样本序列长度不匹配: 期望 {self.past_len + self.pred_len}, 实际 {total_len}")

            # 选择特征列：'S' 使用目标列（在最后 len(target_cols) 列）
            if self.features == 'S':
                target_dim = len(self.target_cols)
                if target_dim <= 0 or target_dim > feature_dim:
                    raise ValueError(f"目标列维度异常: target_dim={target_dim}, feature_dim={feature_dim}")
                samples = samples[:, :, feature_dim - target_dim:feature_dim]
                feature_dim = target_dim

            # 标准化：以训练集拟合
            if self.scale:
                if hasattr(self.args, 'train_data_path'):
                    train_samples = np.load(self.args.train_data_path)
                    if train_samples.ndim != 3:
                        raise ValueError(f"train_samples 期望为3维 (N, T, F)，实际为 {train_samples.shape}")
                    if self.features == 'S':
                        target_dim = len(self.target_cols)
                        train_samples = train_samples[:, :, train_samples.shape[2]-target_dim:train_samples.shape[2]]
                    fit_source = train_samples
                else:
                    fit_source = samples
                self.scaler.fit(fit_source.reshape(-1, fit_source.shape[-1]))
                samples = self.scaler.transform(samples.reshape(-1, feature_dim)).reshape(num_samples, total_len, feature_dim)

            # 时间戳编码
            def encode_timestamps(ts_array):
                ts_pd = pd.to_datetime(ts_array)
                if self.timeenc == 0:
                    df_ts = pd.DataFrame({'timestamp': ts_pd})
                    df_ts['month'] = df_ts['timestamp'].dt.month
                    df_ts['day'] = df_ts['timestamp'].dt.day
                    df_ts['weekday'] = df_ts['timestamp'].dt.weekday
                    df_ts['hour'] = df_ts['timestamp'].dt.hour
                    df_ts['minute'] = df_ts['timestamp'].dt.minute
                    df_ts['minute'] = df_ts['minute'].map(lambda x: x // 15)
                    return df_ts.drop(['timestamp'], axis=1).values
                else:
                    # time_features 需要 pandas DatetimeIndex，不是 numpy 数组
                    feats = time_features(ts_pd, freq=self.freq)
                    return feats.transpose(1, 0)

            data_stamp_list = []
            for i in range(num_samples):
                ts_row = index_samples[i].astype(str)
                data_stamp_list.append(encode_timestamps(ts_row))
            data_stamp = np.stack(data_stamp_list, axis=0)  # (N, T, E)

            self.origin_data = None
            self.data_x = samples  # (N, T, F)
            self.data_y = samples
            self.data_stamp = data_stamp
        print(self.set_type,'读取处理完成')
    def datasets_index(self):
        """
        返回各个数据集的起止时间戳索引
        
        Returns:
            dict: 包含train、val、test数据集起止时间戳的字典
                格式: {'train': (start_timestamp, end_timestamp), 'val': (start_timestamp, end_timestamp), 'test': (start_timestamp, end_timestamp)}
        """
        if self.args.data_mode == 3:
            results = {}
            path_mapping = {
                'train': getattr(self.args, 'train_index_path', None),
                'val': getattr(self.args, 'val_index_path', None),
                'test': getattr(self.args, 'test_index_path', None)
            }
            
            for split_name, index_path in path_mapping.items():
                if index_path is None:
                    raise ValueError(f"data_mode==3 需要设置 {split_name}_index_path 参数")
                idx_arr = np.load(index_path)
                start_ts = str(idx_arr[0, 0])
                end_ts = str(idx_arr[-1, -1])
                results[split_name] = (start_ts, end_ts)
            return {
                'train': results['train'],
                'val': results['val'],
                'test': results['test']
            }

        # 读取原始数据获取时间戳
        if self.data_path.endswith('.csv'):
            df_raw = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl') or self.data_path.endswith('.pickle'):
            df_raw = pd.read_pickle(self.data_path)
        elif self.data_path.endswith('.parquet'):
            df_raw = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format for {self.data_path}")
        df_raw.set_index('timestamp', inplace=True)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.reset_index()
        
        if self.args.data_mode == 0:
            # 模式0：按固定长度划分
            df_raw = df_raw.loc[:self.args.data_len['end']]
            border1s = [0, self.train_len-self.past_len, self.train_len+self.valid_len-self.past_len] 
            border2s = [self.train_len, self.train_len+self.valid_len, self.train_len+self.valid_len+self.test_len]
            
        elif self.args.data_mode == 1:
            # 模式1：按时间范围划分
            train_row_start = df_raw.index.get_loc(self.args.data_time.train.start)
            train_row_end = df_raw.index.get_loc(self.args.data_time.train.end)
            valid_row_start = df_raw.index.get_loc(self.args.data_time.valid.start)
            valid_row_end = df_raw.index.get_loc(self.args.data_time.valid.end)
            test_row_start = df_raw.index.get_loc(self.args.data_time.test.start)
            test_row_end = df_raw.index.get_loc(self.args.data_time.test.end)
            
            border1s = [train_row_start, valid_row_start-self.past_len, test_row_start-self.past_len]
            border2s = [train_row_end+1, valid_row_end+1, test_row_end+1]
            
        elif self.args.data_mode == 2:
            # 模式2：按比例划分
            total_data_len = len(df_raw)
            
            train_ratio, valid_ratio, test_ratio = self.args.train_val_test_ratio
            train_len_ratio = int(total_data_len * train_ratio)
            valid_len_ratio = int(total_data_len * valid_ratio)
            test_len_ratio = total_data_len - train_len_ratio - valid_len_ratio
            
            border1s = [0, train_len_ratio-self.past_len, train_len_ratio+valid_len_ratio-self.past_len]
            border2s = [train_len_ratio, train_len_ratio+valid_len_ratio, train_len_ratio+valid_len_ratio+test_len_ratio]
        
        # 获取对应的时间戳
        train_start_time = df_raw.iloc[border1s[0]]['timestamp']
        train_end_time = df_raw.iloc[border2s[0]-1]['timestamp']
        val_start_time = df_raw.iloc[border1s[1]]['timestamp']
        val_end_time = df_raw.iloc[border2s[1]-1]['timestamp']
        test_start_time = df_raw.iloc[border1s[2]]['timestamp']
        test_end_time = df_raw.iloc[border2s[2]-1]['timestamp']
        
        return {
            'train': (train_start_time, train_end_time),
            'val': (val_start_time, val_end_time),
            'test': (test_start_time, test_end_time)
        }


    def __getitem__(self, index):
        #验证集、测试集每隔一天推理96点
        # index = self.sample_index[id]
        if self.args.data_mode == 3:
            sample_x = self.data_x[index]
            sample_stamp = self.data_stamp[index]

            s_begin = 0
            s_end = self.past_len
            r_begin = s_end - self.label_len
            r_end = s_end + self.pred_len

            seq_x = sample_x[s_begin:s_end]
            seq_y = sample_x[r_begin:r_end]
            seq_x_mark = sample_stamp[s_begin:s_end]
            seq_y_mark = sample_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            s_begin = index
            s_end = s_begin + self.past_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.args.data_mode == 3:
            return self.data_x.shape[0]
        return len(self.data_x) - self.past_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.args.data_mode == 3:
            # data_mode==3 中数据形状为 (N, T, F)，需要reshape为 (N*T, F) 进行逆变换，然后恢复原形状
            if data.ndim == 3:
                N, T, F = data.shape
                data_2d = data.reshape(-1, F)
                inverse_2d = self.scaler.inverse_transform(data_2d)
                return inverse_2d.reshape(N, T, F)
            else:
                # 如果已经是2D，直接处理
                return self.scaler.inverse_transform(data)
        else:
            return self.scaler.inverse_transform(data)
