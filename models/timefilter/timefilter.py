from data_provider.data_factory import data_provider
from models.timefilter.units.tools import EarlyStopping, adjust_learning_rate, visual
from models.timefilter.units.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import argparse
import pytz
import pickle as pkl
from datetime import datetime
warnings.filterwarnings('ignore')
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.timefilter import TimeFilter_architecture
from utils.metrics import MAE_np, RMSE_np, MAPE_np, SPEARMAN_np, PEARSON_np,R2_np, SMAPE_np

class Exp_Basic(object):
    def __init__(self, config):
        self.configs=config
        self.model_config=config['model_config']
        self.training_config=config['training_config']
        self.data_config=config['data_config']
        # 合并所有配置项
        merged_config = {**self.model_config, **self.training_config,**self.data_config}
        def flatten_dict(d, parent_key=''):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key + '_'))
                else:
                    items[new_key] = v
            return items

        merged_config = flatten_dict(merged_config)
        self.args = argparse.Namespace(**merged_config)

        #对于电价数据集，为了能够使用未来一天的日前特征，将原先的past_len推理pred_len修改为past_len+1推理pred_len，再在dataloader中进行相应的处理
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong' or self.args.data=='Electricity_shandong_dk':
            self.args.past_len+=1
        self.args.seq_len=self.args.past_len*self.args.unit_len
        print('args:')
        print(self.args)
        self.model_dict = {
            'TimeFilter': TimeFilter_architecture
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):

        device = torch.device(self.args.device)  # 直接使用 self.args.device
        print(f"Use GPU: {self.args.device}")
        # print(f"Available GPUs: {torch.cuda.device_count()}")
        # if self.args.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(
        #         self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        #     device = torch.device('cuda:{}'.format(self.args.gpu))
        #     print('Use GPU: cuda:{}'.format(self.args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class timefilter(Exp_Basic):
    def __init__(self, config):
        super(timefilter, self).__init__(config)
        self.config=config
        self.model_config=config['model_config']
        self.training_config=config['training_config']
        self.data_config=config['data_config']
        # 合并所有配置项
        merged_config = {**self.model_config, **self.training_config,**self.data_config}
        def flatten_dict(d, parent_key=''):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key + '_'))
                else:
                    items[new_key] = v
            return items

        merged_config = flatten_dict(merged_config)
        self.args = argparse.Namespace(**merged_config)
        #对于电价数据集，为了能够使用未来一天的日前特征，将原先的past_len推理pred_len修改为past_len+1推理pred_len，再在dataloader中进行相应的处理
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong' or self.args.data=='Electricity_shandong_dk':
            self.args.past_len+=1
        self.args.seq_len=self.args.past_len*self.args.unit_len
        # 创建文件夹用于存储，文件夹名为 "数据集名字_时间"
        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = (
            datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H:%M:%S")
        )

        foldername = (
            self.config['training_config']["save_path"]
            +  self.config['data_config']["data_path"].split("/")[-2] + "/"
            + current_time
            + "/"
        )

        print("model folder:", foldername)
        foldername = os.path.join(config['root'], foldername)
        os.makedirs(foldername, exist_ok=True)
        self.foldername = foldername
        self.masks = self._get_mask()

        # TensorBoard writer
        tensorboard_dir = os.path.join(foldername, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logs saved to: {tensorboard_dir}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        dtype = torch.float32
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.args.c_out, N, self.args.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        mask1 = torch.kron(torch.ones(self.args.c_out, self.args.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        masks = torch.cat([mask0, mask1, mask2], dim=0)  # [3, 1, L, L]
        return masks

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self.foldername
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, moe_loss = self.model(batch_x, self.masks, is_training=True)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:].to(self.device)
                alpha = 0.05
                loss = criterion(outputs, batch_y) + alpha * moe_loss
                train_loss.append(loss.item())

                # TensorBoard: 记录 batch loss 和 lr
                global_step = epoch * train_steps + i
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/LearningRate', model_optim.param_groups[0]['lr'], global_step)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # TensorBoard: 记录 epoch 级别指标
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalars('Loss/Train_Valid', {'Train': train_loss, 'Valid': vali_loss}, epoch)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'best_model_0.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.test(test=0)
        return self.model

    def test(self, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.foldername, 'best_model_0.pth')))
        folder_path = self.foldername
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len*self.args.unit_len:, :]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # 使用新的指标计算函数
        mae = MAE_np(preds, trues)
        rmse = RMSE_np(preds, trues)
        mape = MAPE_np(preds, trues)
        r2 = R2_np(preds, trues)
        spearman = SPEARMAN_np(preds, trues)
        pearson = PEARSON_np(preds, trues)
        smape = SMAPE_np(preds, trues)
        # 保留原有的 metric 函数用于兼容性
        mae_old, mse, rmse_old, mape_old, mspe = metric(preds, trues)
        
        print('MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}, R2: {:.6f}, Spearman: {:.6f}, Pearson: {:.6f}'.format(mae, rmse, mape, r2, spearman, pearson))
        print('MSE: {:.6f}, MSPE: {:.6f}, SMAPE: {:.6f}'.format(mse, mspe, smape))

        # TensorBoard: 记录测试集指标
        self.writer.add_scalar('Test/MAE', mae, 0)
        self.writer.add_scalar('Test/RMSE', rmse, 0)
        self.writer.add_scalar('Test/MAPE', mape, 0)
        self.writer.add_scalar('Test/R2', r2, 0)
        self.writer.add_scalar('Test/Spearman', spearman, 0)
        self.writer.add_scalar('Test/Pearson', pearson, 0)
        self.writer.add_scalar('Test/MSE', mse, 0)
        self.writer.add_scalar('Test/SMAPE', smape, 0)
        self.writer.close()

        f = open("result_timexer.txt", 'a')
        f.write(self.args.model_path + "  \n")
        f.write('MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}, R2: {:.6f}, Spearman: {:.6f}, Pearson: {:.6f}, MSE: {:.6f}, MSPE: {:.6f}, SMAPE: {:.6f}'.format(mae, rmse, mape, r2, spearman, pearson, mse, mspe, smape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2, spearman, pearson, smape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # 额外保存result.npz文件
        np.savez(
            os.path.join(folder_path, "result.npz"),
            real_y=trues,
            pred_y=preds,
        )
        datasets_index = test_data.datasets_index()
        import pickle 
        with open(folder_path + 'datasets_index.pkl', 'wb') as file:
            pickle.dump(datasets_index, file)
