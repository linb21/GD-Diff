from data_provider.data_factory import data_provider
import argparse
from models.D3U.D3U_util.utils.tools import EarlyStopping
from models.D3U.D3U_util.utils.metrics import metric
import gc
import math
# from models.ns_models import ns_Transformer
# from models.exp.exp_basic import Exp_Basic
from models.D3U.D3U_util.diffusion_models import diffuMTS
from models.D3U.D3U_util.diffusion_models.diffusion_utils import *
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models.D3U.D3U_util.samplers.dpm_sampler import DPMSolverSampler
import os
import time
import pytz
from datetime import datetime
from models.D3U.D3U_util.utils.metrics import  calc_quantile_CRPS_sum
from multiprocessing import Pool
import CRPS.CRPS as pscore
from models.D3U.D3U_util.layers.Decompose import moving_avg
import warnings
from utils.metrics import MAE_np, RMSE_np, MAPE_np, SPEARMAN_np, PEARSON_np,R2_np, SMAPE_np
from utils.metrics import MAE_torch, RMSE_torch, MAPE_torch, R2_torch, SPEARMAN_torch, PEARSON_torch, SMAPE_torch
warnings.filterwarnings('ignore')
from models.D3U.D3U_util.layers.Decompose import series_decomp, FourierLayer
from models.D3U.D3U_util.condition_models import ns_Transformer, PatchTST_TS, SVQ, iTransformer
class Exp_Basic(object):
    def __init__(self, config):
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
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong' or self.args.data=='Electricity_dk':
            self.args.past_len+=1
        self.args.seq_len=self.args.past_len#*self.args.unit_len
        self.cond_model_dict = {
            'ns_Transformer': ns_Transformer,
            'PatchTST':PatchTST_TS,
            'SVQ': SVQ,
            'iTransformer': iTransformer
        }
        self.device = self._acquire_device()
        model,cond_pred_model = self._build_model()
        self.model = model.to(self.device)
        self.cond_pred_model = cond_pred_model.to(self.device)
        

    def _build_model(self):
        raise NotImplementedError
        return None, None, None

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
def ccc(id, pred, true):

    res_box = np.zeros(len(true))

    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


def calculate_crps_sum_worker(args):
        pred, true = args
        p_in = np.sum(pred, axis=-1).T
        t_in = np.sum(true, axis=-1).reshape(-1)
        crps = ccc(8, p_in, t_in)
        return crps.mean()
        
def calculate_crps_worker(args):

        pred, true = args
        p_in = pred.transpose(1, 0, 2)
        t_in = true
        all_res = []
        for i in range(pred.shape[-1]):
            crps = ccc(8, p_in[:,:,i], t_in[:,i])
            all_res.append(crps)
        all_res= np.array(all_res)
        if isinstance(all_res, np.ndarray):
            return np.mean(all_res, axis=0).mean()
        else:
            return all_res


class d3u(Exp_Basic):
    def __init__(self, config):
        super(d3u, self).__init__(config)
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

        beijing_tz = pytz.timezone("Asia/Shanghai")
        current_time = (
            datetime.now().astimezone(beijing_tz).strftime("%Y-%m-%d_%H-%M-%S_%f")
        )

        data_path = os.path.normpath(self.config['data_config']["data_path"])
        dataset_folder = os.path.basename(os.path.dirname(data_path))
        model_id = str(self.config['training_config'].get("model_id", self.args.model_name))
        model_id = model_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        seed = self.config['training_config'].get("seed", self.args.seed)
        run_name = f"{model_id}_seed{seed}_{current_time}"

        foldername = os.path.join(
            self.config['training_config']["save_path"],
            dataset_folder,
            run_name,
        )

        print("model folder:", foldername)
        foldername = os.path.join(self.config['root'], foldername)
        os.makedirs(foldername, exist_ok=True)
        self.foldername = foldername
        if self.args.data=='Electricity_shanxi' or self.args.data=='Electricity_shandong' or self.args.data=='Electricity_dk':
            self.args.past_len+=1
        self.args.seq_len=self.args.past_len#*self.args.unit_len
        # self.moving_avg = moving_avg(7, stride=1) if args.decomposition else None
        self.decomp = series_decomp(kernel_size=self.args.kernel_size) if self.args.decomposition else None
        self.seasonal = FourierLayer(factor=self.args.fourier_factor) if self.args.decomposition else None

    def _load_condition_checkpoint(self, model, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            if self.args.cond_model == 'PatchTST' and (
                new_key.startswith('encoder.attn_layers.') or new_key.startswith('encoder.norm.')
            ):
                new_key = 'encoder.encoder.' + new_key[len('encoder.'):]
            remapped_state_dict[new_key] = value

        model.load_state_dict(remapped_state_dict)
        
    def _build_model(self):
        model = diffuMTS.Model(self.args).float()

        cond_pred_model = self.cond_model_dict[self.args.cond_model].Model(self.args).float()
        # condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.cond_model)
        # if self.args.decomposition:
        #    best_condition_model_path = condition_path +'/' +str('decomposition') +'/' + self.args.data_name+'/' +str(self.args.pred_len*self.args.unit_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
        # else:
        #    best_condition_model_path = condition_path+'/' +str('all') +'/' + self.args.data_name+'/' +str(self.args.pred_len*self.args.unit_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
        best_condition_model_path = self.args.pretrain_checkpoints
        print(best_condition_model_path)

        if self.args.from_scrach==False:
            self._load_condition_checkpoint(cond_pred_model, best_condition_model_path)
            #
            #cond_pred_model
            if self.args.cond_pred_model_requires_grad:
                for param in cond_pred_model.parameters():
                        param.requires_grad = True
            else:
                for param in cond_pred_model.parameters():
                        param.requires_grad = False

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)

        return model, cond_pred_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            model_optim = optim.Adam([{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}], lr=self.args.learning_rate)
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len*self.args.unit_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :int(self.args.label_len*self.args.unit_len), :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]



                y_T_mean, _, enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  
                if self.args.use_pretraining_condition:
                    enc_out=y_T_mean

                #y_T_mean: [bs x  pred_len x nvars] enc_out: [ bsz x n_vars x patch_num  x d_model ]

                batch_y=batch_y[:, -self.args.pred_len*self.args.unit_len:, :]
                if self.args.bias :
                    y_0=batch_y-y_T_mean #y_bias

                    if self.args.bias_y_0:
                        res, trend = self.decomp(batch_y) # trend/res : [bsz, seq_len, n_vars]
                        seasonal = self.seasonal(res)
                        y_0=res-seasonal
              
                else:
                    y_0=batch_y
                
                
                e = torch.randn_like(y_0).to(self.device)
                

                y_t = self.model.q_sample( y_0, t, noise=e)

                output= self.model( y_t, t,enc_out)


                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                e=e[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                y_0=y_0[:, -self.args.pred_len*self.args.unit_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0

                loss = criterion(output,target)

                loss = loss.detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self.foldername
        # condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.data)
        condition_path = self.args.pretrain_checkpoints

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(condition_path):
            os.makedirs(condition_path)


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()


        criterion = self._select_criterion()
           
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            if self.args.scrach_10_stop:
                if epoch==10:
                     for param in self.cond_pred_model.parameters():
                        param.requires_grad = False
            # Training the diffusion part
            epoch_time = time.time()

            iter_count = 0
            train_loss = []
            self.model.train()
            self.cond_pred_model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len*self.args.unit_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :int(self.args.label_len*self.args.unit_len), :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)

                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n] .to(self.device)             #t: [batch]

                y_T_mean,_,enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  
                if self.args.use_pretraining_condition:
                    enc_out=y_T_mean



                #y_T_mean: [bs x  pred_len x nvars] enc_out: [ bsz x n_vars x patch_num  x d_model ]

   
                batch_y=batch_y[:, -self.args.pred_len*self.args.unit_len:, :]
                if self.args.bias :
                    y_0=batch_y-y_T_mean #y_bias
                    
                    if self.args.bias_y_0:
                        res, trend = self.decomp(batch_y) # trend/res : [bsz, seq_len, n_vars]
                        seasonal = self.seasonal(res)
                        y_0=res-seasonal

                    
                else:
                    y_0=batch_y
                
                
                e = torch.randn_like(y_0).to(self.device)
                

                y_t = self.model.q_sample( y_0, t, noise=e)

                output= self.model( y_t, t,enc_out)


                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                e=e[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                y_0=y_0[:, -self.args.pred_len*self.args.unit_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0
                loss = (target - output).square().mean() 

                train_loss.append(loss.item())

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

                a = 0

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, criterion)
            test_loss= self.vali(test_data, test_loader, criterion)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping( vali_loss, self.model, self.foldername)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.test(test=0)
        return self.model
        
    def calculate_batch_crps(self,pred, true):
            """
            计算单个批次的 CRPS并行化处理。

            """
            # pool = Pool(processes=32)
            # crps_values = pool.map(calculate_crps_worker, zip(pred, true))

            # # crps_sum_values = pool.map(calculate_crps_sum_worker, zip(pred, true))
            # pool.close()
            # pool.join()
            # # print('1',len(crps_values))
            # # print('2',len(crps_sum_values))
            # # 累加所有样本的 CRPS
            # batch_crps = np.sum(crps_values)

            # batch_crps_sum = np.sum(crps_sum_values)
            # return batch_crps
            # 使用列表推导式替代multiprocessing，避免pickle错误
            crps_values = [calculate_crps_worker((p, t)) for p, t in zip(pred, true)]
            
            # 累加所有样本的 CRPS
            batch_crps = np.sum(crps_values)

            return batch_crps

    def test(self, test=1):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def get_effective_sampling_params():
            n_z_samples = int(
                getattr(self.args, 'testing_n_z_samples', self.model.diffusion_config.testing.n_z_samples)
            )
            n_z_samples_depart = int(
                getattr(
                    self.args,
                    'testing_n_z_samples_depart',
                    self.model.diffusion_config.testing.n_z_samples_depart,
                )
            )
            if n_z_samples_depart <= 0:
                raise ValueError('testing.n_z_samples_depart must be positive')
            if n_z_samples % n_z_samples_depart != 0:
                raise ValueError(
                    f'testing.n_z_samples={n_z_samples} must be divisible by '
                    f'testing.n_z_samples_depart={n_z_samples_depart}'
                )
            return n_z_samples, n_z_samples_depart, n_z_samples // n_z_samples_depart

        def exact_y_0(config, sample_count, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            y_0 = y_tile_seq.reshape(
                -1,
                sample_count,
                config.pred_len * config.unit_len,
                y_tile_seq.shape[-1],
            )
            return y_0
        
        def store_gen_y_at_step_t(config, config_diff, sample_count, y_tile_seq, y_true, x_true, fast_sample):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            if fast_sample:
                gen_y_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                y_true_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                x_true_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                for current_t  in range(self.model.sampling_timesteps + 1):
                    gen_y = y_tile_seq[current_t].reshape(
                        -1,
                        sample_count,
                        config.pred_len * config.unit_len,
                        y_tile_seq[current_t].shape[-1],
                    ).cpu().numpy()
                    #print('gen_y',gen_y.shape)(8, 10, 240, 7)
                    # directly modify the dict value by concat np.array instead of append np.array gen_y to list
                    # reduces a huge amount of memory consumption
                    true_y=y_true.cpu().numpy()
                    true_x=x_true.cpu().numpy()
                    if len(gen_y_by_batch_list[current_t]) == 0:
                        gen_y_by_batch_list[current_t] = gen_y
                        y_true_by_batch_list[current_t] = true_y
                        x_true_by_batch_list[current_t] = true_x
                    else:
                        gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
                        y_true_by_batch_list[current_t] = np.concatenate([y_true_by_batch_list[current_t], true_y], axis=0)
                        x_true_by_batch_list[current_t] = np.concatenate([x_true_by_batch_list[current_t], true_x], axis=0)
            else:
                gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                y_true_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                x_true_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

                for current_t  in range(self.model.num_timesteps + 1):
                    gen_y = y_tile_seq[current_t].reshape(
                        -1,
                        sample_count,
                        config.pred_len * config.unit_len,
                        y_tile_seq[current_t].shape[-1],
                    ).cpu().numpy()
                    #print('gen_y',gen_y.shape)(8, 10, 240, 7)
                    # directly modify the dict value by concat np.array instead of append np.array gen_y to list
                    # reduces a huge amount of memory consumption
                    true_y=y_true.cpu().numpy()
                    true_x=x_true.cpu().numpy()
                    if len(gen_y_by_batch_list[current_t]) == 0:
                        gen_y_by_batch_list[current_t] = gen_y
                        y_true_by_batch_list[current_t] = true_y
                        x_true_by_batch_list[current_t] = true_x
                    else:
                        gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
                        y_true_by_batch_list[current_t] = np.concatenate([y_true_by_batch_list[current_t], true_y], axis=0)
                        x_true_by_batch_list[current_t] = np.concatenate([x_true_by_batch_list[current_t], true_x], axis=0)
            return  gen_y_by_batch_list,y_true_by_batch_list,x_true_by_batch_list

        
            """
            Another coverage metric.
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
            # condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.cond_model)
            condition_path = self.args.pretrain_checkpoints
            # if self.args.decomposition:
            #    best_condition_model_path = condition_path +'/' +str('decomposition') +'/' + self.args.data_name+'/' +str(self.args.pred_len*self.args.unit_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
            # else:
            #     best_condition_model_path = condition_path+'/' +str('all') +'/' + self.args.data_name+'/' +str(self.args.pred_len*self.args.unit_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
       
            # print(best_condition_model_path)
            best_condition_model_path=condition_path
            self._load_condition_checkpoint(self.cond_pred_model, best_condition_model_path)
            self.cond_pred_model = self.cond_pred_model.to(self.device)
        preds = []
        trues = []
        folder_path = self.foldername
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        self.sampler = DPMSolverSampler(self.model,self.device,self.args.parameterization)
        total_mse=0.0
        total_mae=0.0
        total_samples=0.0
        sum_crps = 0.0
        sum_crps_sum =0.0 
        real_pred=[]
        real_true=[]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len*self.args.unit_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :int(self.args.label_len*self.args.unit_len), :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
 
                y_T_mean,_,enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  

                #y_T_mean: [bs x  pred_len x nvars] enc_out:  [ bsz x n_vars x patch_num  x d_model ]
                y_true_bias=batch_y[:, -self.args.pred_len*self.args.unit_len:, :]-y_T_mean

                _, n_z_samples_depart, repeat_n = get_effective_sampling_params()
                gen_y_box = []
                gen_y_bias_box=[]
                for _ in range(n_z_samples_depart):

                    # sample
                    x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
                    x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)                           
                    #x_tile: [bs*sample, seq_len, var]
                    
                    enc_out_tile = enc_out.repeat(repeat_n, 1, 1, 1, 1)
                    enc_out_tile = enc_out_tile.transpose(0, 1).flatten(0, 1).to(self.device)             # enc_out_tile:[100*batch, nvar, patch_num, d_model_c]


                    y_T_mean_tile = y_T_mean.repeat(repeat_n, 1, 1, 1)
                    y_T_mean_tile = y_T_mean_tile.transpose(0, 1).flatten(0, 1).to(self.device) 

                    y_shape=(x_tile.shape[0],self.args.pred_len*self.args.unit_len,x_tile.shape[-1])                         # y_shape: [bs*sample, pred_len, var]
                    
                    if self.args.use_pretraining_condition:
                            enc_out_tile=y_T_mean_tile


                    if self.args.type_sampler == "none":
                        y_tile_seq = self.model.p_sample_loop(y_T_mean_tile,enc_out_tile, y_shape) #(如果预测x_0重新更改)
                    elif self.args.type_sampler == "DDIM":
                        y_tile_seq =self.model.fast_sample(y_T_mean_tile,enc_out_tile, y_shape,self.args.eta) #(如果预测x_0重新更改)
                    elif self.args.type_sampler == "DPM_solver":
                        
                        y_tile_seq = self.sampler.sample(S=self.args.DPMsolver_step,
                                        conditioning=enc_out_tile,
                                        shape=y_shape,
                                        verbose=False,
                                        unconditional_guidance_scale=1.0,
                                        unconditional_conditioning=None,
                                        eta=0.,
                                        x_T=None)
                    
                    y_tile_bias=y_tile_seq
                    if self.args.bias :
                    
                        y_tile_seq=y_tile_seq+y_T_mean_tile
                    else:
                        y_tile_seq=y_tile_seq
                    

                    
                    gen_y_bias = exact_y_0(
                        config=self.model.args,
                        sample_count=repeat_n,
                        y_tile_seq=y_tile_bias,
                    )        #gen_y: [bs,sample, pred_len, var]
                    gen_y = exact_y_0(
                        config=self.model.args,
                        sample_count=repeat_n,
                        y_tile_seq=y_tile_seq,
                    )        #gen_y: [bs,sample, pred_len, var]

                    gen_y_bias_box.append(gen_y_bias.cpu().numpy())
                    gen_y_box.append(gen_y.cpu().numpy())
                outputs = np.concatenate(gen_y_box, axis=1)
                outputs_bias= np.concatenate(gen_y_bias_box, axis=1)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len*self.args.unit_len:, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                batch_crps = self.calculate_batch_crps(pred, true)
                sum_crps += batch_crps

                f_dim = -1 if self.args.features == 'MS' else 0
                pred_ns = np.mean(pred, axis=1)
                print('test2 shape:', pred_ns.shape, true.shape)
                # 完成去归一化（反归一化）- 参考timexer的实现
                if test_data.scale and self.args.inverse:
                    # 获取原始特征维度
                    original_feature_dim = len(self.data_config['columns'])
                    # 如果是MS模式，先复制特征维度使其与scaler维度匹配
                    if self.args.features == 'MS':
                        pred_ns = np.tile(pred_ns, [1, 1, int(original_feature_dim / pred_ns.shape[-1])])
                        true = np.tile(true, [1, 1, int(original_feature_dim / true.shape[-1])])
                        # pred = np.tile(pred, [1, 1, 1, int(original_feature_dim / pred.shape[-1])])  # 对pred也进行扩充
                    # 将3D数据reshape成2D (batch*time, features)，进行反归一化，再reshape回原shape
                    shape_preds = pred_ns.shape
                    shape_trues = true.shape
                    pred_ns = test_data.inverse_transform(pred_ns.reshape(shape_preds[0] * shape_preds[1], -1)).reshape(shape_preds)
                    true = test_data.inverse_transform(true.reshape(shape_trues[0] * shape_trues[1], -1)).reshape(shape_trues)
                    # pred是(bs, sample, t, f)，对第3、4维reshape→(bs*sample*t, f)，反归一化后再reshape回来
                    pred_shape = pred.shape
                    # pred = test_data.inverse_transform(pred.reshape(pred_shape[0]*pred_shape[1]*pred_shape[2], -1)).reshape(pred_shape)
                pred_ns = pred_ns[:, :, f_dim:]
                true = true[:, :, f_dim:]
                mae, mse, rmse, mape, mspe = metric(pred_ns, true)
                print('mae_mse', mae, mse)
                total_mse += mse * pred_ns.shape[0]
                total_mae += mae * pred_ns.shape[0]
                total_samples += pred_ns.shape[0]

                preds.append(pred.sum(-1))  # pred已经做过去归一化                     
                trues.append(true.sum(-1))
                real_pred.append(pred_ns)
                real_true.append(true)
                del outputs 
                gc.collect() 

                if i % 1 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()


        print('total_samples',total_samples)
        avg_crps = sum_crps / total_samples
        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        rmse_total = math.sqrt(mse_total)
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: mse:{:.4f}, rmse:{:.4f}, mae:{:.4f} '.format(mse_total, rmse_total, mae_total))
        preds = np.concatenate(preds, axis=0)
        real_pred = np.concatenate(real_pred, axis=0)
        real_true = np.concatenate(real_true, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_save = np.array(preds)
        trues_save = np.array(trues)
        crps_sum=calc_quantile_CRPS_sum(preds,trues)
        print('NT metrc: CRPS_sum:{:.4f}'.format(crps_sum))

        metrics = np.array([mae_total, mse_total, rmse_total, avg_crps, crps_sum], dtype=np.float32)

        np.save(self.foldername + 'pred.npy', preds_save)
        np.save(self.foldername + 'true.npy', trues_save)
        np.save(self.foldername + 'real_pred.npy', real_pred)
        np.save(self.foldername + 'real_true.npy', real_true)
        np.save(self.foldername + 'metrics.npy', metrics)
