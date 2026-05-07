import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from linear_attention_transformer import LinearAttentionTransformer


class Forecasting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mean = None # 数据均值
        self.std = None # 数据标准差

        self.device = config['training_config']["device"]
        self.target_dim = len(config['data_config']["columns"])
        self.emb_time_dim = config['model_config']["model"]["timeemb"]
        self.emb_feature_dim = config['model_config']["model"]["featureemb"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.target_dim_base = self.target_dim
        self.num_sample_features = config['model_config']["model"]["num_sample_features"]
        self.pred_len = config['data_config']["pred_len"]*config['data_config']['unit_len']
        self.past_len = config['data_config']["past_len"]*config['data_config']['unit_len']

        self.version = config['model_config']["version"]

        self.gamma = config['model_config']["model"]["gamma"]
        self.alpha1 = config['model_config']["model"]["alpha1"]

        self.epoch_num = 0

        # self.config = config

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config['model_config']["diffusion"]
        # config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2

        self.diffmodel = DIFF(config_diff, input_dim, config['model_config']["version"], self.pred_len,self.emb_total_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        # print(self.alpha_torch)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, is_finetune, 
        observed_data0, cond_mask0
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, is_finetune=is_finetune,
                observed_data0=observed_data0, cond_mask0=cond_mask0
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1, 
        is_finetune=0, observed_data0=None, 
        cond_mask0=None
    ):

        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(torch.ones((B, K, L))).to(self.device)

        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask

        num_eval = target_mask.sum()
        num_eval = num_eval if num_eval > 0 else 1

        residual = (noise - predicted) * target_mask
        loss_t = (residual ** 2).sum() / num_eval

        if self.version >= 1:
            if is_finetune and is_train:
                # weight_t = torch.log(1 + self.alpha1 * torch.sum(torch.abs(noise - predicted)[:,0,-self.pred_len:],dim=-1))
                residual_t_s = torch.sum(torch.abs((noise - predicted)[:,0,-self.pred_len:]) ** 2, dim=-1)
                residual_t = torch.sum(torch.abs((noise - predicted)[:,0,-self.pred_len:]), dim=-1)

                noise_fft = torch.fft.fft(noise[:,0,-self.pred_len:])
                pred_fft = torch.fft.fft(predicted[:,0,-self.pred_len:])
                residual_fft_s = torch.sum((self.gamma * torch.abs(noise_fft - pred_fft)) ** 2, dim=-1)
                residual_fft = torch.sum((self.gamma * torch.abs(noise_fft - pred_fft)), dim=-1)
                # weight_f = torch.log(1 + self.alpha1 * torch.sum(residual_fft,dim=-1))

                residual = residual_t + residual_fft
                weight = torch.log(1 + self.alpha1 * residual)
                residual_s = residual_t_s + residual_fft_s

                # loss_t = (weight_t.unsqueeze(-1)*(residual ** 2)).sum() / num_eval
                # loss_f = (weight_f.unsqueeze(-1)*(residual_fft ** 2)).sum() / num_eval

                loss = (weight * residual_s).sum() / num_eval
            
            else:
                noise_fft = torch.fft.fft(noise[:,0,-self.pred_len:])
                pred_fft = torch.fft.fft(predicted[:,0,-self.pred_len:])
                residual_fft = self.gamma * torch.abs(noise_fft - pred_fft)

                loss_f = (residual_fft ** 2).sum() / num_eval

                loss = loss_f + loss_t
        else:
            if is_finetune and is_train:
                residual_t_s = torch.sum(torch.abs((noise - predicted)[:,0,-self.pred_len:]) ** 2, dim=-1)
                residual_t = torch.sum(torch.abs((noise - predicted)[:,0,-self.pred_len:]), dim=-1)
                weight = torch.log(1 + self.alpha1 * residual_t)
                loss = (weight * residual_t_s).sum() / num_eval
            else:
                loss = loss_t

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)

        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in tqdm(range(n_samples)):
            current_sample = torch.randn_like(torch.ones((B, K, L))).to(self.device)

            for t in range(self.num_steps - 1, -1, -1):

                diff_input = self.set_input_to_diffmodel(current_sample, observed_data, cond_mask)

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features_forecast(self, observed_data, observed_mask, feature_id, gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for k in range(len(observed_data)):
            ind = np.arange(1, self.target_dim_base)
            np.random.shuffle(ind)
            ind = np.concatenate(([0], ind[:size-1]))
            extracted_data.append(observed_data[k, ind])
            extracted_mask.append(observed_mask[k, ind])
            extracted_feature_id.append(feature_id[k, ind])
            extracted_gt_mask.append(gt_mask[k, ind])

        extracted_data = torch.stack(extracted_data, 0)
        extracted_mask = torch.stack(extracted_mask, 0)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)

        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim, device=self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1, is_finetune=0):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            feature_id, 
        ) = self.process_data(batch)
        temp1, temp2, temp3, temp4 = observed_data, observed_mask, observed_tp, gt_mask

        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features_forecast(observed_data, observed_mask, feature_id, gt_mask)    
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        cond_mask = observed_mask * gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, 
                         is_finetune=is_finetune, observed_data0=temp1, cond_mask0=temp2 * temp4)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
        ) = self.process_data(batch)

        self.target_dim = self.target_dim_base  

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class DIFF(nn.Module):
    def __init__(self, config, inputdim=2, version=0, pred_len=24,side_dim=None):
        super().__init__()
        self.channels = config["channels"]

        self.version = version

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    version=version,
                    pred_len=pred_len,
                    trans_version=config['trans_version'],
                    layer_num=0
                )
                for k in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        if self.version == 0:
            x = F.relu(x) 
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False, 
                 version=0, pred_len=24, trans_version=0, layer_num=-1):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # if layer_num == 0:
        #     self.mid_projection = Conv1d_with_init(2 * channels, 2 * channels, 1)
        # else:
        #     self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.version = version
        self.pred_len = pred_len
        self.trans_version = trans_version
        self.layer_num = layer_num

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            if self.version != 0 and self.layer_num == 0: # 是否仅在第一个残差层进行频域操作
                if self.trans_version == 0:
                    self.freq_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
                elif self.trans_version == 1:
                    self.freq_layer = get_linear_trans(heads=nheads,layers=1,channels=2*channels)
                    self.freq_feature_layer = get_linear_trans(heads=nheads,layers=1,channels=2*channels)
                elif self.trans_version == 2:
                    self.freq_layer = get_linear_trans(heads=nheads,layers=1,channels=2*channels)
                elif self.trans_version == 3:
                    self.freq_layer = get_linear_trans(heads=nheads,layers=1,channels=2*channels)
                    self.freq_feature_layer = get_linear_trans(heads=nheads,layers=1,channels=2*channels)
                else:
                    self.freq_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
                    self.freq_feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)       
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            if self.version != 0 and self.layer_num == 0:
                if self.trans_version == 0:
                    self.freq_layer = get_torch_trans(heads=nheads,layers=1,channels=channels)
                elif self.trans_version == 1:
                    self.freq_layer = get_torch_trans(heads=nheads,layers=1,channels=2*channels)
                    self.freq_feature_layer = get_torch_trans(heads=nheads,layers=1,channels=2*channels)
                elif self.trans_version == 2:
                    self.freq_layer = get_torch_trans(heads=nheads,layers=1,channels=2*channels)
                else:
                    self.freq_layer = get_torch_trans(heads=nheads,layers=1,channels=channels)
                    self.freq_feature_layer = get_torch_trans(heads=nheads,layers=1,channels=channels)   

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y
    
    def forward_freq(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y

        if self.trans_version == 0:
            y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            y = torch.abs(torch.fft.fft(y))

            if self.is_linear:
                y = self.freq_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

            y = torch.abs(torch.fft.ifft(y)).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        elif self.trans_version == 1:
            y = y.reshape(B, 2 * channel, K, L).permute(0, 2, 1, 3).reshape(B * K, 2 * channel, L)
            if self.is_linear:
                y = self.freq_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

            y = y.reshape(B * K, channel, 2, L)
            y = y.permute(0, 1, 3, 2)
            real = y[:,:,:,0]
            imag = y[:,:,:,1]
            y = torch.complex(real, imag)
            y = torch.abs(torch.fft.ifft(y)).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        elif self.trans_version == 2:
            y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            y = torch.fft.fft(y)

            real = torch.real(y)
            imag = torch.imag(y)
            y = torch.stack([real, imag], axis=-1)
            y = y.permute(0, 1, 3, 2)
            y = y.reshape(B * K, 2 * channel, L)

            if self.is_linear:
                y = self.freq_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

            y = y.reshape(B * K, channel, 2, L)
            y = y.permute(0, 1, 3, 2)
            real = y[:,:,:,0]
            imag = y[:,:,:,1]
            y = torch.complex(real, imag)
            y = torch.abs(torch.fft.ifft(y)).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        elif self.trans_version == 3:
            L = L//2+1
            y = y.reshape(B, 2 * channel, K, L).permute(0, 2, 1, 3).reshape(B * K, 2 * channel, L)
            if self.is_linear:
                y = self.freq_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

            y = y.reshape(B * K, channel, 2, L)
            y = y.permute(0, 1, 3, 2)
            real = y[:,:,:,0]
            imag = y[:,:,:,1]
            y = torch.complex(real, imag)
            L = (L-1)*2
            y = torch.fft.irfft(y).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        else:
            y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            if self.is_linear:
                y = self.freq_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

            y = torch.abs(torch.fft.ifft(y)).reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_freq_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        
        if K == 1:
            return y
        
        if self.trans_version == 1:
            y = torch.fft.fft(y.reshape(B, channel, K, L))
            real = torch.real(y)
            imag = torch.imag(y)
            y = torch.stack([real, imag], axis=-1) #(B, channel, K, L, 2)
            y = y.permute(0, 3, 1, 4, 2)
            y = y.reshape(B * L, 2 * channel, K)

            if self.is_linear:
                y = self.freq_feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            y = y.reshape(B, L, 2 * channel, K).permute(0, 2, 3, 1).reshape(B, 2 * channel, K * L)

        elif self.trans_version == 2:
            y = torch.abs(torch.fft.fft(y.reshape(B, channel, K, L)))
            y = y.permute(0, 3, 1, 2).reshape(B * L, channel, K)

            if self.is_linear:
                y = self.freq_feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = self.freq_feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        elif self.trans_version==3:
            y=torch.fft.rfft(y.reshape(B, channel,K,L))
            L= L//2+1
            real = torch.real(y)
            imag = torch.imag(y)
            y=torch.stack([real,imag],axis=-1)#(B,channel,K,L,2)
            y=y.permute(0,3,1,4,2)
            y=y.reshape(B*L,2*channel,K)
            if self.is_linear:
                y= self.freq_feature_layer(y.permute(0,2,1)).permute(0,2,1)
            else:
                y = self.freq_feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            y = y.reshape(B, L, 2 * channel, K).permute(0, 2, 3, 1).reshape(B, 2 * channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        if self.version != 0:
            if self.trans_version == 0:
                y_t = self.forward_time(y, base_shape)
                y_f = self.forward_freq(y, base_shape)
                y = y_t + y_f

                y = self.forward_feature(y, base_shape)

            elif self.trans_version == 1:
                y_t = self.forward_feature(y, base_shape)  # (B,channel,K*L)
                y_f = self.forward_freq_feature(y, base_shape)  # (B,channel,K*L)
                y_t = self.forward_time(y_t, base_shape)
                y_f = self.forward_freq(y_f, base_shape)

                y = y_t + y_f

            elif self.trans_version == 2:
                y_t = self.forward_time(y, base_shape)
                y_f = self.forward_freq(y, base_shape)
                y = y_t + y_f

                y = self.forward_feature(y, base_shape)
            elif self.trans_version == 3:
                y_t = self.forward_feature(y, base_shape)  # (B,channel,K*L)
                y_f = self.forward_freq_feature(y, base_shape)  # (B,channel,K*L)
                y_t = self.forward_time(y_t, base_shape)
                y_f = self.forward_freq(y_f, base_shape)

                y = (y_t + y_f)
            else:
                y_t = self.forward_feature(y, base_shape)  # (B,channel,K*L)
                y_f = self.forward_freq_feature(y, base_shape)  # (B,channel,K*L)
                y_t = self.forward_time(y_t, base_shape)
                y_f = self.forward_freq(y_f, base_shape)

                y = y_t + y_f

        else:
            y = self.forward_time(y, base_shape)
            y = self.forward_feature(y, base_shape)

        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip
