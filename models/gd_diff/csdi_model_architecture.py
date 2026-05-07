import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gd_diff.graph_layer import GatedDynamicGCN


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
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
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / max(dim - 1, 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, graph_config, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.graph_type = graph_config["graph_type"]
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = conv1d_with_init(channels, 2 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.graph_layer = GatedDynamicGCN(channels, channels, graph_config["top_k"])
        self.mix_gate = nn.Linear(channels, 1)

    def forward_time(self, y, base_shape):
        batch_size, channels, num_nodes, seq_len = base_shape
        if seq_len == 1:
            return y

        y = y.reshape(batch_size, channels, num_nodes, seq_len)
        y = y.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, channels)
        y = self.time_layer(y)
        y = y.reshape(batch_size, num_nodes, seq_len, channels)
        y = y.permute(0, 3, 1, 2).reshape(batch_size, channels, num_nodes * seq_len)
        return y

    def forward_graph(self, y, base_shape, graph):
        batch_size, channels, num_nodes, seq_len = base_shape
        y = y.reshape(batch_size, channels, num_nodes, seq_len)
        y = y.permute(0, 3, 2, 1)
        y = self.graph_layer(y, graph)
        y = y.permute(0, 3, 2, 1).reshape(batch_size, channels, num_nodes * seq_len)
        return y

    def forward(self, x, cond_info, diffusion_emb, graph):
        batch_size, channels, num_nodes, seq_len = x.shape
        base_shape = x.shape
        x = x.reshape(batch_size, channels, num_nodes * seq_len)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        temporal_branch = self.forward_time(y, base_shape)
        graph_branch = self.forward_graph(y, base_shape, graph)

        combined = temporal_branch + graph_branch
        gate = torch.sigmoid(
            self.mix_gate(combined.permute(0, 2, 1).reshape(batch_size * num_nodes * seq_len, channels))
        ).reshape(batch_size, 1, num_nodes * seq_len)
        y = gate * temporal_branch + (1.0 - gate) * graph_branch
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(batch_size, cond_dim, num_nodes * seq_len)
        y = y + self.cond_projection(cond_info)

        gate_part, filter_part = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate_part) * torch.tanh(filter_part)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        x = x.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class diff_CSDI(nn.Module):
    def __init__(self, config, graph_config, inputdim=2, side_dim=None):
        super().__init__()
        self.channels = config["channels"]
        self.real_time_emb_dim = config.get("real_time_emb_dim", 0)
        self.graph_type = graph_config["graph_type"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.all_channels = self.channels + self.real_time_emb_dim
        self.input_projection = conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = conv1d_with_init(self.all_channels, self.all_channels, 1)
        self.output_projection2 = conv1d_with_init(self.all_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        if self.graph_type == "sg":
            self.node_embeddings = nn.Parameter(
                torch.randn(graph_config["num_nodes"], graph_config["embed_dim"]),
                requires_grad=True,
            )
        else:
            self.node_embeddings = None

        if self.real_time_emb_dim > 0:
            self.hour_embedding = nn.Embedding(24, self.real_time_emb_dim)
        else:
            self.hour_embedding = None

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    graph_config=graph_config,
                    side_dim=side_dim,
                    channels=self.all_channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def _build_graph(self, batch_size, seq_len, device, graph):
        if self.graph_type == "sg":
            learned_graph = F.softmax(
                F.relu(torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))),
                dim=-1,
            )
            return learned_graph.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).to(device)
        if self.graph_type == "time":
            if graph is None:
                raise ValueError("Time-varying graph input is required when graph_type='time'.")
            return graph.to(device)
        if self.graph_type == "static":
            if graph is None:
                raise ValueError("Static graph input is required when graph_type='static'.")
            return graph.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).to(device)
        raise ValueError(f"Unsupported graph_type: {self.graph_type}")

    def forward(self, x, cond_info, diffusion_step, graph=None, real_time_embed=None):
        batch_size, inputdim, num_nodes, seq_len = x.shape

        x = x.reshape(batch_size, inputdim, num_nodes * seq_len)
        x = F.relu(self.input_projection(x))
        x = x.reshape(batch_size, self.channels, num_nodes, seq_len)

        if real_time_embed is not None and self.hour_embedding is not None:
            hour = real_time_embed[:, 1, :].long()
            hour_emb = self.hour_embedding(hour).permute(0, 2, 1)
            hour_emb = hour_emb.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            x = torch.cat([x, hour_emb], dim=1)

        graph = self._build_graph(batch_size, seq_len, x.device, graph)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diffusion_emb, graph)
            skip_connections.append(skip)

        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(skip_connections))
        x = x.reshape(batch_size, self.all_channels, num_nodes * seq_len)
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x)
        return x.reshape(batch_size, num_nodes, seq_len)


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        model_cfg = config["model_config"]["model"]
        diffusion_cfg = config["model_config"]["diffusion"]
        graph_cfg = config["model_config"]["graph"]

        self.emb_time_dim = model_cfg["timeemb"]
        self.emb_feature_dim = model_cfg["featureemb"]
        self.is_unconditional = model_cfg["is_unconditional"]
        self.real_time_emb = config["model_config"].get("real_time_emb", 0)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1

        self.embed_layer = nn.Embedding(target_dim, self.emb_feature_dim)
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(diffusion_cfg, graph_cfg, input_dim, self.emb_total_dim)

        self.num_steps = diffusion_cfg["num_steps"]
        if diffusion_cfg["schedule"] == "quad":
            self.beta = np.linspace(
                diffusion_cfg["beta_start"] ** 0.5,
                diffusion_cfg["beta_end"] ** 0.5,
                self.num_steps,
            ) ** 2
        else:
            self.beta = np.linspace(
                diffusion_cfg["beta_start"],
                diffusion_cfg["beta_end"],
                self.num_steps,
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        batch_size, num_nodes, seq_len = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, num_nodes, -1)

        if feature_id is None:
            feature_embed = self.embed_layer(torch.arange(num_nodes, device=self.device))
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, seq_len, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1).permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, graph, real_time_embed, set_t=-1, is_train=1):
        batch_size = observed_data.shape[0]
        if is_train == 1:
            t = torch.randint(0, self.num_steps, [batch_size], device=self.device)
        else:
            t = torch.full((batch_size,), set_t, device=self.device, dtype=torch.long)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = current_alpha.sqrt() * observed_data + (1.0 - current_alpha).sqrt() * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t, graph=graph, real_time_embed=real_time_embed)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        return (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, graph, real_time_embed):
        loss_sum = 0
        for t in range(self.num_steps):
            loss_sum += self.calc_loss(
                observed_data,
                cond_mask,
                observed_mask,
                side_info,
                graph,
                real_time_embed,
                set_t=t,
                is_train=0,
            ).detach()
        return loss_sum / self.num_steps

    def impute(self, observed_data, cond_mask, side_info, n_samples, graph, real_time_embed):
        batch_size, num_nodes, seq_len = observed_data.shape
        imputed_samples = torch.zeros(batch_size, n_samples, num_nodes, seq_len, device=self.device)

        for sample_idx in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t], device=self.device),
                    graph=graph,
                    real_time_embed=real_time_embed,
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample = current_sample + sigma * noise

            imputed_samples[:, sample_idx] = current_sample.detach()

        return imputed_samples


class Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super().__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model_config"]["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float().permute(0, 2, 1)
        observed_mask = batch["observed_mask"].to(self.device).float().permute(0, 2, 1)
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
        feature_id = torch.arange(self.target_dim_base, device=self.device).unsqueeze(0).expand(observed_data.shape[0], -1)

        graph = batch.get("graph")
        if graph is not None:
            graph = graph.to(self.device).float()

        if self.real_time_emb:
            real_time_embed = torch.stack(
                [
                    batch["day"].to(self.device),
                    batch["hour"].to(self.device),
                    batch["minute"].to(self.device),
                ],
                dim=1,
            )
        else:
            real_time_embed = None

        return observed_data, observed_mask, observed_tp, gt_mask, feature_id, graph, real_time_embed

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        sample_size = self.num_sample_features
        self.target_dim = sample_size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for idx in range(len(observed_data)):
            indices = np.arange(self.target_dim_base)
            np.random.shuffle(indices)
            indices = indices[:sample_size]
            extracted_data.append(observed_data[idx, indices])
            extracted_mask.append(observed_mask[idx, indices])
            extracted_feature_id.append(feature_id[idx, indices])
            extracted_gt_mask.append(gt_mask[idx, indices])

        return (
            torch.stack(extracted_data, 0),
            torch.stack(extracted_mask, 0),
            torch.stack(extracted_feature_id, 0),
            torch.stack(extracted_gt_mask, 0),
        )

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, gt_mask, feature_id, graph, real_time_embed = self.process_data(batch)

        if is_train == 1 and self.target_dim_base > self.num_sample_features:
            observed_data, observed_mask, feature_id, gt_mask = self.sample_features(
                observed_data, observed_mask, feature_id, gt_mask
            )
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        if is_train == 1:
            return self.calc_loss(observed_data, cond_mask, observed_mask, side_info, graph, real_time_embed, is_train=1)
        return self.calc_loss_valid(observed_data, cond_mask, observed_mask, side_info, graph, real_time_embed)

    def evaluate(self, batch, n_samples):
        observed_data, observed_mask, observed_tp, gt_mask, _, graph, real_time_embed = self.process_data(batch)
        cond_mask = gt_mask
        target_mask = observed_mask * (1 - gt_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)

        with torch.no_grad():
            samples = self.impute(observed_data, cond_mask, side_info, n_samples, graph, real_time_embed)

        return samples, observed_data, target_mask, observed_mask, observed_tp
