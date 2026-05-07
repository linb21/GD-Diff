import torch
import torch.nn as nn
from models.tmdm.units.Embed import DataEmbedding
import yaml
import argparse
from models.tmdm.units.tmdm_utils import *
import torch.nn as nn
import torch.nn.functional as F




class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        # out = gamma.view(-1, self.num_out) * out

        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out
class ConditionalGuidedModel(nn.Module):
    def __init__(self, config, MTS_args):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        data_dim = 2*config.real_target_dim

        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, config.real_target_dim)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=-1)
            else:
                eps_pred = torch.cat((y_t, x), dim=2)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=2)
            else:
                eps_pred = y_t
        if y_t.device.type == 'mps':
            eps_pred = self.lin1(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin2(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin3(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

        else:
            eps_pred = F.softplus(self.lin1(eps_pred, t))
            eps_pred = F.softplus(self.lin2(eps_pred, t))
            eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred = self.lin4(eps_pred)
        return eps_pred

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Model(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs, device):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config['model_config'])

        diffusion_config.diffusion.timesteps = configs.timesteps

        # Respect runtime overrides from both flattened args and nested Hydra nodes.
        runtime_diffusion = getattr(configs, "diffusion", None)
        runtime_testing = getattr(configs, "testing", None)
        if hasattr(configs, "diffusion_beta_start"):
            diffusion_config.diffusion.beta_start = float(configs.diffusion_beta_start)
        elif runtime_diffusion is not None and hasattr(runtime_diffusion, "beta_start"):
            diffusion_config.diffusion.beta_start = float(runtime_diffusion.beta_start)
        if hasattr(configs, "diffusion_beta_end"):
            diffusion_config.diffusion.beta_end = float(configs.diffusion_beta_end)
        elif runtime_diffusion is not None and hasattr(runtime_diffusion, "beta_end"):
            diffusion_config.diffusion.beta_end = float(runtime_diffusion.beta_end)
        if hasattr(configs, "testing_sampling_timesteps"):
            diffusion_config.testing.sampling_timesteps = int(configs.testing_sampling_timesteps)
        elif runtime_testing is not None and hasattr(runtime_testing, "sampling_timesteps"):
            diffusion_config.testing.sampling_timesteps = int(runtime_testing.sampling_timesteps)
        if hasattr(configs, "testing_n_z_samples"):
            diffusion_config.testing.n_z_samples = int(configs.testing_n_z_samples)
        elif runtime_testing is not None and hasattr(runtime_testing, "n_z_samples"):
            diffusion_config.testing.n_z_samples = int(runtime_testing.n_z_samples)
        self.args = configs
        self.device = device
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.vis_step = diffusion_config.diffusion.vis_step
        self.num_figs = diffusion_config.diffusion.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        self.diffussion_model = ConditionalGuidedModel(diffusion_config, self.args)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout)

        a = 0

    def forward(self, x, x_mark, y, y_t, y_0_hat, t):
        enc_out = self.enc_embedding(x, x_mark)
        dec_out = self.diffussion_model(enc_out, y_t, y_0_hat, t)

        return dec_out

