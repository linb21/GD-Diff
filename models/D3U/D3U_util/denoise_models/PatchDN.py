

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import argparse
from einops import repeat
from .dm_layers.embedders import Time_series_PatchEmbed
from .dm_layers.PatchTST_layers import positional_encoding
# from model9_NS_transformer.ns_layers.RevIN import RevIN

try:
    from timm.models.vision_transformer import Attention, Mlp
except ModuleNotFoundError:
    class Attention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, **kwargs):
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            bsz, seq_len, dim = x.shape
            qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.embedding_table = nn.Linear(d_model, hidden_size , bias=True)

    def forward(self, y):
        embeddings=self.embedding_table(y)

        return embeddings


#################################################################################
#                                 Core PatchDN Model                                #
#################################################################################

class PatchDNBlock(nn.Module):
    """
    A PatchDN block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 处理大批次：如果批次大小过大（>65535），分批处理以避免 CUDA kernel 配置错误
        B, N, C = x.shape
        max_batch_size = 1024  # CUDA kernel 通常支持的最大批次大小
        
        if B > max_batch_size:
            # 分批处理 Attention
            norm1_x = self.norm1(x)
            attn_outputs = []
            for i in range(0, B, max_batch_size):
                end_idx = min(i + max_batch_size, B)
                attn_input = modulate(norm1_x[i:end_idx], shift_msa[i:end_idx], scale_msa[i:end_idx])
                attn_outputs.append(self.attn(attn_input))
            attn_output = torch.cat(attn_outputs, dim=0)
        else:
            attn_output = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP 部分也可以分批处理（如果需要）
        if B > max_batch_size:
            norm2_x = self.norm2(x)
            mlp_outputs = []
            for i in range(0, B, max_batch_size):
                end_idx = min(i + max_batch_size, B)
                mlp_input = modulate(norm2_x[i:end_idx], shift_mlp[i:end_idx], scale_mlp[i:end_idx])
                mlp_outputs.append(self.mlp(mlp_input))
            mlp_output = torch.cat(mlp_outputs, dim=0)
        else:
            mlp_output = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PatchDN.
    """
    def __init__(self, hidden_size, patch_num, context_window):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_size*patch_num, context_window , bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        c : condition, [bsz, d_model]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PatchDN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        MTS_args,
        depth=2,
        mlp_ratio=4.0,
        learn_sigma=False,
        pe='zeros',
        learn_pe=True,

    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels =MTS_args.enc_in
        self.out_channels =self.in_channels * 2 if learn_sigma else self.in_channels
        self.num_heads = MTS_args.n_heads_d
        self.context_window= MTS_args.pred_len*MTS_args.unit_len
        self.cond_seq_len = MTS_args.seq_len*MTS_args.unit_len
        self.condition_hidden_size= MTS_args.d_model_c
        self.hidden_size= MTS_args.d_model_d
        self.use_pretraining_condition = MTS_args.use_pretraining_condition
        self.cond_model = getattr(MTS_args, "cond_model", "PatchTST")

        ###patch parameters
        self.patch_size = MTS_args.patch_size
        self.stride = MTS_args.stride
        self.padding_patch = MTS_args.padding_patch

        self.x_embedder = Time_series_PatchEmbed( self.context_window,self.patch_size,self.stride,self.padding_patch,self.hidden_size)
        patch_num= self.x_embedder.patch_num
        self.condition_patch_num = int((self.cond_seq_len - self.patch_size) / self.stride + 2)

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        if self.use_pretraining_condition:
          self.y_embedder = LabelEmbedder(self.context_window, self.hidden_size)
        else:
          cond_input_dim = self.condition_hidden_size if self.cond_model == "iTransformer" else self.condition_patch_num * self.condition_hidden_size
          self.y_embedder = LabelEmbedder(cond_input_dim, self.hidden_size)

        # self.pos_embed = nn.Parameter(torch.zeros(1, patch_num, self.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            PatchDNBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, patch_num, self.context_window)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.patch_num ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.normal_(self.x_embedder.value_embedding.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self,x,t,y):
        # print(f'y.shape: {y.shape}')
        """
        Forward pass of DiT.
        x: y^k or r^k, [bsz, pred_len, n_vars]
        t: time step,  [bsz, ]                          
        y: condition,  [bsz, pred_len, n_vars] if self.use_pretraining_condition else [bsz, n_vars, patch_num, d_model]

        """
        x = x.permute(0,2,1)                 # [bsz, n_vars, pred_len]
        bsz, n_vars, pred_len=x.shape

        # patching
        x = self.x_embedder(x)               # [bsz*n_vars, patch_num, d_model]
        # time step embedding
        t = repeat(t, "b -> b d", d=n_vars).reshape(-1)                           # t: [bsz * n_vars]
        t = self.t_embedder(t)                                                    # t: [bsz * n_vars, D]
            
        if self.use_pretraining_condition:
            # y:[bsz, pred_len, n_vars]
            y = y.permute(0,2,1).reshape(bsz * n_vars, pred_len)
        else:
            # y:[bsz, n_vars, patch_num, d_model]
            _, _, patch_num, d_model = y.shape
            y = y.reshape(bsz * n_vars, patch_num * d_model)
        
        # print(f'y.shape: {y.shape}')
        y = self.y_embedder(y)                                                    # y: [bs * n_vars, D]
        c = t + y                                                                 # c: [bs * n_vars, D]
         
        for block in self.blocks:
            x = block(x, c)                                                       # u: [bs * n_vars x patch_num x d_model]

        x = self.final_layer(x, c)                                                # x: [bs * n_vars x context_window]

        x = x.reshape(bsz, n_vars, pred_len).permute(0,2,1)                       # x: [bs, n_vars, pred_len]
        return x






    
