import torch
import torch.nn as nn
from models.tmdm.units.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.tmdm.units.SelfAttention_Family import DSAttention, AttentionLayer
from models.tmdm.units.Embed import DataEmbedding


class Model(nn.Module):
    """
    Plain Transformer without NS normalization / VAE.
    Drop-in replacement for ns_Transformer in TMDM.
    Returns same 4-tuple interface: (pred, full_out, kl_loss, z_sample)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = int(configs.pred_len * configs.unit_len)
        self.seq_len = configs.seq_len
        self.label_len = int(configs.label_len * configs.unit_len)
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # No normalization — feed raw data directly

        # Decoder input: label_len from encoder + zeros for pred_len
        x_dec_new = torch.cat(
            [x_enc[:, -self.label_len:, :],
             torch.zeros_like(x_dec[:, -self.pred_len:, :])],
            dim=1
        ).to(x_enc.device).clone()

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # No de-normalization
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], dec_out, torch.tensor(0.0, device=x_enc.device), None
