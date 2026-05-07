import torch
import torch.nn as nn
from models.itransformer import iTransformer


class Model(nn.Module):
    """
    iTransformer condition model adapter for TMDM.

    The native iTransformer returns only the forecast horizon. TMDM expects the
    condition model to return (pred, full_out, kl_loss, z_sample), where
    full_out has the same label_len + pred_len time length as batch_y.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = int(configs.pred_len * configs.unit_len)
        self.backbone = iTransformer.Model(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        pred = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        history_len = x_dec.shape[1] - self.pred_len
        history_label = x_enc[:, -history_len:, :]
        full_out = torch.cat([history_label, pred], dim=1)
        kl_loss = torch.zeros((), device=x_enc.device, dtype=x_enc.dtype)
        return pred, full_out, kl_loss, None
