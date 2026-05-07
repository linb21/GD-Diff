import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def SMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred) / (torch.abs(true) + torch.abs(pred)))


def R2_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    true_mean = torch.mean(true)
    tss = torch.sum((true - true_mean) ** 2)
    rss = torch.sum((true - pred) ** 2)
    return 1 - (rss / tss)


def SPEARMAN_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    correlation, _ = spearmanr(
        pred.detach().cpu().numpy().flatten(),
        true.detach().cpu().numpy().flatten(),
    )
    return torch.tensor(correlation, dtype=pred.dtype, device=pred.device)


def PEARSON_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    correlation, _ = pearsonr(
        pred.detach().cpu().numpy().flatten(),
        true.detach().cpu().numpy().flatten(),
    )
    return torch.tensor(correlation, dtype=pred.dtype, device=pred.device)


def MAE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    return np.mean(np.abs(pred - true))


def RMSE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    return np.sqrt(np.mean((pred - true) ** 2))


def MAPE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    return np.mean(np.abs((true - pred) / true))


def SMAPE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    return np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))


def R2_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    true_mean = np.mean(true)
    tss = np.sum((true - true_mean) ** 2)
    rss = np.sum((true - pred) ** 2)
    return 1 - (rss / tss)


def SPEARMAN_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    correlation, _ = spearmanr(pred.flatten(), true.flatten())
    return correlation


def PEARSON_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = pred[mask]
        true = true[mask]
    correlation, _ = pearsonr(pred.flatten(), true.flatten())
    return correlation
