import argparse
import multiprocessing
import os
import sys

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from loguru import logger as log
from omegaconf import open_dict
from torch.utils.data import DataLoader

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def init_dataset(cfg):
    """Get the dataloaders from the selected data provider."""
    import importlib

    loader_module = importlib.import_module(f"data_provider.{cfg.dataset_name}")
    get_dataloaders = getattr(loader_module, "get_dataloaders")
    return get_dataloaders(log, cfg)


def init_base(cfg):
    """Initialize logging and attach the Hydra output directory."""
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(log_path, "train.log"))

    with open_dict(cfg):
        cfg.log_path = log_path

    log.info(cfg)
    return cfg


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    """Load model."""
    model_name = cfg.model_name.lower()

    if model_name == "gru":
        from models import GRU

        return GRU(cfg)
    elif model_name == "csdi":
        from models.CSDI.csdi import csdi

        return csdi(cfg)
    elif model_name == "fedformer":
        from models.FEDformer.fedformer import fedformer

        return fedformer(cfg)
    elif model_name == "timemixer":
        from models.TimeMixer.timemixer import timemixer

        return timemixer(cfg)
    elif model_name == "patchtst":
        from models.PatchTST.patchtst import patchtst

        return patchtst(cfg)
    elif model_name == "diffusion":
        from models.diffusion.diffusion import diffusion

        return diffusion(cfg)
    elif model_name == "diffusion_ja":
        from models.diffusion_ja.diffusion_ja import diffusion_ja

        return diffusion_ja(cfg)
    elif model_name == "diffusion_mr":
        from models.diffusion_mr.diffusion_mr import diffusion_mr

        return diffusion_mr(cfg)
    elif model_name == "nbeats":
        from models import NBeats

        return NBeats(cfg)
    elif model_name == "csdi_job_attention":
        from models.CSDI_job_attention.csdi_job_attention import csdi_job_attention

        return csdi_job_attention(cfg)
    elif model_name == "diffusion_wt":
        from models.diffusion_wt.diffusion_wt import diffusion_wt

        return diffusion_wt(cfg)
    elif model_name == "timefilter":
        from models.timefilter.timefilter import timefilter

        return timefilter(cfg)
    elif model_name == "timesnet":
        from models.timesnet.timesnet import timesnet

        return timesnet(cfg)
    elif model_name == "agcrn":
        from models.AGCRN.agcrn_arch import AGCRN

        return AGCRN(cfg)

    elif model_name == "itransformer":
        from models.itransformer.itransformer import itransformer

        return itransformer(cfg)
    elif model_name == "d3u":
        from models.D3U.d3u import d3u

        return d3u(cfg)
    elif model_name == "tmdm":
        from models.tmdm.tmdm import tmdm

        return tmdm(cfg)
    elif model_name == "gd_diff":
        from models.gd_diff.gd_diff import gd_diff

        return gd_diff(cfg)
    else:
        raise NotImplementedError


def build_trainer(model, scaler, cfg):
    # if cfg.trainer_name == "UTS_trainer":
    #     from Trainer.UTS_trainer import UTS_trainer

    #     return UTS_trainer(model, scaler, cfg)
    # elif cfg.trainer_name == "MTS_trainer":
    #     from Trainer.MTS_trainer import MTS_trainer

        return MTS_trainer(model, scaler, cfg)
    if cfg.trainer_name == "AGCRN_trainer":
        from Trainer.AGCRN_trainer import AGCRN_trainer

        return AGCRN_trainer(model, scaler, cfg)
    elif cfg.trainer_name in ["CSDI_trainer", "FEDformer_trainer", "TimeMixer_trainer", "gd_diff_trainer"]:
        from Trainer.CSDI_trainer import CSDI_trainer

        return CSDI_trainer(model, scaler, cfg)
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    group_name = list(cfg.keys())[0]
    cfg = init_base(cfg[group_name])
    seed_everything(cfg.seed)
    model_name = cfg.model_name.lower()

    builtin_dataloader_models = [
        "csdi",
        "fedformer",
        "timemixer",
        "patchtst",
        "diffusion",
        "diffusion_ja",
        "diffusion_mr",
        "csdi_job_attention",
        "diffusion_wt",
        "timefilter",
        "timesnet",
        "itransformer",
        "d3u",
        "tmdm",
        "gd_diff",
    ]

    if model_name in builtin_dataloader_models:
        train_loader, valid_loader, test_loader, scaler = None, None, None, None
        log.info(f"{cfg.model_name.upper()} model detected, using built-in dataloader")
    else:
        train_loader, valid_loader, test_loader, scaler = init_dataset(cfg)

    model = build_model(cfg)
    trainer = build_trainer(model, scaler, cfg)
    model = trainer.train(train_loader, valid_loader, test_loader)

    if model_name not in builtin_dataloader_models:
        torch.save(model.state_dict(), os.path.join(cfg.log_path, "model.pt"))

        metrics, real_y, pred_y = trainer.infer(test_loader)
        if cfg.save_results:
            np.savez(
                os.path.join(cfg.log_path, "result.npz"),
                real_y=real_y,
                pred_y=pred_y,
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
