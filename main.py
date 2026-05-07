import multiprocessing
import os

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger as log
from omegaconf import open_dict

from models.gd_diff.gd_diff import gd_diff

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


def init_base(cfg):
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    group_name = next(iter(cfg.keys()))
    cfg = init_base(cfg[group_name])
    seed_everything(cfg.seed)

    model = gd_diff(cfg)
    mode = cfg.training_config.mode.lower()

    if mode == "train":
        model.train()
    elif mode == "pred":
        model.test(test=1)
    else:
        raise ValueError(f"Unsupported mode: {cfg.training_config.mode}")


if __name__ == "__main__":
    main()
