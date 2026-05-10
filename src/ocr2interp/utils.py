from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_config(cfg, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
