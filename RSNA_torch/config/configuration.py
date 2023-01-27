import hydra
import logging
from typing import List
from pathlib import Path
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def register_configs(path: Path = None, overrides: List = None) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=path):
        cfg = hydra.compose(config_name="default", overrides=overrides)
        log.info("Configuration loading complete.")
        return cfg
