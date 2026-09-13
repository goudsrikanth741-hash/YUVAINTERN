from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ExperimentConfig:
    name: str
    model: str
    augmentation: bool
    label_smoothing: float
    pretrained: bool


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    return cfg
