"""Geo-Elasticity-MLP 的集中配置。"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    data_dir: Path = ROOT / "elasticity" / "Meshes"
    output_dir: Path = ROOT / "outputs"
    seed: int = 42
    n_points: int = 972
    n_samples: int = 2000
    train_pool: int = 1000
    n_train: int = 900
    n_val: int = 100
    test_start: int = 1800
    input_dim: int = 12
    hidden_dim: int = 128
    hidden_layers: int = 3
    batch_size: int = 16
    epochs: int = 1000
    patience: int = 15
    learning_rate: float = 1e-3
    num_workers: int = 0


CFG = Config()
