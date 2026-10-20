"""按完整几何样本读取、划分和标准化 Geo-FNO Elasticity。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CFG, Config


NAMES = {
    "xy": "Random_UnitCell_XY_10.npy",
    "theta": "Random_UnitCell_theta_10.npy",
    "sigma": "Random_UnitCell_sigma_10.npy",
    "rr": "Random_UnitCell_rr_10.npy",
}


@dataclass(frozen=True)
class Statistics:
    input_mean: np.ndarray  # (12,)
    input_std: np.ndarray  # (12,)
    target_mean: float
    target_std: float


class ElasticityArrays:
    def __init__(self, config: Config = CFG) -> None:
        self.config = config
        self.xy = np.load(config.data_dir / NAMES["xy"], mmap_mode="r")
        self.theta = np.load(config.data_dir / NAMES["theta"], mmap_mode="r")
        self.sigma = np.load(config.data_dir / NAMES["sigma"], mmap_mode="r")
        expected = {
            "xy": (config.n_points, 2, config.n_samples),
            "theta": (config.n_samples, 10),
            "sigma": (config.n_points, config.n_samples),
        }
        for name, shape in expected.items():
            array = getattr(self, name)
            if array.shape != shape or not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{name}: 预期数值数组 {shape}，实际 {array.shape}, {array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} 存在 NaN 或 Inf")

    def load_rr(self, sample_id: int | None = None) -> np.ndarray:
        """仅供检查或绘图；模型输入不会使用 rr。"""
        rr = np.load(self.config.data_dir / NAMES["rr"], mmap_mode="r")
        if rr.shape != (42, self.config.n_samples):
            raise ValueError(f"rr 形状错误: {rr.shape}")
        return rr if sample_id is None else np.asarray(rr[:, sample_id])


def expected_splits(config: Config = CFG) -> dict[str, np.ndarray]:
    if config.n_train + config.n_val != config.train_pool:
        raise ValueError("训练和验证样本数必须合计为前 1000 个样本")
    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(config.train_pool)
    return {
        "train": shuffled[: config.n_train],
        "val": shuffled[config.n_train :],
        "test": np.arange(config.test_start, config.n_samples),
    }


def load_or_create_splits(config: Config = CFG) -> dict[str, np.ndarray]:
    path = config.output_dir / "split_indices.npz"
    expected = expected_splits(config)
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files) != set(expected):
                raise ValueError(f"划分文件内容异常: {path}")
            splits = {key: np.asarray(saved[key], dtype=np.int64) for key in expected}
        if any(not np.array_equal(splits[key], expected[key]) for key in expected):
            raise ValueError(f"划分文件与当前配置不一致: {path}")
        return splits
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **expected)
    return expected


def compute_statistics(arrays: ElasticityArrays, train_ids: np.ndarray) -> Statistics:
    # 原始数组中的最后一维是样本编号；只索引训练几何样本。
    xy = np.asarray(arrays.xy[:, :, train_ids], dtype=np.float64)
    theta = np.asarray(arrays.theta[train_ids, :], dtype=np.float64)
    sigma = np.asarray(arrays.sigma[:, train_ids], dtype=np.float64)
    input_mean = np.concatenate((xy.mean(axis=(0, 2)), theta.mean(axis=0)))
    input_std = np.concatenate((xy.std(axis=(0, 2)), theta.std(axis=0)))
    if np.any(input_std <= 0):
        raise ValueError("某个输入特征在训练集中没有变化，无法标准化")
    target_std = float(sigma.std())
    if target_std <= 0:
        raise ValueError("训练集标签没有变化，无法标准化")
    return Statistics(input_mean, input_std, float(sigma.mean()), target_std)


def load_or_create_statistics(
    arrays: ElasticityArrays, train_ids: np.ndarray, config: Config = CFG
) -> Statistics:
    path = config.output_dir / "normalization.npz"
    if path.exists():
        with np.load(path) as saved:
            stats = Statistics(
                np.asarray(saved["input_mean"]),
                np.asarray(saved["input_std"]),
                float(saved["target_mean"]),
                float(saved["target_std"]),
            )
            saved_train_ids = np.asarray(saved["train_ids"], dtype=np.int64)
        if not np.array_equal(saved_train_ids, train_ids):
            raise ValueError(f"标准化参数对应不同的训练集: {path}")
        if stats.input_mean.shape != (12,) or stats.input_std.shape != (12,):
            raise ValueError(f"标准化参数形状错误: {path}")
        return stats
    stats = compute_statistics(arrays, train_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        input_mean=stats.input_mean,
        input_std=stats.input_std,
        target_mean=stats.target_mean,
        target_std=stats.target_std,
        train_ids=train_ids,
    )
    return stats


class ElasticityDataset(Dataset):
    def __init__(self, arrays: ElasticityArrays, ids: np.ndarray, stats: Statistics) -> None:
        self.arrays = arrays
        self.ids = np.asarray(ids, dtype=np.int64)
        self.mean = stats.input_mean.astype(np.float32)
        self.std = stats.input_std.astype(np.float32)
        self.target_mean = np.float32(stats.target_mean)
        self.target_std = np.float32(stats.target_std)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_id = int(self.ids[index])
        xy = np.asarray(self.arrays.xy[:, :, sample_id], dtype=np.float32)
        theta = np.asarray(self.arrays.theta[sample_id, :], dtype=np.float32)
        features = np.empty((self.arrays.config.n_points, 12), dtype=np.float32)
        features[:, :2] = xy
        features[:, 2:] = theta
        features = (features - self.mean) / self.std
        target = np.asarray(self.arrays.sigma[:, sample_id], dtype=np.float32)
        target = ((target - self.target_mean) / self.target_std)[:, None]
        return torch.from_numpy(features), torch.from_numpy(target), sample_id


def make_loader(
    dataset: ElasticityDataset, batch_size: int, shuffle: bool, seed: int = CFG.seed
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
