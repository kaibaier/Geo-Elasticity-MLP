"""绘制测试几何的真实、预测与绝对误差应力点云。"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG
from data import ElasticityArrays, ElasticityDataset, load_or_create_splits, load_or_create_statistics
from model import load_checkpoint


def main(sample_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用；按项目要求停止")
    checkpoint_path = CFG.output_dir / "main" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"请先运行正式训练: {checkpoint_path}")
    arrays = ElasticityArrays()
    splits = load_or_create_splits()
    test_ids = set(splits["test"].tolist())
    if len(sample_ids) < 3 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("至少选择 3 个互不相同的测试样本")
    if any(sample_id not in test_ids for sample_id in sample_ids):
        raise ValueError("所有可视化样本编号都必须属于独立测试集")
    stats = load_or_create_statistics(arrays, splits["train"])
    dataset = ElasticityDataset(arrays, np.asarray(sample_ids), stats)
    device = torch.device("cuda")
    model, _ = load_checkpoint(str(checkpoint_path), device)
    output_dir = CFG.output_dir / "main" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample_id in enumerate(sample_ids):
        features, labels, _ = dataset[index]
        with torch.inference_mode():
            prediction = model(features[None].to(device))[0, :, 0].cpu().numpy()
        prediction = prediction * stats.target_std + stats.target_mean
        truth = labels[:, 0].numpy() * stats.target_std + stats.target_mean
        error = np.abs(prediction - truth)
        xy = np.asarray(arrays.xy[:, :, sample_id])
        stress_min = min(float(truth.min()), float(prediction.min()))
        stress_max = max(float(truth.max()), float(prediction.max()))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, values, title in zip(
            axes,
            (truth, prediction, error),
            ("Ground truth", "Prediction", "Absolute error"),
        ):
            kwargs = {"vmin": 0, "vmax": float(error.max())} if title == "Absolute error" else {"vmin": stress_min, "vmax": stress_max}
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=9, cmap="viridis", linewidths=0, **kwargs)
            ax.set(title=title, xlabel="x", ylabel="y")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(scatter, ax=ax, shrink=0.75)
        path = output_dir / f"sample_{sample_id}.png"
        fig.suptitle(f"Test geometry {sample_id}")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[1800, 1900, 1999])
    main(parser.parse_args().sample_ids)
