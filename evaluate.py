"""只使用独立测试集评估正式训练得到的最佳模型。"""

import csv
import json

import numpy as np
import torch

from config import CFG
from data import ElasticityArrays, ElasticityDataset, load_or_create_splits, load_or_create_statistics, make_loader
from model import load_checkpoint


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用；按项目要求停止")
    checkpoint_path = CFG.output_dir / "main" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"请先运行正式训练: {checkpoint_path}")
    arrays = ElasticityArrays()
    splits = load_or_create_splits()
    stats = load_or_create_statistics(arrays, splits["train"])
    loader = make_loader(ElasticityDataset(arrays, splits["test"], stats), CFG.batch_size, False)
    device = torch.device("cuda")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    absolute_sum = 0.0
    square_sum = 0.0
    point_count = 0
    per_sample: list[dict] = []
    with torch.inference_mode():
        for features, labels, ids in loader:
            predicted = model(features.to(device)).cpu().numpy()
            truth = labels.numpy()
            predicted = predicted * stats.target_std + stats.target_mean
            truth = truth * stats.target_std + stats.target_mean
            for k, sample_id in enumerate(ids.tolist()):
                diff = predicted[k, :, 0].astype(np.float64) - truth[k, :, 0].astype(np.float64)
                reference = truth[k, :, 0].astype(np.float64)
                abs_sum = float(np.abs(diff).sum())
                sq_sum = float(np.square(diff).sum())
                reference_norm = float(np.linalg.norm(reference))
                if reference_norm == 0:
                    raise ValueError(f"样本 {sample_id} 的真实应力场 L2 范数为 0")
                absolute_sum += abs_sum
                square_sum += sq_sum
                point_count += diff.size
                per_sample.append(
                    {
                        "sample_id": sample_id,
                        "mae": abs_sum / diff.size,
                        "rmse": float(np.sqrt(sq_sum / diff.size)),
                        "relative_l2": float(np.sqrt(sq_sum) / reference_norm),
                    }
                )
    metrics = {
        "test_geometries": len(per_sample),
        "test_points": point_count,
        "mae": absolute_sum / point_count,
        "rmse": float(np.sqrt(square_sum / point_count)),
        "mean_relative_l2": float(np.mean([row["relative_l2"] for row in per_sample])),
        "checkpoint_best_epoch": checkpoint["best_epoch"],
    }
    output_dir = CFG.output_dir / "main"
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "test_per_sample.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
