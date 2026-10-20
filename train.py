"""监督学习训练、验证与 GPU 冒烟测试。"""

import argparse
import csv
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import CFG
from data import (
    ElasticityArrays,
    ElasticityDataset,
    load_or_create_splits,
    load_or_create_statistics,
    make_loader,
)
from model import ConditionalMLP, load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(
    model: ConditionalMLP,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_elements = 0
    for features, target, _ in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                prediction = model(features)
                loss = criterion(prediction, target)
        count = target.numel()
        total_loss += float(loss.item()) * count
        total_elements += count
    return total_loss / total_elements


def train(smoke_test: bool = False) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用；按项目要求停止，不自动切换 CPU")
    device = torch.device("cuda")
    set_seed(CFG.seed)
    arrays = ElasticityArrays()
    splits = load_or_create_splits()
    stats = load_or_create_statistics(arrays, splits["train"])
    train_ids = splits["train"][:32] if smoke_test else splits["train"]
    val_ids = splits["val"][:16] if smoke_test else splits["val"]
    train_loader = make_loader(ElasticityDataset(arrays, train_ids, stats), CFG.batch_size, True)
    val_loader = make_loader(ElasticityDataset(arrays, val_ids, stats), CFG.batch_size, False)
    model = ConditionalMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.MSELoss()
    output_dir = CFG.output_dir / ("smoke" if smoke_test else "main")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    print(f"设备: {torch.cuda.get_device_name(0)} | 参数量: {model.parameter_count}")
    if smoke_test:
        features, targets, _ = next(iter(train_loader))
        with torch.no_grad():
            predicted = model(features.to(device))
        assert features.shape == (CFG.batch_size, CFG.n_points, CFG.input_dim)
        assert targets.shape == predicted.shape == (CFG.batch_size, CFG.n_points, 1)
        print(f"Batch 形状: X={tuple(features.shape)}, y={tuple(targets.shape)}")
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    log_rows: list[dict] = []
    started = time.perf_counter()
    epochs = 2 if smoke_test else CFG.epochs
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, None)
        log_rows.append({"epoch": epoch, "train_mse_normalized": train_loss, "val_mse_normalized": val_loss})
        print(f"Epoch {epoch:03d}/{epochs:03d} | train={train_loss:.6f} | val={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": CFG.input_dim,
                        "hidden_dim": CFG.hidden_dim,
                        "hidden_layers": CFG.hidden_layers,
                    },
                    "best_epoch": epoch,
                    "best_val_mse_normalized": val_loss,
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1
            if no_improvement >= CFG.patience:
                print(f"验证损失连续 {CFG.patience} 轮未改善，提前停止")
                break
    elapsed = time.perf_counter() - started
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in log_rows], [row["train_mse_normalized"] for row in log_rows], label="Train")
    ax.plot([row["epoch"] for row in log_rows], [row["val_mse_normalized"] for row in log_rows], label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (normalized stress)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
    summary = {
        "mode": "smoke" if smoke_test else "full",
        "best_epoch": best_epoch,
        "best_val_mse_normalized": best_loss,
        "epochs_run": len(log_rows),
        "training_seconds": elapsed,
        "parameter_count": model.parameter_count,
        "train_geometry_count": len(train_ids),
        "val_geometry_count": len(val_ids),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if smoke_test:
        reloaded, _ = load_checkpoint(str(checkpoint_path), device)
        with torch.inference_mode():
            result = reloaded(features.to(device))
        assert result.shape == targets.shape and torch.isfinite(result).all()
        print(f"模型保存与重新加载成功: {checkpoint_path}")
    print(f"最佳 Epoch: {best_epoch} | 用时: {elapsed:.1f} 秒 | 输出: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="仅训练 32/16 个几何样本、2 个 Epoch")
    train(parser.parse_args().smoke_test)
