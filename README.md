# Geo-Elasticity-MLP

用一个条件 MLP，从不规则孔洞的二维点坐标和 10 个傅里叶形状系数预测逐点标量应力。项目重点是可复现的数据划分、PyTorch 监督训练、独立测试与可视化。没有物理约束损失或复杂神经算子。

## 数据

原始文件位于项目根目录下不纳入 Git 的 `elasticity/Meshes/`，脚本直接读取，不会复制或移动：

| 文件 | 形状 | 含义 |
| --- | --- | --- |
| `Random_UnitCell_XY_10.npy` | `(972, 2, 2000)` | `XY[:, :, k]` 是第 `k` 个几何样本的 972 个二维节点坐标 |
| `Random_UnitCell_theta_10.npy` | `(2000, 10)` | `theta[k, :]` 是控制孔洞边界的 10 个傅里叶系数 |
| `Random_UnitCell_sigma_10.npy` | `(972, 2000)` | `sigma[:, k]` 是对应节点顺序的标量应力标签 |
| `Random_UnitCell_rr_10.npy` | `(42, 2000)` | 孔洞半径的 42 个角度采样，仅用于数据检查或几何绘图 |

样本 `k` 的输入是 `concat(XY[:, :, k], repeat(theta[k, :], 972))`，形状 `(972, 12)`；标签是 `sigma[:, k]`，形状 `(972, 1)`。不改变节点顺序。标签只称为数据集给出的标量应力，不额外假定它是 von Mises 应力或应力张量的某个分量。

前 1000 个完整几何样本（编号 0–999）使用种子 42 随机划分成 900 个训练样本与 100 个验证样本。最后 200 个（1800–1999）是独立测试集。编号 1000–1799 的 800 个样本暂不使用。划分保存到 `outputs/split_indices.npz`。输入的 12 个特征和标签都用**仅训练集**计算的均值、标准差标准化，参数保存到 `outputs/normalization.npz`。验证与测试复用这些参数；所有最终指标在原始应力尺度上计算。

## 环境

适用 WSL2、Python 3.12 和 NVIDIA RTX 5060。使用 Windows 主机提供给 WSL2 的驱动，不需在 WSL2 安装 Linux NVIDIA 驱动。PyTorch 2.13.0 官方 CUDA 13.0 构建适用于 Blackwell，具体版本和命令见 [PyTorch 安装页面](https://pytorch.org/get-started/previous-versions/) 与 [PyTorch 2.13 发布说明](https://pytorch.org/blog/pytorch-2-13-release-blog/)。

```bash
cd ~/pinn
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_env.py
```

以后每次新开 WSL2 终端，可进入项目并激活已有环境：

```bash
cd ~/pinn
source .venv/bin/activate
which python                 # 应显示 /home/zsn/pinn/.venv/bin/python
python check_env.py
deactivate                   # 完成后退出虚拟环境
```

激活只影响当前终端，不会重新安装依赖。也可以不激活，继续使用本 README 中的 `.venv/bin/python ...` 命令，效果相同。

`requirements.txt` 包含官方 PyTorch CUDA wheel 索引和必要的 NumPy、matplotlib。它会按 PyTorch wheel 的依赖安装 CUDA 运行库；无需单独执行 `apt install nvidia-driver` 或安装系统 CUDA Toolkit。若 `check_env.py` 中 `torch.cuda.is_available()` 为 `False`，先修复 WSL2/GPU 环境，不要改为 CPU 运行训练。

## 模型与运行

MLP 使用 12 维输入、3 个宽度为 128 的隐藏层、ReLU 和 1 维输出。每个 Batch 的输入与标签分别为 `(B, 972, 12)`、`(B, 972, 1)`。训练使用 MSE、Adam、学习率 0.001、Batch Size 16、最多 1000 Epoch、验证损失早停 patience 15。最优模型保存在 `outputs/main/best_model.pt`。

```bash
cd ~/pinn
.venv/bin/python train.py --smoke-test  # 32/16 个几何样本，2 Epoch，仅验证流程
.venv/bin/python train.py               # 正式训练：900/100 个几何样本
.venv/bin/python evaluate.py            # 正式模型，独立测试集 200 个样本
.venv/bin/python visualize.py           # 默认绘制测试样本 1800、1900、1999
```

可用 `visualize.py --sample-ids 1801 1901 1998` 指定至少 3 个不同的测试样本。冒烟测试使用单独的 `outputs/smoke/`，不会作为正式评估的模型。正式训练不会在创建项目时自动启动。

训练日志在 `outputs/main/training_log.csv`，曲线在 `loss_curve.png`，参数量、最优 Epoch 和用时在 `summary.json`。`evaluate.py` 生成 `test_metrics.json` 和 `test_per_sample.csv`，其中 MAE、RMSE 汇总所有测试节点；Relative L2 先逐个完整几何样本计算，再对 200 个样本取平均。`visualize.py` 生成 `outputs/main/figures/sample_*.png`，每张显示真实场、预测场、绝对误差场，按原始不规则坐标绘制散点，不 reshape 成规则网格。

## 文件

- `config.py`：集中配置数据路径、划分、模型和训练参数。
- `data.py`：数组校验、划分、训练集标准化、Dataset、DataLoader；可单独读取 `rr`。
- `model.py`：条件 MLP、参数量和模型重载。
- `train.py`：训练、验证、最佳模型保存、CSV、曲线和冒烟测试。
- `evaluate.py`：独立测试与原始尺度指标。
- `visualize.py`：至少三个测试样本的点云应力图。
- `check_env.py`：环境、CUDA 和实际 GPU 矩阵乘法检查。
- `requirements.txt`：项目依赖。

## Git 提交建议

仓库已在 `~/pinn` 初始化；由项目使用者执行 `git add` 与 `git commit`，不推送。原始数据、已有的 Geo-FNO 独立仓库、虚拟环境、模型权重与运行结果由 `.gitignore` 排除。

```bash
git status --short
git add .gitignore requirements.txt config.py data.py model.py check_env.py README.md
git commit -m "Set up reproducible elasticity dataset and MLP"
git add train.py evaluate.py visualize.py
git commit -m "Add supervised training, evaluation and visualizations"
```

建议提交前查看 `git diff --cached --stat`，确认没有加入原始数据或大文件。README 里的冒烟测试结果可在本人运行后补入另一次提交。
