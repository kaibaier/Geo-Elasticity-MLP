"""检查虚拟环境、软件版本以及实际 GPU 矩阵计算。"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    print(f"虚拟环境: {sys.prefix != sys.base_prefix}")
    print(f"平台: {platform.platform()}")
    for command in ([sys.executable, "-m", "pip", "--version"], ["git", "--version"], ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]):
        if shutil.which(command[0]) or Path(command[0]).is_file():
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            print(f"{' '.join(command)}: {result.stdout.strip() or result.stderr.strip()}")
    import numpy
    import matplotlib
    import torch

    print(f"NumPy: {numpy.__version__} | matplotlib: {matplotlib.__version__}")
    print(f"PyTorch: {torch.__version__} | PyTorch CUDA: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False；停止 GPU 验证")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.device.type == "cuda" and torch.isfinite(c).all()
    print(f"GPU 矩阵乘法成功: {tuple(c.shape)}, device={c.device}")


if __name__ == "__main__":
    main()
