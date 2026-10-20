"""以坐标和傅里叶几何参数为条件的逐点 MLP。"""

import torch
from torch import nn

from config import CFG


class ConditionalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = CFG.input_dim,
        hidden_dim: int = CFG.hidden_dim,
        hidden_layers: int = CFG.hidden_layers,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != CFG.input_dim:
            raise ValueError(f"输入应为 (B, N, {CFG.input_dim})，实际为 {tuple(x.shape)}")
        return self.network(x)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_checkpoint(path: str, device: torch.device) -> tuple[ConditionalMLP, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ConditionalMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
