from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchTrainingConfig:
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 48
    dropout: float = 0.1
    random_state: int = 42
    device: str = "cpu"


class TorchSequenceRegressor:
    expects_sequence_input = True

    def __init__(
        self,
        model_type: str,
        window_shape: tuple[int, int],
        config: TorchTrainingConfig,
    ) -> None:
        self.model_type = model_type
        self.window_shape = window_shape
        self.config = config
        self.model: Any | None = None
        self.training_loss_: float | None = None
        self.target_mean_: float = 0.0
        self.target_scale_: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchSequenceRegressor:
        torch, nn, data = _torch_modules()
        torch.manual_seed(self.config.random_state)
        x_seq = self._as_sequence(x).astype(np.float32, copy=False)
        y_array = np.asarray(y, dtype=np.float32)
        self.target_mean_ = float(np.mean(y_array))
        target_std = float(np.std(y_array))
        self.target_scale_ = target_std if target_std > 0 else 1.0
        y_values = ((y_array - self.target_mean_) / self.target_scale_).reshape(-1, 1)
        self.model = self._build_model(nn)
        self.model.to(self.config.device)
        dataset = data.TensorDataset(torch.from_numpy(x_seq), torch.from_numpy(y_values))
        generator = torch.Generator().manual_seed(self.config.random_state)
        loader = data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        final_loss = 0.0
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())
        self.training_loss_ = final_loss
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        torch, _, _ = _torch_modules()
        x_seq = self._as_sequence(np.asarray(x)).astype(np.float32, copy=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_seq), 512):
                batch = torch.from_numpy(x_seq[start : start + 512]).to(self.config.device)
                predictions.append(self.model(batch).detach().cpu().numpy().reshape(-1))
        scaled = np.concatenate(predictions).astype(float)
        return scaled * self.target_scale_ + self.target_mean_

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        torch, _, _ = _torch_modules()
        torch.save(
            {
                "model_type": self.model_type,
                "window_shape": self.window_shape,
                "config": self.config,
                "state_dict": self.model.state_dict(),
                "training_loss": self.training_loss_,
                "target_mean": self.target_mean_,
                "target_scale": self.target_scale_,
            },
            path,
        )

    def _as_sequence(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x)
        if array.ndim == 3:
            return array
        if array.ndim == 2:
            return array.reshape(array.shape[0], *self.window_shape)
        raise ValueError(f"Expected 2D or 3D window array, got shape {array.shape}")

    def _build_model(self, nn: Any) -> Any:
        window_length, feature_count = self.window_shape
        if self.model_type == "lstm":
            return _LstmRulRegressor(
                nn,
                feature_count,
                self.config.hidden_size,
                self.config.dropout,
            )
        if self.model_type == "gru":
            return _GruRulRegressor(
                nn,
                feature_count,
                self.config.hidden_size,
                self.config.dropout,
            )
        if self.model_type == "transformer":
            return _TransformerRulRegressor(
                nn,
                window_length,
                feature_count,
                self.config.hidden_size,
                self.config.dropout,
            )
        raise ValueError(f"Unknown sequence model type: {self.model_type}")


class _LstmRulRegressor:
    def __new__(cls, nn: Any, feature_count: int, hidden_size: int, dropout: float) -> Any:
        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.LSTM(
                    input_size=feature_count,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, x: Any) -> Any:
                output, _ = self.encoder(x)
                return self.head(output[:, -1, :])

        return Network()


class _GruRulRegressor:
    def __new__(cls, nn: Any, feature_count: int, hidden_size: int, dropout: float) -> Any:
        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.GRU(
                    input_size=feature_count,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, x: Any) -> Any:
                output, _ = self.encoder(x)
                return self.head(output[:, -1, :])

        return Network()


class _TransformerRulRegressor:
    def __new__(
        cls,
        nn: Any,
        window_length: int,
        feature_count: int,
        hidden_size: int,
        dropout: float,
    ) -> Any:
        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(feature_count, hidden_size)
                self.position = nn.Parameter(_zeros_position(window_length, hidden_size))
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 2,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, x: Any) -> Any:
                encoded = self.input_projection(x) + self.position
                encoded = self.encoder(encoded)
                return self.head(encoded[:, -1, :])

        return Network()


def _zeros_position(window_length: int, hidden_size: int) -> Any:
    torch, _, _ = _torch_modules()
    return torch.zeros(1, window_length, hidden_size)


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils import data
    except ImportError as exc:
        raise RuntimeError(
            "Torch sequence models require installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return torch, nn, data
