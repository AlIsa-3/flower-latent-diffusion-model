import torch
from typing import Literal
import os
import pathlib


def device_select() -> Literal["cuda", "mps", "cpu"]:
    """
    Returns the device to run calculations on based on availability.
    Defaults to cpu if cuda and mps are unavailable.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class EarlyStop:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 10,
        checkpoint_path: str | os.PathLike = "best_model.ckpt",
    ):
        if patience < 1:
            raise ValueError("patience must be >= 1")

        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        # Create parent directories if they don't exist
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_metric = float("inf")

        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def _is_improvement(self, metric: float) -> bool:
        return metric < self.best_metric - self.min_delta

    def __call__(
        self,
        metric: float,
        model: torch.nn.Module,
        epoch: int,
        optimizer=None,
    ) -> bool:
        # Avoid NaN or infinite metrics
        if not torch.isfinite(torch.tensor(metric)):
            raise ValueError(f"Invalid metric: {metric}")

        # If improvement then reset early stop counter
        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0

            checkpoint = {
                "epoch": epoch,
                "metric": metric,
                "model_state_dict": model.state_dict(),
            }

            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(checkpoint, self.checkpoint_path)

        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def status(self):
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "patience": self.patience,
            "early_stop": self.early_stop,
        }
