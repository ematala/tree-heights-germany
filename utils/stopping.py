import os

from numpy import Inf

from ..models.base import BaseModel


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        model: BaseModel,
        model_dir: str,
        model_name: str,
        patience: int = 10,
        delta: float = 0.0,
    ):
        """
        Args:
            model (Module): The model to save.
            model_path (str): Path to save the model.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0.0
        """
        self.model = model
        self.path = os.path.join(model_dir, f"{model_name}.pt")
        self.patience = patience
        self.delta = delta

        self.best_score = Inf
        self.stop = False
        self.counter = 0

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
            self.model.save(self.path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.model.save(self.path)
