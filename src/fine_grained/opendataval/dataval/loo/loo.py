import random
from opendataval.dataval.progress import ProgressBar, progress_range
from typing import Optional
from opendataval.dataval.progress import ProgressBar, progress_range
import numpy as np
from opendataval.dataval.progress import ProgressBar, progress_range
import torch
import pandas as pd
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, ModelLessMixin

class LOORemovalRanker(DataEvaluator, ModelLessMixin):
    def __init__(
        self,
        min_subset_size: int = 10,
        max_steps: Optional[int] = None,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        random_state: Optional[RandomState] = None,
    ):
        self.min_subset_size = min_subset_size
        self.max_steps = max_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = check_random_state(random_state)

    def evaluate_model(self, x_train, y_train, x_valid, y_valid):
        input_dim = x_train.shape[1]
        num_classes = len(torch.unique(y_train))

        model = torch.nn.Linear(input_dim, num_classes).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        # Ensure y_train and y_valid are 1D tensors of class indices
        if y_train.ndim > 1:
            y_train = torch.argmax(y_train, dim=1)
        y_train = y_train.type(torch.long)

        if y_valid.ndim > 1:
            y_valid = torch.argmax(y_valid, dim=1)
        y_valid = y_valid.type(torch.long)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = model(x_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = model(x_valid)
            acc = (preds.argmax(dim=1) == y_valid).float().mean().item()

        return acc

    def train_data_values(self, save_csv_path: Optional[str] = None, *args, **kwargs):
        x_train, y_train = self.x_train.to(self.device), self.y_train.to(self.device)
        x_valid, y_valid = self.x_valid.to(self.device), self.y_valid.to(self.device)

        n = x_train.shape[0]
        current_indices = list(range(n))

        removed_indices = []
        loo_scores = []
        val_accuracies = []

        step = 0
        pbar = ProgressBar(desc="LOO Pruning", total=self.max_steps or n - self.min_subset_size)
        while len(current_indices) > self.min_subset_size:
            base_acc = self.evaluate_model(x_train[current_indices], y_train[current_indices], x_valid, y_valid)

            # Evaluate impact of removing each point
            impacts = []
            for idx in current_indices:
                reduced = [i for i in current_indices if i != idx]
                acc = self.evaluate_model(x_train[reduced], y_train[reduced], x_valid, y_valid)
                loo = base_acc - acc
                impacts.append((idx, loo, acc))

            # Remove point with smallest impact
            idx_to_remove, loo_val, acc_after = min(impacts, key=lambda x: x[1])
            current_indices.remove(idx_to_remove)

            removed_indices.append(idx_to_remove)
            loo_scores.append(loo_val)
            val_accuracies.append(acc_after)

            step += 1
            pbar.update(1)
            if self.max_steps and step >= self.max_steps:
                break

        pbar.close()

        self.removal_ranking = removed_indices  # most important first
        self.loo_values = loo_scores
        self.val_acc_history = val_accuracies

        if save_csv_path:
            df = pd.DataFrame({
                "removed_index": self.removal_ranking,
                "loo_value": self.loo_values,
                "val_accuracy_after_removal": self.val_acc_history
            })
            df.to_csv(save_csv_path, index=False)
            print(f"📁 Saved LOO pruning log to '{save_csv_path}'")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        values = np.zeros(len(self.x_train))
        for rank, idx in enumerate(self.removal_ranking):
            values[idx] = len(self.removal_ranking) - rank
        return values / values.max()

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
