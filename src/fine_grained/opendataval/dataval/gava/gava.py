import random
from typing import Optional
import numpy as np
import torch
from opendataval.dataval.progress import progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class GAVA(DataEvaluator, ModelLessMixin):
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 100,
        subset_size_ratio: float = 0.5,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        batch_size: int = 64,
        epochs: int = 100,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.subset_size_ratio = subset_size_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)

    def evaluate_subset(self, subset_indices):
        indices = list(subset_indices)

        x_subset = torch.stack([self.x_train[i] for i in indices])
        y_subset = torch.stack([torch.as_tensor(self.y_train[i], dtype=torch.float32) for i in indices])

        input_dim = x_subset.shape[1]
        output_dim = y_subset.shape[1]

        model = torch.nn.Linear(input_dim, output_dim)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = model(x_subset)
            loss = loss_fn(preds, y_subset.argmax(dim=1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x_val = torch.stack([self.x_valid[i] for i in range(len(self.x_valid))])
            y_val = torch.stack([torch.as_tensor(self.y_valid[i], dtype=torch.float32) for i in range(len(self.y_valid))])
            val_preds = model(x_val)
            acc = (val_preds.argmax(dim=1) == y_val.argmax(dim=1)).float().mean()

        return acc.item()

    def train_data_values(self, *args, **kwargs):
        n = len(self.x_train)
        subset_size = int(n * self.subset_size_ratio)
        freq = np.zeros(n)

        # Initialize population
        population = [self.random_state.choice(n, subset_size, replace=False) for _ in range(self.population_size)]

        for gen in progress_range(self.generations, "GAVA Generations"):
            scored = [(subset, self.evaluate_subset(subset)) for subset in population]
            scored.sort(key=lambda x: -x[1])
            top_k = scored[:self.population_size // 2]

            for subset, _ in top_k:
                freq[subset] += 1

            top_subsets = [subset for subset, _ in top_k]

            next_gen = []
            for _ in range(self.population_size):
                parent1 = random.choice(top_subsets)
                parent2 = random.choice(top_subsets)

                cut = self.random_state.randint(1, subset_size - 1)
                child = np.concatenate([parent1[:cut], parent2[cut:]])

                if self.random_state.rand() < self.mutation_rate:
                    idx = self.random_state.randint(0, len(child))
                    child[idx] = self.random_state.randint(0, n)

                child = np.unique(child)
                if len(child) > subset_size:
                    child = child[:subset_size]
                elif len(child) < subset_size:
                    needed = subset_size - len(child)
                    extra = self.random_state.choice(
                        np.setdiff1d(np.arange(n), child),
                        size=needed,
                        replace=False
                    )
                    child = np.concatenate([child, extra])

                next_gen.append(child)

            population = next_gen

        self.data_values = freq / freq.max() if freq.max() > 0 else freq
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_values
