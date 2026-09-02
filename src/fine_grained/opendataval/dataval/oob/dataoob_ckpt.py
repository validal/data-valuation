"""DataOob with progressive checkpoints for data valuation."""
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
import time
from opendataval.dataval.progress import progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.oob.oob import GroupingIndex


class DataOobCKPT(DataEvaluator, ModelMixin):
    """Data Out-of-Bag with progressive checkpoint snapshots.

    Trains bagged models while capturing data value snapshots at intermediate
    checkpoint counts (e.g., after 10, 50, 100 bags).

    Parameters
    ----------
    num_models : int, optional
        Number of models to bag/aggregate, by default 1000
    proportion : float, optional
        Proportion of data points in the in-bag sample, by default 1.0
    checkpoint_models : list[int], optional
        Model counts at which to save snapshots (e.g., [10, 50, 100]).
        Auto-appends num_models if not included.
    random_state : RandomState, optional
        Random initial state, by default None
    verbose : bool, optional
        Print training progress, by default True
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 1.0,
        checkpoint_models: Optional[list] = None,
        random_state: Optional[RandomState] = None,
        verbose: bool = True,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.checkpoint_models_input = checkpoint_models

        self.checkpoints = {}
        self.checkpoint_memory_reports = {}

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data."""
        self.x_train = x_train
        self.y_train = y_train
        _ = x_valid, y_valid

        self.num_points = len(x_train)
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape
        self.max_samples = round(self.proportion * self.num_points)

        # Initialize checkpoint tracking
        if self.checkpoint_models_input:
            checkpoint_models = sorted(set(self.checkpoint_models_input))
            if self.num_models not in checkpoint_models:
                checkpoint_models.append(self.num_models)
            checkpoint_models.sort()

            for ckpt in checkpoint_models:
                if not isinstance(ckpt, int) or ckpt <= 0 or ckpt > self.num_models:
                    raise ValueError(
                        f"checkpoint_models must contain positive ints <= num_models ({self.num_models}), got {ckpt}"
                    )
            self.checkpoint_models = checkpoint_models
        else:
            self.checkpoint_models = [self.num_models]

        if self.verbose:
            print(f"[DataOobCKPT] Validated checkpoints: {self.checkpoint_models}")

        return self

    def _snapshot_values(self, oob_pred, oob_indices) -> np.ndarray:
        """Compute data values from current OOB predictions and indices."""
        data_values = np.zeros(self.num_points)
        for i, indices in oob_indices.items():
            try:
                y_train_i = self.y_train[i]
                oob_labels = y_train_i.expand((len(indices), *self.label_dim))
                oob_preds = oob_pred[indices]

                # Compute accuracy/loss between OOB predictions and true labels
                if len(oob_labels.shape) > 1:
                    accuracy = (oob_labels.argmax(dim=1) == oob_preds.argmax(dim=1)).float()
                else:
                    accuracy = (oob_labels == oob_preds.squeeze()).float()

                data_values[i] = accuracy.mean().item()
            except Exception:
                data_values[i] = 0.0

        return data_values

    def train_data_values(self, *args, **kwargs):
        """Train bagged models with checkpoint snapshots."""
        if self.verbose:
            print("\n" + "="*70)
            print("[DataOobCKPT] TRAINING PHASE")
            print("="*70)
            print(f"  Number of models: {self.num_models}")
            print(f"  In-bag proportion: {self.proportion:.1%}")
            print(f"  Checkpoints at: {self.checkpoint_models}")
            print("="*70 + "\n")

        # Reset checkpoint storage for fresh training
        self.checkpoints = {}
        self.checkpoint_memory_reports = {}

        # Memory/time tracking
        try:
            self._reset_peak_memory_stats()
        except RuntimeError:
            pass
        ckpt_start = self._memory_snapshot()
        ckpt_t0 = time.perf_counter()

        # Generate random subsets for all models upfront
        sample_dim = (self.num_models, self.max_samples)
        subsets = self.random_state.randint(0, self.num_points, size=sample_dim)

        # Initialize OOB tracking
        oob_pred = torch.zeros((0, *self.label_dim), requires_grad=False)
        oob_indices = GroupingIndex()

        for i in progress_range(self.num_models):
            in_bag = subsets[i]
            out_bag = (np.bincount(in_bag, minlength=self.num_points) == 0).nonzero()[0]

            if not out_bag.any():
                continue

            curr_model = self.pred_model.clone()

            # Extract subset as actual tensors (not Subset objects)
            x_subset = self.x_train[in_bag] if isinstance(self.x_train, torch.Tensor) else torch.tensor(self.x_train[in_bag])
            y_subset = self.y_train[in_bag] if isinstance(self.y_train, torch.Tensor) else torch.tensor(self.y_train[in_bag])

            curr_model.fit(
                x_subset,
                y_subset,
                *args,
                **kwargs,
            )

            # Batched prediction to avoid OOM
            y_hat_list = []
            batch_size = 1024
            for batch_start in range(0, len(out_bag), batch_size):
                batch_end = min(batch_start + batch_size, len(out_bag))
                batch_indices = out_bag[batch_start:batch_end]
                batch_pred = curr_model.predict(Subset(self.x_train, indices=batch_indices))
                y_hat_list.append(batch_pred)
            y_hat = torch.cat(y_hat_list, dim=0)

            if len(y_hat.shape) == 1:
                y_hat = y_hat.unsqueeze(1)

            if oob_pred.dtype != y_hat.dtype:
                y_hat = y_hat.to(oob_pred.dtype)

            oob_pred = torch.cat((oob_pred, y_hat.detach().cpu()), dim=0)
            oob_indices.add_indices(out_bag)

            # Check for checkpoint
            current_count = i + 1
            if current_count in self.checkpoint_models:
                snapshot = self._snapshot_values(oob_pred, oob_indices)
                self.checkpoints[current_count] = snapshot
                mem_report = self._build_memory_report(
                    ckpt_start,
                    self._memory_snapshot(),
                    time.perf_counter() - ckpt_t0,
                )
                self.checkpoint_memory_reports[current_count] = mem_report
                if self.verbose:
                    elapsed = mem_report.get('elapsed_seconds', 0)
                    print(
                        f"[DataOobCKPT] 📍 Checkpoint @ {current_count} models: "
                        f"mean={snapshot.mean():.4f}, std={snapshot.std():.4f}, "
                        f"time={elapsed:.1f}s"
                    )

        if self.verbose:
            print("\n" + "="*70)
            print("[DataOobCKPT] TRAINING SUMMARY")
            print("="*70)
            print(f"  Total models trained: {self.num_models}")
            print(f"  Checkpoints saved: {len(self.checkpoints)}")
            print("="*70 + "\n")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for final checkpoint."""
        return self.checkpoints[self.num_models]

    @property
    def data_values(self) -> np.ndarray:
        """Return cached checkpoint scores without nested memory_report transformation."""
        return self.evaluate_data_values()

    def get_checkpoint_evaluators(self) -> list["DataOobCKPTView"]:
        """Get lightweight proxy evaluators for each checkpoint."""
        if len(self.checkpoints) <= 1:
            return []

        return [
            DataOobCKPTView(self, count)
            for count in sorted(self.checkpoints.keys())
        ]


class DataOobCKPTView(DataEvaluator):
    """Read-only proxy view of a DataOobCKPT checkpoint."""

    def __init__(self, parent: DataOobCKPT, checkpoint_at: int):
        super().__init__()
        self.parent = parent
        self.checkpoint_at = checkpoint_at
        self.pred_model = parent.pred_model
        # Set flat training phase memory report
        self.memory_report = parent.checkpoint_memory_reports.get(checkpoint_at) or {
            'elapsed_seconds': 0.0,
            'cpu_maxrss_kb_start': None,
            'cpu_maxrss_kb_end': None,
            'cpu_rss_kb_start': None,
            'cpu_rss_kb_end': None,
            'cpu_rss_delta_kb': None,
            'cpu_phase_peak_kb': None,
            'cuda_device': None,
            'gpu_allocated_bytes_end': None,
            'gpu_reserved_bytes_end': None,
            'gpu_peak_allocated_bytes': None,
            'gpu_peak_reserved_bytes': None,
        }

    def train_data_values(self, *args, **kwargs):
        """No-op: training already done by parent."""
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return the snapshot scores at this checkpoint."""
        return self.parent.checkpoints[self.checkpoint_at]

    @property
    def data_values(self) -> np.ndarray:
        """Return cached checkpoint scores without nested memory_report transformation."""
        return self.evaluate_data_values()

    def __repr__(self) -> str:
        return (
            f"DataOobCKPT@{self.checkpoint_at}"
            f"(num_models={self.checkpoint_at}, proportion={self.parent.proportion})"
        )

    __str__ = __repr__
