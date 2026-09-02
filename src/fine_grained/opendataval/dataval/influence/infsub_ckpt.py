"""InfluenceSubsample with progressive checkpoints.

Computes influence via subsampled models (Feldman & Zhang 2020) while capturing
data-value snapshots at intermediate model counts (e.g., after 10, 50, 100 models).
Each snapshot is materialized as a separate evaluator for downstream pipelines.
"""
from typing import Optional

import numpy as np
import torch
import time
from opendataval.dataval.progress import ProgressBar, progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.influence.infsub import _resolve_subset_size


class InfluenceSubsampleCKPT(DataEvaluator, ModelMixin):
    """Influence via subsamples with progressive checkpoint snapshots.

    Compute influence of each training example on validation dataset through
    subsampled influence, capturing data-value snapshots at intermediate model counts.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    num_models : int, optional
        Number of models to fit to find data values, by default 1000
    proportion : float, optional
        Proportion of data points to be in each sample, cardinality of each subset is
        :math:`(p)(num_points)`, by default 0.7
    subset_size : int, optional
        Fixed subset size (overrides proportion if specified), by default None
    random_state : RandomState, optional
        Random initial state, by default None
    verbose : bool, optional
        Print training progress and summary statistics, by default False
    perf_metric_type : str, optional
        Performance metric to use: "accuracy" (default) or "neg_loss", by default "accuracy"
    checkpoint_models : list[int], optional
        Model counts at which to save influence scores (e.g., [10, 50, 100]).
        Allows tracking how influence scores converge as more models train.
        By default None (only final result computed, no checkpoints)
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 0.7,
        subset_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
        perf_metric_type: str = "accuracy",
        checkpoint_models: Optional[list] = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.checkpoint_models_input = checkpoint_models

        if perf_metric_type not in ["accuracy", "neg_loss"]:
            raise ValueError(f"perf_metric_type must be 'accuracy' or 'neg_loss', got {perf_metric_type}")
        self.perf_metric_type = perf_metric_type

        self.checkpoints = {}
        self.checkpoint_memory_reports = {}

        print(f"[InfluenceSubsampleCKPT] Using performance metric: {self.perf_metric_type}")
        if checkpoint_models:
            print(f"[InfluenceSubsampleCKPT] Checkpoints requested at models: {sorted(checkpoint_models)}")

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self._resolved_subset_size = _resolve_subset_size(
            self.num_points, self.proportion, self.subset_size
        )
        self.influence_matrix = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))

        # Validate and setup checkpoints
        if self.checkpoint_models_input:
            checkpoint_models = sorted(set(self.checkpoint_models_input))
            # Auto-append num_models if missing
            if self.num_models not in checkpoint_models:
                checkpoint_models.append(self.num_models)
            checkpoint_models.sort()

            # Validate all checkpoints are in valid range
            for ckpt in checkpoint_models:
                if not isinstance(ckpt, int) or ckpt <= 0 or ckpt > self.num_models:
                    raise ValueError(
                        f"checkpoint_models must contain positive ints <= num_models ({self.num_models}), "
                        f"got {ckpt}"
                    )
            self.checkpoint_models = checkpoint_models
        else:
            self.checkpoint_models = [self.num_models]

        if self.verbose:
            print(f"[InfluenceSubsampleCKPT] Validated checkpoints: {self.checkpoint_models}")

        return self

    def _compute_current_values(self) -> np.ndarray:
        """Compute current influence scores from accumulated matrices."""
        msr = np.divide(
            self.influence_matrix,
            self.sample_counts,
            out=np.zeros_like(self.influence_matrix),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values with checkpoint snapshots."""
        if self.verbose:
            print("\n" + "="*70)
            print("[InfluenceSubsampleCKPT] TRAINING PHASE")
            print("="*70)
            print(f"  Number of models: {self.num_models}")
            print(f"  Subset size: {self._resolved_subset_size} ({self.proportion:.1%} of {self.num_points} points)")
            print(f"  Checkpoints at: {self.checkpoint_models}")
            print(f"  Performance metric: {self.perf_metric_type}")
            print("="*70 + "\n")

        # Start memory/time tracking for cumulative reports
        try:
            self._reset_peak_memory_stats()
        except RuntimeError:
            pass  # CUDA not available
        self._ckpt_start = self._memory_snapshot()
        self._ckpt_t0 = time.perf_counter()

        model_performances = []

        for model_idx in progress_range(self.num_models):
            subset = self.random_state.choice(
                self.num_points, self._resolved_subset_size, replace=False
            )

            if self.verbose and model_idx % 100 == 0:
                print(f"  [Model {model_idx+1}/{self.num_models}]")

            curr_model = self.pred_model.clone()

            # Extract subset as actual tensors (not Subset objects)
            x_subset = self.x_train[subset] if isinstance(self.x_train, torch.Tensor) else torch.tensor(self.x_train[subset])
            y_subset = self.y_train[subset] if isinstance(self.y_train, torch.Tensor) else torch.tensor(self.y_train[subset])

            curr_model.fit(
                x_subset,
                y_subset,
                *args,
                **kwargs,
            )

            y_valid_hat = curr_model.predict(self.x_valid)

            if self.perf_metric_type == "accuracy":
                curr_perf = self.evaluate(self.y_valid, y_valid_hat)
                metric_label = "Validation accuracy"
            else:  # neg_loss
                import torch.nn.functional as F
                if not isinstance(y_valid_hat, torch.Tensor):
                    y_valid_hat = torch.tensor(y_valid_hat, dtype=torch.float32)
                if not isinstance(self.y_valid, torch.Tensor):
                    y_valid = torch.tensor(self.y_valid, dtype=torch.long)
                else:
                    y_valid = self.y_valid
                    if y_valid.dim() > 1 and y_valid.shape[1] > 1:
                        y_valid = y_valid.argmax(dim=1)

                loss = F.cross_entropy(y_valid_hat, y_valid)
                curr_perf = -loss.item()
                metric_label = "Negative loss"

            model_performances.append(curr_perf)

            # Update influence matrices
            included = (np.bincount(subset, minlength=self.num_points) != 0).astype(int)
            self.influence_matrix[range(self.num_points), included] += curr_perf
            self.sample_counts[range(self.num_points), included] += 1

            # Check for checkpoint hit
            current_count = model_idx + 1
            if current_count in self.checkpoint_models:
                snapshot = self._compute_current_values()
                self.checkpoints[current_count] = snapshot
                self.checkpoint_memory_reports[current_count] = self._build_memory_report(
                    self._ckpt_start,
                    self._memory_snapshot(),
                    time.perf_counter() - self._ckpt_t0,
                )
                if self.verbose:
                    print(
                        f"[InfluenceSubsampleCKPT] 📍 Checkpoint @ {current_count} models: "
                        f"mean={snapshot.mean():.4f}, std={snapshot.std():.4f}"
                    )

        if self.verbose:
            metric_name = "validation accuracy" if self.perf_metric_type == "accuracy" else "negative loss"
            print("\n" + "="*70)
            print("[InfluenceSubsampleCKPT] TRAINING SUMMARY")
            print("="*70)
            print(f"  Total models trained: {len(model_performances)}")
            print(f"  Metric: {metric_name}")
            print(f"  Mean: {np.mean(model_performances):.4f}")
            print(f"  Std: {np.std(model_performances):.4f}")
            print(f"  Min: {np.min(model_performances):.4f}")
            print(f"  Max: {np.max(model_performances):.4f}")
            print(f"  Checkpoints saved: {len(self.checkpoints)}")
            print("="*70 + "\n")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point (final checkpoint)."""
        return self.checkpoints[self.num_models]

    def get_checkpoint_evaluators(self) -> list["InfluenceSubsampleCKPTView"]:
        """Get lightweight proxy evaluators for each checkpoint.

        Returns
        -------
        list[InfluenceSubsampleCKPTView]
            One proxy per checkpoint, materialized in sorted order.
            Empty list if only one checkpoint (i.e., no expansion needed).
        """
        if len(self.checkpoints) <= 1:
            return []

        return [
            InfluenceSubsampleCKPTView(self, count)
            for count in sorted(self.checkpoints.keys())
        ]


class InfluenceSubsampleCKPTView(DataEvaluator):
    """Read-only proxy view of a InfluenceSubsampleCKPT checkpoint.

    Represents influence scores from a training run at a specific model count.
    Used by the evaluation pipeline (noisy_detection, remove_high_low, save_dataval, etc.)
    to report checkpoint snapshots as separate evaluators with their own memory/time reports.
    """

    def __init__(self, parent: InfluenceSubsampleCKPT, checkpoint_at: int):
        super().__init__()
        self.parent = parent
        self.checkpoint_at = checkpoint_at
        self.pred_model = parent.pred_model
        # Copy training phase memory report from checkpoint (flat dict like normal evaluators)
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
        # Eval phase will be populated by downstream pipeline (remains None until then)
        self.memory_report_eval = None

    def train_data_values(self, *args, **kwargs):
        """No-op: training already done by parent."""
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return the snapshot scores at this checkpoint."""
        return self.parent.checkpoints[self.checkpoint_at]

    @property
    def data_values(self) -> np.ndarray:
        """Return cached checkpoint scores without nested memory_report transformation.

        Overrides parent to keep memory_report flat (like InfluenceSubsample),
        avoiding the 'train'/'eval'/'combined' nesting that happens during evaluation.
        """
        return self.evaluate_data_values()

    def __repr__(self) -> str:
        return (
            f"InfluenceSubsampleCKPT@{self.checkpoint_at}"
            f"(num_models={self.checkpoint_at}, proportion={self.parent.proportion})"
        )

    __str__ = __repr__
