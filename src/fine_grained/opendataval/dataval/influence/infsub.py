from typing import Optional

import numpy as np
import torch
from opendataval.dataval.progress import ProgressBar, progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin


def _resolve_subset_size(num_points: int, proportion: float, subset_size: Optional[int]) -> int:
    """Resolve final subset size.

    If `subset_size` is provided, it takes precedence over `proportion`.
    """
    if num_points <= 0:
        return 0
    if subset_size is not None:
        try:
            resolved = int(subset_size)
        except Exception as exc:
            raise TypeError("subset_size must be an int") from exc
        if resolved <= 0:
            raise ValueError("subset_size must be >= 1")
        if resolved > num_points:
            raise ValueError(
                f"subset_size={resolved} cannot exceed num_points={num_points}"
            )
        return resolved

    # Fall back to proportion
    try:
        p = float(proportion)
    except Exception as exc:
        raise TypeError("proportion must be a float") from exc
    if not (0.0 < p <= 1.0):
        raise ValueError("proportion must be in (0, 1]")
    resolved = int(round(p * num_points))
    resolved = max(1, min(num_points, resolved))
    return resolved


class InfluenceSubsample(DataEvaluator, ModelMixin):
    """Influence computed through subsamples implementation.

    Compute influence of each training example on for the validation dataset
    through closely-related subsampled influence.

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
        :math:`(p)(num_points)`, by default 0.7 as specified by V. Feldman and C. Zhang
    subset_size : int, optional
        Fixed subset size (overrides proportion if specified), by default None
    random_state : RandomState, optional
        Random initial state, by default None
    verbose : bool, optional
        Print training progress and summary statistics, by default False
    perf_metric_type : str, optional
        Performance metric to use: "accuracy" (default) or "neg_loss" (negative cross-entropy loss).
        Accuracy measures model correctness; neg_loss measures prediction confidence.
        By default "accuracy"
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 0.7,
        subset_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
        perf_metric_type: str = "accuracy",
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        # Validate perf_metric_type
        if perf_metric_type not in ["accuracy", "neg_loss"]:
            raise ValueError(f"perf_metric_type must be 'accuracy' or 'neg_loss', got {perf_metric_type}")
        self.perf_metric_type = perf_metric_type
        print(f"[InfluenceSubsample] Using performance metric: {self.perf_metric_type}")

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Subsample Data Valuation.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self._resolved_subset_size = _resolve_subset_size(
            self.num_points, self.proportion, self.subset_size
        )
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.influence_matrix = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Influence Subsample Data Valuator by sampling from subsets of
        :math:`(p)(num_points)` cardinality and computing the performance with the
        :math:`i` data point and without the :math:`i` data point. The form of sampling
        is similar to the shapely value when :math:`p` is :math:`0.5: (V. Feldman).
        Likewise, if we sample not from the subsets of a specific cardinality but the
        uniform across all subsets, it is similar to the Banzhaf value.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        if self.verbose:
            print("\n" + "="*70)
            print("[InfluenceSubsample] TRAINING PHASE")
            print("="*70)
            print(f"  Number of models: {self.num_models}")
            print(f"  Subset size: {self._resolved_subset_size} ({self.proportion:.1%} of {self.num_points} points)")
            print(f"  Subset ratio: {self._resolved_subset_size}/{self.num_points}")
            print(f"  Performance metric: {self.perf_metric_type}")
            print("="*70 + "\n")

        model_performances = []

        for model_idx in progress_range(self.num_models):
            subset = self.random_state.choice(
                self.num_points, self._resolved_subset_size, replace=False
            )

            if self.verbose:
                print(f"\n  [Model {model_idx+1}/{self.num_models}]")
                print(f"    Subset indices: {self._resolved_subset_size} samples")

            curr_model = self.pred_model.clone()

            # Extract subset as actual tensors (not Subset objects)
            x_subset = self.x_train[subset] if isinstance(self.x_train, torch.Tensor) else torch.tensor(self.x_train[subset])
            y_subset = self.y_train[subset] if isinstance(self.y_train, torch.Tensor) else torch.tensor(self.y_train[subset])

            # Train and capture performance
            curr_model.fit(
                x_subset,
                y_subset,
                *args,
                **kwargs,
            )

            # Evaluate on validation set
            y_valid_hat = curr_model.predict(self.x_valid)

            # Compute metric based on perf_metric_type
            if self.perf_metric_type == "accuracy":
                curr_perf = self.evaluate(self.y_valid, y_valid_hat)
                metric_label = "Validation accuracy"
            else:  # neg_loss
                import torch.nn.functional as F
                # Compute cross-entropy loss
                if not isinstance(y_valid_hat, torch.Tensor):
                    y_valid_hat = torch.tensor(y_valid_hat, dtype=torch.float32)
                if not isinstance(self.y_valid, torch.Tensor):
                    y_valid = torch.tensor(self.y_valid, dtype=torch.long)
                else:
                    y_valid = self.y_valid
                    # Handle one-hot encoded labels
                    if y_valid.dim() > 1 and y_valid.shape[1] > 1:
                        y_valid = y_valid.argmax(dim=1)

                loss = F.cross_entropy(y_valid_hat, y_valid)
                curr_perf = -loss.item()  # Negative loss (higher is better)
                metric_label = "Negative loss"

            model_performances.append(curr_perf)

            if self.verbose:
                print(f"    {metric_label}: {curr_perf:.4f}")

            # Update influence matrices
            included = (np.bincount(subset, minlength=self.num_points) != 0).astype(int)
            self.influence_matrix[range(self.num_points), included] += curr_perf
            self.sample_counts[range(self.num_points), included] += 1

        if self.verbose:
            metric_name = "validation accuracy" if self.perf_metric_type == "accuracy" else "negative loss"
            print("\n" + "="*70)
            print("[InfluenceSubsample] TRAINING SUMMARY")
            print("="*70)
            print(f"  Total models trained: {len(model_performances)}")
            print(f"  Metric: {metric_name}")
            print(f"  Mean: {np.mean(model_performances):.4f}")
            print(f"  Std: {np.std(model_performances):.4f}")
            print(f"  Min: {np.min(model_performances):.4f}")
            print(f"  Max: {np.max(model_performances):.4f}")
            print("="*70 + "\n")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Influence Subsample data valuator. Finds
        the difference of average performance of all sets including data point minus
        not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.influence_matrix,
            self.sample_counts,
            out=np.zeros_like(self.influence_matrix),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point
