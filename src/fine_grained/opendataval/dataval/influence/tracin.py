"""TracIn - Tracing Gradient Descent for Data Valuation.

TracIn computes data influence by tracking model gradients across multiple
checkpoints during training. It measures how much each training point
contributes to reducing validation loss across the entire training trajectory.

Implementation note: This is a simplified checkpoint-based variant (TracInCP)
that uses an aggregated validation gradient (sum of per-example gradients)
rather than per-validation-example computation.
"""

from functools import partial
from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.progress import ProgressBar
from opendataval.model import GradientModel


class TracIn(DataEvaluator, ModelMixin):
    """TracIn Data Valuation via Checkpoint-based Gradient Tracing.

    Trains a model over multiple epochs, taking evenly-spaced checkpoints of
    model weights. At each checkpoint, computes the influence of each training
    point based on the selected influence mode.

    Two modes supported:

    1. **Validation mode (default)**: Measures training sample contribution to validation loss.
       - Formula: influence_i = lr * <grad_i(train), sum(grad_j(valid))>  [NO negation]
       - Sign: Positive = helps validation (good) | Negative = hurts validation (mislabeled)
       - Best for: Detecting harmful/mislabeled samples when validation set is clean
       - Requires: Clean validation set for accurate signal
       - Reference: Pruthi et al. 2020 (TracIn)

    2. **Self-influence mode (KOH2017)**: Measures training sample complexity/hardness.
       - Formula: influence_i = -lr * ||grad_i(train)||^2  [negated]
       - Sign: ~0 = easy sample (good) | Negative = hard to fit (potential mislabel)
       - Best for: Identifying mislabeled/outlier samples independent of validation set
       - Advantage: No validation set required, works even with noisy validation data
       - Reference: KOH2017, data valuation via gradient matching

    Final data value = sum over all checkpoints of influence_i(checkpoint).

    Parameters
    ----------
    train_kwargs : dict
        Training arguments (epochs, lr, batch_size, etc.) passed to model.fit()
        for each epoch. Must include 'epochs' or default to 1.
    num_checkpoints : int, optional
        Checkpoint interval - checkpoints at multiples of this value (e.g., 3 → checkpoints at epochs 3, 6, 9, ...), by default 5
    random_state : RandomState, optional
        Random state for reproducibility (inherited from DataEvaluator)
    checkpoints : dict, optional
        Pre-trained checkpoints {epoch: model_state} to use instead of re-training, by default None
    grad_args : tuple, optional
        Positional arguments passed to model.grad()
    grad_kwargs : dict, optional
        Keyword arguments passed to model.grad()

    References
    ----------
    .. [1] P. Pruthi, R. Dhillon, B. Feldman, R. Liang, and D. Weller,
        "Estimating Training Data Influence by Tracing Gradient Descent",
        NeurIPS, 2020. https://arxiv.org/abs/2002.08484
    """

    def __init__(
        self,
        train_kwargs: Optional[dict] = None,
        num_checkpoints: int = 5,
        random_state: Optional[RandomState] = None,
        checkpoints: Optional[dict] = None,
        exper_med=None,
        checkpoint_lrs: Optional[dict] = None,
        influence_mode: str = "validation",
        *grad_args,
        **grad_kwargs
    ):
        """Initialize TracIn evaluator.

        Parameters
        ----------
        train_kwargs : dict, optional
            Training arguments to pass to model.fit(). If None and exper_med provided, uses exper_med.train_kwargs
        num_checkpoints : int, optional
            Checkpoint interval - checkpoints at multiples of this value (e.g., 3 → epochs 3, 6, 9, ...), by default 5
        random_state : RandomState, optional
            Random state for reproducibility
        checkpoints : dict, optional
            Pre-trained checkpoints {epoch: model_state}. If None and exper_med provided, uses exper_med.checkpoints
        exper_med : ExperimentMediator, optional
            ExperimentMediator instance to auto-configure from (provides train_kwargs, num_checkpoints, checkpoints, checkpoint_lrs)
        checkpoint_lrs : dict, optional
            Learning rates at each checkpoint {epoch: lr}. If None and exper_med provided, uses exper_med.checkpoint_lrs
        influence_mode : str, optional
            "validation" (default) - influence on validation set gradient sum
            "self" - self-influence (L2 norm of training gradient, KOH2017 style)
        grad_args : tuple
            Positional arguments for model.grad()
        grad_kwargs : dict
            Keyword arguments for model.grad()
        """
        super().__init__(random_state=random_state)

        # Validate influence_mode
        if influence_mode not in ("validation", "self"):
            raise ValueError(f"influence_mode must be 'validation' or 'self', got '{influence_mode}'")
        self.influence_mode = influence_mode

        # Auto-configure from exper_med if provided
        if exper_med is not None:
            self.train_kwargs = train_kwargs or exper_med.train_kwargs
            self.num_checkpoints = num_checkpoints if num_checkpoints != 5 else exper_med.num_checkpoints
            self.checkpoints = checkpoints or exper_med.get_checkpoints()
            self.checkpoint_lrs = checkpoint_lrs or exper_med.get_checkpoint_lrs()
        else:
            self.train_kwargs = train_kwargs or {}
            self.num_checkpoints = num_checkpoints
            self.checkpoints = checkpoints
            self.checkpoint_lrs = checkpoint_lrs or {}

        self.grad_args = grad_args
        self.grad_kwargs = grad_kwargs

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store training and validation data.

        Parameters
        ----------
        x_train : torch.Tensor
            Training features
        y_train : torch.Tensor
            Training labels
        x_valid : torch.Tensor
            Validation features
        y_valid : torch.Tensor
            Validation labels

        Returns
        -------
        self : TracIn
            Returns self for method chaining
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.influence = np.zeros(len(x_train))
        return self

    def input_model(self, pred_model: GradientModel):
        """Set the prediction model with gradient capability.

        Parameters
        ----------
        pred_model : GradientModel
            Model that can compute gradients

        Returns
        -------
        self : TracIn
            Returns self for method chaining
        """
        assert (
            isinstance(pred_model, GradientModel)
            or callable(getattr(pred_model, "grad"))
        ), "Model with gradient required for TracIn."

        self.pred_model = pred_model.clone()
        return self

    def train_data_values(self, *args, **kwargs):
        """Train model and compute TracIn data values.

        If pre-trained checkpoints are provided, uses them to compute influence values
        without re-training. Otherwise, trains the model and takes checkpoints at
        specified intervals.

        The training loop:
        1. Extracts epochs and lr from train_kwargs, overrides with kwargs if provided
        2. Computes checkpoint epochs: multiples of num_checkpoints in [num_checkpoints, epochs]
        3. For each checkpoint (either pre-trained or during training):
           - Loads checkpoint model state
           - Computes aggregated validation gradients (sum of per-example gradients)
           - Accumulates influence for each training point: lr * dot(train_grad, valid_grad_sum)

        Parameters
        ----------
        args : tuple
            Training positional arguments
        kwargs : dict
            Training keyword arguments (may override epochs/lr from self.train_kwargs)

        Returns
        -------
        self : TracIn
            Returns self for method chaining
        """
        # Extract and merge training parameters
        merged_kwargs = dict(self.train_kwargs)
        merged_kwargs.update(kwargs)

        epochs = int(merged_kwargs.pop("epochs", 1))
        lr = float(merged_kwargs.pop("lr", 0.01))

        # Compute checkpoint epochs (every num_checkpoints epochs: num_checkpoints, 2*num_checkpoints, ...)
        checkpoint_epochs = set(range(self.num_checkpoints, epochs + 1, self.num_checkpoints))

        # If pre-trained checkpoints provided, use them (optimization path)
        if self.checkpoints is not None:
            print(f"[TracIn] Using {len(self.checkpoints)} pre-trained checkpoints")
            print(f"[TracIn] Pre-trained checkpoint epochs: {sorted(self.checkpoints.keys())}")
            if self.checkpoint_lrs:
                print(f"[TracIn] Using actual learning rates from training")
            return self._compute_from_checkpoints(checkpoint_epochs, lr)

        print(f"[TracIn] Training for {epochs} epochs with {len(checkpoint_epochs)} checkpoints")
        print(f"[TracIn] Checkpoint epochs: {sorted(checkpoint_epochs)}")

        # Get gradient iterator factory
        iter_grad = partial(self.pred_model.grad, *self.grad_args, **self.grad_kwargs)

        # Training loop with progress bar
        with ProgressBar(total=epochs, desc="TracIn Training") as pbar:
            for epoch in range(1, epochs + 1):
                # Train one epoch
                self.pred_model.fit(
                    self.x_train,
                    self.y_train,
                    *args,
                    epochs=1,
                    lr=lr,
                    **merged_kwargs
                )

                pbar.update(1)

                # Compute influence at checkpoint epochs
                if epoch in checkpoint_epochs:
                    # Set model to eval mode for gradient computation
                    if hasattr(self.pred_model, 'model'):
                        self.pred_model.model.eval()
                    elif hasattr(self.pred_model, 'eval'):
                        self.pred_model.eval()
                    self._compute_influence_at_epoch(iter_grad, lr, train_pbar=None)
                    pbar.update(0)  # Refresh display
        print(f"[TracIn] Training complete. Final mean influence: {self.influence.mean():.4f}")
        return self

    def _compute_influence_at_epoch(self, iter_grad, lr, train_pbar=None, checkpoint_epoch=None):
        """Compute and accumulate influence at current model state."""
        if self.influence_mode == "validation":
            self._compute_validation_influence(iter_grad, lr, train_pbar, checkpoint_epoch)
        elif self.influence_mode == "self":
            self._compute_self_influence(iter_grad, lr, train_pbar, checkpoint_epoch)

    def _compute_validation_influence(self, iter_grad, lr, train_pbar=None, checkpoint_epoch=None):
        """Compute influence based on validation set gradients.

        Sign convention (NO negation in this mode):
        - Positive influence = sample helps reduce validation loss (good)
        - Negative influence = sample hurts validation loss (mislabeled/harmful)
        """
        # Compute aggregated validation gradients (sum of per-example gradients)
        valid_grad_sum = None
        num_valid_samples = 0
        for valid_grads in iter_grad(self.x_valid, self.y_valid):
            num_valid_samples += 1
            if valid_grad_sum is None:
                # Initialize with first gradient
                valid_grad_sum = tuple(v.clone() for v in valid_grads)
            else:
                # Accumulate: element-wise add
                valid_grad_sum = tuple(
                    vs + vg for vs, vg in zip(valid_grad_sum, valid_grads)
                )

        if valid_grad_sum is None:
            print("[TracIn] Warning: No validation gradients computed")
            return

        # Normalize validation gradients by number of samples (convert sum to average)
        valid_grad_avg = tuple(v / max(num_valid_samples, 1) for v in valid_grad_sum)

        # Debug: compute validation gradient scale
        valid_grad_scales = [torch.norm(g).item() for g in valid_grad_avg]
        valid_grad_max = max(valid_grad_scales) if valid_grad_scales else 0
        valid_grad_mean = sum(valid_grad_scales) / len(valid_grad_scales) if valid_grad_scales else 0
        valid_grad_min = min(valid_grad_scales) if valid_grad_scales else 0
        epoch_str = f"Epoch {checkpoint_epoch}" if checkpoint_epoch else ""
        print(f"[TracIn] {epoch_str} MODE=validation LR={lr:.2e} | Valid: {num_valid_samples} samples, max_norm={valid_grad_max:.4f}, mean_norm={valid_grad_mean:.4f}, min_norm={valid_grad_min:.4f}")

        # Compute influence for each training point
        train_grad_stats = []
        influence_stats = []
        for i, train_grads in enumerate(iter_grad(self.x_train, self.y_train)):
            # Dot product with averaged validation gradients
            inf = sum(torch.sum(t * v) for t, v in zip(train_grads, valid_grad_avg))
            inf_value =  lr * inf.item()
            # Accumulate influence
            self.influence[i] += inf_value

            # Debug: track training gradient scales and influence
            if i < 5:  # First 5 samples
                train_norms = [torch.norm(t).item() for t in train_grads]
                train_grad_stats.append({
                    'sample': i,
                    'max_norm': max(train_norms),
                    'mean_norm': sum(train_norms) / len(train_norms),
                    'influence': inf_value
                })
                influence_stats.append(inf_value)

            if train_pbar:
                train_pbar.update(1)

        # Debug: print training gradient and influence statistics
        if train_grad_stats:
            print(f"[TracIn]   Training samples (first 5):")
            for stat in train_grad_stats:
                print(f"[TracIn]     Sample {stat['sample']}: max_norm={stat['max_norm']:.4f}, mean_norm={stat['mean_norm']:.4f}, influence={stat['influence']:.6f}")
            print(f"[TracIn]   Influence range: [{min(influence_stats):.6f}, {max(influence_stats):.6f}]")

    def _compute_self_influence(self, iter_grad, lr, train_pbar=None, checkpoint_epoch=None):
        """Compute self-influence (negated L2 norm of training gradients, KOH2017 style).

        Self-influence measures training sample complexity/hardness. Large gradients
        indicate samples that are hard to fit (potential mislabels or outliers).

        Formula: influence_i = -lr * ||grad_train[i]||^2

        Negation convention: Hard samples (large gradients) → negative influence (bad)
                            Easy samples (small gradients) → close to 0 (good)

        Reference: KOH2017, data valuation via gradient matching
        """
        train_grad_stats = []
        influence_stats = []

        epoch_str = f"Epoch {checkpoint_epoch}" if checkpoint_epoch else ""
        print(f"[TracIn] {epoch_str} MODE=self (KOH2017, negated) LR={lr:.2e}")

        for i, train_grads in enumerate(iter_grad(self.x_train, self.y_train)):
            # Self-influence: negative L2 norm squared of training gradient
            # Negation: hard-to-fit samples (large gradients) get negative scores
            grad_norm_sq = sum(torch.sum(t * t) for t in train_grads)
            inf_value = -lr * grad_norm_sq.item()  # Negated!
            # Accumulate influence
            self.influence[i] += inf_value

            # Debug: track training gradient scales and influence
            if i < 5:  # First 5 samples
                train_norms = [torch.norm(t).item() for t in train_grads]
                train_grad_stats.append({
                    'sample': i,
                    'max_norm': max(train_norms),
                    'mean_norm': sum(train_norms) / len(train_norms),
                    'influence': inf_value
                })
                influence_stats.append(inf_value)

            if train_pbar:
                train_pbar.update(1)

        # Debug: print training gradient and influence statistics
        if train_grad_stats:
            print(f"[TracIn]   Training samples (first 5):")
            for stat in train_grad_stats:
                print(f"[TracIn]     Sample {stat['sample']}: max_norm={stat['max_norm']:.4f}, mean_norm={stat['mean_norm']:.4f}, influence={stat['influence']:.6f}")
            print(f"[TracIn]   Influence range: [{min(influence_stats):.6f}, {max(influence_stats):.6f}]")

    def _compute_from_checkpoints(self, checkpoint_epochs, lr):
        """Compute influence using pre-trained checkpoints."""
        iter_grad = partial(self.pred_model.grad, *self.grad_args, **self.grad_kwargs)

        checkpoints_list = sorted(self.checkpoints.keys())

        # Debug: inspect checkpoint state
        print(f"\n[TracIn] Checkpoint state inspection:")
        print(f"[TracIn] Total checkpoints: {len(self.checkpoints)}")
        checkpoint_hashes = {}
        for ep in checkpoints_list:
            cp = self.checkpoints[ep]
            if isinstance(cp, dict):
                # Hash the state_dict values
                hash_val = sum(v.sum().item() if torch.is_tensor(v) else 0 for v in cp.values())
            else:
                hash_val = 0
            checkpoint_hashes[ep] = hash_val
            print(f"[TracIn]   Epoch {ep}: hash={hash_val:.6f}")

        # Check if all checkpoints are identical
        unique_hashes = set(checkpoint_hashes.values())
        if len(unique_hashes) == 1:
            print(f"[TracIn] ERROR: All {len(checkpoints_list)} checkpoints are IDENTICAL! Bug in checkpoint saving.")
        elif len(unique_hashes) < len(checkpoints_list):
            print(f"[TracIn] WARNING: Only {len(unique_hashes)} unique checkpoints out of {len(checkpoints_list)}")
        else:
            print(f"[TracIn] ✓ All checkpoints are unique")
        print()

        prev_param_hash = None
        for epoch in ProgressBar(iterable=checkpoints_list, desc="TracIn Checkpoints"):
            # Load checkpoint state
            self.pred_model.load_state_dict(self.checkpoints[epoch])

            # Set model to eval mode for gradient computation (batch norm needs this)
            if hasattr(self.pred_model, 'model'):
                self.pred_model.model.eval()
            elif hasattr(self.pred_model, 'eval'):
                self.pred_model.eval()

            # Debug: verify checkpoint was loaded (check if parameters changed)
            current_params = []
            if hasattr(self.pred_model, 'model'):
                for p in self.pred_model.model.parameters():
                    current_params.append(p.data.clone())
            else:
                for p in self.pred_model.parameters():
                    current_params.append(p.data.clone())

            current_param_hash = sum(p.sum().item() for p in current_params)
            if prev_param_hash is not None and abs(current_param_hash - prev_param_hash) < 1e-6:
                print(f"[TracIn] ERROR: Epoch {epoch} parameters unchanged from previous! Checkpoints not being saved correctly.")
            prev_param_hash = current_param_hash

            # Use actual learning rate at this checkpoint if available, otherwise use provided lr
            epoch_lr = self.checkpoint_lrs.get(epoch, lr) if self.checkpoint_lrs else lr

            # Compute influence at this checkpoint (no nested progress bar)
            self._compute_influence_at_epoch(iter_grad, epoch_lr, train_pbar=None, checkpoint_epoch=epoch)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return computed data values (TracIn influence).

        Returns
        -------
        np.ndarray
            Data values for each training point (accumulated TracIn influence across checkpoints)
        """
        return self.influence

    def print_influence_statistics(self):
        """Print detailed statistics about computed influence values for diagnostics."""
        print("\n[TracIn] === INFLUENCE STATISTICS ===")
        print(f"[TracIn] Total samples: {len(self.influence)}")
        print(f"[TracIn] Mean influence: {self.influence.mean():.6f}")
        print(f"[TracIn] Std influence: {self.influence.std():.6f}")
        print(f"[TracIn] Min influence: {self.influence.min():.6f}")
        print(f"[TracIn] Max influence: {self.influence.max():.6f}")

        # Distribution analysis
        positive_count = np.sum(self.influence > 0)
        negative_count = np.sum(self.influence < 0)
        zero_count = np.sum(self.influence == 0)

        print(f"\n[TracIn] Distribution:")
        print(f"[TracIn]   Positive influence: {positive_count} ({100*positive_count/len(self.influence):.1f}%)")
        print(f"[TracIn]   Negative influence: {negative_count} ({100*negative_count/len(self.influence):.1f}%)")
        print(f"[TracIn]   Zero influence: {zero_count} ({100*zero_count/len(self.influence):.1f}%)")

        # Percentiles
        print(f"\n[TracIn] Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(self.influence, p)
            print(f"[TracIn]   {p:2d}th: {val:.6f}")

        # Most and least influential samples
        top_indices = np.argsort(self.influence)[-5:][::-1]
        bottom_indices = np.argsort(self.influence)[:5]

        print(f"\n[TracIn] Most influential (helpful) samples:")
        for idx in top_indices:
            print(f"[TracIn]   Sample {idx}: influence={self.influence[idx]:.6f}")

        print(f"\n[TracIn] Least influential (harmful) samples:")
        for idx in bottom_indices:
            print(f"[TracIn]   Sample {idx}: influence={self.influence[idx]:.6f}")

    def validate_algorithm(self):
        """Validate that TracIn algorithm is working correctly.

        Checks:
        1. Gradients are being computed (not all zeros)
        2. Influence values are diverse (not all the same)
        3. No NaN or Inf values
        """
        print("\n[TracIn] === ALGORITHM VALIDATION ===")

        # Check for NaN/Inf
        if np.any(np.isnan(self.influence)):
            print("[TracIn] ✗ ERROR: NaN values detected in influence")
        elif np.any(np.isinf(self.influence)):
            print("[TracIn] ✗ ERROR: Inf values detected in influence")
        else:
            print("[TracIn] ✓ No NaN/Inf values")

        # Check if values are diverse
        if self.influence.std() < 1e-6:
            print("[TracIn] ✗ WARNING: Very low std - all influence values are similar")
            print(f"[TracIn]    Std: {self.influence.std():.2e}")
        else:
            print(f"[TracIn] ✓ Good diversity in influence values (std={self.influence.std():.6f})")

        # Check if all zeros
        if np.all(self.influence == 0):
            print("[TracIn] ✗ ERROR: All influence values are zero - gradients not computed")
        else:
            print("[TracIn] ✓ Non-zero influence values present")

        # Check for validation set issues
        print(f"\n[TracIn] Recommendations for mislabel detection:")
        if np.sum(self.influence < 0) < 0.1 * len(self.influence):
            print("[TracIn] ⚠ Low proportion of negative influence (<10%)")
            print("[TracIn]   This may indicate:")
            print("[TracIn]   1. Validation set also contains label noise")
            print("[TracIn]   2. Model hasn't learned clean decision boundary yet")
            print("[TracIn]   3. Mislabeled samples aren't clearly harmful to model")
        else:
            print(f"[TracIn] ✓ Good proportion of negative influence ({100*np.sum(self.influence < 0)/len(self.influence):.1f}%)")
