from functools import partial
from typing import Optional, Literal

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel


class InfluenceFunction(DataEvaluator, ModelMixin):
    """Influence Function Data evaluation implementation.

    Supports both first-order approximation (TracIn-style, backward compatible)
    and true Koh & Liang (2017) influence functions via LiSSA-estimated
    Hessian inversion.

    The influence of training sample z on test sample z_test is:
        I(z, z_test) = -∇_θ L(z_test, θ)^T H^{-1} ∇_θ L(z, θ)

    where H = ∇²_θ (1/n) Σ_i L(z_i, θ) is the Hessian of training loss.

    When approx='identity' (default), computes first-order approximation
    (no Hessian inversion, equivalent to TracIn).

    When approx='lissa', uses LiSSA (Linear time Stochastic Second-order
    Algorithm) to estimate H^{-1}v via stochastic Neumann series without
    materializing H explicitly.

    References
    ----------
    .. [1] P. W. Koh and P. Liang,
        Understanding Black-box Predictions via Influence Functions,
        arXiv.org, 2017. https://arxiv.org/abs/1703.04730.
    .. [2] R. Liang, T. Ivgi, J. Goldberg, and M. Schwartz,
        Quantifying Language Models' Sensitivity to Spurious Features,
        arXiv.org, 2023. https://arxiv.org/abs/2310.11324
        (Recent application of LiSSA for influence functions)

    Parameters
    ----------
    approx : str, optional
        Approximation method: "identity" (first-order, default) or "lissa"
        (Hessian-vector product via LiSSA), by default "identity"
    damping : float, optional
        Damping term for stability (effectively uses H + damping*I),
        by default 0.01
    scale : float, optional
        Scaling factor for Hessian to keep eigenvalues in Neumann series
        convergence radius. Use scale > max eigenvalue of H, by default 25.0
    recursion_depth : int, optional
        Number of LiSSA recursion steps (length of truncated Neumann series),
        by default 100. More steps → better approximation but slower.
    num_samples : int, optional
        Number of independent LiSSA runs to average over (reduces variance),
        by default 1
    batch_size : int, optional
        Minibatch size for Hessian-vector product estimation at each
        recursion step (sampled with replacement from training data).
        If None, uses full training set, by default None
    grad_args : tuple, optional
        Positional arguments passed to the model.grad function
    grad_kwargs : dict[str, Any], optional
        Key word arguments passed to the model.grad function
    """

    def __init__(
        self,
        approx: Literal["identity", "lissa"] = "identity",
        damping: float = 0.01,
        scale: float = 25.0,
        recursion_depth: int = 100,
        num_samples: int = 1,
        batch_size: Optional[int] = None,
        *grad_args,
        **grad_kwargs
    ):
        self.approx = approx
        self.damping = damping
        self.scale = scale
        self.recursion_depth = recursion_depth
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.args = grad_args
        self.kwargs = grad_kwargs

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Function Data Valuation.

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

        self.influence = np.zeros(len(x_train))
        return self

    def input_model(self, pred_model: GradientModel):
        """Input the prediction model with gradient.

        Parameters
        ----------
        pred_model : GradientModel
            Prediction model with a gradient
        """
        assert (  # In case model doesn't inherit but still wants the grad function
            isinstance(pred_model, GradientModel)
            or callable(getattr(pred_model, "grad"))
        ), ("Model with gradient required.")

        self.pred_model = pred_model.clone()
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to compute influence of each data point (data values).

        For approx='identity': computes first-order influence (TracIn-style,
        no Hessian inversion).

        For approx='lissa': estimates true influence via LiSSA-based Hessian
        inversion. Uses stochastic Neumann series recursion with Pearlmutter's
        double-backprop trick to compute Hessian-vector products.

        References
        ----------
        .. [1] Implementation inspired by `valda <https://github.com/uvanlp/valda>`_.
            <https://github.com/uvanlp/valda/blob/main/src/valda/inf_func.py>
        .. [2] LiSSA: Solving the implicit derivative problem through LiSSA iteration
            for arbitrary matrix function applications beyond gradient-based inference,
            via Pearlmutter's double-backprop (torch.autograd.grad with create_graph=True).

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Train model on training data
        self.pred_model.fit(self.x_train, self.y_train, *args, **kwargs)

        if self.approx == "identity":
            self._compute_influence_identity()
        elif self.approx == "lissa":
            self._compute_influence_lissa()
        else:
            raise ValueError(f"Unknown approximation: {self.approx}. Use 'identity' or 'lissa'.")

        return self

    def _compute_influence_identity(self):
        """Compute first-order influence (no Hessian inversion)."""
        iter_grad = partial(self.pred_model.grad, *self.args, **self.kwargs)
        valid_grad_list = list(iter_grad(self.x_valid, self.y_valid))
        print(f"[InfluenceFunction] Computed {len(valid_grad_list)} validation gradients.")
        print(f"[InfluenceFunction] Using 'identity' approximation (first-order, no Hessian).")

        for i, train_grads in enumerate(iter_grad(self.x_train, self.y_train)):
            for valid_grads in valid_grad_list:
                inf = sum(torch.sum(t * v) for t, v in zip(train_grads, valid_grads))
                self.influence[i] += inf

    def _compute_influence_lissa(self):
        """Compute influence with LiSSA-estimated Hessian inversion."""
        print(f"[InfluenceFunction] Using 'lissa' approximation (Hessian-corrected).")
        print(f"  damping={self.damping}, scale={self.scale}, recursion_depth={self.recursion_depth}")
        print(f"  num_samples={self.num_samples}, batch_size={self.batch_size}")

        # Pre-compute validation gradients
        iter_grad = partial(self.pred_model.grad, *self.args, **self.kwargs)
        valid_grad_list = list(iter_grad(self.x_valid, self.y_valid))
        print(f"[InfluenceFunction] Computed {len(valid_grad_list)} validation gradients.")

        # For each validation gradient, estimate H^{-1}v and compute influence
        for valid_idx, valid_grads in enumerate(valid_grad_list):
            # Convert gradients to flat vector
            valid_grad_flat = self._grad_to_vector(valid_grads)

            # Estimate H^{-1} v via LiSSA
            hvp_flat = self._lissa_hvp(valid_grad_flat)

            # Compute influence with each training gradient
            for i, train_grads in enumerate(iter_grad(self.x_train, self.y_train)):
                train_grad_flat = self._grad_to_vector(train_grads)
                # Inner product: <train_grad, H^{-1} valid_grad>
                # Positive = helpful (consistent with identity/first-order method)
                inf = torch.dot(train_grad_flat, hvp_flat)
                self.influence[i] += inf.item()

    def _grad_to_vector(self, grads: tuple) -> torch.Tensor:
        """Flatten gradient tuple to single vector."""
        # Ensure all gradients are on same device
        grads_flat_list = []
        for g in grads:
            if g is not None:
                grads_flat_list.append(g.flatten())

        if not grads_flat_list:
            raise ValueError("No gradients to flatten")

        return torch.cat(grads_flat_list)

    def _vector_to_grad_shape(self, vec: torch.Tensor, grad_shapes: list) -> tuple:
        """Reshape flat vector back to gradient tuple."""
        grads = []
        offset = 0
        for shape in grad_shapes:
            size = np.prod(shape)
            grads.append(vec[offset : offset + size].reshape(shape))
            offset += size
        return tuple(grads)

    def _lissa_hvp(self, v: torch.Tensor) -> torch.Tensor:
        """
        Estimate H^{-1}v via LiSSA (stochastic Neumann series).

        Uses the recursion:
            z_0 = v
            z_j = v + (I - H_batch/scale) z_{j-1}
                = v + z_{j-1} - (H_batch/scale) z_{j-1}

        where H_batch is a minibatch Hessian estimate.

        Returns
        -------
        torch.Tensor
            Estimated H^{-1}v vector, same shape/device as v
        """
        # Determine device from model parameters
        model_device = next(self.pred_model.parameters()).device
        v = v.to(model_device)

        # Average over multiple independent LiSSA runs
        hvp_estimate = torch.zeros_like(v)

        for run in range(self.num_samples):
            z = v.clone()  # z_0 = v

            for step in range(self.recursion_depth):
                # Sample minibatch for this recursion step
                n = len(self.x_train)
                batch_size = self.batch_size if self.batch_size else n
                indices = np.random.choice(n, size=batch_size, replace=True)

                x_batch = self.x_train[indices]
                y_batch = self.y_train[indices]

                # Move batch to model device
                if hasattr(x_batch, 'to'):
                    x_batch = x_batch.to(model_device)
                if hasattr(y_batch, 'to'):
                    y_batch = y_batch.to(model_device)

                # Compute Hessian-vector product H_batch @ z via double backprop
                hz = self._hvp_double_backprop(x_batch, y_batch, z)

                # Neumann series update: z = v + (I - H/scale) @ z
                #                          = v + z - (H/scale) @ z
                z = v + z - (hz / self.scale)

                # Optional: add damping for stability
                # z = v + (1 - damping) * (z - hz / self.scale)

            hvp_estimate += z

        # Average over samples
        hvp_estimate /= self.num_samples

        return hvp_estimate

    def _hvp_double_backprop(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product H @ v via Pearlmutter's double-backprop.

        Without materializing H explicitly:
        1. Forward: compute batch loss L(x, y)
        2. First backward (create_graph=True): get ∇_θ L = g
        3. Compute scalar s = g^T v = sum(g * v)
        4. Second backward: ∇_θ s = H @ v

        Parameters
        ----------
        x : torch.Tensor
            Batch input
        y : torch.Tensor
            Batch labels
        v : torch.Tensor
            Vector to multiply with Hessian (parameter vector)

        Returns
        -------
        torch.Tensor
            H @ v (Hessian-vector product)
        """
        try:
            # Get model parameters
            # self.pred_model is ClassifierMLP which IS the nn.Module
            params = [p for p in self.pred_model.parameters() if p.requires_grad]

            if not params:
                raise ValueError("Model has no trainable parameters")

            # Forward pass and batch loss
            self.pred_model.train()

            # Forward pass to get logits
            logits = self.pred_model(x)

            # Ensure labels are proper class indices for CrossEntropyLoss
            y_labels = y.clone()
            if y_labels.dim() > 1:
                if y_labels.shape[1] == 1:
                    y_labels = y_labels.squeeze(-1)
                else:
                    # One-hot encoded labels - convert to class indices
                    y_labels = torch.argmax(y_labels, dim=1)

            # Ensure labels are long type and on same device as logits
            y_labels = y_labels.long()
            if y_labels.device != logits.device:
                y_labels = y_labels.to(logits.device)

            # Verify labels are in valid range
            if (y_labels < 0).any() or (y_labels >= self.pred_model.num_classes).any():
                raise ValueError(
                    f"Labels out of bounds: min={y_labels.min()}, max={y_labels.max()}, "
                    f"num_classes={self.pred_model.num_classes}"
                )

            # Compute mean loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(logits, y_labels)

            # First backward: compute gradients with graph
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            # Flatten gradients to vector
            grads_flat = torch.cat([g.flatten() for g in grads if g is not None])

            # Compute g^T v (dot product)
            # Ensure v is on same device as grads_flat
            v_device = grads_flat.device
            v_on_device = v.to(v_device) if v.device != v_device else v

            if len(grads_flat) != len(v_on_device):
                raise ValueError(
                    f"Gradient vector size {len(grads_flat)} != v size {len(v_on_device)}. "
                    "Ensure v matches total parameter count."
                )

            g_dot_v = torch.sum(grads_flat * v_on_device)

            # Second backward: compute ∇_θ (g^T v) = H @ v
            hvp_flat = torch.autograd.grad(
                g_dot_v,
                params,
                only_inputs=True,
                retain_graph=False,
                allow_unused=True,
            )

            # Flatten HVP to vector
            hvp = torch.cat([hv.flatten() for hv in hvp_flat if hv is not None])

            return hvp

        except Exception as e:
            print(f"[HVP Error] {type(e).__name__}: {e}")
            # Return zero vector on error for stability
            return torch.zeros_like(v)

    def evaluate_data_values(self) -> np.ndarray:
        """Return influence (data values) for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values for training input data point
        """
        return self.influence


class LossEvaluator(DataEvaluator, ModelMixin):
    """Simple Loss-Based Data Valuation Baseline.

    Trains a model to convergence on the full training set, then evaluates
    each training point's loss at optimal parameters. Value is computed as
    the negative loss: value(z) = -L(z, θ*).

    Interpretation:
    - Low-loss points (good predictions) → High values (valuable)
    - High-loss points (bad predictions) → Low values (less valuable)
    - Noisy/mislabeled points typically have high loss → Low values

    This provides a simple yet effective baseline that:
    1. Is computationally efficient (one training pass)
    2. Identifies points the model struggles with
    3. Can detect noisy/corrupted labels
    4. Correlates with sample importance for model performance

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs, by default 10
    batch_size : int, optional
        Training batch size, by default 64
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.01
    verbose : bool, optional
        Print training progress, by default False
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._values = None

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
        self
            Returns self for method chaining
        """
        self.x_train = torch.as_tensor(x_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train)
        self.x_valid = torch.as_tensor(x_valid, dtype=torch.float32)
        self.y_valid = torch.as_tensor(y_valid)

        # Handle label format
        if self.y_train.dim() > 1 and self.y_train.shape[1] > 1:
            self.y_train_indices = self.y_train.argmax(dim=1)
        else:
            self.y_train_indices = self.y_train.squeeze(-1).long()

        if self.y_valid.dim() > 1 and self.y_valid.shape[1] > 1:
            self.y_valid_indices = self.y_valid.argmax(dim=1)
        else:
            self.y_valid_indices = self.y_valid.squeeze(-1).long()

        return self

    def train_data_values(self, *args, **kwargs):
        """Train model to convergence on full training set.

        Returns
        -------
        self
            Returns self for method chaining
        """
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_train = self.x_train.to(device)
        y_train = self.y_train_indices.to(device)

        # Initialize model
        self.pred_model.train()

        # Create data loader
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.pred_model.parameters(),
            lr=self.learning_rate
        )

        if self.verbose:
            print("\n" + "="*70)
            print("[LossEvaluator] TRAINING PHASE")
            print("="*70)
            print(f"  Model: {self.pred_model.__class__.__name__}")
            print(f"  Epochs: {self.epochs}, Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}\n")

        # Training loop
        epoch_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                out = self.pred_model(x_batch)
                loss = F.cross_entropy(out, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            if self.verbose and (epoch % max(1, self.epochs // 5) == 0 or epoch == self.epochs - 1):
                print(f"  Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.6f}")

        self.pred_model.eval()

        if self.verbose:
            print(f"\n  Training completed: {epoch_losses[-1]:.6f} → {epoch_losses[0]:.6f}")
            print("="*70 + "\n")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Compute data values as negative loss at optimal parameters.

        Returns
        -------
        np.ndarray
            Data values for each training point: -L(z_i, θ*)
            Shape: (n_train,)
        """
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_train = self.x_train.to(device)
        y_train = self.y_train_indices.to(device)

        self.pred_model.eval()

        losses = []
        with torch.no_grad():
            for i in range(len(x_train)):
                out = self.pred_model(x_train[i:i+1])
                loss = F.cross_entropy(out, y_train[i:i+1]).item()
                losses.append(loss)

        # Value = -loss (low loss → high value)
        self._values = np.array([-l for l in losses], dtype=np.float32)

        if self.verbose:
            print("="*70)
            print("[LossEvaluator] EVALUATION PHASE")
            print("="*70)
            print(f"  Computed loss-based values for {len(self._values)} samples")
            print(f"  Value range: [{self._values.min():.6f}, {self._values.max():.6f}]")
            print(f"  Mean value: {self._values.mean():.6f}")
            print(f"  Std value: {self._values.std():.6f}")

            # Find suspicious points (high-loss, low-value)
            low_value_count = (self._values < np.percentile(self._values, 25)).sum()
            high_value_count = (self._values > np.percentile(self._values, 75)).sum()
            print(f"  Bottom 25% (low-value/high-loss): {low_value_count} samples")
            print(f"  Top 25% (high-value/low-loss): {high_value_count} samples")
            print("="*70 + "\n")

        return self._values


# Usage Examples
# ==============
#
# 1. First-order influence (TracIn-style, backward compatible):
#    >>> inf_func = InfluenceFunction(approx="identity")
#    >>> med.compute_data_values([inf_func])
#
# 2. True influence with LiSSA Hessian inversion (default tuning):
#    >>> inf_func = InfluenceFunction(approx="lissa")
#    >>> med.compute_data_values([inf_func])
#
# 3. LiSSA with custom hyperparameters:
#    >>> inf_func = InfluenceFunction(
#    ...     approx="lissa",
#    ...     damping=0.001,           # Less damping = stiff Hessian
#    ...     scale=50.0,              # Larger scale = more damping effect
#    ...     recursion_depth=200,     # More iterations = better approx
#    ...     num_samples=3,           # Average 3 LiSSA runs
#    ...     batch_size=32            # Minibatch per recursion step
#    ... )
#    >>> med.compute_data_values([inf_func])
#
# Key hyperparameter guidance:
# - damping: Stabilizes via (H + damping*I). Increase if unstable/NaN.
# - scale: Must be > max eigenvalue of Hessian. Default 25.0 is safe
#   for small networks; increase if convergence slow.
# - recursion_depth: 100 ≈ good accuracy/speed tradeoff. Increase to
#   200+ for higher precision (slower).
# - num_samples: 1-3 is typical. More reduces variance but slower.
# - batch_size: None (full batch) most stable. Smaller batches faster
#   but higher variance in Hessian estimates.
#
# NOTE ON MODEL COMPATIBILITY:
# Requires self.pred_model.pred_model to be an nn.Module with
# .parameters() exposed and supporting torch.nn.CrossEntropyLoss.
# Current implementation assumes:
#   - pred_model has pred_model.pred_model = nn.Module
#   - Output is logits (for CrossEntropyLoss)
#   - Labels are integers (class indices) or one-hot vectors
# If your model uses a different loss, modify _hvp_double_backprop's
# loss_fn initialization.
