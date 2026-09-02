"""LoGRA - Influence functions via Low-Rank Approximation (LogIX integration).

Computes Hessian-preconditioned influence function scores using LogIX's
efficient LoRA-PCA gradient projection and K-FAC Hessian approximation.

Implementation based on the verified debugged notebook where:
- LoRA-PCA projects gradients to lower-rank subspace (memory efficient)
- K-FAC approximates inverse Hessian (curvature-aware)
- Influence aggregated over balanced validation batch
- Per-sample scores directly comparable to trajectory methods

Key differences from In-Run Shapley:
- Hessian-aware (curvature-based)
- Checkpoint-based (one final model, not trajectory)
- LoRA projection (memory efficient for large models)
"""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from numpy.random import RandomState
from sklearn.utils import check_random_state
import sys
import os
from io import StringIO
import uuid
import time

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.progress import ProgressBar


class LoGRA_old(DataEvaluator, ModelMixin):
    """LoGRA: Influence functions with LoRA-PCA and K-FAC (LogIX-based).

    Trains a model to convergence, then computes per-training-sample influence
    on each validation target using LogIX's efficient Hessian approximation.

    Key properties:
    - Curvature-aware: includes Hessian (Fisher Information Matrix) approximation
    - K-FAC: efficient Kronecker-factored approximate curvature
    - LoRA-PCA: low-rank gradient projection for memory efficiency
    - Checkpoint-based: scores computed at final model state
    - Aggregated: influence summed over balanced validation batch

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs, by default 10
    batch_size : int, optional
        Training batch size, by default 64
    learning_rate : float, optional
        Learning rate for SGD optimizer (same as ClassifierMLP), by default 0.01
    lora : str, optional
        LoRA projection type: "none", "random", or "pca" (recommended), by default "pca"
    hessian : str, optional
        Hessian approximation: "none" (first-order), "raw" (exact Fisher), "kfac", "ekfac",
        by default "kfac"
    random_state : RandomState, optional
        Random state for reproducibility, by default None

    References
    ----------
    PDF: inrunShapley_and_LoGRA_merged.ipynb
    LogIX: https://github.com/logix-project/logix
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        lora: str = "pca",
        hessian: str = "kfac",
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lora = lora
        self.hessian = hessian
        self.verbose = verbose
        self._values = None
        self._seed = None  # Store seed for consistency across train and evaluate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x_train : torch.Tensor or Subset
            Training features
        y_train : torch.Tensor or Subset
            Training labels
        x_valid : torch.Tensor or Subset
            Validation features
        y_valid : torch.Tensor or Subset
            Validation labels

        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.verbose:
            print("\n" + "="*70)
            print("[A] DATA INPUT PHASE")
            print("="*70)

        # Convert Subset objects to tensors (from DataFetcher)
        from torch.utils.data import Subset

        if isinstance(x_train, Subset):
            x_train = torch.stack([x_train.dataset[i][0] if isinstance(x_train.dataset[i], tuple) else x_train.dataset[i] for i in x_train.indices])
        if isinstance(y_train, Subset):
            y_train = torch.stack([torch.tensor(x_train.dataset[i][1]) if isinstance(x_train.dataset[i], tuple) else x_train.dataset[i] for i in x_train.indices])
        if isinstance(x_valid, Subset):
            x_valid = torch.stack([x_valid.dataset[i][0] if isinstance(x_valid.dataset[i], tuple) else x_valid.dataset[i] for i in x_valid.indices])
        if isinstance(y_valid, Subset):
            y_valid = torch.stack([torch.tensor(y_valid.dataset[i][1]) if isinstance(y_valid.dataset[i], tuple) else y_valid.dataset[i] for i in y_valid.indices])

        # Ensure tensors
        self.x_train = torch.as_tensor(x_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train)
        self.x_valid = torch.as_tensor(x_valid, dtype=torch.float32)
        self.y_valid = torch.as_tensor(y_valid)

        if self.verbose:
            print(f"  x_train shape: {self.x_train.shape}, dtype: {self.x_train.dtype}, device: {self.x_train.device}")
            print(f"  y_train shape: {self.y_train.shape}, dtype: {self.y_train.dtype}, device: {self.y_train.device}")
            print(f"  x_valid shape: {self.x_valid.shape}, dtype: {self.x_valid.dtype}, device: {self.x_valid.device}")
            print(f"  y_valid shape: {self.y_valid.shape}, dtype: {self.y_valid.dtype}, device: {self.y_valid.device}")
            print(f"  Train data range: X=[{self.x_train.min():.4f}, {self.x_train.max():.4f}]")
            print(f"  Valid data range: X=[{self.x_valid.min():.4f}, {self.x_valid.max():.4f}]")

        # Handle label format
        if self.y_train.dim() > 1:
            if self.y_train.shape[1] > 1:  # one-hot
                self.y_train_indices = self.y_train.argmax(dim=1)
                if self.verbose:
                    print(f"  y_train: one-hot encoded ({self.y_train.shape[1]} classes) → argmax")
            else:  # [N, 1]
                self.y_train_indices = self.y_train.squeeze(-1)
                if self.verbose:
                    print(f"  y_train: squeezed from {self.y_train.shape}")
        else:
            self.y_train_indices = self.y_train.long()
            if self.verbose:
                print(f"  y_train: already 1D")

        if self.y_valid.dim() > 1:
            if self.y_valid.shape[1] > 1:  # one-hot
                self.y_valid_indices = self.y_valid.argmax(dim=1)
                if self.verbose:
                    print(f"  y_valid: one-hot encoded ({self.y_valid.shape[1]} classes) → argmax")
            else:  # [N, 1]
                self.y_valid_indices = self.y_valid.squeeze(-1)
                if self.verbose:
                    print(f"  y_valid: squeezed from {self.y_valid.shape}")
        else:
            self.y_valid_indices = self.y_valid.long()
            if self.verbose:
                print(f"  y_valid: already 1D")

        if self.verbose:
            # Verify class balance
            train_classes, train_counts = torch.unique(self.y_train_indices, return_counts=True)
            valid_classes, valid_counts = torch.unique(self.y_valid_indices, return_counts=True)
            print(f"\n  Training class distribution:")
            for cls, count in zip(train_classes, train_counts):
                print(f"    Class {cls.item()}: {count.item()} samples ({100*count.item()/len(self.y_train_indices):.1f}%)")
            print(f"  Validation class distribution:")
            for cls, count in zip(valid_classes, valid_counts):
                print(f"    Class {cls.item()}: {count.item()} samples ({100*count.item()/len(self.y_valid_indices):.1f}%)")
            print("="*70 + "\n")

        return self

    def train_data_values(self, *args, **kwargs):
        """Train model to convergence.

        Trains a fresh model without computing influence values.
        Influence computation happens in evaluate_data_values().

        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.verbose:
            print("\n" + "="*70)
            print("[B] TRAINING PHASE")
            print("="*70)

        # Set random seeds for reproducibility
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(0, 2**31 - 1)
            self._seed = seed  # Store for reuse in evaluate_data_values
        else:
            seed = None
            self._seed = None

        if seed is not None:
            # Set all random sources
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # Enable deterministic behavior (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if self.verbose:
                print(f"  Random seed: {seed}")

        # Move data to device
        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train_indices.to(self.device)

        if self.verbose:
            print(f"  x_train device: {x_train.device}, shape: {x_train.shape}")
            print(f"  y_train device: {y_train.device}, shape: {y_train.shape}")

        # # Initialize model with a forward pass to resolve LazyLinear layers
        self.pred_model.train()
        with torch.no_grad():
             _ = self.pred_model(x_train[:min(1, len(x_train))])

        # Ensure non-inplace ReLU (LogIX LoRA hooks break with in-place ops)
        self._make_relu_safe()

        # Training loop
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.verbose:
            print(f"  Model: {self.pred_model.__class__.__name__}")
            print(f"  Optimizer: Adam, lr={self.learning_rate}")
            print(f"  Batch size: {self.batch_size}, Epochs: {self.epochs}")
            total_params = sum(p.numel() for p in self.pred_model.parameters())
            trainable_params = sum(p.numel() for p in self.pred_model.parameters() if p.requires_grad)
            print(f"  Parameters: {total_params} total, {trainable_params} trainable")

        optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.learning_rate)

        # Debug: Initial model state
        if self.verbose:
            print("\n  [B.1] Initial Model State:")
            with torch.no_grad():
                init_out = self.pred_model(x_train)
                init_loss = F.cross_entropy(init_out, y_train).item()
                init_acc = (init_out.argmax(dim=1) == y_train).float().mean().item()
            print(f"    Loss: {init_loss:.6f}")
            print(f"    Accuracy: {init_acc:.4f}")

            # Check initial gradient magnitude
            init_grad_norm = 0.0
            for p in self.pred_model.parameters():
                if p.requires_grad:
                    init_grad_norm += (p.grad ** 2).sum().item() if p.grad is not None else 0
            init_grad_norm = init_grad_norm ** 0.5
            print(f"    Gradient norm: {init_grad_norm:.6e}")

            initial_params = [p.clone() for p in self.pred_model.parameters()]

        self.pred_model.train()
        epoch_losses = []
        epoch_accs = []
        batch_losses_all = []

        with ProgressBar(total=self.epochs, desc="LoGRA Training") as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                epoch_acc = 0.0
                num_batches = 0

                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    out = self.pred_model(x_batch)
                    loss = F.cross_entropy(out, y_batch)
                    loss.backward()

                    # Track gradient norm per batch
                    batch_grad_norm = 0.0
                    for p in self.pred_model.parameters():
                        if p.grad is not None:
                            batch_grad_norm += (p.grad ** 2).sum().item()
                    batch_grad_norm = batch_grad_norm ** 0.5

                    optimizer.step()

                    batch_acc = (out.argmax(dim=1) == y_batch).float().mean().item()
                    epoch_loss += loss.item()
                    epoch_acc += batch_acc
                    batch_losses_all.append(loss.item())
                    num_batches += 1

                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                avg_epoch_acc = epoch_acc / max(num_batches, 1)
                epoch_losses.append(avg_epoch_loss)
                epoch_accs.append(avg_epoch_acc)

                if self.verbose and (epoch % max(1, self.epochs // 5) == 0 or epoch == self.epochs - 1):
                    print(f"  Epoch {epoch+1}/{self.epochs}: loss={avg_epoch_loss:.6f}, acc={avg_epoch_acc:.4f}")

                pbar.update(1)

        self.pred_model.eval()

        # Debug: Final model state
        if self.verbose:
            print(f"\n  [B.2] Final Model State:")
            with torch.no_grad():
                final_out = self.pred_model(x_train)
                final_loss = F.cross_entropy(final_out, y_train).item()
                final_acc = (final_out.argmax(dim=1) == y_train).float().mean().item()
            print(f"    Loss: {final_loss:.6f}")
            print(f"    Accuracy: {final_acc:.4f}")
            print(f"    Loss reduction: {(init_loss - final_loss):.6f} ({100*(init_loss - final_loss)/max(init_loss, 1e-10):.1f}%)")

            # Check weight changes
            total_weight_change = 0.0
            max_weight_change = 0.0
            for p_init, p_final in zip(initial_params, self.pred_model.parameters()):
                change = (p_final - p_init).abs().sum().item()
                total_weight_change += change
                max_weight_change = max(max_weight_change, (p_final - p_init).abs().max().item())

            print(f"    Total weight change: {total_weight_change:.6f}")
            print(f"    Max single weight change: {max_weight_change:.6f}")

            if total_weight_change < 1e-6:
                print("    ⚠ WARNING: Weights almost unchanged - model might be stuck!")
            else:
                print("    ✓ Model is learning")

            # Training convergence summary
            print(f"\n  [B.3] Training Summary:")
            print(f"    Min loss: {min(epoch_losses):.6f}, Max loss: {max(epoch_losses):.6f}")
            print(f"    Final accuracy: {epoch_accs[-1]:.4f}")
            if len(epoch_losses) > 1:
                trend = "decreasing ✓" if epoch_losses[-1] < epoch_losses[-2] else "increasing ⚠"
                print(f"    Loss trend (last epoch): {trend}")

            print("="*70 + "\n")

        # Store trained model state
        self._trained_model = self.pred_model.clone()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Compute influence functions using LogIX.

        Runs LogIX to compute LoRA-PCA gradient projections and K-FAC Hessian,
        then aggregates influence over the validation batch.

        Returns
        -------
        np.ndarray
            Influence scores for each training point (shape: n_train,)
        """
        # Use stored seed from training for consistency
        seed = self._seed

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            from logix import LogIX, LogIXScheduler
            from logix.utils import DataIDGenerator
        except ImportError as e:
            raise ImportError(
                "LogIX not installed. Install with: "
                "pip install git+https://github.com/logix-project/logix"
            ) from e

        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train_indices.to(self.device)
        x_valid = self.x_valid.to(self.device).float()
        y_valid = self.y_valid_indices.to(self.device)

        # Initialize LogIX with unique project name to avoid interference
        # Use combination of timestamp and UUID for uniqueness across parallel jobs
        unique_suffix = f"{int(time.time() * 1e6)}_{uuid.uuid4().hex[:8]}"
        project_name = f"opendataval_logra_{unique_suffix}"
        #project_name = f"opendataval_logra"
        logix = LogIX(project=project_name, config=None)
        scheduler = LogIXScheduler(logix, lora=self.lora, hessian=self.hessian, save="grad")
        logix.watch(self.pred_model)

        idg = DataIDGenerator()

        if self.verbose:
            print(f"[LoGRA] Computing influence with lora={self.lora}, hessian={self.hessian}...")

        # Single consolidated progress bar
        progress = ProgressBar(total=2, desc="LoGRA")

        # Phase 1: Log per-sample training gradients
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        self.pred_model.eval()
        for _ in scheduler:
            for x_batch, y_batch in train_loader:  # No nested bar
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with logix(data_id=idg(x_batch)):
                    self.pred_model.zero_grad()
                    out = self.pred_model(x_batch)
                    loss = F.cross_entropy(out, y_batch, reduction="sum")
                    loss.backward()

        logix.finalize()
        progress.update(1)  # Phase 1 complete

        # Phase 2: Compute influence of all training points on each validation point
        log_loader = logix.build_log_dataloader()
        logix.eval()
        logix.setup({"grad": ["log"]})

        influence = torch.zeros(len(x_train))
        valid_dataset = TensorDataset(x_valid, y_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

        for x_val, y_val in valid_loader:  # No nested bar
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

            with logix(data_id=idg(x_val)):
                self.pred_model.zero_grad()
                out = self.pred_model(x_val)
                loss = F.cross_entropy(out, y_val, reduction="sum")
                loss.backward()

            tlog = logix.get_log()

            # Suppress LogIX progress bars if not verbose
            if not self.verbose:
                # Suppress both stdout and stderr (LogIX uses tqdm on stderr)
                import contextlib
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    result = logix.influence.compute_influence_all(tlog, log_loader)
            else:
                result = logix.influence.compute_influence_all(tlog, log_loader)

            influence_val = torch.as_tensor(result["influence"]).flatten().cpu()
            influence += influence_val

        progress.update(1)  # Phase 2 complete

        self._values = influence.numpy().astype(np.float32)
        return self._values

    def _make_relu_safe(self):
        """Ensure no in-place ReLU (LogIX LoRA hooks incompatible with in-place ops)."""
        for module in self.pred_model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
