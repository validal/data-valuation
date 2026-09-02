"""
Simplified In-Run Data Shapley for ResNet18.

- Uses ONLY Adam optimizer + StepLR scheduler (ResNet18 config)
- Computes AUC mislabel detection during training
- Clean, focused implementation for reproducibility
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from opendataval.dataval.api import DataEvaluator, ModelMixin


class IndexedDataset(Dataset):
    """Wraps data to return (x, y, global_index) for shuffle safety."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], i


class InRunDataShapleySimple(DataEvaluator, ModelMixin):
    """Simplified In-Run Data Shapley for ResNet18 training.

    Configuration:
    - Optimizer: Adam (fixed)
    - Scheduler: StepLR (fixed) - reduces LR by gamma every step_size epochs
    - Validation: Class-balanced batches
    - Metrics: Tracks AUC for mislabel detection

    Parameters
    ----------
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization (default: 5e-4, ResNet18 value)
    step_size : int
        Reduce LR every N epochs (default: 10)
    step_gamma : float
        LR multiplier at each step (default: 0.1)
    val_batch_size : int, optional
        Validation batch size (default: auto)
    random_state : int, optional
        Random seed
    verbose : bool
        Print debug info
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        step_size: int = 10,
        step_gamma: float = 0.1,
        val_batch_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.val_batch_size = val_batch_size
        self.verbose = verbose
        self._values = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tracking
        self.debug_history = {
            "step": [],
            "epoch": [],
            "val_loss": [],
            "frac_pos_dot": [],
            "mean_values": [],
            "auc_mislabel": [],  # NEW: AUC for mislabel detection
        }
        self.noisy_labels_indices = None

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        noisy_train_indices: Optional[np.ndarray] = None,
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
        noisy_train_indices : np.ndarray, optional
            Indices of mislabeled training samples (for AUC tracking)

        Returns
        -------
        self
        """
        self.x_train = torch.as_tensor(x_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train)
        self.x_valid = torch.as_tensor(x_valid, dtype=torch.float32)
        self.y_valid = torch.as_tensor(y_valid)
        self.noisy_labels_indices = noisy_train_indices

        # Convert labels to indices if needed
        if self.y_train.dim() > 1 and self.y_train.shape[1] > 1:
            self.y_train = self.y_train.argmax(dim=1)
        if self.y_valid.dim() > 1 and self.y_valid.shape[1] > 1:
            self.y_valid = self.y_valid.argmax(dim=1)

        self.y_train = self.y_train.long()
        self.y_valid = self.y_valid.long()

        if self.verbose:
            print(f"\n[InRunDataShapleySimple] Data loaded:")
            print(f"  Train: {self.x_train.shape}, Labels: {self.y_train.shape}")
            print(f"  Valid: {self.x_valid.shape}, Labels: {self.y_valid.shape}")
            if noisy_train_indices is not None:
                print(f"  Mislabeled samples: {len(noisy_train_indices)} ({100*len(noisy_train_indices)/len(self.x_train):.1f}%)\n")

        return self

    def train_data_values(self, *args, **kwargs):
        """Train with GradDotProdEngine and compute trajectory-aware values."""

        try:
            from ghostEngines import GradDotProdEngine
        except ImportError as e:
            raise ImportError(
                "GhostSuite not installed. Install with: "
                "pip install git+https://github.com/Jiachen-T-Wang/GhostSuite"
            ) from e

        # Setup device and seeds
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Move to device
        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train.to(self.device)
        x_valid = self.x_valid.to(self.device).float()
        y_valid = self.y_valid.to(self.device)

        # Validation batch size: default to 64 for memory efficiency (not full validation set)
        val_batch_size = self.val_batch_size or min(64, len(x_valid))
        val_batch_size = min(val_batch_size, len(x_valid))

        # Initialize model
        self.pred_model.train()
        with torch.no_grad():
            _ = self.pred_model(x_train[:min(1, len(x_train))])

        # Freeze BatchNorm (per-sample gradients invalid in train mode)
        self._freeze_batchnorm(self.pred_model)

        # Setup: Adam optimizer + StepLR scheduler (ResNet18 config)
        optimizer = torch.optim.Adam(
            self.pred_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.step_gamma
        )

        # Attach GhostSuite engine
        engine = GradDotProdEngine(
            module=self.pred_model,
            val_batch_size=val_batch_size,
            loss_reduction="mean"
        )
        engine.attach(optimizer)

        # Data loaders
        dataset = IndexedDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = IndexedDataset(x_valid, y_valid)
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True
        )

        # Initialize values
        values = torch.zeros(len(x_train))

        if self.verbose:
            print(f"\n[InRunDataShapleySimple] Training Configuration:")
            print(f"  Optimizer: Adam (lr={self.learning_rate}, wd={self.weight_decay})")
            print(f"  Scheduler: StepLR (step_size={self.step_size}, gamma={self.step_gamma})")
            print(f"  Epochs: {self.epochs}, Batch: {self.batch_size}")
            print(f"  Total steps: {len(loader) * self.epochs}\n")

        step = 0
        total_steps = len(loader) * self.epochs

        step_pbar = tqdm(
            total=total_steps,
            desc="Training",
            disable=not self.verbose,
            leave=True
        )

        self.pred_model.train()
        for epoch in range(self.epochs):
            val_iter = iter(val_loader)
            epoch_values_change = 0.0

            for x_batch, y_batch, idx in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Get validation batch
                try:
                    x_val_batch, y_val_batch, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    x_val_batch, y_val_batch, _ = next(val_iter)

                x_val_batch = x_val_batch.to(self.device)
                y_val_batch = y_val_batch.to(self.device)

                B_train = x_batch.size(0)
                B_val = x_val_batch.size(0)
                B_tot = B_train + B_val

                # Combined batch [train; val]
                cx = torch.cat([x_batch, x_val_batch])
                cy = torch.cat([y_batch, y_val_batch])

                # Attach batch info
                engine.attach_train_batch(X_train=idx, Y_train=y_batch, iter_num=step)

                optimizer.zero_grad(set_to_none=True)

                # Forward and backward
                with engine.saved_tensors_context():
                    out = self.pred_model(cx)
                    loss = F.cross_entropy(out, cy, reduction="mean")
                    loss.backward()

                # Get dot products
                engine.aggregate_and_log()
                dot = engine.dot_product_log[-1]["dot_product"].float().cpu()

                # Use CURRENT learning rate from scheduler
                current_lr = scheduler.get_last_lr()[0]

                # Accumulate values
                scaling = (current_lr * (B_tot ** 2)) / (B_val * B_train)
                values[idx] += scaling * dot
                epoch_values_change += (scaling * dot).abs().sum().item()

                # Track debug info every 10 steps
                if step % 10 == 0:
                    with torch.no_grad():
                        val_loss = F.cross_entropy(self.pred_model(x_valid), y_valid).item()

                    self.debug_history["step"].append(step)
                    self.debug_history["epoch"].append(epoch)
                    self.debug_history["val_loss"].append(val_loss)
                    self.debug_history["frac_pos_dot"].append((dot > 0).float().mean().item())
                    self.debug_history["mean_values"].append(values.mean().item())

                    # Compute AUC for mislabel detection
                    if self.noisy_labels_indices is not None:
                        auc = self._compute_auc_mislabel(values.numpy())
                        self.debug_history["auc_mislabel"].append(auc)

                # Optimizer step
                engine.prepare_gradients()
                optimizer.step()
                engine.clear_gradients()

                step_pbar.update(1)
                step_pbar.set_postfix({
                    'epoch': f'{epoch+1}/{self.epochs}',
                    'val_loss': f'{val_loss:.4f}' if step % 10 == 0 else '...',
                })

            # Update scheduler at epoch level
            scheduler.step()

            if self.verbose:
                print(f"  Epoch {epoch+1}/{self.epochs}: "
                      f"val_loss={self.debug_history['val_loss'][-1]:.4f}, "
                      f"mean_val={values.mean().item():.6f}", end="")
                if self.noisy_labels_indices is not None:
                    print(f", auc_mislabel={self.debug_history['auc_mislabel'][-1]:.4f}")
                else:
                    print()

        step_pbar.close()
        engine.detach()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[InRunDataShapleySimple] TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"  Final values mean: {values.mean().item():.6f}")
            print(f"  Final values range: [{values.min().item():.6f}, {values.max().item():.6f}]")
            if self.noisy_labels_indices is not None:
                print(f"  Final AUC mislabel: {self.debug_history['auc_mislabel'][-1]:.4f}")
            print(f"{'='*70}\n")

        self._values = values.numpy()
        return self

    def _compute_auc_mislabel(self, values: np.ndarray) -> float:
        """Compute AUC for mislabel detection.

        Parameters
        ----------
        values : np.ndarray
            Current data values for all training samples

        Returns
        -------
        float
            AUC score (0-1): how well data values identify mislabeled samples
            - 1.0 = perfect mislabel detection
            - 0.5 = random
            - 0.0 = inverted
        """
        if self.noisy_labels_indices is None:
            return 0.0

        # Create binary labels: 1 = mislabeled, 0 = correct
        mislabel_binary = np.zeros(len(values))
        mislabel_binary[self.noisy_labels_indices] = 1

        try:
            # AUC: higher data values should correlate with lower mislabel probability
            # So we negate values for the computation (lower values = likely mislabeled)
            auc = roc_auc_score(mislabel_binary, -values)
            return auc
        except Exception:
            return 0.0

    def evaluate_data_values(self) -> np.ndarray:
        """Return computed in-run Shapley data values.

        Returns
        -------
        np.ndarray
            Data values for each training point
        """
        if self._values is None:
            raise RuntimeError("No computed values. Call train_data_values() first.")
        return self._values.astype(np.float32)

    def _freeze_batchnorm(self, model: nn.Module):
        """Freeze BatchNorm layers (eval mode)."""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = False
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = False

    def get_debug_history(self) -> dict:
        """Get training debug history.

        Returns
        -------
        dict
            Dictionary with training history including:
            - step, epoch, val_loss, frac_pos_dot, mean_values
            - auc_mislabel (if mislabeled samples provided)
        """
        return self.debug_history
