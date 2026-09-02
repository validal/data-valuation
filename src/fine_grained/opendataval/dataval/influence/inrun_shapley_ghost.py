"""In-Run Data Shapley via GhostSuite's GradDotProdEngine.

Accumulates gradient dot products between training samples and validation loss
across all training steps, computing trajectory-aware data values without
requiring Hessian estimation.

Implementation based on the verified debugged notebook where:
- Sign is +: values[idx] += lr * (B_tot**2)/(n_val*B_train) * dot
- Class-balanced validation batch prevents class-identity domination
- Global indices preserved under shuffle=True via IndexedDataset
- BatchNorm frozen during scoring (per-sample gradients invalid in train mode)
"""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState
from sklearn.utils import check_random_state
from tqdm import tqdm

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.progress import ProgressBar


class IndexedDataset(Dataset):
    """Wraps data to return (x, y, global_index) for shuffle safety."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], i


class BalancedBatchSampler:
    """Sampler that ensures class-balanced batches."""

    def __init__(self, labels, batch_size, drop_last=True, random_state=None):
        """
        Parameters
        ----------
        labels : torch.Tensor
            Class labels
        batch_size : int
            Target batch size
        drop_last : bool
            If True, drop incomplete batches
        random_state : int
            Random seed for reproducibility
        """
        self.labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_state = random_state

        # Get unique classes and group indices by class
        unique_classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0] for c in unique_classes}
        self.num_classes = len(unique_classes)

    def __iter__(self):
        """Generate balanced batches."""
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

        # Shuffle indices within each class
        shuffled_class_indices = {}
        for c, indices in self.class_indices.items():
            shuffled = rng.permutation(indices)
            shuffled_class_indices[c] = shuffled.copy()

        # Track position in each class
        positions = {c: 0 for c in self.class_indices.keys()}

        batch = []
        while True:
            # Cycle through classes, adding one sample per class
            for c in sorted(self.class_indices.keys()):
                if positions[c] >= len(shuffled_class_indices[c]):
                    # Reshuffle this class if exhausted
                    shuffled = rng.permutation(self.class_indices[c])
                    shuffled_class_indices[c] = shuffled.copy()
                    positions[c] = 0

                batch.append(shuffled_class_indices[c][positions[c]])
                positions[c] += 1

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Check if we have any samples left
            if not batch:
                break

            if not self.drop_last and len(batch) > 0:
                yield batch
                batch = []

    def __len__(self):
        """Total number of batches."""
        total_samples = len(self.labels)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class InRunDataShapleyGhost(DataEvaluator, ModelMixin):
    """In-Run Data Shapley using GhostSuite's GradDotProdEngine.

    Computes trajectory-aware data values by accumulating gradient dot products
    between each training sample and the validation loss gradient across every
    training step. No Hessian required.

    Key properties:
    - Values are first-order: positive = beneficial, negative = harmful
    - Trajectory-dependent: uses all training steps, not just final checkpoint
    - Class-balanced: requires balanced validation batch (no class dominance)
    - Shuffle-safe: uses indexed dataset to preserve global indices

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs, by default 10
    batch_size : int, optional
        Training batch size, by default 64
    learning_rate : float, optional
        Base learning rate, by default 0.01
    val_batch_size : int, optional
        Size of balanced validation batch, by default None (auto from x_valid)
    random_state : RandomState, optional
        Random state for reproducibility, by default None
    momentum : float, optional
        SGD momentum. If provided, enables ResNet-optimized mode (SGD + cyclic LR),
        by default None (uses Adam). Set to match ResNet9 training.
    weight_decay : float, optional
        L2 regularization strength for SGD, by default 0.0
    lr_peak_epoch : int, optional
        Epoch at which cyclic LR reaches peak (only in SGD mode), by default 5
    max_steps : int, optional
        Maximum number of training steps to run, by default None (run all)
        Useful for quick testing or limiting computation, by default 100
    scheduler_type : str, optional
        Learning rate scheduler type, by default 'onecycle'
        - 'onecycle': OneCycleLR (matches ResNet9, default for backward compatibility)
        - 'cosine': CosineAnnealingLR (matches ResNet18)
        - 'step': StepLR (set step_gamma=1.0 for an effectively fixed LR)
        - 'none': no scheduler at all; optimizer uses a fixed learning_rate
    dynamic_val_batch : bool, optional
        If True, draw a fresh validation batch each training step.
        If False (default), use the same validation batch for all steps.
        - True: Scores samples against multiple validation perspectives (robust)
        - False: Scores samples against fixed validation batch (efficient)
    balanced_val_batch : bool, optional
        If True (default), ensure validation batch is class-balanced.
        If False, use random sampling (may have class imbalance).
        - True: Each batch has ~equal samples per class (prevents class bias)
        - False: Random sampling (faster, but may have imbalanced batches)

    Training Modes
    ---------------
    Mode 1 (Default, momentum=None):
        Uses Adam optimizer with fixed learning rate

    Mode 2 (ResNet-optimized, momentum set):
        Uses SGD + momentum with cyclic (triangular) LR schedule
        Matches ResNet9 training configuration exactly
        Learning rate changes every batch: [0 → peak @ lr_peak_epoch → 0]

    References
    ----------
    GhostSuite: https://github.com/Jiachen-T-Wang/GhostSuite
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        val_batch_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
        save_plots: bool = True,
        plot_dir: Optional[str] = None,
        momentum: Optional[float] = None,
        weight_decay: float = 0.0,
        lr_peak_epoch: int = 5,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        max_steps: Optional[int] = None,
        scheduler_type: str = "onecycle",
        dynamic_val_batch: bool = False,
        balanced_val_batch: bool = True,
        step_size: int = 10,
        step_gamma: float = 0.1,
    ):
        super().__init__(random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_batch_size = val_batch_size
        self.verbose = verbose
        self.save_plots = save_plots
        self.plot_dir = plot_dir or "."
        self._values = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamic_val_batch = dynamic_val_batch
        self.balanced_val_batch = balanced_val_batch

        # ResNet-optimized mode detection
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_peak_epoch = lr_peak_epoch
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.use_sgd_mode = momentum is not None
        self.max_steps = max_steps
        self.step_size = step_size
        self.step_gamma = step_gamma

        # LR scheduler type
        if scheduler_type not in ("onecycle", "cosine", "step", "none"):
            raise ValueError(f"scheduler_type must be 'onecycle', 'cosine', 'step', or 'none', got '{scheduler_type}'")
        self.scheduler_type = scheduler_type

        # Debug tracking
        self.debug_history = {
            "step": [],
            "val_loss": [],
            "frac_pos_dot": [],
            "mean_abs_dot": [],
            "mean_values": [],
            "epoch": []
        }

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

        # Handle label format
        if self.y_train.dim() > 1:
            if self.y_train.shape[1] > 1:  # one-hot
                self.y_train_indices = self.y_train.argmax(dim=1)
            else:  # [N, 1]
                self.y_train_indices = self.y_train.squeeze(-1)
        else:
            self.y_train_indices = self.y_train.long()

        if self.y_valid.dim() > 1:
            if self.y_valid.shape[1] > 1:  # one-hot
                self.y_valid_indices = self.y_valid.argmax(dim=1)
            else:  # [N, 1]
                self.y_valid_indices = self.y_valid.squeeze(-1)
        else:
            self.y_valid_indices = self.y_valid.long()

        # Debug output
        if self.verbose:
            print("\n[DEBUG InRunDataShapleyGhost.input_data]")
            print(f"  Training set shape: {self.x_train.shape}, labels: {self.y_train_indices.shape}")
            print(f"  Validation set shape: {self.x_valid.shape}, labels: {self.y_valid_indices.shape}")

            # Check class balance in validation set
            num_classes = len(torch.unique(self.y_valid_indices))
            print(f"  Validation set class distribution (num_classes={num_classes}):")
            for c in range(num_classes):
                count = (self.y_valid_indices == c).sum().item()
                pct = 100.0 * count / len(self.y_valid_indices)
                print(f"    Class {c}: {count} samples ({pct:.1f}%)")

            # Check if balanced (each class has similar count)
            class_counts = [(self.y_valid_indices == c).sum().item() for c in range(num_classes)]
            min_count = min(class_counts)
            max_count = max(class_counts)
            is_balanced = (max_count - min_count) <= 1  # Allow 1 sample difference
            balance_status = "✓ BALANCED" if is_balanced else "⚠ IMBALANCED"
            print(f"  Validation balance status: {balance_status}")
            print()

        return self

    def train_data_values(self, *args, **kwargs):
        """Train with GradDotProdEngine and compute trajectory-aware values.

        Returns
        -------
        self
            Returns self for method chaining
        """
        try:
            from ghostEngines import GradDotProdEngine
        except ImportError as e:
            raise ImportError(
                "GhostSuite not installed. Install with: "
                "pip install git+https://github.com/Jiachen-T-Wang/GhostSuite"
            ) from e

        # Set random seeds for reproducibility
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = None

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

        # Move data to device
        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train_indices.to(self.device)
        x_valid = self.x_valid.to(self.device).float()
        y_valid = self.y_valid_indices.to(self.device)

        val_batch_size = self.val_batch_size or len(x_valid)
        val_batch_size = min(val_batch_size, len(x_valid))
        # Initialize model with a forward pass to resolve LazyLinear layers
        self.pred_model.train()
        with torch.no_grad():
            _ = self.pred_model(x_train[:min(1, len(x_train))])

        # Freeze BatchNorm if present (per-sample gradients invalid in train mode)
        self._freeze_batchnorm(self.pred_model)

        # Initialize values
        values = torch.zeros(len(x_train))

        # Mode selection and optimizer setup
        if self.use_sgd_mode:
            # Mode 2: ResNet-optimized (SGD + momentum + cyclic LR)
            optimizer = torch.optim.SGD(
                self.pred_model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            use_cyclic_lr = self.scheduler_type in ("onecycle", "cosine", "step")
            if self.verbose:
                print(f"[InRunDataShapleyGhost] Using SGD mode: lr={self.learning_rate}, momentum={self.momentum}, wd={self.weight_decay}, scheduler={self.scheduler_type}")
        else:
            # Mode 1/3/4: Adam (with or without scheduler)
            optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # Enable scheduler if scheduler_type is explicitly set (not default)
            use_cyclic_lr = self.scheduler_type in ("onecycle", "cosine", "step")
            if self.verbose:
                if use_cyclic_lr:
                    print(f"[InRunDataShapleyGhost] Using Adam mode with {self.scheduler_type} scheduler: lr={self.learning_rate}")
                else:
                    print(f"[InRunDataShapleyGhost] Using default Adam mode (fixed LR): lr={self.learning_rate}")

        # Attach engine
        engine = GradDotProdEngine(
            module=self.pred_model,
            val_batch_size=val_batch_size,
            loss_reduction="mean"
        )
        engine.attach(optimizer)

        # Training loop
        dataset = IndexedDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Create validation DataLoader for batch sampling.
        # drop_last is REQUIRED: the engine's hooks capture val_batch_size at
        # attach time, so every combined batch must contain exactly that many
        # val rows. A short final val batch crashes (or silently corrupts values).
        val_dataset = IndexedDataset(x_valid, y_valid)

        if self.balanced_val_batch:
            # Class-balanced sampler: ensures all classes represented in each batch
            sampler = BalancedBatchSampler(
                labels=y_valid,
                batch_size=val_batch_size,
                drop_last=True,
                random_state=seed + 1 if seed is not None else None
            )
            val_loader = DataLoader(val_dataset, batch_sampler=sampler)
        else:
            # Random sampling (original behavior)
            val_loader = DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True
            )

        val_gen = None

        # Debug: Validation set size and class balance
        if self.verbose:
            y_valid_cpu = y_valid.cpu().numpy() if isinstance(y_valid, torch.Tensor) else y_valid
            unique, counts = np.unique(y_valid_cpu, return_counts=True)
            class_dist_full = {int(c): int(cnt) for c, cnt in zip(unique, counts)}

            n_val_full = len(y_valid)
            n_batches = n_val_full // val_batch_size
            n_val_after_drop = n_batches * val_batch_size
            n_dropped = n_val_full - n_val_after_drop

            print(f"[DEBUG InRunDataShapleyGhost] Validation set:")
            print(f"  Full validation set: {n_val_full} samples")
            print(f"  Class distribution: {class_dist_full}")
            print(f"  Batch size: {val_batch_size}")
            print(f"  Full batches (with drop_last=True): {n_batches}")
            print(f"  Samples after drop_last: {n_val_after_drop}")
            print(f"  Samples dropped: {n_dropped}")

            # Check first batch
            print(f"  First validation batch class balance:")
            for i, (x_val_batch, y_val_batch, _) in enumerate(val_loader):
                if i == 0:
                    y_batch_cpu = y_val_batch.cpu().numpy() if isinstance(y_val_batch, torch.Tensor) else y_val_batch
                    unique_batch, counts_batch = np.unique(y_batch_cpu, return_counts=True)
                    class_dist_batch = {int(c): int(cnt) for c, cnt in zip(unique_batch, counts_batch)}
                    print(f"    Batch 1 (size={len(y_val_batch)}): {class_dist_batch}")
                break

        # Create scheduler based on scheduler_type
        scheduler = None
        if use_cyclic_lr:
            if self.scheduler_type == "onecycle":
                # OneCycleLR: matches ResNet9 (default for backward compatibility)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    epochs=self.epochs,
                    steps_per_epoch=len(loader),
                    pct_start=self.lr_peak_epoch / self.epochs,
                    anneal_strategy='cos',
                    div_factor=self.div_factor,
                    final_div_factor=self.final_div_factor
                )
            elif self.scheduler_type == "cosine":
                # CosineAnnealingLR: matches ResNet18
                # Note: This is called once per epoch, not per batch
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs,
                    eta_min=self.learning_rate / self.final_div_factor
                )
                # Mark as epoch-level scheduler for proper step calls
                scheduler.is_epoch_level = True
            elif self.scheduler_type == "step":
                # StepLR: matches ResNet18 exactly
                # Reduces LR by step_gamma every step_size epochs
                from torch.optim.lr_scheduler import StepLR
                scheduler = StepLR(
                    optimizer,
                    step_size=self.step_size,
                    gamma=self.step_gamma
                )
                # Mark as epoch-level scheduler for proper step calls
                scheduler.is_epoch_level = True

        # Debug output
        if self.verbose:
            print("[DEBUG InRunDataShapleyGhost.train_data_values]")
            print(f"  Training setup:")
            print(f"    Epochs: {self.epochs}, Batch size: {self.batch_size}, LR: {self.learning_rate}")
            print(f"    Training set: {len(x_train)} samples")
            print(f"    Validation set: {len(x_valid)} samples (batch_size={val_batch_size})")
            print(f"    Total batches per epoch: {len(loader)}")
            val_mode = "DYNAMIC (fresh batch each step)" if self.dynamic_val_batch else "FIXED (same batch per epoch)"
            bal_mode = "BALANCED (class-balanced)" if self.balanced_val_batch else "RANDOM (may be imbalanced)"
            print(f"    Validation batch mode: {val_mode}")
            print(f"    Validation batch sampling: {bal_mode}")
            print(f"    Device: {self.device}")
            print()

        step = 0
        total_batches = len(loader) * self.epochs

        # Create step-level progress bar
        step_pbar = tqdm(
            total=total_batches,
            desc="Training steps",
            unit="batch",
            disable=not self.verbose,
            leave=True
        )

        # Debug: Initial values
        if self.verbose:
            print("[DEBUG InRunDataShapleyGhost.train_data_values] Training start:")
            print(f"  Initial values sum: {values.sum().item():.6f}")
            print(f"  Initial values range: [{values.min().item():.6f}, {values.max().item():.6f}]")
            print(f"  Total batches to process: {total_batches}\n")

        self.pred_model.train()
        for epoch in range(self.epochs):
            epoch_values_change = 0.0

            if self.dynamic_val_batch:
                # Dynamic mode: fresh DataLoader each epoch for maximum randomness
                val_iter = iter(val_loader)
            else:
                # Fixed mode: reuse validation batches across epochs
                val_iter = iter(val_loader)

            for x_batch, y_batch, idx in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Sample validation batch
                if self.dynamic_val_batch:
                    # Dynamic: try to get next from current loader, refresh if exhausted
                    try:
                        x_val_batch, y_val_batch, _ = next(val_iter)
                    except StopIteration:
                        # Refresh validation loader for next epoch of dynamic sampling
                        val_iter = iter(val_loader)
                        x_val_batch, y_val_batch, _ = next(val_iter)
                else:
                    # Fixed: cycle through validation batches (original behavior)
                    try:
                        x_val_batch, y_val_batch, _ = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        x_val_batch, y_val_batch, _ = next(val_iter)

                x_val_batch = x_val_batch.to(self.device)
                y_val_batch = y_val_batch.to(self.device)

                # Engine hooks were sized for val_batch_size at attach time;
                # mutating engine.val_batch_size has no effect. Enforce instead.
                B_val = x_val_batch.size(0)
                assert B_val == val_batch_size, (
                    f"val batch {B_val} != engine val_batch_size {val_batch_size}"
                )

                # Build combined batch [train; val]
                cx = torch.cat([x_batch, x_val_batch])
                cy = torch.cat([y_batch, y_val_batch])

                B_train = x_batch.size(0)
                B_tot = cx.size(0)

                # Debug output for first batch
                if self.verbose and step == 0:
                    print("[DEBUG InRunDataShapleyGhost] First batch:")
                    print(f"  Training batch: {B_train} samples")
                    print(f"  Validation batch: {B_val} samples (from val_batch_size={val_batch_size})")
                    print(f"  Combined batch: {B_tot} samples ({B_train} train + {B_val} val)")
                    # Check class distribution in training batch
                    y_batch_np = y_batch.cpu().numpy()
                    for c in range(len(torch.unique(y_batch))):
                        count = (y_batch == c).sum().item()
                        print(f"    Train batch class {c}: {count} samples")
                    print()

                # Attach batch info for indexing
                engine.attach_train_batch(X_train=idx, Y_train=y_batch, iter_num=step)

                optimizer.zero_grad(set_to_none=True)

                # Forward and backward with engine context
                with engine.saved_tensors_context():
                    out = self.pred_model(cx)
                    loss = F.cross_entropy(out, cy, reduction="mean")
                    loss.backward()

                # Aggregate and extract dot products
                engine.aggregate_and_log()
                dot = engine.dot_product_log[-1]["dot_product"].float().cpu()

                # CRITICAL: Use CURRENT learning rate (not initial LR)
                # For cyclic schedules, LR changes every batch → Shapley weights must match
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = self.learning_rate

                # Accumulate values: positive sign convention
                # Scaling: current_lr * (B_tot^2) / (B_val * B_train)
                scaling = (current_lr * (B_tot ** 2)) / (B_val * B_train)
                values[idx] += scaling * dot
                epoch_values_change += (scaling * dot).abs().sum().item()

                # Track debug info
                if step % 10 == 0:  # Log every 10 steps to avoid clutter
                    with torch.no_grad():
                        vl = F.cross_entropy(self.pred_model(x_valid), y_valid).item()
                    self.debug_history["step"].append(step)
                    self.debug_history["val_loss"].append(vl)
                    self.debug_history["frac_pos_dot"].append((dot > 0).float().mean().item())
                    self.debug_history["mean_abs_dot"].append(dot.abs().mean().item())
                    self.debug_history["mean_values"].append(values.mean().item())
                    self.debug_history["epoch"].append(epoch)

                # Optimizer step
                engine.prepare_gradients()
                optimizer.step()
                # Update LR scheduler (OneCycleLR updates per batch)
                if scheduler is not None and not getattr(scheduler, 'is_epoch_level', False):
                    scheduler.step()  # Update LR for next batch (OneCycleLR)
                engine.clear_gradients()

                # Update progress bar with step count and metrics
                step_pbar.update(1)
                step_pbar.set_postfix({
                    'epoch': f'{epoch+1}/{self.epochs}',
                    'val_loss': f'{self.debug_history["val_loss"][-1]:.4f}' if self.debug_history["val_loss"] else 'N/A',
                    'mean_val': f'{values.mean().item():.6f}'
                })

                step += 1

                # Check if max_steps reached
                if self.max_steps is not None and step >= self.max_steps:
                    if self.verbose:
                        print(f"\n  ⚠️  Max steps ({self.max_steps}) reached, stopping training")
                    break

            # Break outer epoch loop if max_steps reached
            if self.max_steps is not None and step >= self.max_steps:
                break

            # Update LR scheduler at epoch level (CosineAnnealingLR updates per epoch)
            if scheduler is not None and getattr(scheduler, 'is_epoch_level', False):
                scheduler.step()  # Update LR for next epoch (CosineAnnealingLR)

            # Debug: Epoch progress
            if self.verbose:
                print(f"\n  ✓ Epoch {epoch+1}/{self.epochs} complete:")
                print(f"    Values change this epoch: {epoch_values_change:.6f}")
                print(f"    Values sum: {values.sum().item():.6f}")
                print(f"    Values range: [{values.min().item():.6f}, {values.max().item():.6f}]")
                print(f"    Mean value: {values.mean().item():.6f}\n")

        step_pbar.close()
        engine.detach()

        # Debug: Final values
        if self.verbose:
            print(f"\n" + "="*70)
            print(f"[InRunDataShapleyGhost] TRAINING COMPLETE")
            print(f"="*70)
            print(f"  Total steps: {step}")
            print(f"  Final values sum: {values.sum().item():.6f}")
            print(f"  Final values range: [{values.min().item():.6f}, {values.max().item():.6f}]")
            print(f"  Mean value: {values.mean().item():.6f}")
            print(f"  Std value: {values.std().item():.6f}")

            # Print debug stats
            stats = self.get_debug_stats()
            if stats:
                print(f"\n  [Training Statistics]")
                print(f"    Final val loss: {stats.get('final_val_loss', 0):.6f}")
                print(f"    Min val loss: {stats.get('min_val_loss', 0):.6f}")
                print(f"    Avg frac positive dots: {stats.get('avg_frac_pos_dot', 0):.4f}")
                print(f"    Avg mean abs dot: {stats.get('avg_mean_abs_dot', 0):.6f}")

            if values.abs().sum().item() < 1e-6:
                print(f"\n  ⚠️  WARNING: All values near zero - might not be learning!")
            else:
                print(f"\n  ✓ Values are being accumulated")

            if self.save_plots:
                print(f"  ✓ Training plots saved to: {self.plot_dir}")

            print(f"="*70)

        self._values = values.numpy()

        # Save debug CSV if requested (skip matplotlib plots to avoid backend issues)
        if self.save_plots:
            self._save_debug_csv()

        return self

    def _get_config_suffix(self) -> str:
        """Generate suffix with epochs, batch_size, and learning_rate."""
        return f"e{self.epochs}_b{self.batch_size}_lr{self.learning_rate}"


    def _save_debug_csv(self):
        """Save debug history to CSV with config suffix."""
        try:
            import pandas as pd
            import os
        except ImportError:
            if self.verbose:
                print("[InRunDataShapleyGhost] pandas not available - skipping CSV save")
            return

        if not self.debug_history["step"]:
            return

        # Create CSV directory if it doesn't exist
        os.makedirs(self.plot_dir, exist_ok=True)

        # Create DataFrame from debug history
        df = pd.DataFrame({
            "step": self.debug_history["step"],
            "epoch": self.debug_history["epoch"],
            "val_loss": self.debug_history["val_loss"],
            "frac_pos_dot": self.debug_history["frac_pos_dot"],
            "mean_abs_dot": self.debug_history["mean_abs_dot"],
            "mean_values": self.debug_history["mean_values"],
        })

        # Save with config suffix
        suffix = self._get_config_suffix()
        csv_path = os.path.join(self.plot_dir, f"inrun_shapley_debug_{suffix}.csv")
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"  ✓ Saved debug CSV to: {csv_path}")

    def get_debug_stats(self) -> dict:
        """Get debug statistics from training.

        Returns
        -------
        dict
            Dictionary with training statistics
        """
        if not self.debug_history["step"]:
            return {}

        return {
            "total_steps": len(self.debug_history["step"]),
            "final_val_loss": self.debug_history["val_loss"][-1] if self.debug_history["val_loss"] else None,
            "min_val_loss": min(self.debug_history["val_loss"]) if self.debug_history["val_loss"] else None,
            "avg_frac_pos_dot": np.mean(self.debug_history["frac_pos_dot"]) if self.debug_history["frac_pos_dot"] else None,
            "avg_mean_abs_dot": np.mean(self.debug_history["mean_abs_dot"]) if self.debug_history["mean_abs_dot"] else None,
            "final_mean_value": self.debug_history["mean_values"][-1] if self.debug_history["mean_values"] else None,
        }

    def evaluate_data_values(self) -> np.ndarray:
        """Return computed in-run Shapley data values.

        Returns
        -------
        np.ndarray
            Data values for each training point (shape: n_train,)
        """
        if self._values is None:
            raise RuntimeError(
                "No computed values. Call train_data_values() first."
            )
        return self._values.astype(np.float32)

    def _freeze_batchnorm(self, model):
        """Freeze BatchNorm layers (eval mode + no grad on affine params)."""
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = False
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = False
