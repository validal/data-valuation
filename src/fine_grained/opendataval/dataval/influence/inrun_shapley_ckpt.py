"""InRunDataShapley with validation batch size checkpoints.

Trains once with full per-example ghost dot matrices captured independently,
then materializes checkpoint proxy views for nested validation batch sizes.
All checkpoint logic is self-contained here — InRunDataShapleyGhost is untouched.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.utils import check_random_state

from opendataval.dataval.influence.inrun_shapley_ghost import (
    InRunDataShapleyGhost, IndexedDataset
)
from opendataval.dataval.api import DataEvaluator


def _balanced_nested_order(y_valid: torch.Tensor, v_max: int, seed: Optional[int]) -> torch.Tensor:
    """Round-robin-over-classes ordering where every prefix [:k] is class-balanced."""
    y_np = y_valid.cpu().numpy() if isinstance(y_valid, torch.Tensor) else y_valid
    classes = np.unique(y_np)
    rng = np.random.RandomState(seed)
    per_class_idx = {c: rng.permutation(np.where(y_np == c)[0]) for c in classes}
    order, positions = [], {c: 0 for c in classes}
    while len(order) < v_max:
        for c in classes:
            if positions[c] < len(per_class_idx[c]):
                order.append(int(per_class_idx[c][positions[c]]))
                positions[c] += 1
                if len(order) == v_max:
                    break
    return torch.tensor(order, dtype=torch.long)


class _GhostDotMulti:
    """Independent per-example ghost dot matrix D[i, v] = <g_i, g_v> for all
    nn.Linear layers, captured from same backward pass as GradDotProdEngine
    (multiple hooks per module fire independently). Read-only."""

    def __init__(self, module: nn.Module, n_val: int):
        self.n_val = n_val
        self.layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        assert self.layers, "no nn.Linear layers found in scored module"
        self._h = []
        for layer in self.layers:
            self._h.append(layer.register_forward_hook(self._fwd))
            self._h.append(layer.register_full_backward_hook(self._bwd))

    @staticmethod
    def _fwd(mod, inp, _):
        mod._ghost_A = inp[0].detach()

    @staticmethod
    def _bwd(mod, _, gout):
        mod._ghost_B = gout[0].detach()

    def dot_matrix(self) -> torch.Tensor:
        """Returns D of shape (B_train, n_val), on same device as captured tensors."""
        D = None
        for m in self.layers:
            A, B = m._ghost_A, m._ghost_B
            A_t, A_v = A[:-self.n_val], A[-self.n_val:]
            B_t, B_v = B[:-self.n_val], B[-self.n_val:]
            layer_D = (B_t @ B_v.T) * (A_t @ A_v.T)
            if m.bias is not None:
                layer_D = layer_D + (B_t @ B_v.T)
            D = layer_D if D is None else D + layer_D
        return D

    def remove(self):
        for h in self._h:
            h.remove()


class InRunShapleyCKPT(InRunDataShapleyGhost):
    """In-Run Data Shapley with validation batch size checkpoints.

    Trains ONCE against a fixed class-balanced-ordered validation pool of
    size `max(checkpoint_val_batch_sizes)`, capturing genuine per-example
    ghost dot matrices each step via independent hooks. Every checkpoint size
    gets its own independently-computed attribution (not scalar-rescaled).

    Parameters
    ----------
    checkpoint_val_batch_sizes : list[int]
        Validation batch sizes to track, e.g. [16, 32, 128, 512]. The
        LARGEST value defines the validation pool used for training.
    (all other params inherited from InRunDataShapleyGhost)
    """

    def __init__(self, checkpoint_val_batch_sizes: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_val_batch_sizes_input = checkpoint_val_batch_sizes or []
        self.checkpoints = {}
        self.checkpoint_memory_reports = {}
        self.conserve_start = {}
        self.conserve_end = {}
        self.debug_history_per_k = {}

    def input_data(self, x_train, y_train, x_valid, y_valid):
        super().input_data(x_train, y_train, x_valid, y_valid)

        sizes = sorted(set(self.checkpoint_val_batch_sizes_input))
        if not sizes:
            raise ValueError("checkpoint_val_batch_sizes must be non-empty")
        for bs in sizes:
            if not isinstance(bs, int) or bs <= 0 or bs > len(x_valid):
                raise ValueError(
                    f"checkpoint_val_batch_sizes must contain positive ints <= "
                    f"validation samples ({len(x_valid)}), got {bs}")
        self.checkpoint_val_batch_sizes = sizes
        self.v_max = sizes[-1]

        if self.verbose:
            print(f"[InRunShapleyCKPT] checkpoints={sizes}  v_max={self.v_max}")
        return self

    def train_data_values(self, *args, **kwargs):
        """Train with full per-example dot matrices, checkpoint at each requested size."""
        try:
            from ghostEngines import GradDotProdEngine
        except ImportError as e:
            raise ImportError(
                "GhostSuite not installed. Install with: "
                "pip install git+https://github.com/Jiachen-T-Wang/GhostSuite"
            ) from e

        if self.verbose:
            print("\n" + "=" * 70)
            print("[InRunShapleyCKPT] TRAINING WITH CLASS-BALANCED CHECKPOINTS")
            print(f"  checkpoints={self.checkpoint_val_batch_sizes}  v_max={self.v_max}")
            print("=" * 70 + "\n")

        try:
            self._reset_peak_memory_stats()
        except (RuntimeError, AttributeError):
            pass
        ckpt_start = self._memory_snapshot() if hasattr(self, "_memory_snapshot") else None
        ckpt_t0 = time.perf_counter()

        seed = self._set_seeds()

        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train_indices.to(self.device)
        x_valid = self.x_valid.to(self.device).float()
        y_valid = self.y_valid_indices.to(self.device)

        # ONE fixed, class-balanced-ordered val pool
        order = _balanced_nested_order(y_valid, self.v_max, seed)
        X_val = x_valid[order].contiguous()
        Y_val = y_valid[order].contiguous()

        self.pred_model.train()
        with torch.no_grad():
            _ = self.pred_model(x_train[:1])
        self._freeze_batchnorm(self.pred_model)

        n_train = len(x_train)
        ck_sizes = self.checkpoint_val_batch_sizes
        values = {k: torch.zeros(n_train, device=self.device) for k in ck_sizes}
        for k in ck_sizes:
            self.debug_history_per_k[k] = {"step": [], "val_loss": [], "mean_values": []}

        optimizer = self._make_optimizer()
        engine = GradDotProdEngine(module=self.pred_model, val_batch_size=self.v_max,
                                   loss_reduction="mean")
        engine.attach(optimizer)

        # Independent hook set for full per-example dot matrix
        multi = _GhostDotMulti(self.pred_model, n_val=self.v_max)

        with torch.no_grad():
            for k in ck_sizes:
                self.conserve_start[k] = F.cross_entropy(
                    self.pred_model(X_val[:k]), Y_val[:k]).item()

        loader = DataLoader(IndexedDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)
        scheduler = self._make_scheduler(optimizer, len(loader))

        from tqdm import tqdm
        step, total_batches = 0, len(loader) * self.epochs
        pbar = tqdm(total=total_batches, desc="[checkpoint mode] steps",
                   disable=not self.verbose, leave=True)

        self.pred_model.train()
        for epoch in range(self.epochs):
            for x_batch, y_batch, idx in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                idx_t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx)
                idx_t = idx_t.to(self.device)

                B_tr = x_batch.size(0)
                cx = torch.cat([x_batch, X_val])
                cy = torch.cat([y_batch, Y_val])

                engine.attach_train_batch(X_train=idx, Y_train=y_batch, iter_num=step)
                optimizer.zero_grad(set_to_none=True)
                with engine.saved_tensors_context():
                    out = self.pred_model(cx)
                    loss = F.cross_entropy(out, cy, reduction="mean")
                    loss.backward()

                engine.aggregate_and_log()

                # Full per-example dot matrix D[i, v] before engine's collapse
                D_step = multi.dot_matrix()

                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else self.learning_rate
                for k in ck_sizes:
                    B_tot_k = B_tr + k
                    scale = current_lr * (B_tot_k ** 2) / (k * B_tr)
                    contrib = scale * D_step[:, :k].sum(dim=1)
                    values[k].index_add_(0, idx_t, contrib)

                if step % 10 == 0:
                    with torch.no_grad():
                        for k in ck_sizes:
                            vl = F.cross_entropy(self.pred_model(X_val[:k]), Y_val[:k]).item()
                            self.debug_history_per_k[k]["step"].append(step)
                            self.debug_history_per_k[k]["val_loss"].append(vl)
                            self.debug_history_per_k[k]["mean_values"].append(values[k].mean().item())

                engine.prepare_gradients()
                optimizer.step()
                if scheduler is not None and not getattr(scheduler, 'is_epoch_level', False):
                    scheduler.step()
                engine.clear_gradients()

                pbar.update(1)
                pbar.set_postfix({'epoch': f'{epoch + 1}/{self.epochs}'})
                step += 1
                if self.max_steps is not None and step >= self.max_steps:
                    break
            if scheduler is not None and getattr(scheduler, 'is_epoch_level', False):
                scheduler.step()
            if self.max_steps is not None and step >= self.max_steps:
                break

        pbar.close()
        multi.remove()
        engine.detach()

        with torch.no_grad():
            for k in ck_sizes:
                self.conserve_end[k] = F.cross_entropy(
                    self.pred_model(X_val[:k]), Y_val[:k]).item()

        # Materialize checkpoints
        for k in ck_sizes:
            self.checkpoints[k] = values[k].detach().cpu()

        elapsed = time.perf_counter() - ckpt_t0
        for k in ck_sizes:
            if hasattr(self, "_build_memory_report") and ckpt_start is not None:
                self.checkpoint_memory_reports[k] = self._build_memory_report(
                    ckpt_start, self._memory_snapshot(), elapsed)
            if self.verbose:
                v = self.checkpoints[k].numpy()
                claimed, actual = v.sum(), self.conserve_start[k] - self.conserve_end[k]
                gap = abs(claimed - actual) / (abs(actual) + 1e-12)
                print(f"[InRunShapleyCKPT] k={k:4d}  mean={v.mean():.4e}  std={v.std():.4e}  "
                     f"Sigma_phi={claimed:.4e}  DeltaL_val={actual:.4e}  rel_gap={gap:.2%}")

        if self.verbose:
            print("\n" + "=" * 70)
            print("[InRunShapleyCKPT] CHECKPOINTS MATERIALIZED")
            print(f"  Checkpoints: {sorted(self.checkpoints.keys())}")
            print("=" * 70 + "\n")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        if not self.checkpoints:
            raise RuntimeError("No computed values. Call train_data_values() first.")
        return self.checkpoints[max(self.checkpoints.keys())].numpy()

    def get_checkpoint_evaluators(self) -> list:
        if len(self.checkpoints) <= 1:
            return []
        return [InRunShapleyCKPTView(self, k) for k in sorted(self.checkpoints.keys())]

    @property
    def data_values(self) -> np.ndarray:
        return self.evaluate_data_values()

    def conservation_report(self) -> dict:
        """Per-checkpoint efficiency-axiom audit."""
        out = {}
        for k in sorted(self.checkpoints.keys()):
            v = self.checkpoints[k].numpy()
            claimed = float(v.sum())
            actual = self.conserve_start[k] - self.conserve_end[k]
            out[k] = {
                "sigma_phi": claimed,
                "delta_L_val": actual,
                "rel_gap": abs(claimed - actual) / (abs(actual) + 1e-12),
            }
        return out

    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(0, 2 ** 31 - 1)
        else:
            seed = None
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return seed

    def _make_optimizer(self):
        """Create optimizer (SGD or Adam based on use_sgd_mode)."""
        if self.use_sgd_mode:
            return torch.optim.SGD(
                self.pred_model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        return torch.optim.Adam(
            self.pred_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def _make_scheduler(self, optimizer, steps_per_epoch):
        """Create learning rate scheduler."""
        if self.scheduler_type == "onecycle":
            pct_start = min(0.99, self.lr_peak_epoch / max(1, self.epochs))
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate, epochs=self.epochs,
                steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                anneal_strategy='cos', div_factor=self.div_factor,
                final_div_factor=self.final_div_factor
            )
        elif self.scheduler_type == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs,
                eta_min=self.learning_rate / self.final_div_factor
            )
            sch.is_epoch_level = True
            return sch
        elif self.scheduler_type == "step":
            sch = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.step_gamma
            )
            sch.is_epoch_level = True
            return sch
        return None


class InRunShapleyCKPTView(DataEvaluator):
    """Read-only proxy view of InRunShapleyCKPT checkpoint."""

    def __init__(self, parent: InRunShapleyCKPT, batch_size: int):
        super().__init__()
        self.parent = parent
        self.batch_size = batch_size
        self.pred_model = parent.pred_model
        self.memory_report = parent.checkpoint_memory_reports.get(batch_size) or {}

    def train_data_values(self, *args, **kwargs):
        return self

    def evaluate_data_values(self) -> np.ndarray:
        vals = self.parent.checkpoints[self.batch_size]
        return vals.cpu().numpy() if isinstance(vals, torch.Tensor) else vals

    @property
    def data_values(self) -> np.ndarray:
        return self.evaluate_data_values()

    def __repr__(self) -> str:
        return f"InRunShapleyCKPT@bs{self.batch_size}(epochs={self.parent.epochs}, val_batch_size={self.batch_size})"

    __str__ = __repr__
