"""LoGRA - Influence functions via Low-Rank Approximation (LogIX integration).

Computes Hessian-preconditioned influence function scores using LogIX's
efficient LoRA-PCA gradient projection and K-FAC Hessian approximation.

Score:  I(q, z_i) = < q, (H + lambda*I)^{-1} g_i >
Aggregate over a validation set is EXACT with a single summed query,
because the score is linear in q and the preconditioner depends only on
the training logs:   sum_v I(q_v, z_i) = I(sum_v q_v, z_i).

Two query modes
---------------
aggregate_query=True  (default): build ONE summed validation gradient,
    run ONE scoring sweep of the train-gradient log store. Cheapest exact
    path for opendataval-style aggregate valuation (detection AUROC,
    point removal, cleanup-by-value). Does NOT produce per-test-point
    scores -> cannot feed LDS.
aggregate_query=False: batched per-test-point queries (query_batch_size
    rows per sweep); additionally stores self._influence_matrix of shape
    (n_valid, n_train) for LDS. Same aggregate values (linearity), at the
    cost of a fatter scoring matmul.

Fixes carried in this version
-----------------------------
1. logix.watch() operates on a PRIVATE deepcopy of pred_model. watch()
   mutates the model in place (LoRA-wraps every Linear); reusing the
   shared model across evaluators in one process injects stale
   logix_lora_* modules and stale Hessian state (the "size of tensor a
   (32) must match ... (1024)" crash). A guard assert refuses to watch
   an already-wrapped model.
2. logix.finalize() INSIDE the scheduler loop (required for lora="pca"
   multi-pass fitting: covariance fit -> LoRA add -> gradient save).
3. build_log_dataloader(batch_size=log_loader_batch_size): LogIX's
   default of 16 makes every scoring sweep dataloader-overhead-bound
   (62,500 iterations over a 1M-sample store). Default here: 65536.
4. Hard asserts on influence shape/finiteness and on logged-sample count
   (DataIDGenerator hashes inputs; duplicate rows collide and silently
   shrink the store).
5. Subset conversion bug fix (y_train previously indexed x_train.dataset).
6. Optional on-disk log-store cleanup after evaluation.
7. Float64 accumulation of the summed query and of aggregate scores.

Caveat: the summed-query identity holds for plain dot-product scoring
only. Any query-side normalization (relative / cosine influence) breaks
sum-of-scores = score-of-sum; with such configs use aggregate_query=False.

References
----------
LoGra: Choe et al., arXiv:2405.13954.  LogIX: github.com/logix-project/logix
"""

from typing import Optional
import contextlib
import copy
import shutil
import time
import uuid
from io import StringIO

import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, TensorDataset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.progress import ProgressBar


class LoGRA(DataEvaluator, ModelMixin):
    """LoGRA: Influence functions with LoRA-PCA and K-FAC (LogIX-based).

    Trains a model to convergence, then computes the influence of every
    training sample on the validation set at the final checkpoint.

    Three training modes:
    - Mode 1 (momentum set): SGD + momentum + scheduler (OneCycleLR or CosineAnnealingLR)
    - Mode 2 (momentum=None, scheduler_type!='step'): Adam + fixed LR
    - Mode 3 (momentum=None, scheduler_type='step'): Adam + StepLR

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs, by default 10
    batch_size : int, optional
        Training (and gradient-logging) batch size, by default 64
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.01
    lora : str, optional
        Projection: "none", "random", or "pca" (recommended; adds extra
        scheduler passes), by default "pca". WARNING: "none" writes
        UNPROJECTED per-sample gradients to disk; Phase 1 will stall on
        flushes for anything beyond toy sizes.
    hessian : str, optional
        "none" (plain grad dot product), "raw" (exact Fisher in the
        projected subspace), "kfac", "ekfac", by default "kfac".
        (ekfac adds one more scheduler pass than kfac.)
    aggregate_query : bool, optional
        True: single summed-validation-gradient query, one scoring sweep
        (no per-test matrix). False: batched per-test-point queries,
        keeps (n_valid, n_train) matrix for LDS. By default True.
    query_batch_size : int, optional
        Validation batch size for query backward passes (both modes) and
        per scoring sweep (batched mode). Set to n_valid when memory
        allows. By default 1024.
    log_loader_batch_size : int, optional
        Batch size for sweeping the train-gradient log store during
        scoring. Never leave at LogIX's tiny default. By default 65536.
    cleanup : bool, optional
        Delete the LogIX project directory after evaluation, by default True
    random_state : RandomState, optional
        Random state for reproducibility, by default None
    verbose : bool, optional
        Enable debug probes and prints, by default False
    momentum : float, optional
        SGD momentum (enables Mode 1 if set). By default None (Adam mode)
    weight_decay : float, optional
        L2 regularization, by default 0.0
    lr_peak_epoch : int, optional
        Peak epoch for OneCycleLR, by default 5
    div_factor : float, optional
        OneCycleLR div_factor, by default 25.0
    final_div_factor : float, optional
        OneCycleLR final_div_factor, by default 10000.0
    scheduler_type : str, optional
        Learning rate scheduler type, by default 'onecycle'
        - 'onecycle': OneCycleLR (Mode 1, batch-level, ResNet9)
        - 'cosine': CosineAnnealingLR (Mode 1, epoch-level, ResNet18)
        - 'step': StepLR (Mode 3, epoch-level, Adam+StepLR)
    step_size : int, optional
        StepLR step size (reduces LR every N epochs), by default 10
    step_gamma : float, optional
        StepLR multiplier at each step, by default 0.1
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        lora: str = "pca",
        hessian: str = "kfac",
        aggregate_query: bool = True,
        query_batch_size: int = 1024,
        log_loader_batch_size: int = 1024,
        cleanup: bool = True,
        random_state: Optional[RandomState] = None,
        verbose: bool = False,
        momentum: Optional[float] = None,
        weight_decay: float = 0.0,
        lr_peak_epoch: int = 5,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        scheduler_type: str = "onecycle",
        step_size: int = 10,
        step_gamma: float = 0.1,
    ):
        super().__init__(random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lora = lora
        self.hessian = hessian
        self.aggregate_query = aggregate_query
        self.query_batch_size = query_batch_size
        self.log_loader_batch_size = log_loader_batch_size
        self.cleanup = cleanup
        self.verbose = verbose
        self._values = None
        self._influence_matrix = None  # (n_valid, n_train), batched mode only
        self._seed = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ResNet-optimized mode detection
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_peak_epoch = lr_peak_epoch
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.use_sgd_mode = momentum is not None

        # LR scheduler type
        if scheduler_type not in ("onecycle", "cosine", "step"):
            raise ValueError(f"scheduler_type must be 'onecycle', 'cosine', or 'step', got '{scheduler_type}'")
        self.scheduler_type = scheduler_type

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _dbg(self, msg: str):
        if self.verbose:
            print(f"[LoGRA] {msg}")

    def _set_seeds(self, seed):
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _labels_to_indices(y: torch.Tensor) -> torch.Tensor:
        """One-hot [N, C] -> argmax; [N, 1] -> squeeze; [N] -> long."""
        if y.dim() > 1:
            return y.argmax(dim=1) if y.shape[1] > 1 else y.squeeze(-1).long()
        return y.long()

    @staticmethod
    def _make_relu_safe(model):
        """No in-place ReLU (LogIX LoRA hooks incompatible with in-place ops)."""
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        return model

    def _coverage_report(self, model):
        """Watched-parameter fraction (Linear proxy): structural-bias number
        to report per architecture — LogIX only hooks supported modules."""
        total_p = sum(p.numel() for p in model.parameters())
        lin_mods = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear)]
        lin_p = sum(m.weight.numel()
                    + (m.bias.numel() if m.bias is not None else 0)
                    for _, m in lin_mods)
        self._dbg(f"coverage (Linear proxy): {lin_p}/{total_p} params "
                  f"= {100 * lin_p / max(total_p, 1):.1f}% "
                  f"| modules: {[n for n, _ in lin_mods]}")
        if lin_p == 0:
            self._dbg("WARNING: no Linear modules found — logix.watch() "
                      "will capture nothing.")

    def _dump_log_shapes(self, log, tag):
        """Print module/key/shape of a LogIX log tuple (best effort)."""
        try:
            _, feats = log
            for mod, d in feats.items():
                for k, v in d.items():
                    self._dbg(f"  {tag}: module={mod} key={k} "
                              f"shape={tuple(v.shape)} dtype={v.dtype}")
        except Exception as e:
            self._dbg(f"  ({tag} introspection unavailable: {e})")

    def _score_call(self, logix, qlog, log_loader):
        """compute_influence_all with LogIX's tqdm suppressed unless verbose."""
        if self.verbose:
            return logix.influence.compute_influence_all(qlog, log_loader)
        with contextlib.redirect_stdout(StringIO()), \
             contextlib.redirect_stderr(StringIO()):
            return logix.influence.compute_influence_all(qlog, log_loader)

    # ------------------------------------------------------------------ #
    # opendataval API
    # ------------------------------------------------------------------ #
    def input_data(self, x_train, y_train, x_valid, y_valid):
        """Store training and validation data. Returns self."""
        from torch.utils.data import Subset

        def _unpack(subset, item_idx):
            return torch.stack([
                subset.dataset[i][item_idx]
                if isinstance(subset.dataset[i], tuple)
                else torch.as_tensor(subset.dataset[i])
                for i in subset.indices
            ])

        if isinstance(x_train, Subset):
            x_train = _unpack(x_train, 0)
        if isinstance(y_train, Subset):
            y_train = _unpack(y_train, 1)  # FIX: was indexing x_train.dataset
        if isinstance(x_valid, Subset):
            x_valid = _unpack(x_valid, 0)
        if isinstance(y_valid, Subset):
            y_valid = _unpack(y_valid, 1)

        self.x_train = torch.as_tensor(x_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train)
        self.x_valid = torch.as_tensor(x_valid, dtype=torch.float32)
        self.y_valid = torch.as_tensor(y_valid)
        self.y_train_indices = self._labels_to_indices(self.y_train)
        self.y_valid_indices = self._labels_to_indices(self.y_valid)

        if self.verbose:
            self._dbg(f"x_train {tuple(self.x_train.shape)} | "
                      f"x_valid {tuple(self.x_valid.shape)}")
            for name, yi in (("train", self.y_train_indices),
                             ("valid", self.y_valid_indices)):
                cls, cnt = torch.unique(yi, return_counts=True)
                self._dbg(f"{name} classes: "
                          f"{ {int(c): int(n) for c, n in zip(cls, cnt)} }")
        return self

    def train_data_values(self, *args, **kwargs):
        """Train pred_model with Adam. Influence computed in
        evaluate_data_values(). Returns self."""
        t0 = time.time()
        # pred_model may live on a non-default CUDA device (e.g. cuda:2);
        # match self.device to it instead of the hardcoded cuda:0 default.
        self.device = next(self.pred_model.parameters()).device
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            self._seed = rng.randint(0, 2**31 - 1)
        else:
            self._seed = None
        self._set_seeds(self._seed)
        self._dbg(f"seed={self._seed}")

        x_train = self.x_train.to(self.device).float()
        y_train = self.y_train_indices.to(self.device)

        # Resolve LazyLinear layers, disable in-place ReLU
        self.pred_model.train()
        with torch.no_grad():
            _ = self.pred_model(x_train[:1])
        self._make_relu_safe(self.pred_model)

        loader = DataLoader(TensorDataset(x_train, y_train),
                            batch_size=self.batch_size, shuffle=True)

        # Mode selection: SGD + scheduler, Adam + fixed LR, Adam + StepLR, or Adam + CosineAnnealingLR
        if self.use_sgd_mode:
            # Mode 1: SGD + momentum with scheduler
            optimizer = torch.optim.SGD(self.pred_model.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

            # Create scheduler based on scheduler_type
            if self.scheduler_type == "onecycle":
                # OneCycleLR: matches ResNet9 (batch-level updates)
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
                # CosineAnnealingLR: matches ResNet18 (epoch-level updates)
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs,
                    eta_min=self.learning_rate / self.final_div_factor
                )
                # Mark as epoch-level scheduler for proper step calls
                scheduler.is_epoch_level = True

            mode_str = f"SGD lr={self.learning_rate} momentum={self.momentum} wd={self.weight_decay} scheduler={self.scheduler_type}"

        elif self.scheduler_type == "cosine":
            # Mode 4: Adam + CosineAnnealingLR (NEW - when scheduler_type='cosine' and momentum=None)
            optimizer = torch.optim.Adam(self.pred_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.learning_rate / self.final_div_factor
            )
            scheduler.is_epoch_level = True
            mode_str = f"Adam+CosineAnnealingLR lr={self.learning_rate} wd={self.weight_decay} epochs={self.epochs}"

        elif self.scheduler_type == "step":
            # Mode 3: Adam + StepLR (when scheduler_type='step' and momentum=None)
            optimizer = torch.optim.Adam(self.pred_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.step_gamma
            )
            scheduler.is_epoch_level = True
            mode_str = f"Adam+StepLR lr={self.learning_rate} wd={self.weight_decay} step_size={self.step_size} gamma={self.step_gamma}"

        else:
            # Mode 2: Adam + fixed LR (original default)
            optimizer = torch.optim.Adam(self.pred_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
            scheduler = None
            mode_str = f"Adam (fixed) lr={self.learning_rate} wd={self.weight_decay}"

        if self.verbose:
            total = sum(p.numel() for p in self.pred_model.parameters())
            self._dbg(f"model={self.pred_model.__class__.__name__} "
                      f"({total} params) | {mode_str} "
                      f"bs={self.batch_size} epochs={self.epochs}")

        self.pred_model.train()
        epoch_losses = []
        with ProgressBar(total=self.epochs, desc="LoGRA Training") as pbar:
            for epoch in range(self.epochs):
                ep_loss, nb = 0.0, 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = F.cross_entropy(self.pred_model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    # Update LR scheduler (OneCycleLR updates per batch)
                    if scheduler is not None and not getattr(scheduler, 'is_epoch_level', False):
                        scheduler.step()  # Update LR for next batch (OneCycleLR)
                    ep_loss += loss.item()
                    nb += 1

                # Update LR scheduler at epoch level (CosineAnnealingLR updates per epoch)
                if scheduler is not None and getattr(scheduler, 'is_epoch_level', False):
                    scheduler.step()  # Update LR for next epoch (CosineAnnealingLR)

                epoch_losses.append(ep_loss / max(nb, 1))
                if self.verbose and (epoch % max(1, self.epochs // 5) == 0
                                     or epoch == self.epochs - 1):
                    self._dbg(f"epoch {epoch + 1}/{self.epochs}: "
                              f"loss={epoch_losses[-1]:.4f}")
                pbar.update(1)

        self.pred_model.eval()

        if self.verbose:
            # Convergence level = what IF is evaluated AT: record it for the
            # matched-convergence protocol (near-init checkpoints inflate
            # gradient-similarity detection scores).
            # NOTE: batched (not one giant forward over all of x_train/x_valid) —
            # a single unbatched pass over 40k CIFAR-resolution samples can need
            # tens of GB for one activation tensor alone (e.g. a [40000, 256, 32, 32]
            # layer1 activation is ~42GB in fp32) and OOMs regardless of the
            # batch_size used for actual training.
            with torch.no_grad():
                def _batched_loss_acc(x, y):
                    loader = DataLoader(TensorDataset(x, y),
                                        batch_size=self.batch_size, shuffle=False)
                    loss_sum, correct, total = 0.0, 0, 0
                    for xb, yb in loader:
                        out = self.pred_model(xb)
                        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
                        correct += (out.argmax(1) == yb).sum().item()
                        total += len(xb)
                    return loss_sum / total, correct / total

                tr_loss, tr_acc = _batched_loss_acc(x_train, y_train)
                v_loss, v_acc = _batched_loss_acc(
                    self.x_valid.to(self.device).float(),
                    self.y_valid_indices.to(self.device))
            self._dbg(f"checkpoint: train loss={tr_loss:.4f} acc={tr_acc:.4f}"
                      f" | val loss={v_loss:.4f} acc={v_acc:.4f}"
                      f" | wall={time.time() - t0:.1f}s")

        self._trained_model = self.pred_model.clone()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Compute LoGra influence per training point.

        Returns
        -------
        np.ndarray
            Aggregate influence per training point, shape (n_train,).
            In batched mode (aggregate_query=False), the per-test-point
            matrix is additionally stored in self._influence_matrix
            with shape (n_valid, n_train).
        """
        self._set_seeds(self._seed)

        def _gpu_mem(tag):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved() / 1024**3
                maxalloc = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[LoGRA-MEM] {tag}: allocated={alloc:.2f}GiB "
                      f"reserved={reserv:.2f}GiB max_allocated={maxalloc:.2f}GiB",
                      flush=True)

        _gpu_mem("evaluate_data_values: entry")

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
        n_train, n_valid = len(x_train), len(x_valid)
        _gpu_mem("after x_train/x_valid moved to device")

        # ---- private copy: logix.watch() mutates the model in place ----
        score_model = self._make_relu_safe(
            copy.deepcopy(self.pred_model)).eval().to(self.device)
        assert not any("logix_lora" in n
                       for n, _ in score_model.named_modules()), (
            "model already LoRA-wrapped by a previous logix.watch(); "
            "instantiate a fresh evaluator/model per run")
        _gpu_mem("after deepcopy(pred_model) -> score_model")

        # Unique project dir: no cross-job interference, no stale-log reuse
        project_name = (f"opendataval_logra_"
                        f"{int(time.time() * 1e6)}_{uuid.uuid4().hex[:8]}")
        logix = LogIX(project=project_name, config=None)
        scheduler = LogIXScheduler(logix, lora=self.lora,
                                   hessian=self.hessian, save="grad")
        _gpu_mem("after LogIX(...)/LogIXScheduler(...) constructed")
        # FIX: LogIX Conv2d hooks mutate captured activations in place; clone every
        # hooked module's input so hooks touch private copies (differentiable identity)
        def _decouple_pre(_mod, _inputs):
            return tuple(i.clone() if isinstance(i, torch.Tensor) else i
                         for i in _inputs)
        self._decouple_handles = [
            _m.register_forward_pre_hook(_decouple_pre)
            for _m in score_model.modules()
            if isinstance(_m, (torch.nn.Conv2d, torch.nn.Linear))
        ]
        _gpu_mem("after registering _decouple_pre hooks")
        logix.watch(score_model)
        _gpu_mem("after logix.watch(score_model)")
        # mode="index": sequential IDs by position, not content hash. Adult's
        # one-hot-encoded tabular rows contain exact duplicates (e.g. two
        # people sharing every categorical feature), and hash-mode collapses
        # those to a single logged gradient, silently shrinking the score
        # count below n_train (see the assert below).
        idg = DataIDGenerator(mode="index")

        if self.verbose:
            mode = "aggregate (1 summed query)" if self.aggregate_query \
                else f"batched (query_bs={self.query_batch_size})"
            self._dbg(f"lora={self.lora} hessian={self.hessian} "
                      f"| mode={mode} | scheduler passes={len(scheduler)} "
                      f"| log_loader_bs={self.log_loader_batch_size}")
            self._coverage_report(score_model)
            if self.lora == "none":
                self._dbg("WARNING: lora='none' writes unprojected "
                          "gradients to disk; expect Phase 1 stalls.")

        progress = ProgressBar(total=2, desc="LoGRA")

        # ------------- Phase 1: log per-sample train gradients ----------
        t1 = time.time()
        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  batch_size=self.batch_size, shuffle=False)
        n_passes = len(scheduler)
        n_logged_last = 0
        for ep_i, _ in enumerate(scheduler):
            n_logged = 0
            for bi, (xb, yb) in enumerate(train_loader):
                if ep_i == 0 and bi < 3:
                    _gpu_mem(f"pass {ep_i+1}/{n_passes} batch {bi} "
                             f"(shape={tuple(xb.shape)}): before forward+backward")
                with logix(data_id=idg(xb)):
                    score_model.zero_grad()
                    F.cross_entropy(score_model(xb), yb,
                                    reduction="sum").backward()
                if ep_i == 0 and bi < 3:
                    _gpu_mem(f"pass {ep_i+1}/{n_passes} batch {bi}: after forward+backward")
                n_logged += len(xb)
                if self.verbose and ep_i == n_passes - 1 and bi == 0:
                    self._dump_log_shapes(logix.get_log(), "train-log")
            _gpu_mem(f"pass {ep_i+1}/{n_passes}: before logix.finalize()")
            logix.finalize()  # INSIDE the loop — required for lora="pca"
            _gpu_mem(f"pass {ep_i+1}/{n_passes}: after logix.finalize()")
            n_logged_last = n_logged
            self._dbg(f"scheduler pass {ep_i + 1}/{n_passes} "
                      f"({n_logged} samples)"
                      + (" [gradient-save]" if ep_i == n_passes - 1
                         else " [fit]"))

        assert n_logged_last == n_train, (
            f"logged {n_logged_last} sample-grads on the save pass, expected "
            f"{n_train}. Duplicate inputs collide in DataIDGenerator "
            f"(id = hash of input) and shrink the store.")
        self._dbg(f"phase 1: {time.time() - t1:.1f}s")
        _gpu_mem("phase 1 complete")
        progress.update(1)

        # ------------- Phase 2: scoring ----------------------------------
        t2 = time.time()
        try:
            log_loader = logix.build_log_dataloader(
                batch_size=self.log_loader_batch_size)
        except TypeError:
            # older LogIX without a batch_size kwarg
            log_loader = logix.build_log_dataloader()
            self._dbg("WARNING: build_log_dataloader() ignored batch_size — "
                      "scoring sweeps will be dataloader-overhead-bound "
                      "(LogIX default bs=16).")
        _gpu_mem("after logix.build_log_dataloader()")
        logix.eval()
        logix.setup({"grad": ["log"]})
        _gpu_mem("after logix.eval()/setup()")

        val_loader = DataLoader(TensorDataset(x_valid, y_valid),
                                batch_size=self.query_batch_size,
                                shuffle=False)

        # Debug: Check validation set class balance
        if self.verbose:
            print(f"[DEBUG] Validation batch class balance:")
            for i, (x_val_batch, y_val_batch) in enumerate(val_loader):
                if i == 0:  # First batch only
                    y_batch_cpu = y_val_batch.cpu().numpy() if isinstance(y_val_batch, torch.Tensor) else y_val_batch
                    unique, counts = np.unique(y_batch_cpu, return_counts=True)
                    class_dist = {int(c): int(cnt) for c, cnt in zip(unique, counts)}
                    print(f"  First validation batch (size={len(y_val_batch)}): {class_dist}")
                    break

        _gpu_mem(f"before phase2_{'aggregate' if self.aggregate_query else 'batched'}()")
        if self.aggregate_query:
            values = self._phase2_aggregate(
                logix, idg, score_model, val_loader, log_loader, n_train)
        else:
            values = self._phase2_batched(
                logix, idg, score_model, val_loader, log_loader,
                n_train, n_valid)
        _gpu_mem(f"after phase2_{'aggregate' if self.aggregate_query else 'batched'}()")

        self._dbg(f"phase 2: {time.time() - t2:.1f}s")
        progress.update(1)

        self._values = values.astype(np.float32)

        if self.verbose:
            v = torch.as_tensor(self._values)
            k = min(10, n_train)
            self._dbg(f"final: mean={v.mean():.3e} std={v.std():.3e} | "
                      f"top-{k} helpful={v.topk(k).indices.tolist()} | "
                      f"top-{k} harmful={(-v).topk(k).indices.tolist()}")

        if self.cleanup:
            log_dir = getattr(logix, "log_dir", None) or project_name
            try:
                del log_loader, logix
                shutil.rmtree(log_dir, ignore_errors=True)
                self._dbg(f"cleaned up log store: {log_dir}")
            except Exception as e:
                self._dbg(f"cleanup failed (non-fatal): {e}")

        return self._values

    # ------------------------------------------------------------------ #
    # Phase-2 variants
    # ------------------------------------------------------------------ #
    def _phase2_aggregate(self, logix, idg, model, val_loader, log_loader,
                          n_train) -> np.ndarray:
        """ONE summed validation gradient -> ONE scoring sweep.

        Exact for plain dot-product scoring: the query log rows are summed
        BEFORE preconditioning, valid because preconditioning is linear.
        """
        acc = None
        n_q = 0
        for x_val, y_val in val_loader:
            with logix(data_id=idg(x_val)):
                model.zero_grad()
                F.cross_entropy(model(x_val), y_val,
                                reduction="sum").backward()
                _, feats = logix.get_log()
            if self.verbose and n_q == 0:
                self._dump_log_shapes((None, feats), "query-log")
            # collapse per-sample rows -> one summed row (float64 accumulator)
            s = {m: {k: v.sum(dim=0, keepdim=True).double()
                     for k, v in d.items()}
                 for m, d in feats.items()}
            if acc is None:
                acc = s
            else:
                for m, d in s.items():
                    for k, v in d.items():
                        acc[m][k] += v
            n_q += len(x_val)

        qlog = (["val_sum"],
                {m: {k: v.float() for k, v in d.items()}
                 for m, d in acc.items()})

        result = self._score_call(logix, qlog, log_loader)
        infl = torch.as_tensor(result["influence"]).flatten()

        assert infl.numel() == n_train, (
            f"got {infl.numel()} scores, expected {n_train}")
        assert torch.isfinite(infl).all(), (
            "non-finite influence — check damping / hessian mode")

        self._dbg(f"aggregate mode: {n_q} val points -> 1 query, 1 sweep")
        self._influence_matrix = None  # not available on this path
        return infl.numpy()

    def _phase2_batched(self, logix, idg, model, val_loader, log_loader,
                        n_train, n_valid) -> np.ndarray:
        """Batched per-test-point queries; keeps (n_valid, n_train) matrix.

        Aggregate values identical to _phase2_aggregate by linearity.
        """
        influence = torch.zeros(n_train, dtype=torch.float64)
        blocks = []
        sweeps = 0
        for qi, (x_val, y_val) in enumerate(val_loader):
            with logix(data_id=idg(x_val)):
                model.zero_grad()
                F.cross_entropy(model(x_val), y_val,
                                reduction="sum").backward()
                tlog = logix.get_log()
            if self.verbose and qi == 0:
                self._dump_log_shapes(tlog, "query-log")

            result = self._score_call(logix, tlog, log_loader)
            sweeps += 1
            block = torch.as_tensor(result["influence"]).reshape(
                len(x_val), -1).cpu()

            assert block.shape[1] == n_train, (
                f"influence block has {block.shape[1]} train columns, "
                f"expected {n_train}")
            assert torch.isfinite(block).all(), (
                f"non-finite influence in query batch {qi} — "
                f"check damping / hessian mode")

            if self.verbose:
                self._dbg(f"query batch {qi}: block {tuple(block.shape)} | "
                          f"mean={block.mean():.3e} std={block.std():.3e}")

            influence += block.sum(dim=0).double()
            blocks.append(block)

        self._influence_matrix = torch.cat(blocks, dim=0).numpy()
        assert self._influence_matrix.shape == (n_valid, n_train)
        self._dbg(f"batched mode: {n_valid} queries in {sweeps} sweep(s) "
                  f"(per-point looping: {n_valid})")
        return influence.numpy()