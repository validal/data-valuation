"""SAVA and Hierarchical OT data valuation.

Implements batch-wise LAVA (SAVA) and Hierarchical Optimal Transport (HOT)
data valuation methods.

References
----------
    .. [1] S. Kessler et al.,
        SAVA: Scalable Learning-Agnostic Data Valuation,
        https://github.com/skezle/sava
"""
import time
from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from opendataval.dataval.progress import ProgressBar

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model


def _to_labels(y: torch.Tensor) -> torch.Tensor:
    """Convert one-hot labels to class indices if needed."""
    if isinstance(y, torch.Tensor) and y.dim() == 2 and y.shape[1] > 1:
        return y.argmax(dim=1)
    return y


def _calibrate_gradients(f: torch.Tensor) -> np.ndarray:
    """Apply LAVA calibration to dual potentials.

    Matches the formula in LavaEvaluator.evaluate_data_values including
    the -1 sign flip so that high value = valuable, low value = detrimental,
    consistent with all other opendataval evaluators.
    """
    f = f.squeeze()
    num_points = max(len(f) - 1, 1)
    calibrated = f * (1 + 1 / num_points) - f.sum() / num_points
    return (-1.0 * calibrated).numpy(force=True)


def _squash_and_normalize(values: np.ndarray) -> np.ndarray:
    """Apply tanh squashing to (-1,1) then shift/scale to (0,1) and normalize."""
    squashed = (np.tanh(values) + 1) / 2
    total = squashed.sum()
    if total > 0:
        squashed = squashed / total
    return squashed


def _stratified_batch_order(y: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return an index permutation that interleaves classes for class-balanced batches.

    Each consecutive block of ``batch_size`` indices in the returned permutation
    will contain roughly ``batch_size // n_classes`` samples from every class.
    Within each class the order is randomised.

    Parameters
    ----------
    y : torch.Tensor
        Training labels (class indices **or** one-hot — one-hot is converted
        automatically via :func:`_to_labels`).
    batch_size : int
        Target number of points per batch.

    Returns
    -------
    torch.Tensor
        1-D long tensor of length ``len(y)``.
    """
    labels = _to_labels(y)
    classes = labels.unique()
    n_classes = len(classes)
    per_class = max(1, batch_size // n_classes)

    class_indices = []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx))
        class_indices.append(idx[perm])

    order_parts = []
    pointers = [0] * n_classes
    while True:
        chunk = []
        for i, ci in enumerate(class_indices):
            end = min(pointers[i] + per_class, len(ci))
            if pointers[i] < len(ci):
                chunk.append(ci[pointers[i]:end])
                pointers[i] = end
        if not chunk:
            break
        order_parts.append(torch.cat(chunk))

    return torch.cat(order_parts) if order_parts else torch.arange(len(y))


def _solve_ot_batch(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    feature_cost,
    lam_x: float,
    lam_y: float,
    p: int,
    entreg: float,
    loss: str,
    blur: float,
    scaling: float,
    device: torch.device,
    mode: str,
    outer_debias: bool,
    cached_label_distances: Optional[torch.Tensor] = None,
) -> tuple:
    """Solve OT between one train batch and one val batch.

    Returns
    -------
    dual_sol : tuple[Tensor, Tensor]
        (F_i, G_j) dual potentials.
    label_distances : Tensor or None
        Cached label distance matrix for reuse.
    """
    print("unique labels:", y_tr[:5],y_val[:5])
    print("unique label counts (train):", torch.bincount(_to_labels(y_tr)))
    print("unique label counts (val):", torch.bincount(_to_labels(y_val)))
    print('shape labels:', y_tr.shape, y_val.shape)
    print("sample labels:", _to_labels(y_tr)[:5], _to_labels(y_val)[:5])
    dist = DatasetDistance(
        x_train=x_tr,
        y_train=_to_labels(y_tr),
        x_valid=x_val,
        y_valid=_to_labels(y_val),
        feature_cost=feature_cost,
        lam_x=lam_x,
        lam_y=lam_y,
        p=p,
        entreg=entreg,
        device=device,
        inner_ot_loss=loss,
        blur=blur,
        scaling=scaling,
        mode=mode,
        outer_debias=outer_debias,
    )

    # Inject cached label distances to skip recomputation
    if cached_label_distances is not None:
        dist.label_distances = cached_label_distances

    dual_sol = dist.dual_sol()
    label_distances = dist.label_distances  # may be None if lam_y=0
    return dual_sol, label_distances


class SavaEvaluator(DataEvaluator, ModelLessMixin):
    """SAVA: Scalable Learning-Agnostic Data Valuation (Kessler et al.).

    Two-level hierarchical OT:
    - Inner level: solve OT between each (train_batch, val_batch) pair → cost C_ij
    - Outer level: solve OT on the batch-level cost matrix C_bar → plan P_bar
    - Final score for point l in train batch k:
        s_l = sum_m  P_bar[k, m] * calibrated_gradient[k,m][l]

    Parameters
    ----------
    batch_size : int
        Number of points per train/validation batch.
    lam_x : float
        Weight for feature distance component.
    lam_y : float
        Weight for label distance component.
    p : int
        OT cost order (p-Wasserstein).
    entreg : float
        Entropy regularization strength for Sinkhorn.
    loss : str
        GeomLoss loss type, default "sinkhorn".
    blur : float
        Entropic scale for GeomLoss.
    scaling : float
        GeomLoss epsilon-scaling factor.
    sinkhorn_reg : float
        Regularization for the outer (batch-level) Sinkhorn, default 1e-2.
    cache_label_distances : bool
        Reuse label distance matrix W across batch pairs.
    mode : str
        "cls" for classification, "reg" for regression.
    device : torch.device
        Compute device.
    embedding_model : Model, optional
        Pre-trained embedding model.
    random_state : RandomState, optional
        Random seed.
    debug : bool
        Print timing and shape information.
    outer_debias : bool
        Use debiased Sinkhorn in outer OT problem.
    """

    def __init__(
        self,
        batch_size: int = 512,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        p: int = 2,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        blur: float = 0.05,
        scaling: float = 0.8,
        sinkhorn_reg: float = 1e-2,
        cache_label_distances: bool = True,
        mode: str = "cls",
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        debug: bool = False,
        outer_debias: bool = True,
        feature_cost: Optional[FeatureCost] = None,
        stratified_batches: bool = False,
    ):
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.batch_size = batch_size
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.entreg = entreg
        self.loss = loss
        self.blur = blur
        self.gl_scaling = scaling
        self.sinkhorn_reg = sinkhorn_reg
        self.cache_label_distances = cache_label_distances
        self.mode = mode
        self.device = device
        self.embedding_model = embedding_model
        self.debug = debug
        self.outer_debias = outer_debias
        self.feature_cost = feature_cost
        self.stratified_batches = stratified_batches

    def __repr__(self) -> str:
        embedding_str = "None"
        if self.embedding_model is not None:
            embedding_str = self.embedding_model.__class__.__name__
        return (
            f"SavaEvaluator(batch_size={self.batch_size}, lam_x={self.lam_x}, "
            f"lam_y={self.lam_y}, p={self.p}, embedding_model={embedding_str})"
        )

    __str__ = __repr__

    def _make_feature_cost(self):
        if self.feature_cost is not None:
            return self.feature_cost
        # train_data_values() calls self.embeddings(...) BEFORE this, so the data
        # is already feature vectors here. Building a FeatureCost that embeds a
        # second time with src_dim=(3,32,32) fails on those vectors:
        #   RuntimeError: shape '[-1, 3, 32, 32]' is invalid for input of size N
        # LavaEvaluator handles this by using a plain euclidean cost once the
        # data is embedded; do the same so SAVA and LAVA agree.
        return "euclidean"

    def _iter_batches(self, x: torch.Tensor, y: torch.Tensor):
        n = len(x)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield x[start:end], y[start:end]

    def train_data_values(self, *args, **kwargs):
        """Compute HOT data values via two-level OT.

        Steps (Algorithm 1 from SAVA paper):
        1. For each (train_batch_k, val_batch_m): solve OT → dual_sol, cost C_km
        2. Build batch-level cost matrix C_bar of shape (n_tr_batches, n_val_batches)
        3. Solve outer Sinkhorn on C_bar → plan P_bar
        4. For each point l in train batch k:
           s_l = sum_m P_bar[k, m] * calibrated_gradient[k, m][l]
        """
        try:
            import ot as pot
        except ImportError:
            raise ImportError(
                "POT (Python Optimal Transport) is required for HierarchicalOTEvaluator. "
                "Install it with: pip install POT"
            )

        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        feature_cost = self._make_feature_cost()

        # --- Stratified batching: reorder train data so each batch is class-balanced ---
        if self.stratified_batches:
            perm = _stratified_batch_order(self.y_train, self.batch_size)
            x_train_iter = x_train[perm]
            y_train_iter = self.y_train[perm]
            inv_perm = torch.argsort(perm).numpy()
            val_perm = _stratified_batch_order(self.y_valid, self.batch_size)
            x_valid_iter = x_valid[val_perm]
            y_valid_iter = self.y_valid[val_perm]
        else:
            x_train_iter = x_train
            y_train_iter = self.y_train
            inv_perm = None
            val_perm = torch.randperm(len(x_valid))
            x_valid_iter = x_valid[val_perm]
            y_valid_iter = self.y_valid[val_perm]

        train_batches = list(self._iter_batches(x_train_iter, y_train_iter))
        val_batches   = list(self._iter_batches(x_valid_iter, y_valid_iter))
        n_tr_batches  = len(train_batches)
        n_val_batches = len(val_batches)

        # Storage: dual_sol_dict[i][j] = (F_i, G_j) for train batch i, val batch j
        dual_sol_dict = {i: {} for i in range(n_tr_batches)}
        # Batch-level cost matrix C_bar[i, j] = OT cost between batch i and batch j
        costs_bar = np.zeros((n_tr_batches, n_val_batches), dtype=np.float64)

        label_distances = None
        t0 = time.perf_counter()

        # ---- Inner level: solve OT for all (i, j) pairs -------------------
        for i, (x_tr, y_tr) in enumerate(
            ProgressBar(iterable=train_batches, desc="HOT inner OT (train batches)")
        ):
            for j, (x_val, y_val) in enumerate(val_batches):
                print(y_tr.shape, y_val.shape)
                dual_sol, label_distances_new = _solve_ot_batch(
                    x_tr, y_tr, x_val, y_val,
                    feature_cost=feature_cost,
                    lam_x=self.lam_x,
                    lam_y=self.lam_y,
                    p=self.p,
                    entreg=self.entreg,
                    loss=self.loss,
                    blur=self.blur,
                    scaling=self.gl_scaling,
                    device=self.device,
                    mode=self.mode,
                    outer_debias=self.outer_debias,
                    cached_label_distances=label_distances if self.cache_label_distances else None,
                )

                if self.cache_label_distances and label_distances is None:
                    label_distances = label_distances_new

                dual_sol_dict[i][j] = dual_sol

                # Batch-level cost = mean transport cost (Algorithm 1 line 6)
                # Approximated as mean of dual potentials sum (consistent with Sinkhorn duality)
                fi = dual_sol[0].squeeze().numpy(force=True)   # (batch_tr_size,)
                gj = dual_sol[1].squeeze().numpy(force=True)   # (batch_val_size,)
                costs_bar[i, j] = (fi.mean() + gj.mean()) / 2

        if self.debug:
            print(f"[HOT] inner OT done in {time.perf_counter() - t0:.2f}s")
            print(f"[HOT] C_bar: shape={costs_bar.shape}, "
                  f"min={costs_bar.min():.4g}, max={costs_bar.max():.4g}")

        # ---- Outer level: Sinkhorn on C_bar --------------------------------
        a = np.ones(n_tr_batches,  dtype=np.float64) / n_tr_batches
        b = np.ones(n_val_batches, dtype=np.float64) / n_val_batches

        eps = np.max(np.abs(costs_bar))
        if eps < 1e-12:
            eps = 1.0
        plan_bar = pot.sinkhorn(
            a, b,
            costs_bar / eps,
            reg=self.sinkhorn_reg,
            verbose=False,
        )   # shape (n_tr_batches, n_val_batches)

        if self.debug:
            print(f"[HOT] P_bar: shape={plan_bar.shape}, "
                  f"sum={plan_bar.sum():.4g}")

        # ---- Aggregate per-point values (Algorithm 1 lines 10-14) ---------
        values = []
        for k, (x_tr, y_tr) in enumerate(train_batches):
            batch_tr_size = len(x_tr)
            for l in range(batch_tr_size):
                # Weighted sum over val batches using outer plan row k
                s_l = 0.0
                for m in range(n_val_batches):
                    calibrated = _calibrate_gradients(dual_sol_dict[k][m][0])
                    s_l += plan_bar[k, m] * calibrated[l]
                values.append(s_l)

        values_arr = np.array(values, dtype=np.float64)

        # Restore original training order after stratified permutation
        if inv_perm is not None:
            values_arr = values_arr[inv_perm]

        self.data_values = values_arr

        if self.debug:
            print(f"[HOT] train_data_values completed in {time.perf_counter() - t0:.2f}s")
            v = self.data_values
            print(f"[HOT] values: min={v.min():.4g}, max={v.max():.4g}, mean={v.mean():.4g}")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training point.

        Returns
        -------
        np.ndarray
            Shape (n_train,). Higher = more valuable.
        """
        if self.data_values is None:
            raise ValueError("Call train_data_values() first.")
        return self.data_values
