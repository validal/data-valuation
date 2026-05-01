from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
import time

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class KNNShapley(DataEvaluator, ModelLessMixin):
    """Data valuation using KNNShapley implementation.

    KNN Shapley is a model-less mixin. This means we cannot specify an underlying
    prediction model for the DataEvaluator. However, we can specify a pretrained
    embedding model.

    This implementation processes one validation point at a time to avoid
    materialising the full (n_train × n_valid) distance matrix in memory.
    Peak memory is O(n_train) rather than O(n_train × n_valid).

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors to group the data points, by default 10
    batch_size : int, optional
        Batch size used when embedding x_train in a single forward pass,
        by default 32
    embedding_model : Model, optional
        Pre-trained embedding model used by DataEvaluator, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    debug : bool, optional
        Print timing / shape diagnostics, by default False
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        debug: bool = False,
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        self.debug = debug
        # Optional precomputed distance matrix (n_train × n_valid)
        self.dist_matrix: Optional[torch.Tensor] = None

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[KNNShapley][DEBUG] {msg}")

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """:math:`1.` for all matching rows and :math:`0.` otherwise."""
        return (y == self.y_valid).all(dim=1).float()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        """Convert an arbitrary array-like / dataset to a 2-D float tensor."""
        if isinstance(x, torch.Tensor):
            return x
        try:
            chunks = []
            for batch in DataLoader(x, batch_size=self.batch_size):
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                if not isinstance(xb, torch.Tensor):
                    xb = torch.as_tensor(np.array(xb))
                chunks.append(xb)
            return torch.cat(chunks, dim=0)
        except Exception:
            return torch.as_tensor(np.array(x))

    def _dist_to_one_valid(
        self,
        x_train_flat: torch.Tensor,   # (n, d)
        x_val_point: torch.Tensor,    # (1, d)  – a single validation row
    ) -> torch.Tensor:
        """Return Euclidean distances from all training points to one validation
        point, computed in batches to bound memory use.

        Returns
        -------
        torch.Tensor, shape (n,)
        """
        parts = []
        for x_batch in DataLoader(x_train_flat, batch_size=self.batch_size):
            # x_batch: (bs, d)
            d = torch.cdist(x_batch, x_val_point)   # (bs, 1)
            parts.append(d.squeeze(1))               # (bs,)
        return torch.cat(parts, dim=0)               # (n,)

    def _shapley_one_valid(
        self,
        dist_j: torch.Tensor,   # (n,)  distances to the j-th validation point
        y_val_j: torch.Tensor,  # (1, ...)  label of the j-th validation point
    ) -> torch.Tensor:
        """Compute per-training-point Shapley contribution for a single
        validation point using the O(n) recursion from Theorem 1 of Jia et al.

        Returns
        -------
        torch.Tensor, shape (n,)
        """
        n = dist_j.shape[0]
        K = self.k_neighbors

        # Sort training points by ascending distance to this validation point
        sort_idx = torch.argsort(dist_j, stable=True)          # (n,)
        y_sorted = self.y_train[sort_idx]                       # (n, ...)

        # Binary match vector: 1 if y_sorted[i] == y_val_j, else 0
        match = (y_sorted == y_val_j).all(dim=1).float()       # (n,)

        # Initialise score tensor (indexed by sorted rank for now)
        score_sorted = torch.zeros(n, dtype=torch.float64)

        # Base case: furthest training point (rank n-1)
        score_sorted[n - 1] = match[n - 1] / n

        # Recurse from rank n-2 down to 0
        for i in range(n - 2, -1, -1):
            score_sorted[i] = (
                score_sorted[i + 1]
                + min(K, i + 1) / (K * (i + 1))
                * (match[i] - match[i + 1])
            )

        # Scatter back to original training-point order
        score = torch.zeros(n, dtype=torch.float64)
        score[sort_idx] = score_sorted

        return score   # (n,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_data_values(self, *args, **kwargs):
        """Compute KNN-Shapley data values by iterating one validation point
        at a time.  Peak memory is O(n_train × 1) instead of
        O(n_train × n_valid).
        """
        n = len(self.x_train)

        x_train_emb, x_valid_emb = self.embeddings(self.x_train, self.x_valid)
        x_train_flat = self._to_tensor(x_train_emb).view(n, -1).float()

        m = len(self.x_valid)
        x_valid_flat = self._to_tensor(x_valid_emb).view(m, -1).float()

        self._dbg(
            f"n_train={n}, n_valid={m}, k={self.k_neighbors}, "
            f"feature_dim={x_train_flat.shape[1]}"
        )

        # Accumulate (n,) score vectors, one per validation point
        score_sum = torch.zeros(n, dtype=torch.float64)

        t0 = time.perf_counter()

        for j in tqdm.trange(m, desc="KNN-Shapley (per valid point)"):
            # ── Distance vector for validation point j ───────────────
            if isinstance(self.dist_matrix, torch.Tensor):
                # Use precomputed column
                dist_j = self.dist_matrix[:, j].float()
            else:
                x_val_j = x_valid_flat[j : j + 1]          # (1, d)
                dist_j = self._dist_to_one_valid(x_train_flat, x_val_j)

            # ── Shapley recursion for validation point j ─────────────
            y_val_j = self.y_valid[j : j + 1]              # keep 2-D shape
            score_j = self._shapley_one_valid(dist_j, y_val_j)

            score_sum += score_j

        t1 = time.perf_counter()
        self._dbg(f"Total time for all {m} validation points: {t1 - t0:.2f}s")

        self.data_values = (score_sum / m).numpy()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point."""
        return self.data_values

    def set_distance_matrix(self, dist) -> "KNNShapley":
        """Provide a precomputed distance matrix to avoid recomputation.

        Accepts a ``torch.Tensor`` or ``np.ndarray`` of shape
        ``(n_train, n_valid)``.  Only the column ``[:, j]`` for the current
        validation point is read at a time, so passing a memory-mapped array
        or a column-major layout is particularly efficient.

        Parameters
        ----------
        dist : torch.Tensor | np.ndarray
            Precomputed pairwise distances, entry ``(i, j)`` = distance
            between training sample *i* and validation sample *j*.

        Returns
        -------
        KNNShapley
            Self, for chaining.
        """
        if isinstance(dist, np.ndarray):
            dist_t = torch.as_tensor(dist)
        elif isinstance(dist, torch.Tensor):
            dist_t = dist
        else:
            raise TypeError("dist must be a torch.Tensor or numpy.ndarray")

        n = len(self.x_train) if hasattr(self, "x_train") else None
        m = len(self.x_valid) if hasattr(self, "x_valid") else None
        if n is not None and m is not None:
            if dist_t.dim() != 2 or dist_t.shape != (n, m):
                raise ValueError(
                    f"dist shape {tuple(dist_t.shape)} does not match expected ({n}, {m})."
                )

        self.dist_matrix = dist_t
        size_mb = dist_t.numel() * dist_t.element_size() / 1024 / 1024
        self._dbg(f"set_distance_matrix: shape={tuple(dist_t.shape)}, size≈{size_mb:.1f} MB")
        return self