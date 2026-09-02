"""Vectorised, GPU-capable KNN-Shapley.

``KNNShapleyVec`` computes *exactly* the same quantity as
:class:`~opendataval.dataval.knnshap.knnshap.KNNShapley` but replaces the
per-validation-point Python recursion with a closed-form reverse cumulative
sum, evaluated for many validation points at once and optionally on GPU.

The reference implementation evaluates, for each validation point, the
Theorem-1 recursion of Jia et al.::

    s[n-1] = match[n-1] / n
    s[i]   = s[i+1] + min(K, i+1) / (K * (i+1)) * (match[i] - match[i+1])

Unrolling the recursion gives a suffix sum, since the weight depends only on
the rank ``i``::

    w[i]  = min(K, i+1) / (K * (i+1))
    s[i]  = s[n-1] + sum_{t=i}^{n-2} w[t] * (match[t] - match[t+1])

which is a single ``flip -> cumsum -> flip``.  That removes the
``n_valid x n_train`` interpreter loop (4e8 iterations for CIFAR-10) and makes
the whole computation a handful of batched tensor ops.

This is an exact reformulation, not an approximation.  Results agree with the
reference implementation to floating-point round-off; the only differences come
from summation order in ``cumsum``.
"""

from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
import time

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.progress import progress_range
from opendataval.model.api import Model


class KNNShapleyVec(DataEvaluator, ModelLessMixin):
    """Vectorised KNN-Shapley (exact, GPU-capable).

    Drop-in replacement for :class:`KNNShapley` with identical semantics and
    output, but orders of magnitude faster: the O(n_valid * n_train) Python
    recursion is replaced by batched tensor operations.

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
        by default 1024
    valid_chunk : int, optional
        Number of validation points processed per batched pass.  Controls the
        peak size of the (n_train x valid_chunk) working tensors, by default 512
    device : str | torch.device, optional
        Device for the computation.  Defaults to CUDA when available.
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
        batch_size: int = 1024,
        valid_chunk: int = 512,
        device: Optional[torch.device] = None,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        debug: bool = False,
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.valid_chunk = valid_chunk
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        self.debug = debug
        self.dist_matrix: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        embedding_str = "None"
        if self.embedding_model is not None:
            embedding_str = self.embedding_model.__class__.__name__
        return (
            f"KNNShapleyVec(k_neighbors={self.k_neighbors}, "
            f"batch_size={self.batch_size}, valid_chunk={self.valid_chunk}, "
            f"device={self.device.type}, embedding_model={embedding_str}, "
            f"debug={self.debug})"
        )

    __str__ = __repr__

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[KNNShapleyVec][DEBUG] {msg}")

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

    def _rank_weights(self, n: int) -> torch.Tensor:
        """``w[i] = min(K, i+1) / (K * (i+1))`` for ``i`` in ``[0, n-2]``."""
        K = self.k_neighbors
        idx = torch.arange(1, n, dtype=torch.float64, device=self.device)  # i+1
        return torch.clamp(idx, max=float(K)) / (K * idx)

    def _shapley_chunk(
        self,
        dist_chunk: torch.Tensor,   # (n, b) distances, one column per valid point
        y_val_chunk: torch.Tensor,  # (b, C) labels of those validation points
        weights: torch.Tensor,      # (n-1,) precomputed rank weights
    ) -> torch.Tensor:
        """Shapley contributions for a chunk of validation points.

        Returns
        -------
        torch.Tensor, shape (n, b) in original training-point order.
        """
        n = dist_chunk.shape[0]

        # Rank training points by distance, per validation point (column).
        # stable=True reproduces the reference implementation's tie-breaking.
        sort_idx = torch.argsort(dist_chunk, dim=0, stable=True)          # (n, b)

        # match[i, j] = 1 iff the i-th nearest train point to valid point j
        # carries the same label as valid point j.
        y_sorted = self.y_train_dev[sort_idx]                             # (n, b, C)
        match = (y_sorted == y_val_chunk.unsqueeze(0)).all(dim=2).double()  # (n, b)

        # Closed form of the recursion: a suffix sum of w[t] * (m[t] - m[t+1]).
        #
        # The reference multiplies a *Python scalar* weight by a float32 match
        # difference, so each term is rounded to float32 before being summed
        # into a float64 accumulator. Reproducing that rounding is what makes
        # this bit-faithful: computing the term in full float64 instead drifts
        # by ~1e-9 and can reorder near-tied data values.
        term = weights.unsqueeze(1) * (match[:-1] - match[1:])            # (n-1, b)
        term = term.float().double()                                      # match reference rounding
        suffix = torch.flip(torch.cumsum(torch.flip(term, [0]), dim=0), [0])

        base = match[n - 1] / n                                           # (b,)
        score_sorted = torch.empty_like(match)                            # (n, b)
        score_sorted[: n - 1] = base.unsqueeze(0) + suffix
        score_sorted[n - 1] = base

        # Undo the per-column sort: score[sort_idx[i, j], j] = score_sorted[i, j]
        score = torch.zeros_like(score_sorted)
        score.scatter_(0, sort_idx, score_sorted)
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_data_values(self, *args, **kwargs):
        """Compute KNN-Shapley data values with batched tensor operations."""
        n = len(self.x_train)
        m = len(self.x_valid)

        x_train_emb, x_valid_emb = self.embeddings(self.x_train, self.x_valid)
        x_train_flat = self._to_tensor(x_train_emb).view(n, -1).float().to(self.device)
        x_valid_flat = self._to_tensor(x_valid_emb).view(m, -1).float().to(self.device)

        self.y_train_dev = self._to_tensor(self.y_train).to(self.device)
        y_valid_dev = self._to_tensor(self.y_valid).to(self.device)

        self._dbg(
            f"n_train={n}, n_valid={m}, k={self.k_neighbors}, "
            f"feature_dim={x_train_flat.shape[1]}, device={self.device}, "
            f"valid_chunk={self.valid_chunk}"
        )

        weights = self._rank_weights(n)                                   # (n-1,)
        score_sum = torch.zeros(n, dtype=torch.float64, device=self.device)

        t0 = time.perf_counter()
        n_chunks = (m + self.valid_chunk - 1) // self.valid_chunk

        for c in progress_range(n_chunks, "KNN-Shapley (vectorised, per chunk)"):
            lo = c * self.valid_chunk
            hi = min(lo + self.valid_chunk, m)

            if isinstance(self.dist_matrix, torch.Tensor):
                dist_chunk = self.dist_matrix[:, lo:hi].float().to(self.device)
            else:
                dist_chunk = torch.cdist(x_train_flat, x_valid_flat[lo:hi])  # (n, b)

            score = self._shapley_chunk(dist_chunk, y_valid_dev[lo:hi], weights)
            score_sum += score.sum(dim=1)

            del dist_chunk, score

        t1 = time.perf_counter()
        self._dbg(f"Total time for all {m} validation points: {t1 - t0:.2f}s")

        self.data_values = (score_sum / m).cpu().numpy()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point."""
        return self.data_values

    def set_distance_matrix(self, dist) -> "KNNShapleyVec":
        """Provide a precomputed (n_train, n_valid) distance matrix."""
        if isinstance(dist, np.ndarray):
            dist_t = torch.as_tensor(dist)
        elif isinstance(dist, torch.Tensor):
            dist_t = dist
        else:
            raise TypeError("dist must be a torch.Tensor or numpy.ndarray")
        self.dist_matrix = dist_t
        return self
