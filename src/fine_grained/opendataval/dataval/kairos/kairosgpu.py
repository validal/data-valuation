"""GPU-resident Kairos.

``KairosGPU`` computes exactly the same data values as
:class:`~opendataval.dataval.kairos.kairos.Kairos`, but keeps the tensors on
the accelerator and removes two structural bottlenecks in the reference
implementation:

1. ``_compute_median_heuristic`` draws ``num_samples`` (default 10000) random
   pairs in a Python loop and calls ``torch.norm(...).item()`` on each one.
   Every ``.item()`` is a device synchronisation, so the loop costs 10000
   round-trips.  The same index sequence is drawn here (so the bandwidth is
   identical), but the distances are evaluated as one batched norm.

2. ``train_data_values`` materialises the full kernel block
   ``K`` of shape ``(n_train + n_valid, n_train)``.  For CIFAR-10 that is
   50000 x 40000 float32 = 8 GB, and ``D`` alongside it another 8 GB.  Only two
   reductions of ``K`` are ever used - ``avg_K_train`` (row means) and
   ``avg_K_valid`` (column means) - so this version accumulates both over row
   chunks and never holds more than ``(chunk, n_train)`` at once.

Everything else - the SKLR residual model, embeddings, the unbiased
correction, ``evaluate_data_values`` - is inherited unchanged from ``Kairos``.
"""

from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState

from opendataval.dataval.kairos.kairos import Kairos
from opendataval.model.api import Model


class KairosGPU(Kairos):
    """Kairos with GPU-resident, chunked kernel computation.

    Parameters
    ----------
    device : str | torch.device, optional
        Device for the computation. Defaults to CUDA when available.
    row_chunk : int, optional
        Rows of the stacked matrix processed per pass. Bounds the working set
        at ``(row_chunk, n_train)``. By default 8192.

    All other parameters are identical to :class:`Kairos`.
    """

    def __init__(
        self,
        lambda_weight=0.97,
        sigma_feature=None,
        kernel_type="sigma",
        unbiased=False,
        use_median_heuristic=True,
        num_samples=10000,
        random_state: Optional[RandomState] = None,
        embedding_model: Optional[Model] = None,
        debug: bool = False,
        device: Optional[torch.device] = None,
        row_chunk: int = 8192,
    ):
        super().__init__(
            lambda_weight=lambda_weight,
            sigma_feature=sigma_feature,
            kernel_type=kernel_type,
            unbiased=unbiased,
            use_median_heuristic=use_median_heuristic,
            num_samples=num_samples,
            random_state=random_state,
            embedding_model=embedding_model,
            debug=debug,
        )
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.row_chunk = row_chunk

    def __repr__(self) -> str:
        embedding_str = "None"
        if self.embedding_model is not None:
            embedding_str = self.embedding_model.__class__.__name__
        return (
            f"KairosGPU(lambda_weight={self.lambda_weight}, unbiased={self.unbiased}, "
            f"use_median_heuristic={self.use_median_heuristic}, "
            f"num_samples={self.num_samples}, device={self.device.type}, "
            f"row_chunk={self.row_chunk}, embedding_model={embedding_str}, "
            f"debug={self.debug})"
        )

    __str__ = __repr__

    def input_data(self, x_train, y_train, x_valid, y_valid, debug=True):
        """Delegate to Kairos, then move the tensors onto the device."""
        super().input_data(x_train, y_train, x_valid, y_valid, debug=debug)
        for name in ("X_train", "X_valid", "y_train", "y_valid", "r_train"):
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(self.device))
        return self

    def _compute_median_heuristic(self) -> float:
        """Same sampled pairs as Kairos, but one batched norm instead of a loop."""
        X = self.X_train
        n = X.shape[0]
        num_pairs = min(self.num_samples, n * (n - 1) // 2)

        # Draw the identical index sequence to the reference implementation:
        # two randint calls per iteration, pairs with i == j discarded.
        rng = np.random.RandomState(42)
        ii, jj = [], []
        for _ in range(num_pairs):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                ii.append(i)
                jj.append(j)

        if not ii:
            return 1.0

        idx_i = torch.as_tensor(ii, dtype=torch.long, device=X.device)
        idx_j = torch.as_tensor(jj, dtype=torch.long, device=X.device)
        dists = torch.linalg.vector_norm(X[idx_i] - X[idx_j], dim=1)
        return float(np.median(dists.detach().cpu().numpy()))

    def train_data_values(self, *args, debug=None, **kwargs):
        """Chunked, GPU-resident version of the Kairos kernel pass."""
        if debug is None:
            debug = self.debug

        X = self.X_train                     # (n_train, d)
        V = self.X_valid                     # (n_valid, d)
        n_train = X.shape[0]
        n_valid = V.shape[0]

        if debug:
            print("\n" + "=" * 70)
            print("[KairosGPU] TRAINING PHASE")
            print("=" * 70)
            print(f"  X_train shape: {tuple(X.shape)}  device={X.device}")
            print(f"  X_valid shape: {tuple(V.shape)}  row_chunk={self.row_chunk}")

        # Bandwidth - identical logic to Kairos
        if self.sigma_feature is None and self.use_median_heuristic:
            sigma = self._compute_median_heuristic()
            self.sigma_feature = sigma
            if debug:
                print(f"  ✓ Computed sigma_feature = {sigma:.6f}")
        else:
            sigma = self.sigma_feature if self.sigma_feature is not None else 3.0
            if debug:
                print(f"  Using provided sigma_feature = {sigma:.6f}")

        inv_two_sigma2 = 1.0 / (2.0 * sigma ** 2)

        Z = torch.cat([X, V], dim=0)                 # (n_train + n_valid, d)
        Z_norm_sq = (Z * Z).sum(dim=1)               # (n_train + n_valid,)
        X_norm_sq = Z_norm_sq[:n_train]              # (n_train,)

        # Only two reductions of K are needed, so accumulate them chunk by chunk
        # instead of building the full (n_train + n_valid, n_train) matrix.
        avg_K_train = torch.empty(n_train, dtype=X.dtype, device=X.device)
        valid_col_sum = torch.zeros(n_train, dtype=torch.float64, device=X.device)

        total_rows = n_train + n_valid
        for lo in range(0, total_rows, self.row_chunk):
            hi = min(lo + self.row_chunk, total_rows)

            G = Z[lo:hi] @ X.T                                        # (c, n_train)
            D = Z_norm_sq[lo:hi].unsqueeze(1) + X_norm_sq.unsqueeze(0) - 2.0 * G
            K = torch.exp(-D * inv_two_sigma2)
            del G, D

            # rows of this chunk that belong to the train block -> row means
            tr_hi = min(hi, n_train)
            if lo < n_train:
                avg_K_train[lo:tr_hi] = K[: tr_hi - lo].mean(dim=1)

            # rows that belong to the valid block -> accumulate column sums
            if hi > n_train:
                start = max(lo, n_train) - lo
                valid_col_sum += K[start:].sum(dim=0).double()

            del K

        avg_K_valid = (valid_col_sum / n_valid).to(X.dtype)

        self.avg_K_train = avg_K_train
        self.avg_K_valid = avg_K_valid

        # feature discrepancy - identical to Kairos
        if self.unbiased:
            avg_K_train_unbiased = (self.avg_K_train * n_train - 1) / (n_train - 1)
            feature_metric = self.avg_K_valid - avg_K_train_unbiased
        else:
            feature_metric = self.avg_K_valid - self.avg_K_train

        squared_residual = torch.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        if debug:
            print(f"\n  [Feature Metric] mean={feature_metric.mean():.6f} "
                  f"std={feature_metric.std():.6f} "
                  f"min={feature_metric.min():.6f} max={feature_metric.max():.6f}")
            print(f"  [Squared Residual] mean={squared_residual.mean():.6f} "
                  f"std={squared_residual.std():.6f}")
            print("=" * 70)

        return self
