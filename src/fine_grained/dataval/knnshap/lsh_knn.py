"""
Standalone LSH-based approximate KNN module.

Provides LSHKNNIndex: a self-contained class that builds an LSH index,
returns approximate KNN results, quantifies approximation error (Recall@K),
and exposes tunable parameters so you can control the speed/accuracy trade-off
before plugging the chosen parameters into KNNShapleyLSH.

Typical workflow
----------------
>>> idx = LSHKNNIndex(K=100, n_hash_table=50, t=2.0).fit(X_train)
>>> mean_recall, *_ = idx.recall(X_valid, n_sample=200)
>>> # satisfied? pass the same params to KNNShapleyLSH
>>> evaluator = idx.to_knnshapley(y_train=y_train)
"""

from __future__ import annotations

from math import ceil
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LSH primitive (2-stable Gaussian family)
# ---------------------------------------------------------------------------

def _f_h(x: float | np.ndarray, r: float = 1.0) -> float | np.ndarray:
    """Collision probability for the 2-stable LSH family at normalised distance x."""
    return (
        1
        - 2 * norm.cdf(-r / x)
        - 2 / (np.sqrt(2 * np.pi) * r / x) * (1 - np.exp(-(r ** 2 / (2 * x ** 2))))
    )


def _lsh_hash(x: np.ndarray, w: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Project x through (w, b) and bucket by width t."""
    return np.floor((w @ x + b) / t).astype(int)


# ---------------------------------------------------------------------------
# Internal index store
# ---------------------------------------------------------------------------

class _LSHTable:
    """One hash table: n_hash_bit projections, bucket → list[train_idx]."""

    def __init__(self, n_hash_bit: int, dim: int, t: float, dist_rand: float, rng: np.random.RandomState):
        self.t = t
        self.dist_rand = dist_rand
        self.w = rng.normal(0, 1, (n_hash_bit, dim))
        self.b = rng.uniform(0, t, n_hash_bit)
        self.buckets: dict[str, list[int]] = {}

    def _key(self, x: np.ndarray) -> str:
        code = _lsh_hash(x / self.dist_rand, self.w, self.b, self.t)
        return ".".join(map(str, code))

    def add(self, idx: int, x: np.ndarray) -> None:
        k = self._key(x)
        self.buckets.setdefault(k, []).append(idx)

    def candidates(self, x: np.ndarray) -> list[int]:
        k = self._key(x)
        return self.buckets.get(k, [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LSHKNNIndex:
    """LSH-based approximate KNN index with built-in recall measurement.

    Parameters
    ----------
    K : int
        Default number of nearest neighbours to retrieve.
    n_hash_table : int
        Number of independent hash tables (more → better recall, more memory).
    n_hash_bit : int or None
        Bits (projections) per table.  If None, derived automatically:
        ``ceil(log(N) * alpha / log(1 / f_h(1, t)))``.
    t : float
        LSH bucket width.  Smaller → finer buckets → higher precision,
        lower recall.  Larger → coarser buckets → higher recall, more
        false positives.
    alpha : float
        Scaling factor used when n_hash_bit is auto-computed.
    dist_rand : float or None
        Average random L2 distance used for normalisation.
        If None, estimated from training data at fit time.
    random_state : int or None
        RNG seed for reproducibility.

    Examples
    --------
    >>> idx = LSHKNNIndex(K=100, n_hash_table=100, t=2.0, random_state=42)
    >>> idx.fit(X_train)
    >>> approx_knn, n_candidates = idx.query(X_valid)
    >>> mean_recall, per_point, sampled = idx.recall(X_valid, n_sample=200)
    >>> evaluator = idx.to_knnshapley(y_train=y_train)
    """

    def __init__(
        self,
        K: int = 10,
        n_hash_table: int = 10,
        n_hash_bit: Optional[int] = None,
        t: float = 1.0,
        alpha: float = 0.5,
        dist_rand: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.K = K
        self.n_hash_table = n_hash_table
        self._n_hash_bit_fixed = n_hash_bit
        self.t = t
        self.alpha = alpha
        self._dist_rand_fixed = dist_rand
        self.random_state = random_state

        # set after fit()
        self.x_train: Optional[np.ndarray] = None
        self.dist_rand_: Optional[float] = None
        self.n_hash_bit_: Optional[int] = None
        self._tables: list[_LSHTable] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray) -> "LSHKNNIndex":
        """Build the LSH index on training data.

        Parameters
        ----------
        x_train : np.ndarray, shape (N, d)

        Returns
        -------
        self
        """
        rng = np.random.RandomState(self.random_state)
        self.x_train = np.asarray(x_train, dtype=np.float64)
        N, d = self.x_train.shape

        # ---- dist_rand --------------------------------------------------
        if self._dist_rand_fixed is not None:
            self.dist_rand_ = self._dist_rand_fixed
        else:
            probe_idx = rng.choice(N, size=min(1000, N), replace=False)
            probe = self.x_train[probe_idx]
            query_idx = rng.choice(N, size=min(200, N), replace=False)
            self.dist_rand_ = float(np.mean([
                np.linalg.norm(self.x_train[i] - probe, axis=1).mean()
                for i in query_idx
            ]))
            print(f"[LSHKNNIndex] Estimated dist_rand={self.dist_rand_:.4f}")

        # ---- n_hash_bit -------------------------------------------------
        if self._n_hash_bit_fixed is not None:
            self.n_hash_bit_ = self._n_hash_bit_fixed
        else:
            self.n_hash_bit_ = int(np.ceil(
                np.log(N) * self.alpha / np.log(1.0 / _f_h(1.0, self.t))
            ))

        print(f"[LSHKNNIndex] Building index: N={N}, d={d}, "
              f"n_hash_table={self.n_hash_table}, n_hash_bit={self.n_hash_bit_}, "
              f"t={self.t}, dist_rand={self.dist_rand_:.4f}")

        # ---- Build tables -----------------------------------------------
        self._tables = [
            _LSHTable(self.n_hash_bit_, d, self.t, self.dist_rand_, rng)
            for _ in range(self.n_hash_table)
        ]
        for i in tqdm(range(N), desc="Building LSH index"):
            for table in self._tables:
                table.add(i, self.x_train[i])

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        x_query: np.ndarray,
        K: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve approximate K nearest neighbours for each query point.

        Parameters
        ----------
        x_query : np.ndarray, shape (M, d)
        K : int or None
            Number of neighbours.  Defaults to ``self.K``.

        Returns
        -------
        approx_knn : np.ndarray, shape (M, K)
            Training indices of approximate neighbours, sorted by distance.
            Slots with no candidate are filled with -1.
        n_candidates : np.ndarray, shape (M,)
            Number of candidate points that collided in any table.
        """
        self._check_fitted()
        K = K or self.K
        x_query = np.asarray(x_query, dtype=np.float64)
        M = x_query.shape[0]
        approx_knn  = np.full((M, K), -1, dtype=int)
        n_candidates = np.zeros(M, dtype=int)

        for j in tqdm(range(M), desc="Querying LSH"):
            cands = []
            for table in self._tables:
                cands.extend(table.candidates(x_query[j]))
            cands = np.unique(cands)
            n_candidates[j] = len(cands)

            if len(cands) == 0:
                continue

            dists = np.linalg.norm(
                self.x_train[cands] / self.dist_rand_ - x_query[j] / self.dist_rand_,
                axis=1,
            )
            order = cands[np.argsort(dists)]
            take = min(K, len(order))
            approx_knn[j, :take] = order[:take]

        return approx_knn, n_candidates

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        x_query: np.ndarray,
        K: Optional[int] = None,
        n_sample: int = 200,
        random_state: Optional[int] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Measure Recall@K against exact brute-force KNN on a random sample.

        Parameters
        ----------
        x_query : np.ndarray, shape (M, d)
        K : int or None
        n_sample : int
            Number of query points to sample.
        random_state : int or None

        Returns
        -------
        mean_recall : float
        per_point_recall : np.ndarray, shape (n_sample,)
        sampled_indices : np.ndarray, shape (n_sample,)
            Indices into x_query that were evaluated.
        """
        self._check_fitted()
        K = K or self.K
        x_query = np.asarray(x_query, dtype=np.float64)
        M = x_query.shape[0]

        rng = np.random.RandomState(random_state)
        n_sample = min(n_sample, M)
        sampled_indices = rng.choice(M, size=n_sample, replace=False)
        x_sample = x_query[sampled_indices]

        # Approximate
        approx_knn, _ = self.query(x_sample, K=K)

        # Exact
        print(f"[LSHKNNIndex] Computing exact KNN (brute force, {n_sample} points)...")
        nbrs = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="l2")
        nbrs.fit(self.x_train)
        _, exact_knn = nbrs.kneighbors(x_sample)

        per_point_recall = np.empty(n_sample)
        for j in range(n_sample):
            true_set   = set(exact_knn[j].tolist())
            approx_set = set(approx_knn[j][approx_knn[j] >= 0].tolist())
            per_point_recall[j] = len(true_set & approx_set) / K

        mean_recall = float(per_point_recall.mean())
        print(f"[LSHKNNIndex] Mean Recall@{K}: {mean_recall:.3f}  "
              f"(min={per_point_recall.min():.3f}, max={per_point_recall.max():.3f})")
        return mean_recall, per_point_recall, sampled_indices

    # ------------------------------------------------------------------
    # Parameter report
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of all resolved parameters (after fit)."""
        self._check_fitted()
        N = len(self.x_train)
        p1 = _f_h(1.0, self.t) ** self.n_hash_bit_
        p_recall = 1 - (1 - p1) ** self.n_hash_table
        return {
            "N_train"      : N,
            "n_hash_table" : self.n_hash_table,
            "n_hash_bit"   : self.n_hash_bit_,
            "t"            : self.t,
            "dist_rand"    : self.dist_rand_,
            "K"            : self.K,
            "p_collision_near" : float(p_recall),   # theoretical P[true NN collides]
        }

    # ------------------------------------------------------------------
    # Bridge to KNNShapleyLSH
    # ------------------------------------------------------------------

    def to_knnshapley(self, y_train: np.ndarray, eps: float = 0.01, num_cores: int = 1):
        """Create a KNNShapleyLSH pre-configured with this index's parameters.

        Passes ``dist_rand``, ``t``, ``n_hash_table``, ``alpha``, and
        ``k_neighbors`` so KNNShapleyLSH will reproduce the same LSH setup.

        Parameters
        ----------
        y_train : np.ndarray
            Training labels (needed for KNNShapleyLSH.input_data).
        eps : float
            KNNShapleyLSH eps (controls K_star = ceil(1/eps)).
        num_cores : int

        Returns
        -------
        KNNShapleyLSH (not yet fitted — call input_data + train_data_values)
        """
        from opendataval.dataval.knnshap.scknnshap import KNNShapleyLSH
        self._check_fitted()

        return KNNShapleyLSH(
            k_neighbors  = self.K,
            n_hash_table = self.n_hash_table,
            alpha        = self.alpha,
            eps          = eps,
            dist_rand    = self.dist_rand_,
            t            = self.t,
            num_cores    = num_cores,
            random_state = self.random_state,
        )

    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(x_train) before using the index.")
