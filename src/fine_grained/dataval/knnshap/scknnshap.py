"""
KNN-Shapley with LSH acceleration.

Implements the algorithm from:
  "Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms"
  (Ruoxi Jia et al., VLDB 2019)

"""
import time
from math import ceil
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from opendataval.dataval.api import DataEvaluator, ModelLessMixin


# ---------------------------------------------------------------------------
# Paper helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _compute_distance(i_q, query, x_trn, n_trn, K):
    dist_to_random = np.zeros(n_trn)
    for i_trn in range(n_trn):
        dist_to_random[i_trn] = np.linalg.norm(query - x_trn[i_trn, :], 2)
    dist_to_random_avg = np.mean(dist_to_random)
    dist_to_KNN = np.sort(dist_to_random)[:K]
    return dist_to_random_avg, dist_to_KNN


def _estimate_contrast(x_trn, query, K, num_cores=1):
    n_trn = x_trn.shape[0]
    n_q = query.shape[0]
    result = Parallel(n_jobs=num_cores)(
        delayed(_compute_distance)(i_q, query[i_q, :], x_trn, n_trn, K)
        for i_q in range(n_q)
    )
    dist_to_random_avg = np.array([result[i][0] for i in range(n_q)])
    dist_to_KKN = np.array([result[i][1] for i in range(n_q)])
    assert dist_to_KKN.shape[0] == n_q
    dist_to_KNN_avg_q = np.mean(dist_to_KKN, axis=0)
    dist_to_random_avg_avg = np.mean(dist_to_random_avg)
    contrast = dist_to_random_avg_avg / dist_to_KNN_avg_q
    return dist_to_random_avg_avg, dist_to_KNN_avg_q, contrast


def get_contrast(x_trn, mc_num=5, eps=0.1, num_cores=1):
    """Estimate empirical contrast of the dataset (from original paper)."""
    n_trn = x_trn.shape[0]
    K = int(1 / eps)

    contrast = []
    dist_rand = []
    dist_knn = []
    for mc_i in range(mc_num):
        start = time.time()
        sample_ind_trn = np.random.choice(
            np.arange(n_trn), int(n_trn / 5 * 4), replace=False
        ).astype(int)
        sample_ind_query = np.array(
            list(set(np.arange(n_trn).astype(int).tolist()) - set(sample_ind_trn.tolist()))
        ).astype(int)
        dist_rand_, dist_knn_, contrast_ = _estimate_contrast(
            x_trn[sample_ind_trn, :], x_trn[sample_ind_query, :], K, num_cores
        )
        dist_rand.append(dist_rand_)
        dist_knn.append(dist_knn_)
        contrast.append(contrast_)
        elapsed_time = time.time() - start
        print("monte carlo iteration%s, elapsed time: %s" % (mc_i, elapsed_time))

    dist_knn = np.array(dist_knn)
    contrast = np.array(contrast)
    dist_rand = np.array(dist_rand)
    return dist_rand, dist_knn, contrast


def f_h(x, r):
    y = (
        1
        - 2 * norm.cdf(-r / x)
        - 2 / (np.sqrt(2 * np.pi) * r / x) * (1 - np.exp(-(r ** 2 / (2 * (x ** 2)))))
    )
    return y


def g_normalize(contrast, r):
    y = np.log(f_h(1 / contrast, r)) / np.log(f_h(1, r))
    return y


def g_unnormalize(dist_rand, dist_knn, r):
    y = np.log(f_h(dist_knn, r)) / np.log(f_h(dist_rand, r))
    return y


def find_best_r_normalize(search_range, contrast):
    y = g_normalize(contrast, search_range)
    min_ind = np.argmin(y)
    return search_range[min_ind]


def find_best_r_unnormalize(search_range, dist_rand, dist_knn):
    y = g_unnormalize(dist_rand, dist_knn, search_range)
    min_ind = np.argmin(y)
    return search_range[min_ind]


def lsh_function(t, x, w, b):
    """2-stable LSH projection (from original paper)."""
    h = np.floor((np.dot(w, x) + b) / t).astype(int)
    return h


# ---------------------------------------------------------------------------
# LSH class (from original paper, unchanged)
# ---------------------------------------------------------------------------

class LSH:
    def __init__(self, n_hash_bit, n_hash_table, x_trn, y_trn, dist_rand, equal, t=0.1):
        self.n_hash_bit = n_hash_bit
        self.n_hash_table = n_hash_table
        self.t = t
        self.dist_rand = dist_rand
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.N, self.dim = x_trn.shape
        self.equal = equal
        # draw w from a normal distribution (2-stable)
        self.w = np.random.normal(0, 1, (n_hash_table, n_hash_bit, self.dim))
        # draw b from U[0, t]
        self.b = np.random.uniform(0, self.t, (n_hash_table, n_hash_bit))
        self.x_trn_hash = [dict() for _ in range(n_hash_table)]
        for i in tqdm(range(self.N), desc="Building LSH index"):
            hash_code_all = lsh_function(self.t, x_trn[i] / dist_rand, self.w, self.b)
            for l in range(n_hash_table):
                hash_code_trn = ".".join(map(str, hash_code_all[l, :]))
                if hash_code_trn in self.x_trn_hash[l]:
                    self.x_trn_hash[l][hash_code_trn].append(i)
                else:
                    self.x_trn_hash[l][hash_code_trn] = [i]

    def get_approx_KNN(self, x_tst, K):
        N_tst = x_tst.shape[0]
        x_tst_knn = np.ones((N_tst, K)) * (-1)
        nns_len = np.zeros(N_tst)
        for i_tst in tqdm(range(N_tst), desc="Querying LSH"):
            nns = []
            for l in range(self.n_hash_table):
                hash_code_int = lsh_function(
                    self.t, x_tst[i_tst] / self.dist_rand, self.w[l, :, :], self.b[l, :]
                )
                hash_code_test = ".".join(map(str, hash_code_int))
                if hash_code_test in self.x_trn_hash[l]:
                    nns += self.x_trn_hash[l][hash_code_test]
            nns = np.unique(nns)
            num_collide_elements = len(nns)
            if len(nns) > 0:
                dist = [
                    np.linalg.norm(
                        self.x_trn[i] / self.dist_rand - x_tst[i_tst] / self.dist_rand, 2
                    )
                    for i in nns
                ]
                dist_min_ind = nns[np.argsort(dist)]
                if num_collide_elements < K:
                    x_tst_knn[i_tst, :num_collide_elements] = dist_min_ind[:num_collide_elements]
                else:
                    x_tst_knn[i_tst, :] = dist_min_ind[:K]
            nns_len[i_tst] = len(nns)
        return x_tst_knn.astype(int), nns_len

    def compute_approx_shapley(self, x_tst_knn, y_tst, K):
        N_tst, K_star = x_tst_knn.shape
        # Accumulate into a 1-D sum instead of a (N_val × N_train) matrix.
        # For 10k val × 7M train that matrix would be ~560 GB; the sum is ~56 MB.
        sp_sum = np.zeros(self.N)
        for j in tqdm(range(N_tst), desc="Computing Shapley values"):
            non_nan_index = np.where(x_tst_knn[j, :] >= 0)[0]
            if len(non_nan_index) == 0:
                continue
            K_tot = non_nan_index[-1]
            sp_j = np.zeros(self.N)
            if K_tot == self.N:
                sp_j[x_tst_knn[j, self.N - 1]] = (
                    self.equal(self.y_trn[x_tst_knn[j, self.N - 1]], y_tst[j]) / self.N
                )
            for i in np.arange(K_tot - 1, -1, -1):
                sp_j[x_tst_knn[j, i]] = sp_j[x_tst_knn[j, i + 1]] + (
                    self.equal(self.y_trn[x_tst_knn[j, i]], y_tst[j])
                    - self.equal(self.y_trn[x_tst_knn[j, i + 1]], y_tst[j])
                ) / K * min([K, i + 1]) / (i + 1)
            sp_sum += sp_j
        return sp_sum / N_tst


# ---------------------------------------------------------------------------
# opendataval wrapper
# ---------------------------------------------------------------------------

class KNNShapleyLSH(DataEvaluator, ModelLessMixin):
    """KNN-Shapley with LSH acceleration (Jia et al., VLDB 2019).

    Parameters
    ----------
    k_neighbors : int
        K for the KNN classifier used in Shapley computation.
    eps : float
        Approximation parameter; K_star = ceil(1/eps) neighbors are retrieved.
    alpha : float
        Controls n_hash_bit = ceil(log(n_train) * alpha / log(1/f_h(1,t))).
    n_hash_table : int
        Number of LSH hash tables.
    mc_num : int
        Monte Carlo iterations for contrast estimation.
    num_cores : int
        Parallel workers for contrast estimation.
    dist_rand : float or None
        Pre-computed average random distance. If None, estimated from data.
    t : float or None
        LSH projection width. If None, selected automatically via contrast.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        k_neighbors: int = 1,
        eps: float = 0.1,
        alpha: float = 0.5,
        n_hash_table: int = 10,
        mc_num: int = 5,
        num_cores: int = 1,
        dist_rand: Optional[float] = None,
        t: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.k_neighbors = k_neighbors
        self.eps = eps
        self.alpha = alpha
        self.n_hash_table = n_hash_table
        self.mc_num = mc_num
        self.num_cores = num_cores
        self._dist_rand_fixed = dist_rand
        self._t_fixed = t
        self.random_state = random_state

        self.data_values = None
        self._lsh = None

    # ------------------------------------------------------------------
    # opendataval interface
    # ------------------------------------------------------------------

    def input_data(self, x_train, y_train, x_valid, y_valid):
        self.x_train = self._to_numpy(x_train)
        self.y_train = self._to_numpy_labels(y_train)
        self.x_valid = self._to_numpy(x_valid)
        self.y_valid = self._to_numpy_labels(y_valid)
        self._lsh = None
        self.data_values = None
        return self

    def train_data_values(self, *args, **kwargs):
        """Build LSH index and compute Shapley values (paper algorithm)."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        x_trn = self.x_train
        y_trn = self.y_train
        x_val = self.x_valid
        y_val = self.y_valid

        K = self.k_neighbors
        K_star = max(K, ceil(1 / self.eps))

        # ---- 1. Determine dist_rand and t --------------------------------
        contrast = None

        if self._dist_rand_fixed is not None:
            dist_rand = self._dist_rand_fixed
            print(f"[KNNShapleyLSH] Using provided dist_rand={dist_rand:.4f}")
        else:
            print("[KNNShapleyLSH] Estimating contrast from validation data...")
            dist_rand_mc, dist_knn_mc, contrast_mc = get_contrast(
                x_val, mc_num=self.mc_num, eps=self.eps, num_cores=self.num_cores
            )
            dist_rand = float(np.mean(dist_rand_mc, axis=0))
            contrast = float(np.mean(contrast_mc, axis=0)[K_star - 1])
            print(f"[KNNShapleyLSH] dist_rand={dist_rand:.4f}, contrast={contrast:.4f}")

        if self._t_fixed is not None:
            t = self._t_fixed
            print(f"[KNNShapleyLSH] Using provided t={t:.4f}")
        elif contrast is not None:
            search_range = np.arange(1e-3, 10, 1e-3)
            t = find_best_r_normalize(search_range, contrast)
            print(f"[KNNShapleyLSH] Selected t={t:.4f} via contrast minimization")
        else:
            t = 1.0
            print(f"[KNNShapleyLSH] dist_rand provided but t not set, using default t={t:.4f}. "
                  f"Pass t= explicitly for best results.")

        # ---- 2. Compute n_hash_bit from paper formula --------------------
        n_trn = len(x_trn)
        n_hash_bit = int(np.ceil(
            np.log(n_trn) * self.alpha / np.log(1.0 / f_h(1.0, t))
        ))
        print(f"[KNNShapleyLSH] n_hash_bit={n_hash_bit}, n_hash_table={self.n_hash_table}, K_star={K_star}")

        # ---- 3. Build LSH ------------------------------------------------
        equal_fn = self._make_equal_fn()
        start = time.time()
        self._lsh = LSH(
            n_hash_bit=n_hash_bit,
            n_hash_table=self.n_hash_table,
            x_trn=x_trn,
            y_trn=y_trn,
            dist_rand=dist_rand,
            equal=equal_fn,
            t=t,
        )
        print(f"[KNNShapleyLSH] LSH built in {time.time() - start:.2f}s")

        # ---- 4. Approximate KNN on validation set ------------------------
        start = time.time()
        x_val_knn, _ = self._lsh.get_approx_KNN(x_val, K_star)
        print(f"[KNNShapleyLSH] KNN query done in {time.time() - start:.2f}s")

        # ---- 5. Compute Shapley values ------------------------------------
        start = time.time()
        sp_approx = self._lsh.compute_approx_shapley(x_val_knn, y_val, K)
        print(f"[KNNShapleyLSH] Shapley values computed in {time.time() - start:.2f}s")

        # sp_approx is already averaged over validation points inside compute_approx_shapley
        self.data_values = sp_approx
        return self

    def evaluate_data_values(self) -> np.ndarray:
        if self.data_values is None:
            raise ValueError("Call train_data_values() first.")
        return self.data_values

    def compute_lsh_recall(self, K: Optional[int] = None, n_sample: int = 200, random_state: Optional[int] = None):
        """Empirical Recall\@K on a random sample of validation points.

        Compares the approximate neighbors returned by the LSH index against
        exact brute-force KNN.  Must be called after ``train_data_values()``.

        Parameters
        ----------
        K : int or None
            Number of neighbors to evaluate.  Defaults to ``self.k_neighbors``.
        n_sample : int
            Number of validation points to sample.  Capped at N_val.
        random_state : int or None
            Seed for the sampling RNG.

        Returns
        -------
        mean_recall : float
            Average Recall\@K over the sampled points.
        per_point_recall : np.ndarray, shape (n_sample,)
            Per-point recall values for the sampled subset.
        sampled_indices : np.ndarray, shape (n_sample,)
            Indices into x_valid that were sampled.
        """
        from sklearn.neighbors import NearestNeighbors

        K = K or self.k_neighbors
        K_star = max(K, int(np.ceil(1.0 / self.eps)))
        x_trn = self.x_train
        x_val = self.x_valid
        n_trn = len(x_trn)
        N_val = x_val.shape[0]

        # ---- Sample validation points -----------------------------------
        rng = np.random.RandomState(random_state)
        n_sample = min(n_sample, N_val)
        sampled_indices = rng.choice(N_val, size=n_sample, replace=False)
        x_sample = x_val[sampled_indices]

        # ---- Resolve dist_rand and t ------------------------------------
        t = self._t_fixed if self._t_fixed is not None else 1.0

        if self._dist_rand_fixed is not None:
            dist_rand = self._dist_rand_fixed
        else:
            # cheap estimate: mean L2 from sample points to a random probe set
            probe_idx = rng.choice(n_trn, size=min(500, n_trn), replace=False)
            probe = x_trn[probe_idx]
            dist_rand = float(np.mean([
                np.linalg.norm(x_sample[i] - probe, axis=1).mean()
                for i in range(min(50, n_sample))
            ]))
            print(f"[KNNShapleyLSH] recall: estimated dist_rand={dist_rand:.4f}")

        # ---- Build temporary LSH index (no Shapley) ---------------------
        n_hash_bit = int(np.ceil(
            np.log(n_trn) * self.alpha / np.log(1.0 / f_h(1.0, t))
        ))
        print(f"[KNNShapleyLSH] Building LSH for recall check "
              f"(n_hash_bit={n_hash_bit}, n_hash_table={self.n_hash_table}, K_star={K_star})...")

        tmp_lsh = LSH(
            n_hash_bit   = n_hash_bit,
            n_hash_table = self.n_hash_table,
            x_trn        = x_trn,
            y_trn        = self.y_train,
            dist_rand    = dist_rand,
            equal        = self._make_equal_fn(),
            t            = t,
        )

        # ---- Approximate KNN on sampled points --------------------------
        print(f"[KNNShapleyLSH] Computing recall on {n_sample}/{N_val} validation points...")
        approx_knn, _ = tmp_lsh.get_approx_KNN(x_sample, K_star)  # (n_sample, K_star)

        # ---- Exact KNN via brute force ----------------------------------
        nbrs = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="l2")
        nbrs.fit(x_trn)
        _, exact_knn = nbrs.kneighbors(x_sample)   # (n_sample, K)

        # ---- Recall computation -----------------------------------------
        per_point_recall = np.empty(n_sample)
        for j in range(n_sample):
            true_set   = set(exact_knn[j].tolist())
            approx_set = set(approx_knn[j][approx_knn[j] >= 0].tolist())
            per_point_recall[j] = len(true_set & approx_set) / K

        mean_recall = float(per_point_recall.mean())
        print(f"[KNNShapleyLSH] Mean Recall@{K}: {mean_recall:.3f}")
        return mean_recall, per_point_recall, sampled_indices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_equal_fn(self):
        """Return label equality function matching paper's equal(a, b)."""
        def equal(a, b):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                return int(np.array_equal(a, b))
            return int(a == b)
        return equal

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        import torch
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _to_numpy_labels(data) -> np.ndarray:
        import torch
        if isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
        else:
            arr = np.asarray(data)
        # Convert one-hot to class indices for scalar comparison
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.argmax(axis=1)
        return arr
