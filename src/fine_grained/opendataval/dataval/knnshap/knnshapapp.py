from typing import Optional, Tuple

import numpy as np
import torch
from opendataval.dataval.progress import ProgressBar, progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class KNNShapleyApprox(DataEvaluator, ModelLessMixin):
    """Approximate KNN-Shapley with multiple approximation strategies.

    References
    ----------
    .. [1] R. Jia et al., Efficient Task-Specific Data Valuation for Nearest Neighbor
           Algorithms, 2019. https://arxiv.org/abs/1908.08619

    Parameters
    ----------
    k_neighbors : int
        Number of neighbors to group the data points.
    batch_size : int
        Batch size for processing.
    embedding_model : Model | None
        Pre-trained embedding model.
    random_state : RandomState | None
        Random initial state.
    approx_strategy : str
        One of {'exact','validation_subsample','train_subsample','kdtree','faiss',
        'random_projection','monte_carlo'}
    approx_param : float
        Parameter for approximation (e.g., subsample ratio, projection ratio).
    distance_metric : str
        One of {'euclidean','manhattan','cosine'}
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        approx_strategy: str = "exact",
        approx_param: float = 0.1,
        distance_metric: str = "euclidean",
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        self.approx_strategy = approx_strategy
        self.approx_param = approx_param
        self.distance_metric = distance_metric

        valid_strategies = {
            "exact",
            "validation_subsample",
            "train_subsample",
            "kdtree",
            "faiss",
            "random_projection",
            "monte_carlo",
        }
        if approx_strategy not in valid_strategies:
            raise ValueError(f"approx_strategy must be one of {sorted(valid_strategies)}")

        valid_metrics = {"euclidean", "manhattan", "cosine"}
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {sorted(valid_metrics)}")

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """1 for all matching rows and 0 otherwise, against `self.y_valid`."""
        return (y == self.y_valid).all(dim=1).float()

    def compute_distance_matrix(
        self, x_train: torch.Tensor, x_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute distances per strategy/metric.

        Returns
        -------
        (dist, sort_indices)
            For strategies 'kdtree' and 'faiss', returns both distance matrix and
            precomputed sort indices. For all other strategies, returns (dist, None).
        """

        def _ensure_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(np.array(x))

        x_train = _ensure_tensor(x_train)
        x_valid = _ensure_tensor(x_valid)

        # Random projection dimensionality reduction
        if self.approx_strategy == "random_projection":
            x_train, x_valid = self._random_projection(x_train, x_valid)

        # Subsample validation
        if self.approx_strategy == "validation_subsample":
            m_valid = len(x_valid)
            subset_size = max(1, int(m_valid * self.approx_param))
            idx = self.random_state.choice(m_valid, subset_size, replace=False)
            x_valid = x_valid[idx]
            if hasattr(self, "y_valid"):
                self.y_valid = self.y_valid[idx]

        # Subsample train
        if self.approx_strategy == "train_subsample":
            n_train = len(x_train)
            subset_size = max(1, int(n_train * self.approx_param))
            idx = self.random_state.choice(n_train, subset_size, replace=False)
            x_train = x_train[idx]
            if hasattr(self, "y_train"):
                self.y_train = self.y_train[idx]

        # KDTree/FAISS special fast paths
        if self.approx_strategy == "kdtree":
            return self._kdtree_distances(x_train, x_valid)
        if self.approx_strategy == "faiss":
            return self._faiss_distances(x_train, x_valid)
        if self.approx_strategy == "monte_carlo":
            # Handled separately by caller
            raise RuntimeError("Monte Carlo handled separately")

        # Metric-based exact distances
        if self.distance_metric == "euclidean":
            dist = self._euclidean_distances(x_train, x_valid)
        elif self.distance_metric == "manhattan":
            dist = self._manhattan_distances(x_train, x_valid)
        elif self.distance_metric == "cosine":
            dist = self._cosine_distances(x_train, x_valid)
        else:
            dist = self._euclidean_distances(x_train, x_valid)

        return dist, None

    def _euclidean_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor) -> torch.Tensor:
        n = len(x_train)
        m = len(x_valid)
        x_train_view = x_train.view(n, -1)
        x_valid_view = x_valid.view(m, -1)

        dist_list = []
        for x_train_batch in DataLoader(x_train_view, self.batch_size, shuffle=False):
            dist_row = []
            for x_val_batch in DataLoader(x_valid_view, self.batch_size, shuffle=False):
                dist_row.append(torch.cdist(x_train_batch, x_val_batch, p=2))
            dist_list.append(torch.cat(dist_row, dim=1))
        return torch.cat(dist_list, dim=0)

    def _manhattan_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor) -> torch.Tensor:
        n = len(x_train)
        m = len(x_valid)
        x_train_view = x_train.view(n, -1)
        x_valid_view = x_valid.view(m, -1)

        dist_list = []
        for x_train_batch in DataLoader(x_train_view, self.batch_size, shuffle=False):
            dist_row = []
            for x_val_batch in DataLoader(x_valid_view, self.batch_size, shuffle=False):
                dist_row.append(torch.cdist(x_train_batch, x_val_batch, p=1))
            dist_list.append(torch.cat(dist_row, dim=1))
        return torch.cat(dist_list, dim=0)

    def _cosine_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor) -> torch.Tensor:
        n = len(x_train)
        m = len(x_valid)
        x_train_view = x_train.view(n, -1)
        x_valid_view = x_valid.view(m, -1)

        x_train_norm = torch.nn.functional.normalize(x_train_view, p=2, dim=1)
        x_valid_norm = torch.nn.functional.normalize(x_valid_view, p=2, dim=1)

        dist_list = []
        for x_train_batch in DataLoader(x_train_norm, self.batch_size, shuffle=False):
            dist_row = []
            for x_val_batch in DataLoader(x_valid_norm, self.batch_size, shuffle=False):
                similarity = torch.mm(x_train_batch, x_val_batch.T)
                dist_row.append(1 - similarity)
            dist_list.append(torch.cat(dist_row, dim=1))
        return torch.cat(dist_list, dim=0)

    def _random_projection(self, x_train: torch.Tensor, x_valid: torch.Tensor):
        d_original = x_train.shape[1]
        d_projected = max(1, int(d_original * self.approx_param))
        projection = torch.randn(d_original, d_projected, device=x_train.device)
        projection = projection / torch.sqrt(torch.sum(projection**2, dim=0, keepdim=True))
        return x_train @ projection, x_valid @ projection

    def _kdtree_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor):
        # Import lazily to avoid hard dependency
        try:
            from scipy.spatial import KDTree
        except Exception as ex:
            raise ImportError("scipy.spatial.KDTree is required for 'kdtree' strategy") from ex

        x_train_np = x_train.detach().cpu().numpy()
        x_valid_np = x_valid.detach().cpu().numpy()
        tree = KDTree(x_train_np)

        k_query = min(self.k_neighbors + 1, len(x_train_np))
        distances, indices = tree.query(x_valid_np, k=k_query)

        n_valid = len(x_valid_np)
        n_train = len(x_train_np)
        dist_matrix = torch.full((n_train, n_valid), float("inf"))

        # KDTree returns arrays even for k=1; unify shape
        if k_query == 1:
            distances = distances[None, :]
            indices = indices[None, :]

        for j in range(n_valid):
            drow = distances[j] if distances.ndim > 1 else np.array([distances[j]])
            irow = indices[j] if np.ndim(indices) > 1 else np.array([indices[j]])
            for idx, dist in zip(irow, drow):
                if 0 <= int(idx) < n_train:
                    dist_matrix[int(idx), j] = float(dist)

        sort_indices = torch.argsort(dist_matrix, dim=0)
        return dist_matrix, sort_indices

    def _faiss_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor):
        try:
            import faiss  # type: ignore
        except Exception as ex:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu") from ex

        x_train_np = x_train.detach().cpu().numpy().astype("float32")
        x_valid_np = x_valid.detach().cpu().numpy().astype("float32")
        d = x_train_np.shape[1]

        # Simple heuristic for index choice
        if d <= 256:
            nlist = max(1, min(100, len(x_train_np) // 39))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(x_train_np)
        else:
            index = faiss.IndexFlatL2(d)

        index.add(x_train_np)
        k = max(1, min(self.k_neighbors, len(x_train_np)))
        distances, indices = index.search(x_valid_np, k)

        n_valid = len(x_valid_np)
        n_train = len(x_train_np)
        dist_matrix = torch.full((n_train, n_valid), float("inf"))
        for j in range(n_valid):
            for idx, dist in zip(indices[j], distances[j]):
                if 0 <= int(idx) < n_train:
                    dist_matrix[int(idx), j] = float(dist)

        sort_indices = torch.argsort(dist_matrix, dim=0)
        return dist_matrix, sort_indices

    def _monte_carlo_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor):
        n = len(x_train)
        m = len(x_valid)
        n_permutations = max(1, int(self.approx_param * n))
        scores = torch.zeros(n)

        for _ in progress_range(n_permutations):
            perm = torch.randperm(n)
            x_train_perm = x_train[perm]
            y_train_perm = self.y_train[perm]
            dist = self._euclidean_distances(x_train_perm, x_valid)
            sort_idx = torch.argsort(dist, dim=0)
            y_sorted = y_train_perm[sort_idx]

            for val_idx in range(m):
                match_positions = torch.where(
                    y_sorted[:, val_idx] == self.y_valid[val_idx]
                )[0]
                if len(match_positions) > 0:
                    first_match = int(match_positions[0].item())
                    if first_match < self.k_neighbors:
                        contribution = 1.0 / (first_match + 1)
                        original_idx = perm[sort_idx[first_match, val_idx]]
                        scores[original_idx] += contribution

        scores = scores / n_permutations
        dummy_dist = torch.zeros((n, m))
        dummy_sort = torch.argsort(torch.randn(n, m), dim=0)
        return dummy_dist, dummy_sort, scores

    def train_data_values(self, *args, **kwargs):
        n = len(self.x_train)
        m = len(self.x_valid)
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        if self.approx_strategy == "monte_carlo":
            _, _, scores = self._monte_carlo_distances(x_train, x_valid)
            # Ensure consistent output type across strategies (numpy array)
            self.data_values = scores.detach().cpu().numpy()
            return self

        dist, sort_indices = self.compute_distance_matrix(x_train, x_valid)
        if sort_indices is None:
            sort_indices = torch.argsort(dist, dim=0, stable=True)

        n, m = dist.shape
        y_train_sort = self.y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[n - 1], torch.arange(m)] = self.match(y_train_sort[n - 1]) / n

        for i in progress_range(n - 2, -1, -1):
            score[sort_indices[i], torch.arange(m)] = (
                score[sort_indices[i + 1], torch.arange(m)]
                + min(self.k_neighbors, i + 1)
                / (self.k_neighbors * (i + 1))
                * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )

        self.data_values = score.mean(axis=1).detach().numpy()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_values

    def get_approximation_info(self) -> dict:
        info = {
            "strategy": self.approx_strategy,
            "parameter": self.approx_param,
            "distance_metric": self.distance_metric,
            "k_neighbors": self.k_neighbors,
            "batch_size": self.batch_size,
        }
        if hasattr(self, "data_values"):
            info["num_points"] = len(self.data_values)
            info["value_range"] = (
                float(self.data_values.min()),
                float(self.data_values.max()),
            )
        return info


class KNNShapleyAdaptive(KNNShapleyApprox):
    """Adaptive KNN-Shapley that chooses approximation based on data size."""

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        distance_metric: str = "euclidean",
        memory_limit_mb: float = 1024.0,
        time_limit_seconds: float = 60.0,
    ):
        super().__init__(
            k_neighbors=k_neighbors,
            batch_size=batch_size,
            embedding_model=embedding_model,
            random_state=random_state,
            approx_strategy="exact",
            approx_param=0.1,
            distance_metric=distance_metric,
        )
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_seconds = time_limit_seconds

    def _choose_approximation_strategy(self, n_train: int, n_valid: int, d: int) -> str:
        exact_memory_mb = (n_train * n_valid * 4) / (1024 * 1024)
        exact_time_estimate = (n_train * n_valid * max(1, d)) / 1e9
        if exact_memory_mb < self.memory_limit_mb and exact_time_estimate < self.time_limit_seconds:
            return "exact"
        if n_valid > 1000:
            self.approx_param = min(1000 / max(1, n_valid), 0.5)
            return "validation_subsample"
        if n_train > 10000:
            self.approx_param = min(5000 / max(1, n_train), 0.3)
            return "train_subsample"
        if d > 100:
            self.approx_param = 0.1
            return "random_projection"
        return "kdtree"

    def train_data_values(self, *args, **kwargs):
        n_train = len(self.x_train)
        n_valid = len(self.x_valid)
        x_train_emb, _ = self.embeddings(self.x_train[:1], self.x_valid[:1])
        d = x_train_emb.view(1, -1).shape[1]

        self.approx_strategy = self._choose_approximation_strategy(n_train, n_valid, d)
        print(f"Selected approximation strategy: {self.approx_strategy}")
        if self.approx_param != 0.1:
            print(f"Approximation parameter: {self.approx_param}")
        return super().train_data_values(*args, **kwargs)
