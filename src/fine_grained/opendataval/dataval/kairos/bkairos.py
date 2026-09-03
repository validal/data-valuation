import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from opendataval.dataval import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model
from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from typing import Optional


class bKairos(DataEvaluator, ModelLessMixin):
    def __init__(self, lambda_weight=0.97, sigma_feature=None, kernel_type='sigma', unbiased=False,
                 use_median_heuristic=True, num_samples=10000, batch_size=1024,
                 random_state: Optional = None, embedding_model: Optional[Model] = None):
        """
        Batched/scalable version of Kairos for memory-efficient processing.

        Args:
            lambda_weight (float): Weight for the residual squared term, by default 0.97.
            sigma_feature (float): Bandwidth for the Gaussian (RBF) kernel on features.
                If None and use_median_heuristic=True, computed via median heuristic, by default None.
            kernel_type (str): Only 'sigma' is supported, by default 'sigma'.
            unbiased (bool): Whether to use unbiased feature metric computation, by default False.
            use_median_heuristic (bool): If True, estimate bandwidth from median of sampled pairwise distances, by default True.
            num_samples (int): Number of sampled pairs for median heuristic estimation, by default 10000.
            batch_size (int): Batch size for kernel computation to reduce memory usage, by default 1024.
            random_state (Optional): Random state for reproducibility, by default None.
            embedding_model (Model, optional): Pre-trained embedding model for computing embeddings, by default None.

        Note:
            No training (optimization) is performed in this class.
            The evaluation metric for each training sample is defined as:

                lambda_weight * feature_metric + (1 - lambda_weight) * squared_residual

            where:
                feature_metric = avg_{j in train} k(x_i, x_j) - avg_{j in valid} k(x_j, x_i),
                k is an RBF kernel with bandwidth sigma_feature, and
                squared_residual = sqrt(||(y_train[i] - ŷ_train[i])||²).
        """
        super().__init__(random_state=random_state)
        if kernel_type != 'sigma':
            raise ValueError("bKairos only supports kernel_type 'sigma'.")
        self.lambda_weight = lambda_weight
        self.sigma_feature = sigma_feature
        self.use_median_heuristic = use_median_heuristic
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.embedding_model = embedding_model

        # Data placeholders; these are set by input_data.
        self.X_train = None  # numpy array for efficient kernel computation
        self.X_valid = None  # numpy array for efficient kernel computation
        self.y_train = None
        self.y_valid = None
        self.r_train = None  # residual for training samples

        # These will be computed in train_data_values
        self.avg_K_train = None
        self.avg_K_valid = None

        self.unbiased = unbiased

    def __repr__(self) -> str:
        embedding_str = "None"
        if self.embedding_model is not None:
            embedding_str = self.embedding_model.__class__.__name__
        return (
            f"bKairos(lambda_weight={self.lambda_weight}, unbiased={self.unbiased}, "
            f"use_median_heuristic={self.use_median_heuristic}, num_samples={self.num_samples}, "
            f"batch_size={self.batch_size}, embedding_model={embedding_str}, "
            f"random_state={self.random_state})"
        )

    __str__ = __repr__

    def _compute_median_heuristic(self) -> float:
        """
        Compute bandwidth using median heuristic on sampled pairwise distances.
        Estimates median distance from sampled pairs to avoid O(n²) computation.
        Uses numpy for efficient BLAS computation.

        Returns:
            float: Estimated bandwidth (median of pairwise distances)
        """
        X = self.X_train
        n = X.shape[0]
        num_pairs = min(self.num_samples, n * (n - 1) // 2)

        # Sample random pairs using numpy
        rng = np.random.RandomState(42)
        sampled_dists = []
        for _ in range(num_pairs):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                dist = np.linalg.norm(X[i] - X[j])
                sampled_dists.append(dist)

        sampled_dists = np.array(sampled_dists)
        median_dist = np.median(sampled_dists)
        return median_dist

    def input_data(self, x_train, y_train, x_valid, y_valid) -> "bKairos":
        """
        Prepares the training and validation data and computes residuals on the training set.
        The classifier (SKLR) is trained on the validation data and then used to predict
        on the training set (to get ŷ_train).

        Parameters
        ----------
        x_train : array-like or torch.Tensor
            Training features
        y_train : array-like or torch.Tensor
            Training labels
        x_valid : array-like or torch.Tensor
            Validation features
        y_valid : array-like or torch.Tensor
            Validation labels

        Returns
        -------
        bKairos
            Self for method chaining
        """
        # Call parent to set self.x_train, self.x_valid for embeddings
        super().input_data(x_train, y_train, x_valid, y_valid)

        # Apply embeddings FIRST to original data
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        # Convert inputs to numpy arrays if needed
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.detach().cpu().numpy()
        if isinstance(x_valid, torch.Tensor):
            x_valid = x_valid.detach().cpu().numpy()

        x_train = np.array(x_train, dtype=np.float32)
        x_valid = np.array(x_valid, dtype=np.float32)

        # Convert labels to numpy arrays
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.detach().cpu().numpy()

        y_train = np.array(y_train, dtype=np.float32)
        y_valid = np.array(y_valid, dtype=np.float32)

        # Flatten if image data (4D or higher)
        if x_train.ndim > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_valid = x_valid.reshape(x_valid.shape[0], -1)

        self.input_dim = x_train.shape[1]
        self.num_classes = y_train.shape[1] if y_train.ndim > 1 else 1

        # Store features as numpy arrays for efficient BLAS-backed kernel computation.
        self.X_train = x_train  # numpy array (n_train, d)
        self.X_valid = x_valid  # numpy array (n_valid, d)
        # Store labels as torch tensors for residual calculations
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Train a simple classifier (SKLR) on the validation data to obtain predicted probabilities.
        if self.lambda_weight == 1:
            self.r_train = torch.zeros(y_train.shape, dtype=torch.float32)
        else:
            y_valid_indices = np.argmax(y_valid, axis=1)
            self.classifier = SKLR(random_state=42)
            self.classifier.fit(x_valid, y_valid_indices)
            p_train = self.classifier.predict_proba(x_train)

            # Compute residuals for the training set: residual = y_true - y_pred.
            r_train = y_train - p_train
            self.r_train = torch.tensor(r_train, dtype=torch.float32)
        return self

    def _compute_kernel_averages_batched(self, X_batch, X_all, V_all, sigma):
        """
        Computes RBF kernel averages using numpy+BLAS for efficiency.

        Args:
            X_batch: Batch of training samples (B, d) - numpy array
            X_all: All training samples (n_train, d) - numpy array
            V_all: All validation samples (n_valid, d) - numpy array
            sigma: RBF kernel bandwidth

        Returns:
            avg_train: Average kernel similarity to training set (B,) - numpy array
            avg_valid: Average kernel similarity to validation set (B,) - numpy array
        """
        sigma2 = sigma ** 2
        inv_two_sigma2 = 1.0 / (2.0 * sigma2)

        # Compute squared norms using numpy (efficient BLAS)
        X_batch_norm_sq = np.sum(X_batch * X_batch, axis=1, keepdims=True)  # (B, 1)
        X_all_norm_sq = np.sum(X_all * X_all, axis=1, keepdims=True)        # (n_train, 1)
        V_norm_sq = np.sum(V_all * V_all, axis=1, keepdims=True)            # (n_valid, 1)

        # Compute squared distances using BLAS matrix multiplication
        # D = ||X_batch - X_all||^2 = ||X_batch||^2 + ||X_all||^2 - 2*X_batch @ X_all.T
        D_train = X_batch_norm_sq + X_all_norm_sq.T - 2 * (X_batch @ X_all.T)
        K_train = np.exp(-D_train * inv_two_sigma2)  # (B, n_train)
        avg_train = K_train.mean(axis=1)  # (B,)

        # Compute kernel with validation set
        D_valid = X_batch_norm_sq + V_norm_sq.T - 2 * (X_batch @ V_all.T)
        K_valid = np.exp(-D_valid * inv_two_sigma2)  # (B, n_valid)
        avg_valid = K_valid.mean(axis=1)  # (B,)

        return avg_train, avg_valid

    def train_data_values(self, *args, **kwargs):
        """
        Pre-compute the per-sample average RBF-kernel values using batched computation:
          - avg_K_train[i] = mean_j k(x_i, x_j)
          - avg_K_valid[i] = mean_j k(x_j, x_i)

        This batched version avoids constructing full kernel matrices and scales
        to larger datasets by processing samples in batches.

        Returns
        -------
        bKairos
            Self for method chaining
        """
        X = self.X_train        # (n_train, d) numpy array
        V = self.X_valid        # (n_valid, d) numpy array
        n_train = X.shape[0]

        # Compute or use provided bandwidth
        if self.sigma_feature is None and self.use_median_heuristic:
            sigma = self._compute_median_heuristic()
            self.sigma_feature = sigma
        else:
            sigma = self.sigma_feature if self.sigma_feature is not None else 3.0

        # Initialize numpy arrays for efficient kernel computation
        avg_K_train = np.zeros(n_train, dtype=np.float32)
        avg_K_valid = np.zeros(n_train, dtype=np.float32)

        # Process training samples in batches using numpy+BLAS
        for start_idx in tqdm(range(0, n_train, self.batch_size), desc="Computing kernel averages"):
            end_idx = min(start_idx + self.batch_size, n_train)
            X_batch = X[start_idx:end_idx]  # (B, d) numpy array

            # Compute kernel averages using numpy (efficient BLAS)
            batch_avg_train, batch_avg_valid = self._compute_kernel_averages_batched(
                X_batch, X, V, sigma
            )

            # Store results
            avg_K_train[start_idx:end_idx] = batch_avg_train
            avg_K_valid[start_idx:end_idx] = batch_avg_valid

        # Convert to torch tensors for later metric computation
        self.avg_K_train = torch.tensor(avg_K_train, dtype=torch.float32)
        self.avg_K_valid = torch.tensor(avg_K_valid, dtype=torch.float32)

        # feature discrepancy: train - valid
        if self.unbiased:
            n = len(self.X_train)
            avg_K_train_unbiased = (self.avg_K_train * n - 1) / (n - 1)
            feature_metric = avg_K_train_unbiased - self.avg_K_valid
        else:
            feature_metric = self.avg_K_train - self.avg_K_valid

        # squared residual term
        squared_residual = torch.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        return self

    def _rbf_kernel(self, A, B, sigma):
        """
        Computes the RBF kernel between each row in A and each row in B using numpy+BLAS.
        Supports both numpy arrays and torch tensors as input.
        """
        # Convert to numpy if needed
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        if isinstance(B, torch.Tensor):
            B = B.detach().cpu().numpy()

        # Efficient BLAS computation
        A_norm_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_norm_sq = np.sum(B ** 2, axis=1, keepdims=True)
        dist_sq = A_norm_sq + B_norm_sq.T - 2 * (A @ B.T)
        return np.exp(-dist_sq / (2 * sigma ** 2))

    def evaluate_data_values(self, lambda_weight: Optional[float] = None) -> np.ndarray:
        """
        Uses the pre-computed avg_K_train / avg_K_valid plus residuals
        to form the final metric per training sample.

        Parameters
        ----------
        lambda_weight : float, optional
            Override instance lambda_weight for this evaluation only.
            If None, uses self.lambda_weight. Useful for tuning without recomputation, by default None

        Returns
        -------
        np.ndarray
            Data values for each training sample
        """
        if lambda_weight is None:
            lambda_weight = self.lambda_weight

        # feature discrepancy: train - valid
        if self.unbiased:
            feature_metric = (self.avg_K_train * len(self.X_train) - 1) / (len(self.X_train) - 1) - self.avg_K_valid
        else:
            feature_metric = self.avg_K_train - self.avg_K_valid

        # squared residual term
        squared_residual = torch.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        # combined metric
        metric = lambda_weight * feature_metric + (1 - lambda_weight) * squared_residual

        return -metric.detach().cpu().numpy()