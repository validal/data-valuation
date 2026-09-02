import numpy as np
import torch
from typing import Optional
from sklearn.linear_model import LogisticRegression as SKLR

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class Kairos(DataEvaluator, ModelLessMixin):
    def __init__(self, lambda_weight=0.97, sigma_feature=None, kernel_type='sigma', unbiased=False, use_median_heuristic=True, num_samples=10000, random_state: Optional = None, embedding_model: Optional[Model] = None, debug: bool = False):
        """
        Args:
            lambda_weight (float): Weight for the residual squared term.
            sigma_feature (float): Bandwidth for the Gaussian (RBF) kernel on features.
                If None and use_median_heuristic=True, computed via median heuristic.
            kernel_type (str): Only 'sigma' is supported.
            unbiased (bool): Whether to use unbiased feature metric computation.
            use_median_heuristic (bool): If True, estimate bandwidth from median of sampled pairwise distances.
            num_samples (int): Number of sampled pairs for median heuristic estimation, by default 10000.
            random_state: Random state for reproducibility.
            embedding_model (Model, optional): Pre-trained embedding model for computing embeddings, by default None.
            debug (bool): Print detailed debug information about computed values, by default False.

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
            raise ValueError("Kairos only supports kernel_type 'sigma'.")
        self.lambda_weight = lambda_weight
        self.sigma_feature = sigma_feature
        self.use_median_heuristic = use_median_heuristic
        self.num_samples = num_samples
        self.embedding_model = embedding_model
        self.debug = debug

        # Data placeholders; these are set by input_data.
        self.X_train = None
        self.X_valid = None
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
            if hasattr(self.embedding_model, '__module__'):
                embedding_str = self.embedding_model.__class__.__name__
            else:
                embedding_str = "embedding_model"
        return (
            f"Kairos(lambda_weight={self.lambda_weight}, unbiased={self.unbiased}, "
            f"use_median_heuristic={self.use_median_heuristic}, num_samples={self.num_samples}, "
            f"embedding_model={embedding_str}, random_state={self.random_state}, debug={self.debug})"
        )

    __str__ = __repr__

    def input_data(self, x_train, y_train, x_valid, y_valid, debug=True) -> "Kairos":

        """
        Prepares the training and validation data and computes residuals on the training set.
        The classifier (SKLR) is trained on the validation data and then used to predict
        on the training set (to get ŷ_train).
        """
        # Call parent to set self.x_train, self.x_valid for embeddings
        super().input_data(x_train, y_train, x_valid, y_valid)

        # Apply embeddings FIRST to original data
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        if debug:
            print("\n[DEBUG Kairos.input_data] Input shapes:")
            if isinstance(x_train, torch.Tensor):
                print(f"  x_train (torch): {x_train.shape}")
            else:
                print(f"  x_train (numpy): {np.array(x_train).shape}")
            if isinstance(y_train, torch.Tensor):
                print(f"  y_train (torch): {y_train.shape}")
            else:
                print(f"  y_train (numpy): {np.array(y_train).shape}")

        # Convert inputs to numpy arrays if needed.
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.detach().cpu().numpy()
        if isinstance(x_valid, torch.Tensor):
            x_valid = x_valid.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.detach().cpu().numpy()

        x_train = np.array(x_train, dtype=np.float32)
        x_valid = np.array(x_valid, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_valid = np.array(y_valid, dtype=np.float32)

        if debug:
            print(f"  After numpy conversion: x_train {x_train.shape}, y_train {y_train.shape}")

        # Flatten if image data (4D or higher)
        if x_train.ndim > 2:
            if debug:
                print(f"  ⚠️  x_train is {x_train.ndim}D, flattening to 2D...")
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_valid = x_valid.reshape(x_valid.shape[0], -1)
            if debug:
                print(f"  ✓ After flattening: x_train {x_train.shape}, x_valid {x_valid.shape}")

        self.input_dim = x_train.shape[1]
        self.num_classes = y_train.shape[1] if y_train.ndim > 1 else 1

        if debug:
            print(f"  input_dim={self.input_dim}, num_classes={self.num_classes}\n")

        # Convert to torch tensors.
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.X_valid = torch.tensor(x_valid, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Train a simple classifier (SKLR) on the validation data to obtain predicted probabilities.
        if debug:
            print(f"[DEBUG Kairos.input_data] Training LogisticRegression...")
            print(f"  x_valid shape for fit: {x_valid.shape}")
            print(f"  y_valid_indices shape: {np.argmax(y_valid, axis=1).shape}")

        if self.lambda_weight == 1:
            self.r_train = torch.zeros(y_train.shape, dtype=torch.float32)
        else:
            y_valid_indices = np.argmax(y_valid, axis=1)
            self.classifier = SKLR(random_state=42)
            try:
                self.classifier.fit(x_valid, y_valid_indices)
                if debug:
                    print(f"  ✓ LogisticRegression trained successfully")
            except Exception as e:
                if debug:
                    print(f"  ✗ Error training LogisticRegression: {e}")
                raise
            p_train = self.classifier.predict_proba(x_train)

            # Compute residuals for the training set: residual = y_true - y_pred.
            r_train = y_train - p_train
            self.r_train = torch.tensor(r_train, dtype=torch.float32)
        return self

    def _compute_median_heuristic(self) -> float:
        """
        Compute bandwidth using median heuristic on sampled pairwise distances.
        Estimates median distance from 10000 random pairs to avoid O(n²) computation.

        Returns:
            float: Estimated bandwidth (median of pairwise distances)
        """
        X = self.X_train
        n = X.shape[0]
        num_pairs = min(self.num_samples, n * (n - 1) // 2)

        # Sample random pairs
        rng = np.random.RandomState(42)
        sampled_dists = []
        for _ in range(num_pairs):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                dist = torch.norm(X[i] - X[j]).item()
                sampled_dists.append(dist)

        sampled_dists = np.array(sampled_dists)
        median_dist = np.median(sampled_dists)
        return median_dist

    def train_data_values(self, *args, debug=None, **kwargs):
        """
        Pre-compute the per-sample average RBF-kernel values:
          - avg_K_train[i] = mean_j k(x_i, x_j)
          - avg_K_valid[i] = mean_j k(x_j, x_i)

        This version fuses the two kernel computations into one large matrix
        multiply + one exp, sharing all the common work. If sigma_feature is None,
        uses median heuristic on sampled pairwise distances.
        """
        # Use instance debug flag if not overridden
        if debug is None:
            debug = self.debug

        if debug:
            print(f"\n" + "="*70)
            print(f"[Kairos] TRAINING PHASE")
            print(f"="*70)
            print(f"  X_train shape: {self.X_train.shape}")
            print(f"  X_valid shape: {self.X_valid.shape}")

        X = self.X_train        # (n_train, d)
        V = self.X_valid        # (n_valid, d)
        n_train = X.shape[0]

        # Compute or use provided bandwidth
        if self.sigma_feature is None and self.use_median_heuristic:
            if debug:
                print(f"  Computing Gaussian bandwidth via median heuristic...")
            sigma = self._compute_median_heuristic()
            self.sigma_feature = sigma  # Store for later use
            if debug:
                print(f"  ✓ Computed sigma_feature = {sigma:.6f}")
        else:
            sigma = self.sigma_feature if self.sigma_feature is not None else 3.0
            if debug:
                print(f"  Using provided sigma_feature = {sigma:.6f}")

        # bandwidth parameters for Gaussian kernel
        sigma2 = sigma ** 2
        inv_two_sigma2 = 1.0 / (2.0 * sigma2)

        if debug:
            print(f"\n  [Gaussian Bandwidth Parameters]")
            print(f"    sigma (bandwidth):  {sigma:.6f}")
            print(f"    sigma²:             {sigma2:.6f}")
            print(f"    1/(2σ²):            {inv_two_sigma2:.6f}")

        # 1) stack the data so we do one big matmul instead of two
        Z = torch.cat([X, V], dim=0)     # shape: (n_train + n_valid, d)

        # 2) squared ℓ₂ norms of each row in Z
        #    so we only compute these once
        Z_norm_sq = (Z * Z).sum(dim=1)   # shape: (n_train + n_valid,)

        # 3) one matrix–multiply for all cross terms
        #    Z @ Xᵀ gives a (n_train+n_valid)×n_train Gram block:
        G = Z @ X.T                      # shape: (n_train+n_valid, n_train)

        # 4) pairwise squared distances:
        #    D[i,j] = ‖Z[i]−X[j]‖² = Z_norm_sq[i] + X_norm_sq[j] − 2·G[i,j]
        #    note: X_norm_sq is just the first n_train entries of Z_norm_sq
        D = Z_norm_sq.unsqueeze(1) + Z_norm_sq[:n_train].unsqueeze(0) - 2.0 * G

        # 5) kernel matrix for *all* pairs in one exp
        K = torch.exp(-D * inv_two_sigma2)   # shape: (n_train+n_valid, n_train)

        # 6) split back into train/valid blocks
        K_train = K[:n_train, :]             # (n_train, n_train)
        K_valid = K[n_train:, :]             # (n_valid, n_train)

        # 7) averages
        self.avg_K_train = K_train.mean(dim=1)    # (n_train,)
        self.avg_K_valid = K_valid.mean(dim=0)    # (n_train,)

        # feature discrepancy
        if self.unbiased:
            n = len(self.X_train)
            avg_K_train_unbiased = (self.avg_K_train * n - 1) / (n - 1)
            feature_metric = self.avg_K_valid - avg_K_train_unbiased
            #feature_metric = self.avg_K_train - (self.avg_K_valid * len(self.X_train) - 1) / (len(self.X_train) - 1)
        else:
            feature_metric =  self.avg_K_valid - self.avg_K_train

        # squared residual term
        squared_residual = torch.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        if debug:
            print(f"\n  [Feature Metric Statistics]")
            print(f"    Mean:   {feature_metric.mean():.6f}")
            print(f"    Std:    {feature_metric.std():.6f}")
            print(f"    Min:    {feature_metric.min():.6f}")
            print(f"    Max:    {feature_metric.max():.6f}")
            print(f"    Range:  {feature_metric.max() - feature_metric.min():.6f}")

            print(f"\n  [Squared Residual Term Statistics]")
            print(f"    Mean:   {squared_residual.mean():.6f}")
            print(f"    Std:    {squared_residual.std():.6f}")
            print(f"    Min:    {squared_residual.min():.6f}")
            print(f"    Max:    {squared_residual.max():.6f}")
            print(f"    Range:  {squared_residual.max() - squared_residual.min():.6f}")

            print(f"\n  [Combined Metric (current lambda={self.lambda_weight:.4f})]")
            combined = self.lambda_weight * feature_metric + (1 - self.lambda_weight) * squared_residual
            print(f"    Mean:   {combined.mean():.6f}")
            print(f"    Std:    {combined.std():.6f}")
            print(f"    Variance: {combined.var():.6f}")

            print(f"\n" + "="*70)

        return self

    def _rbf_kernel(self, A, B, sigma):
        """
        Computes the RBF kernel between each row in A and each row in B.
        """
        A_norm_sq = (A ** 2).sum(dim=1).unsqueeze(1)
        B_norm_sq = (B ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = A_norm_sq + B_norm_sq - 2 * (A @ B.T)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def evaluate_data_values(self) -> np.ndarray:
        """
        Uses the pre-computed avg_K_train / avg_K_valid plus residuals
        to form the final metric per training sample.
        """
         # feature discrepancy
        if self.unbiased:
            n = len(self.X_train)
            avg_K_train_unbiased = (self.avg_K_train * n - 1) / (n - 1)
            feature_metric = self.avg_K_valid - avg_K_train_unbiased
            #feature_metric = self.avg_K_train - (self.avg_K_valid * len(self.X_train) - 1) / (len(self.X_train) - 1)
        else:
            feature_metric =  self.avg_K_valid - self.avg_K_train

        # squared residual term
        squared_residual = torch.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        # combined metric
        metric = self.lambda_weight * feature_metric + (1 - self.lambda_weight) * squared_residual

        # Find and print optimal lambda from balancing
        opt_result = self.find_optimal_lambda_weight(verbose=self.debug)

        if self.debug:
            print(f"\n" + "="*70)
            print(f"[Kairos] OPTIMAL LAMBDA COMPUTATION")
            print(f"="*70)
            print(f"  Current lambda:     {opt_result.get('current_lambda', self.lambda_weight):.6f}")
            print(f"  Optimal lambda:     {opt_result.get('optimal_lambda', self.lambda_weight):.6f}")
            print(f"  Improvement:        {opt_result.get('improvement_percent', 0):.2f}%")
            print(f"\n  [Metric Statistics with Current Lambda]")
            print(f"    Mean:             {opt_result.get('current_metric_mean', 0):.6f}")
            print(f"\n  [Metric Statistics with Optimal Lambda]")
            print(f"    Mean:             {opt_result.get('optimal_metric_mean', 0):.6f}")
            print(f"\n  [Average Term Contributions]")
            print(f"    Feature:          {opt_result.get('avg_feature', 0):.6f}")
            print(f"    Residual:         {opt_result.get('avg_residual', 0):.6f}")
            print(f"="*70 + "\n")

        return -metric.detach().cpu().numpy()

    def online_update(self, x_new, y_new) -> "Kairos":
        """
        Incrementally update the values when a new batch (x_new, y_new) arrives.

        Args:
            x_new:  New training features, shape (m, d), numpy array or torch.Tensor.
            y_new:  New training labels (one-hot or probabilities), shape (m, c),
                    numpy array or torch.Tensor.

        Returns:
            self, with all internal buffers updated.
        """
        # 1) Normalize inputs to torch.float32
        import numpy as _np
        import torch as _torch

        # convert to numpy
        if isinstance(x_new, _torch.Tensor):
            x_new = x_new.detach().cpu().numpy()
        if isinstance(y_new, _torch.Tensor):
            y_new = y_new.detach().cpu().numpy()

        x_new = _np.array(x_new, dtype=_np.float32)
        y_new = _np.array(y_new, dtype=_np.float32)

        # to torch
        X_new = _torch.tensor(x_new, dtype=_torch.float32)
        y_new = _torch.tensor(y_new, dtype=_torch.float32)

        # 2) Compute residuals for the new batch
        if self.lambda_weight == 1:
            r_new = _torch.zeros_like(y_new)
        else:
            # classifier was trained on validation in input_data()
            p_new = self.classifier.predict_proba(x_new)  # shape (m, c)
            r_new = _torch.tensor(y_new.numpy() - p_new, dtype=_torch.float32)

        # 3) Prepare sizes
        N_old = self.X_train.shape[0]
        m = X_new.shape[0]
        N_total = N_old + m

        σ = self.sigma_feature

        # 4) Update avg_K_train for old points:
        #    avg_K_train_old_new = mean_j∈new k(x_i_old, x_j_new)
        K_old_new = self._rbf_kernel(self.X_train, X_new, σ)    # (N_old, m)
        sum_old_old = self.avg_K_train * N_old                  # (N_old,)
        sum_old_new = K_old_new.sum(dim=1)                      # (N_old,)
        avg_old_updated = (sum_old_old + sum_old_new) / N_total # (N_old,)

        # 5) Compute avg_K_train for new points:
        #    sum over old + sum over new
        K_new_old = self._rbf_kernel(X_new, self.X_train, σ)    # (m, N_old)
        K_new_new = self._rbf_kernel(X_new, X_new, σ)           # (m, m)
        sum_new = K_new_old.sum(dim=1) + K_new_new.sum(dim=1)    # (m,)
        avg_new = sum_new / N_total                             # (m,)

        # 6) Concatenate to form updated avg_K_train
        self.avg_K_train = _torch.cat([avg_old_updated, avg_new], dim=0)

        # 7) Update avg_K_valid: only need new columns
        #    avg_K_valid_new = mean_i k(x_valid_i, x_new_j)
        K_valid_new = self._rbf_kernel(self.X_valid, X_new, σ)  # (n_valid, m)
        avg_valid_new = K_valid_new.mean(dim=0)                 # (m,)
        self.avg_K_valid = _torch.cat([self.avg_K_valid, avg_valid_new], dim=0)

        # 8) Append new data to X_train, y_train, r_train
        self.X_train = _torch.cat([self.X_train, X_new], dim=0)
        self.y_train = _torch.cat([self.y_train, y_new], dim=0)
        self.r_train = _torch.cat([self.r_train, r_new], dim=0)

        return self

    def find_optimal_lambda_weight(self, verbose: bool = True) -> dict:
        """
        Find optimal lambda_weight that balances feature and residual contributions.

        Uses normalization to equalize the scale of both metrics, then finds lambda
        that minimizes the variance of the combined metric (optimal balance).

        Must be called after train_data_values() and evaluate_data_values().

        Args:
            verbose (bool): Print comparison between current and optimal lambda

        Returns:
            dict: Contains:
                - 'optimal_lambda': Best lambda_weight
                - 'current_lambda': Current lambda_weight
                - 'current_metric_mean': Mean value with current lambda
                - 'optimal_metric_mean': Mean value with optimal lambda
                - 'improvement_percent': Percentage improvement
                - 'avg_feature': Average feature contribution
                - 'avg_residual': Average residual contribution
        """
        if self.feature_metric is None or self.squared_residual is None:
            raise ValueError("Must call train_data_values() and evaluate_data_values() first")

        feature = self.feature_metric.cpu().numpy() if isinstance(self.feature_metric, torch.Tensor) else self.feature_metric
        residual = self.squared_residual.cpu().numpy() if isinstance(self.squared_residual, torch.Tensor) else self.squared_residual

        # Normalize both metrics to [0, 1] range for fair comparison
        feat_min, feat_max = feature.min(), feature.max()
        res_min, res_max = residual.min(), residual.max()

        if feat_max - feat_min > 0:
            feat_norm = (feature - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = np.ones_like(feature)

        if res_max - res_min > 0:
            res_norm = (residual - res_min) / (res_max - res_min)
        else:
            res_norm = np.ones_like(residual)

        # Find lambda that minimizes the variance (best balance)
        lambdas = np.linspace(0, 1, 201)
        variances = []
        for lam in lambdas:
            combined = lam * feat_norm + (1 - lam) * res_norm
            variances.append(np.var(combined))

        optimal_lambda = lambdas[np.argmin(variances)]

        # Compute metrics with current and optimal lambda
        current_metric = self.lambda_weight * feature + (1 - self.lambda_weight) * residual
        optimal_metric = optimal_lambda * feature + (1 - optimal_lambda) * residual

        current_mean = current_metric.mean()
        optimal_mean = optimal_metric.mean()
        current_std = current_metric.std()
        optimal_std = optimal_metric.std()

        # Improvement is based on reduced variance (better balance)
        improvement = ((current_std - optimal_std) / current_std) * 100 if current_std != 0 else 0

        result = {
            'optimal_lambda': float(optimal_lambda),
            'current_lambda': float(self.lambda_weight),
            'current_metric_mean': float(current_mean),
            'optimal_metric_mean': float(optimal_mean),
            'current_metric_std': float(current_std),
            'optimal_metric_std': float(optimal_std),
            'improvement_percent': float(improvement),
            'avg_feature': float(feature.mean()),
            'avg_residual': float(residual.mean()),
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"Kairos: Lambda Weight Optimization Report")
            print(f"{'='*70}")
            print(f"\nCurrent Configuration:")
            print(f"  Lambda weight:              {self.lambda_weight:.4f}")
            print(f"\nOptimal Configuration (minimizes variance):")
            print(f"  Optimal lambda weight:      {optimal_lambda:.4f}")
            print(f"  Difference:                 {abs(optimal_lambda - self.lambda_weight):.4f}")
            print(f"\nMetric Statistics:")
            print(f"  Feature (avg):              {feature.mean():.6f}")
            print(f"  Residual (avg):             {residual.mean():.6f}")
            print(f"\nCombined Metric Performance:")
            print(f"  {'Metric':<30} {'Current':<15} {'Optimal':<15}")
            print(f"  {'-'*60}")
            print(f"  {'Mean':<30} {current_mean:<15.6f} {optimal_mean:<15.6f}")
            print(f"  {'Std Dev':<30} {current_std:<15.6f} {optimal_std:<15.6f}")
            print(f"\nImprovement (variance reduction): {improvement:.2f}%")
            print(f"{'='*70}\n")

        return result

    @staticmethod
    def compute_lambda_weight(avg_feature_metric: float, avg_residual_metric: float, method: str = 'balanced') -> float:
        """
        Compute lambda_weight from average feature and residual metrics.

        Args:
            avg_feature_metric (float): Average feature-based contribution
            avg_residual_metric (float): Average residual-based contribution
            method (str): Method for computing lambda_weight:
                - 'balanced': Equal contribution from both metrics (0.5)
                - 'ratio': Weight by inverse ratio of magnitudes
                - 'normalize': Normalize by sum of magnitudes

        Returns:
            float: Computed lambda_weight in [0, 1]
        """
        if method == 'balanced':
            return 0.5
        elif method == 'ratio':
            if avg_residual_metric == 0:
                return 0.5
            ratio = avg_feature_metric / avg_residual_metric
            lambda_w = ratio / (ratio + 1)
            return float(np.clip(lambda_w, 0, 1))
        elif method == 'normalize':
            total = abs(avg_feature_metric) + abs(avg_residual_metric)
            if total == 0:
                return 0.5
            lambda_w = abs(avg_feature_metric) / total
            return float(np.clip(lambda_w, 0, 1))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'balanced', 'ratio', or 'normalize'.")
