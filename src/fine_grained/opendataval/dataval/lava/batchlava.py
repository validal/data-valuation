from typing import Optional, List, Tuple, Union
import warnings

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from opendataval.dataval.progress import ProgressBar, progress_range
from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model


class BatchwiseLavaEvaluator(DataEvaluator, ModelLessMixin):
    """Batchwise LAVA implementation for scalable data valuation.
    
    This implementation computes LAVA values by processing training-validation
    batch pairs independently and aggregating results, making it suitable for
    large datasets where full OT computation is infeasible.
    
    Parameters
    ----------
    device : torch.device, optional
        Tensor device for acceleration.
    embedding_model : Optional[Model], optional
        A model for computing embeddings, if needed.
    random_state : Optional[RandomState], optional
        Random state for reproducibility.
    lam_x : float, optional
        Regularization weight for features (default=1.0).
    lam_y : float, optional
        Regularization weight for labels (default=1.0).
    p : int, optional
        Power of the cost function (default=2).
    entreg : float, optional
        Entropy regularization (default=1e-1).
    loss : str, optional
        Inner OT loss type (default="sinkhorn").
    feature_cost : Optional[FeatureCost], optional
        Custom feature cost function.
    debug : bool, optional
        Enable debug output (default=False).
    blur : Optional[float], optional
        Blur parameter for geomloss.
    train_batch_size : int, optional
        Training batch size for OT computation (default=100).
    val_batch_size : int, optional
        Validation batch size for OT computation (default=100).
    scaling : float, optional
        Scaling parameter for geomloss (default=0.8).
    backend : Optional[str], optional
        Backend for geomloss computations.
    truncate : Optional[float], optional
        Truncation parameter for geomloss.
    diameter : Optional[float], optional
        Diameter parameter for geomloss.
    outer_debias : bool, optional
        Whether to use outer debiasing (default=True).
    normalize_values : bool, optional
        Optional normalization of per-batch values. When True, applies tanh-based
        squashing to (0,1) and per-batch L1 normalization. Defaults to False to
        preserve LAVA’s original value scales.
    cache_label_distances : bool, optional
        Cache label distances between batches for efficiency (default=False).
    parallel : bool, optional
        Enable parallel processing across GPUs (default=False).
    n_gpu : int, optional
        Number of GPUs for parallel processing (default=1).
    progress_bar : bool, optional
        Show progress bar during computation (default=True).
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        p: int = 2,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        feature_cost: Optional[FeatureCost] = None,
        debug: bool = False,
        blur: Optional[float] = None,
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        scaling: float = 0.8,
        backend: Optional[str] = None,
        truncate: Optional[float] = None,
        diameter: Optional[float] = None,
        outer_debias: bool = True,
        normalize_values: bool = False,
        cache_label_distances: bool = False,
        parallel: bool = False,
        n_gpu: int = 1,
        progress_bar: bool = True,
    ):
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.embedding_model = embedding_model
        self.device = device
        
        # OT parameters
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.entreg = entreg
        self.loss = loss
        self.feature_cost = feature_cost
        self.debug = debug
        self.blur = blur
        
        # Batch parameters
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.normalize_values = normalize_values
        self.cache_label_distances = cache_label_distances
        self.parallel = parallel
        self.n_gpu = n_gpu
        self.progress_bar = progress_bar
        
        # GeomLoss parameters
        self.gl_scaling = scaling
        self.gl_backend = backend
        self.gl_truncate = truncate
        self.gl_diameter = diameter
        self.outer_debias = outer_debias
        
        # Storage for computed values
        self.data_values = None
        self._dual_solutions = []  # Store for debugging/inspection
        
    def _prepare_batches(self, x: torch.Tensor, y: torch.Tensor, 
                        batch_size: int, shuffle: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Split data into batches."""
        n_samples = len(x)
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
        
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]
            batch_x = x[batch_idx]
            batch_y = y[batch_idx] if y is not None else None
            batches.append((batch_x, batch_y))
        
        return batches
    
    def _compute_batch_ot(self, x_tr_batch: torch.Tensor, y_tr_batch: torch.Tensor,
                         x_val_batch: torch.Tensor, y_val_batch: torch.Tensor,
                         feature_cost) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute OT between two batches."""
        # Regression-aware path: if lam_y == 0, concatenate y as last feature
        if float(self.lam_y) == 0.0:
            ytr = y_tr_batch.reshape(-1, 1).to(dtype=x_tr_batch.dtype, device=x_tr_batch.device)
            yva = y_val_batch.reshape(-1, 1).to(dtype=x_val_batch.dtype, device=x_val_batch.device)
            x_tr_batch = torch.cat([x_tr_batch, ytr], dim=1)
            x_val_batch = torch.cat([x_val_batch, yva], dim=1)
            feature_cost = "euclidean"
            if self.debug:
                print(f"[batchwise-lava] Regression mode: concatenated y to features")
        
        dist = DatasetDistance(
            x_train=x_tr_batch,
            y_train=y_tr_batch,
            x_valid=x_val_batch,
            y_valid=y_val_batch,
            feature_cost=feature_cost,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            p=self.p,
            entreg=self.entreg,
            device=self.device,
            inner_ot_loss=self.loss,
            debug=self.debug,
            blur=self.blur,
            scaling=self.gl_scaling,
            backend=self.gl_backend,
            truncate=self.gl_truncate,
            diameter=self.gl_diameter,
            outer_debias=self.outer_debias,
        )
        
        dual_sol = dist.dual_sol()
        return dual_sol
    
    def _calibrate_gradients(self, dual_sol: Tuple[torch.Tensor, torch.Tensor], 
                           training_size: int) -> np.ndarray:
        """Calibrate gradients from dual solution."""
        f, _ = dual_sol
        f = f.squeeze()
        
        # LAVA calibration formula
        num_points = max(len(f) - 1, 1)
        calibrated = f * (1 + 1 / num_points) - f.sum() / num_points

        # Convert to numpy
        calibrated_np = calibrated.detach().cpu().numpy()
        
        # Normalize if requested
        if self.normalize_values:
            # Apply tanh to squash to (-1, 1), then scale to (0, 1)
            calibrated_np = (np.tanh(calibrated_np) + 1) / 2
        
        return calibrated_np
    
    def train_data_values(self, *args, **kwargs):
        """Compute batchwise LAVA values."""
        # Get embeddings
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        y_train, y_valid = self.y_train, self.y_valid
        
        if self.debug:
            print(f"[batchwise-lava] Starting batchwise computation")
            print(f"[batchwise-lava] Training samples: {len(x_train)}, Validation samples: {len(x_valid)}")
            print(f"[batchwise-lava] Train batch size: {self.train_batch_size}, Val batch size: {self.val_batch_size}")
        
        # Prepare feature cost
        feature_cost = None
        if self.embedding_model is not None and float(self.lam_y) != 0.0:
            # Use embedding-based feature cost for image data
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=x_train.shape[1:] if len(x_train.shape) > 2 else (1,),
                tgt_embedding=self.embedding_model,
                tgt_dim=x_valid.shape[1:] if len(x_valid.shape) > 2 else (1,),
                p=2,
                device=self.device.type,
            )
        elif float(self.lam_y) == 0.0:
            feature_cost = "euclidean"
        else:
            feature_cost = self.feature_cost if self.feature_cost else "euclidean"
        
        # Split into batches
        train_batches = self._prepare_batches(x_train, y_train, self.train_batch_size)
        val_batches = self._prepare_batches(x_valid, y_valid, self.val_batch_size)
        
        if self.debug:
            print(f"[batchwise-lava] Number of train batches: {len(train_batches)}")
            print(f"[batchwise-lava] Number of val batches: {len(val_batches)}")
        
        # Initialize accumulation arrays
        n_train = len(x_train)
        values_accum = np.zeros(n_train)
        counts_accum = np.zeros(n_train)
        
        # Track batch indices for accumulation
        train_indices = torch.arange(n_train)
        train_batch_indices = []
        for start_idx in range(0, n_train, self.train_batch_size):
            end_idx = min(start_idx + self.train_batch_size, n_train)
            batch_idx = train_indices[start_idx:end_idx]
            train_batch_indices.append(batch_idx)
        
        # Process each training-validation batch pair
        iterator = range(len(train_batches))
        if self.progress_bar:
            iterator = ProgressBar(iterable=iterator, desc="Processing batch pairs")
        
        # Initialize cache for label distances if requested
        label_distances_cache = None
        
        for i in iterator:
            x_tr_batch, y_tr_batch = train_batches[i]
            batch_values = np.zeros(len(x_tr_batch))
            
            for j, (x_val_batch, y_val_batch) in enumerate(val_batches):
                # Compute OT between batches
                dual_sol = self._compute_batch_ot(
                    x_tr_batch, y_tr_batch, x_val_batch, y_val_batch, feature_cost
                )
                
                # Store dual solution for debugging
                if self.debug and i == 0 and j == 0:
                    self._dual_solutions.append(dual_sol)
                
                # Calibrate gradients
                calibrated_grad = self._calibrate_gradients(dual_sol, len(x_tr_batch))
                
                        # Optional per-batch normalization to equalize contributions
                if self.normalize_values and np.sum(np.abs(calibrated_grad)) > 0:
                    calibrated_grad = calibrated_grad / np.sum(np.abs(calibrated_grad))
                
                batch_values += calibrated_grad
            
            # Average over validation batches
            if len(val_batches) > 0:
                batch_values = batch_values / len(val_batches)
            
            # Accumulate to global array
            batch_idx = train_batch_indices[i].cpu().numpy()
            values_accum[batch_idx] += batch_values
            counts_accum[batch_idx] += 1
        
        # Compute final values (average over contributions)
        mask = counts_accum > 0
        final_values = np.zeros(n_train)
        final_values[mask] = values_accum[mask] / counts_accum[mask]
        
        # Handle points that weren't sampled (shouldn't happen with proper batching)
        if not np.all(mask):
            warnings.warn(f"Some training points ({np.sum(~mask)}) were not included in any batch")
            # Fill missing values with mean of computed values
            mean_val = np.mean(final_values[mask]) if np.any(mask) else 0
            final_values[~mask] = mean_val
        
        # Align sign with standard LAVA (lower values = more detrimental)
        self.data_values = -1 * final_values
        
        if self.debug:
            print(f"[batchwise-lava] Computed values - Min: {self.data_values.min():.4f}, "
                  f"Max: {self.data_values.max():.4f}, Mean: {self.data_values.mean():.4f}")
        
        return self

    
    def evaluate_data_values(self) -> np.ndarray:
        """Return computed data values.
        
        Returns
        -------
        np.ndarray
            Predicted data values for training data points
        """
        if self.data_values is None:
            raise RuntimeError("Must call train_data_values() before evaluate_data_values()")
        
        return self.data_values
    
    def get_dual_solutions(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get stored dual solutions for debugging.
        
        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor]]
            List of dual solutions from batch computations
        """
        return self._dual_solutions


class HierarchicalLavaEvaluator(BatchwiseLavaEvaluator):
    """Hierarchical LAVA implementation with two-level optimal transport.
    
    This implementation:
    1. Computes OT between training and validation batches
    2. Uses batch-level costs to compute a global transport plan between batches
    3. Aggregates sample-level values using the global batch weights
    
    Parameters
    ----------
    batch_reg : float, optional
        Regularization for batch-level OT (default=0.1).
    use_checkpoint : bool, optional
        Enable checkpointing for resumable computation (default=False).
    checkpoint_path : Optional[str], optional
        Path for checkpoint file (default=None).
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        p: int = 2,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        feature_cost: Optional[FeatureCost] = None,
        debug: bool = False,
        blur: Optional[float] = None,
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        scaling: float = 0.8,
        backend: Optional[str] = None,
        truncate: Optional[float] = None,
        diameter: Optional[float] = None,
        outer_debias: bool = True,
        batch_reg: float = 0.1,
        use_checkpoint: bool = False,
        checkpoint_path: Optional[str] = None,
        progress_bar: bool = True,
        **kwargs
    ):
        super().__init__(
            device=device,
            embedding_model=embedding_model,
            random_state=random_state,
            lam_x=lam_x,
            lam_y=lam_y,
            p=p,
            entreg=entreg,
            loss=loss,
            feature_cost=feature_cost,
            debug=debug,
            blur=blur,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            scaling=scaling,
            backend=backend,
            truncate=truncate,
            diameter=diameter,
            outer_debias=outer_debias,
            normalize_values=False,  # Don't normalize in hierarchical version
            progress_bar=progress_bar,
            **kwargs
        )
        
        self.batch_reg = batch_reg
        self.use_checkpoint = use_checkpoint
        self.checkpoint_path = checkpoint_path
        
        # Storage for hierarchical computation
        self.batch_costs = None
        self.batch_plan = None
        self.batch_dual_solutions = None

    def _sinkhorn_plan_torch(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        reg: float,
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> np.ndarray:
        """Compute entropic OT transport plan using torch-only Sinkhorn.

        Parameters
        ----------
        a : np.ndarray
            Source histogram (non-negative, sums to 1), shape (n,).
        b : np.ndarray
            Target histogram (non-negative, sums to 1), shape (m,).
        C : np.ndarray
            Cost matrix, shape (n, m).
        reg : float
            Entropic regularization strength (> 0).
        max_iter : int
            Maximum Sinkhorn iterations.
        tol : float
            Convergence tolerance on scaling vector change.

        Returns
        -------
        np.ndarray
            Transport plan of shape (n, m) as a numpy array.
        """
        # Move data to torch with stable dtype and device
        dtype = torch.float64
        dev = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))

        a_t = torch.as_tensor(a, dtype=dtype, device=dev)
        b_t = torch.as_tensor(b, dtype=dtype, device=dev)
        C_t = torch.as_tensor(C, dtype=dtype, device=dev)

        # Ensure valid inputs
        eps = torch.tensor(1e-300, dtype=dtype, device=dev)
        reg_t = torch.tensor(max(reg, 1e-12), dtype=dtype, device=dev)

        # Kernel
        K = torch.exp(-C_t / reg_t)
        K = torch.clamp(K, min=eps.item())

        # Initialize scaling vectors
        u = torch.ones_like(a_t)
        v = torch.ones_like(b_t)

        # Iterative scaling
        for _ in range(max_iter):
            u_prev = u
            Kv = K @ v
            Kv = torch.where(Kv > eps, Kv, eps)
            u = a_t / Kv

            KT_u = K.transpose(0, 1) @ u
            KT_u = torch.where(KT_u > eps, KT_u, eps)
            v = b_t / KT_u

            if torch.max(torch.abs(u - u_prev)) < tol:
                break

        # Transport plan P = diag(u) K diag(v)
        P = (u.unsqueeze(1) * K) * v.unsqueeze(0)
        return P.detach().cpu().numpy()
        
    def train_data_values(self, *args, **kwargs):
        """Compute hierarchical LAVA values."""
        import pickle
        import os
        
        # Get embeddings
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        y_train, y_valid = self.y_train, self.y_valid
        
        if self.debug:
            print(f"[hierarchical-lava] Starting hierarchical computation")
        
        # Prepare feature cost
        feature_cost = None
        if self.embedding_model is not None and float(self.lam_y) != 0.0:
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=x_train.shape[1:] if len(x_train.shape) > 2 else (1,),
                tgt_embedding=self.embedding_model,
                tgt_dim=x_valid.shape[1:] if len(x_valid.shape) > 2 else (1,),
                p=2,
                device=self.device.type,
            )
        elif float(self.lam_y) == 0.0:
            feature_cost = "euclidean"
        else:
            feature_cost = self.feature_cost if self.feature_cost else "euclidean"
        
        # Split into batches
        train_batches = self._prepare_batches(x_train, y_train, self.train_batch_size)
        val_batches = self._prepare_batches(x_valid, y_valid, self.val_batch_size)
        
        n_train_batches = len(train_batches)
        n_val_batches = len(val_batches)
        
        # Initialize storage
        batch_costs = np.zeros((n_train_batches, n_val_batches))
        batch_dual_solutions = [[None for _ in range(n_val_batches)] 
                               for _ in range(n_train_batches)]
        
        # Load checkpoint if exists and requested
        checkpoint_loaded = False
        if self.use_checkpoint and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                batch_costs = checkpoint['batch_costs']
                batch_dual_solutions = checkpoint['batch_dual_solutions']
                checkpoint_loaded = True
                if self.debug:
                    print(f"[hierarchical-lava] Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                if self.debug:
                    print(f"[hierarchical-lava] Failed to load checkpoint: {e}")
        
        # Process batch pairs
        iterator = range(n_train_batches)
        if self.progress_bar:
            iterator = ProgressBar(iterable=iterator, desc="Computing batch OT")
        
        for i in iterator:
            x_tr_batch, y_tr_batch = train_batches[i]
            
            for j in range(n_val_batches):
                # Skip if already computed
                if checkpoint_loaded and batch_dual_solutions[i][j] is not None:
                    continue
                
                x_val_batch, y_val_batch = val_batches[j]
                
                # Compute OT between batches
                dual_sol = self._compute_batch_ot(
                    x_tr_batch, y_tr_batch, x_val_batch, y_val_batch, feature_cost
                )
                
                # Store dual solution
                batch_dual_solutions[i][j] = dual_sol
                
                # Compute batch cost (average transport cost)
                # We could extract this from the distance value or compute from plan
                f, g = dual_sol
                # Simple approximation: use norm of potentials
                batch_costs[i, j] = (torch.abs(f).mean() + torch.abs(g).mean()).item()
                
                # Save checkpoint periodically
                if self.use_checkpoint and self.checkpoint_path and (i * n_val_batches + j) % 10 == 0:
                    with open(self.checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'batch_costs': batch_costs,
                            'batch_dual_solutions': batch_dual_solutions,
                        }, f)
        
        # Save final checkpoint
        if self.use_checkpoint and self.checkpoint_path:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump({
                    'batch_costs': batch_costs,
                    'batch_dual_solutions': batch_dual_solutions,
                }, f)
        
        # Compute batch-level transport plan
        a = np.ones(n_train_batches) / n_train_batches
        b = np.ones(n_val_batches) / n_val_batches
        
        if self.debug:
            print(f"[hierarchical-lava] Computing batch-level OT with reg={self.batch_reg}")
        
        # Compute batch-level plan using internal torch-based Sinkhorn (no POT)
        batch_plan = self._sinkhorn_plan_torch(a, b, batch_costs, self.batch_reg, max_iter=1000)
        
        if self.debug:
            plan_sum = batch_plan.sum()
            print(f"[hierarchical-lava] Batch plan sum: {plan_sum:.6f}")
        
        # Aggregate sample-level values
        n_train = len(x_train)
        final_values = np.zeros(n_train)
        
        # Track batch indices
        train_indices = torch.arange(n_train)
        train_batch_indices = []
        for start_idx in range(0, n_train, self.train_batch_size):
            end_idx = min(start_idx + self.train_batch_size, n_train)
            batch_idx = train_indices[start_idx:end_idx]
            train_batch_indices.append(batch_idx.cpu().numpy())
        
        # Compute weighted aggregation
        for i in range(n_train_batches):
            batch_values = np.zeros(len(train_batch_indices[i]))
            
            for j in range(n_val_batches):
                weight = batch_plan[i, j]
                if weight > 1e-10:  # Skip negligible contributions
                    dual_sol = batch_dual_solutions[i][j]
                    calibrated_grad = self._calibrate_gradients(dual_sol, len(train_batch_indices[i]))
                    batch_values += weight * calibrated_grad
            
            # Assign to global array
            batch_idx = train_batch_indices[i]
            final_values[batch_idx] = batch_values
        
        # Align sign with standard LAVA
        self.data_values = -1 * final_values
        self.batch_costs = batch_costs
        self.batch_plan = batch_plan
        self.batch_dual_solutions = batch_dual_solutions
        
        if self.debug:
            print(f"[hierarchical-lava] Final values - Min: {self.data_values.min():.4f}, "
                  f"Max: {self.data_values.max():.4f}, Mean: {self.data_values.mean():.4f}")
        
        return self
    
    def get_batch_metrics(self) -> dict:
        """Get batch-level metrics for analysis.
        
        Returns
        -------
        dict
            Dictionary containing batch costs and transport plan
        """
        return {
            'batch_costs': self.batch_costs,
            'batch_plan': self.batch_plan,
            'n_train_batches': self.batch_costs.shape[0] if self.batch_costs is not None else 0,
            'n_val_batches': self.batch_costs.shape[1] if self.batch_costs is not None else 0,
        }