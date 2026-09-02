from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
import json
from opendataval.dataval.progress import ProgressBar, progress_range
import time
import traceback
from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import torch.multiprocessing as torch_mp

# Set up for maximum parallelization
torch.set_num_threads(1)  # Each process uses 1 thread to avoid oversubscription


def macos_fix():
    """Geomloss package has a bug on MacOS remedied as follows.

    `Link to similar bug: https://github.com/NVlabs/stylegan3/issues/75`_.
    """
    import os
    import sys

    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class LavaEvaluator(DataEvaluator, ModelLessMixin):
    """Data valuation using LAVA implementation.

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
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        mode: str = "cls",
        p: int = 2,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        feature_cost: Optional[FeatureCost] = None,
        debug: bool = False,
        blur: Optional[float] = None,
        # GeomLoss passthrough knobs
        scaling: float = 0.8,
        backend: Optional[str] = None,
        truncate: Optional[float] = None,
        diameter: Optional[float] = None,
        outer_debias: bool = True,

    ):
        macos_fix()
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.embedding_model = embedding_model
        self.device = device

        # Mode: 'cls' for classification (discrete labels), 'reg' for regression (continuous targets)
        self.mode = mode

        # OTDD-related parameters for convergence studies
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.entreg = entreg
        self.loss = loss
        self.feature_cost = feature_cost
        # Keep a handle to the OTDD distance object for later access (e.g., transport plan)
        self._dist: Optional[DatasetDistance] = None
        self.debug = bool(debug)
        self.blur = blur
        # GeomLoss control
        self.gl_scaling = float(scaling)
        self.gl_backend = backend
        self.gl_truncate = truncate
        self.gl_diameter = diameter
        self.outer_debias = bool(outer_debias)

    def __repr__(self) -> str:
        embedding_str = "None"
        if self.embedding_model is not None:
            embedding_str = self.embedding_model.__class__.__name__
        return (
            f"LavaEvaluator(device={self.device}, embedding_model={embedding_str}, "
            f"lam_x={self.lam_x}, lam_y={self.lam_y}, mode={self.mode}, p={self.p})"
        )

    __str__ = __repr__

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values using class-wise Wasserstein distance."""
        import torch as _torch
        from torch.utils.data import Dataset, Subset

        def _ensure_tensor(data):
            """Convert Dataset/Subset objects to tensors."""
            if isinstance(data, (Dataset, Subset)):
                items = []
                for i in range(len(data)):
                    item = data[i]
                    # Handle tuple returns from datasets
                    if isinstance(item, (tuple, list)):
                        item = item[0]  # Take first element if it's a tuple
                    items.append(_torch.as_tensor(item, dtype=_torch.float32))
                return _torch.stack(items)
            return data

        # Embed data ONCE if embedding model is provided
        if self.embedding_model is not None:
            x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
            # Use euclidean distance (data already embedded)
            feature_cost = "euclidean"
        else:
            # No embedding, use original data
            x_train = self.x_train
            x_valid = self.x_valid
            feature_cost = None

        # Ensure data is converted from Dataset/Subset to tensors
        x_train = _ensure_tensor(x_train)
        x_valid = _ensure_tensor(x_valid)
        y_train = _ensure_tensor(self.y_train)
        y_valid = _ensure_tensor(self.y_valid)

        # Note: we no longer use lam_y==0 concatenation hack. Use `mode='reg'` to
        # indicate regression and let DatasetDistance handle label distances.

        if self.debug:
            def _stat(t, name):
                try:
                    t_cpu = t.detach().cpu()
                    print(f"[lava] {name}: shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, "
                          f"device={t.device}, min={t_cpu.min().item():.4g}, max={t_cpu.max().item():.4g}")
                except Exception:
                    print(f"[lava] {name}: shape={getattr(t,'shape',None)}, device={getattr(t,'device',None)}")
            print(f"[lava] train_data_values: lam_x={self.lam_x}, lam_y={self.lam_y}, p={self.p}, "
                  f"entreg={self.entreg}, loss={self.loss}, feature_cost={'custom' if self.feature_cost else feature_cost}")
            data_note = "(embedded)" if self.embedding_model else "(raw)"
            _stat(x_train, f'x_train {data_note}')
            _stat(y_train, 'y_train')
            _stat(x_valid, f'x_valid {data_note}')
            _stat(y_valid, 'y_valid')

        # Prefer explicitly provided feature_cost; otherwise use embedding-based one; else euclidean
        _feat_cost = self.feature_cost if self.feature_cost else (feature_cost if feature_cost else "euclidean")

        # Build and solve the OT distance, timing and verbose debug for troubleshooting
        if self.debug:
            print(f"[lava] building DatasetDistance with feature_cost={_feat_cost}, device={self.device}, blur={self.blur}, backend={self.gl_backend}")
        t0 = time.perf_counter()

        # Convert one-hot labels to class indices — otdd expects 1D integer labels
        def _to_labels(y):
            if isinstance(y, _torch.Tensor) and y.dim() == 2 and y.shape[1] > 1:
                return y.argmax(dim=1)
            return y
        y_train_labels = _to_labels(y_train)
        y_valid_labels = _to_labels(y_valid)

        try:
            dist = DatasetDistance(
                x_train=x_train,
                y_train=y_train_labels,
                x_valid=x_valid,
                y_valid=y_valid_labels,
                feature_cost=_feat_cost,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                p=self.p,
                entreg=self.entreg,
                device=self.device,
                inner_ot_loss=self.loss,
                debug=self.debug,
                blur=self.blur,
                scaling=self.gl_scaling,
                gl_backend=self.gl_backend,
                gl_truncate=self.gl_truncate,
                gl_diameter=self.gl_diameter,
                mode=self.mode,
                outer_debias=self.outer_debias,
            )
        except Exception as ex:
            t1 = time.perf_counter()
            print(f"[lava][error] DatasetDistance construction failed after {t1 - t0:.4f}s: {ex}")
            traceback.print_exc()
            raise
        t1 = time.perf_counter()
        if self.debug:
            print(f"[lava] DatasetDistance constructed in {t1 - t0:.4f}s")

        # Store for downstream usage (e.g., saving transport plan)
        self._dist = dist

        # Call dual_sol() and time the OT solve
        t2 = time.perf_counter()
        try:
            if self.debug:
                print("[lava] starting otdd.dual_sol() (this may take time depending on dataset size and geomloss backend)")
            dual = dist.dual_sol()
            self.dual_sol = dual
        except Exception as ex:
            t3 = time.perf_counter()
            print(f"[lava][error] dual_sol() failed after {t3 - t2:.4f}s: {ex}")
            traceback.print_exc()
            raise
        t3 = time.perf_counter()
        if self.debug:
            try:
                f, g = self.dual_sol
                print(f"[lava] dual potentials: f.shape={tuple(f.shape)}, g.shape={tuple(g.shape)}")
            except Exception:
                print("[lava] dual_sol obtained")
            print(f"[lava] dual_sol() completed in {t3 - t2:.4f}s (construct {t1 - t0:.4f}s, total {t3 - t0:.4f}s)")
        return self

    def save_transport_plan(
        self,
        save_path: str,
        epsilon: Optional[float] = None,
        max_iters: int = 500,
        tol: float = 1e-9,
    ) -> torch.Tensor:
        """Compute and save the OT transport plan using the same augmented cost.

        Parameters
        ----------
        save_path : str
            Path where to save the coupling. Uses .pt (torch.save) or .npy (numpy) based on suffix.
        epsilon : float, optional
            Entropic regularization; if None, uses the default tied to the OTDD blur/p.
        max_iters : int, default 500
            Maximum Sinkhorn iterations.
        tol : float, default 1e-9
            Convergence tolerance for scaling vectors.

        Returns
        -------
        torch.Tensor
            The (N, M) transport plan on CPU.
        """
        # Ensure we have a distance object prepared
        if self._dist is None:
            import torch as _torch
            from torch.utils.data import Dataset, Subset

            def _ensure_tensor(data):
                """Convert Dataset/Subset objects to tensors."""
                if isinstance(data, (Dataset, Subset)):
                    items = []
                    for i in range(len(data)):
                        item = data[i]
                        # Handle tuple returns from datasets
                        if isinstance(item, (tuple, list)):
                            item = item[0]  # Take first element if it's a tuple
                        items.append(_torch.as_tensor(item, dtype=_torch.float32))
                    return _torch.stack(items)
                return data

            # Lazily prepare embeddings and distance if train_data_values wasn't called
            x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
            x_train = _ensure_tensor(x_train)
            x_valid = _ensure_tensor(x_valid)
            y_train = _ensure_tensor(self.y_train)
            y_valid = _ensure_tensor(self.y_valid)

            # Do not augment features for regression here; DatasetDistance supports
            # regression mode via `mode='reg'` and combines lam_x/lam_y accordingly.
            feat_cost = self.feature_cost if self.feature_cost else "euclidean"
            self._dist = DatasetDistance(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                feature_cost=feat_cost,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                p=self.p,
                entreg=self.entreg,
                device=self.device,
                inner_ot_loss=self.loss,
                blur=self.blur,
                scaling=self.gl_scaling,
                gl_backend=self.gl_backend,
                gl_truncate=self.gl_truncate,
                gl_diameter=self.gl_diameter,
                mode=self.mode,
                outer_debias=self.outer_debias,
            )
        if self.debug:
            print(f"[lava] save_transport_plan: path={save_path}, epsilon={epsilon}, max_iters={max_iters}, tol={tol}")
        return self._dist.transport_plan(
            save_path=save_path, epsilon=epsilon, max_iters=max_iters, tol=tol
        )

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Gets the calibrated gradient of the dual solution, which can be interpreted as
        the data values.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        f1k = self.dual_sol[0].squeeze()
        num_points = len(f1k) - 1
        train_gradient = f1k * (1 + 1 / (num_points)) - f1k.sum() / num_points

        # We multiply -1 to align LAVA with other data valuation algorithms
        # Low values should indicate detrimental data points
        train_gradient =  -1 * train_gradient
        return train_gradient.numpy(force=True)


class LavaOOBEvaluator(DataEvaluator, ModelLessMixin):
    """Approximate LAVA via bootstrapped OT on subsets (Out-Of-Bag style).

    Computes per-point values by repeatedly sampling smaller train/valid subsets,
    solving the OT problem on each subset, extracting the train-side potentials,
    calibrating them (as in LavaEvaluator), and aggregating the values back to the
    original training indices. This reduces compute versus full OT on all points.

    Parameters
    ----------
    device : torch.device, optional
        Tensor device for acceleration.
    embedding_model : Optional[Model], optional
        Optional model used to compute feature embeddings for distance.
    random_state : Optional[RandomState], optional
        Random state for reproducibility.
    lam_x : float, optional
        Weight for feature distance component (default=1.0).
    lam_y : float, optional
        Weight for label distance component (default=1.0).
    p : int, optional
        OT order for outer problem (default=4 to match LavaEvaluator default).
    entreg : float, optional
        Entropy regularization (default=1e-1).
    loss : str, optional
        GeomLoss loss (default="sinkhorn").
    n_bootstrap : int, optional
        Number of bootstrap replicates (default=20).
    train_sample_size : Optional[int], optional
        Number of training points per replicate; if None, uses fraction.
    valid_sample_size : Optional[int], optional
        Number of validation points per replicate; if None, uses fraction.
    train_sample_frac : float, optional
        Fraction of training points per replicate (used if size is None), default 0.2.
    valid_sample_frac : float, optional
        Fraction of validation points per replicate (used if size is None), default 0.2.
    replace : bool, optional
        Sample with replacement (bootstrap) if True; subsample without replacement otherwise.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        mode: str = "cls",
        p: int = 4,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        feature_cost: Optional[FeatureCost] = None,
        n_bootstrap: int = 100,
        train_sample_size: Optional[int] = None,
        valid_sample_size: Optional[int] = None,
        train_sample_frac: float = 0.2,
        valid_sample_frac: float = 0.2,
        replace: bool = True,
        debug: bool = False,
        blur: float = 0.05,
        # GeomLoss passthrough knobs
        scaling: float = 0.8,
        backend: Optional[str] = None,
        truncate: Optional[float] = None,
        diameter: Optional[float] = None,
        outer_debias: bool = True,
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
        self.debug = bool(debug)
        self.progress_bar = bool(progress_bar)

        # Bootstrap parameters
        self.n_bootstrap = int(n_bootstrap)
        self.train_sample_size = train_sample_size
        self.valid_sample_size = valid_sample_size
        self.train_sample_frac = float(train_sample_frac)
        self.valid_sample_frac = float(valid_sample_frac)
        self.replace = bool(replace)
        self.blur = float(blur)
        # GeomLoss control
        self.gl_scaling = float(scaling)
        self.gl_backend = backend
        self.gl_truncate = truncate
        self.gl_diameter = diameter
        self.outer_debias = bool(outer_debias)
        # Mode: 'cls' or 'reg'
        self.mode = mode
        # Storage for bootstrap indices (train/valid) captured during training
        self._bootstrap_train_indices = []  # list[np.ndarray]
        self._bootstrap_valid_indices = []  # list[np.ndarray]

    def _make_feature_cost(self):
        feature_cost = None
        # If a custom feature cost was provided, prefer it
        # If a custom feature_cost was provided, prefer it
        if getattr(self, "feature_cost", None) is not None:
            return self.feature_cost
        if self.embedding_model is not None:
            resize = 32
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=(3, resize, resize),
                tgt_embedding=self.embedding_model,
                tgt_dim=(3, resize, resize),
                p=2,
                device=self.device.type,
            )
        return feature_cost if feature_cost else "euclidean"

    def _calibrate_potentials(self, f: torch.Tensor) -> torch.Tensor:
        """Apply the same calibration as LavaEvaluator on a subset's potentials."""
        n = len(f)
        num_points = max(n - 1, 1)
        return f * (1 + 1 / num_points) - f.mean()

    def train_data_values(self, *args, **kwargs):
        import torch as _torch
        from torch.utils.data import Dataset, Subset

        def _ensure_tensor(data):
            """Convert Dataset/Subset objects to tensors."""
            if isinstance(data, (Dataset, Subset)):
                items = []
                for i in range(len(data)):
                    item = data[i]
                    # Handle tuple returns from datasets
                    if isinstance(item, (tuple, list)):
                        item = item[0]  # Take first element if it's a tuple
                    items.append(_torch.as_tensor(item, dtype=_torch.float32))
                return _torch.stack(items)
            return data

        # Prepare embeddings (no-op for non-image datasets)
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        x_train = _ensure_tensor(x_train)
        x_valid = _ensure_tensor(x_valid)
        y_train = _ensure_tensor(self.y_train)
        y_valid = _ensure_tensor(self.y_valid)

        # Use DatasetDistance' built-in `mode` handling to account for labels.

        if self.debug:
            print(f"[lava-oob] train_data_values: n_bootstrap={self.n_bootstrap}, "
                  f"train_frac={self.train_sample_frac}, valid_frac={self.valid_sample_frac}, "
                  f"lam_x={self.lam_x}, lam_y={self.lam_y}, p={self.p}, entreg={self.entreg}")

        N_tr = len(y_train)
        N_va = len(y_valid)

        # Determine sample sizes
        s_tr = self.train_sample_size or max(2, int(np.ceil(self.train_sample_frac * N_tr)))
        s_va = self.valid_sample_size or max(2, int(np.ceil(self.valid_sample_frac * N_va)))

        # Aggregators
        agg = torch.zeros(N_tr, dtype=torch.float32, device=self.device)
        cnt = torch.zeros(N_tr, dtype=torch.int32, device=self.device)

        rng = check_random_state(None)

        feature_cost = self._make_feature_cost()

        # Move full tensors to CPU first (DatasetDistance will handle device moves)
        x_tr_full, y_tr_full = x_train, y_train
        x_va_full, y_va_full = x_valid, y_valid

        # Reset stored bootstrap indices for this training run
        self._bootstrap_train_indices = []
        self._bootstrap_valid_indices = []

        for b in range(self.n_bootstrap):
            # Sample indices
            tr_idx = rng.choice(N_tr, size=s_tr, replace=self.replace)
            va_idx = rng.choice(N_va, size=s_va, replace=self.replace)
            if self.debug and b == 0:
                print(f"[lava-oob] subset sizes: train={len(tr_idx)}, valid={len(va_idx)}; feature_cost={'custom' if callable(feature_cost) else feature_cost}")

            # Store sampled indices
            self._bootstrap_train_indices.append(np.array(tr_idx, copy=True))
            self._bootstrap_valid_indices.append(np.array(va_idx, copy=True))

            # Slice subsets
            x_tr_sub = x_tr_full[tr_idx]
            y_tr_sub = y_tr_full[tr_idx]
            x_va_sub = x_va_full[va_idx]
            y_va_sub = y_va_full[va_idx]

            # Compute OT potentials on the subset
            dist = DatasetDistance(
                x_train=x_tr_sub,
                y_train=y_tr_sub,
                x_valid=x_va_sub,
                y_valid=y_va_sub,
                feature_cost=feature_cost,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                p=self.p,
                entreg=self.entreg,
                device=self.device,
                inner_ot_loss=self.loss,
                blur=self.blur,
                scaling=self.gl_scaling,
                gl_backend=self.gl_backend,
                gl_truncate=self.gl_truncate,
                gl_diameter=self.gl_diameter,
                mode=self.mode,
                outer_debias=self.outer_debias,
            )
            # Print scalar OT distance per bootstrap when debugging
            try:
                if self.debug:
                    dval = dist.distance_value()
                    print(f"[lava-oob] bootstrap {b}: OT distance (train subset vs valid subset) = {dval:.6g}")
            except Exception:
                pass
            F_i, _ = dist.dual_sol()

            f = F_i.squeeze()
            f = self._calibrate_potentials(f)

            # Accumulate back to original indices
            # If sampling with replacement, multiple occurrences sum naturally
            f_device = f.to(device=agg.device, dtype=agg.dtype)
            idx_tensor = torch.as_tensor(tr_idx, device=agg.device, dtype=torch.long)
            agg.index_add_(0, idx_tensor, f_device)
            cnt.index_add_(0, idx_tensor, torch.ones_like(idx_tensor, dtype=cnt.dtype))

        # Store aggregated results
        self._agg_potentials = agg
        self._agg_counts = cnt
        return self

    def evaluate_data_values(self) -> np.ndarray:
        # Compute mean calibrated potential per point (avoid division by zero)
        agg = self._agg_potentials
        cnt = self._agg_counts.clamp_min(1)
        mean_potential = agg / cnt.to(dtype=agg.dtype)

        # Align sign with LavaEvaluator: lower means more detrimental → multiply by -1
        values = (-1.0 * mean_potential).detach().cpu().numpy()
        return values

    def get_bootstrap_indices(self):
        """Return the stored bootstrap indices.

        Returns
        -------
        dict
            A dict with keys 'train' and 'valid', each a list of numpy arrays
            containing the sampled indices per bootstrap replicate.
        """
        return {
            "train": self._bootstrap_train_indices,
            "valid": self._bootstrap_valid_indices,
        }

    def save_bootstrap_indices(self, save_path: str):
        """Save stored bootstrap indices to disk.

        If save_path ends with .npz, saves each replicate as train_{i}, valid_{i} arrays
        and includes n_bootstrap metadata. If ends with .json, saves a JSON with two lists
        of lists under keys 'train' and 'valid'.

        Parameters
        ----------
        save_path : str
            Destination path (.npz or .json)
        """
        if not self._bootstrap_train_indices:
            raise RuntimeError("No bootstrap indices stored. Run train_data_values() first.")

        if save_path.endswith('.npz'):
            data = {**{f"train_{i}": arr for i, arr in enumerate(self._bootstrap_train_indices)},
                    **{f"valid_{i}": arr for i, arr in enumerate(self._bootstrap_valid_indices)},
                    "n_bootstrap": np.array(len(self._bootstrap_train_indices))}
            np.savez(save_path, **data)
        elif save_path.endswith('.json'):
            payload = {
                "train": [arr.tolist() for arr in self._bootstrap_train_indices],
                "valid": [arr.tolist() for arr in self._bootstrap_valid_indices],
                "n_bootstrap": len(self._bootstrap_train_indices),
            }
        
            with open(save_path, 'w') as f:
                json.dump(payload, f)
        else:
            raise ValueError("Unsupported file format. Use .npz or .json")
    
class ParallelLavaOOBEvaluator(LavaOOBEvaluator):
    """LavaOOB with full parallelization across 56 cores."""
    
    def __init__(
        self,
        n_jobs: int = -1,  # -1 uses all cores
        backend: str = "loky",  # or "multiprocessing"
        chunk_size: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.backend = backend
        self.chunk_size = chunk_size
        self.debug = kwargs.get('debug', False)
        # propagate/override mode if provided in kwargs
        self.mode = kwargs.get('mode', getattr(self, 'mode', 'cls'))
        
    def _compute_bootstrap_ot(self, b: int):
        """Compute OT for a single bootstrap replicate identified by index b."""
        rng = np.random.RandomState(b)  # Seed each bootstrap
        
        # Sample indices
        N_tr = len(self.y_train_full)
        N_va = len(self.y_valid_full)
        s_tr = self.train_sample_size or max(2, int(np.ceil(self.train_sample_frac * N_tr)))
        s_va = self.valid_sample_size or max(2, int(np.ceil(self.valid_sample_frac * N_va)))
        
        tr_idx = rng.choice(N_tr, size=s_tr, replace=self.replace)
        va_idx = rng.choice(N_va, size=s_va, replace=self.replace)
        
        # Slice subsets
        x_tr_sub = self.x_train_full[tr_idx]
        y_tr_sub = self.y_train_full[tr_idx]
        x_va_sub = self.x_valid_full[va_idx]
        y_va_sub = self.y_valid_full[va_idx]
        
        # Compute OT
        dist = DatasetDistance(
            x_train=x_tr_sub,
            y_train=y_tr_sub,
            x_valid=x_va_sub,
            y_valid=y_va_sub,
            feature_cost=self.feature_cost,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            p=self.p,
            entreg=self.entreg,
            device=self.device,
            inner_ot_loss=self.loss,
            blur=self.blur,
            scaling=self.gl_scaling,
            gl_backend=self.gl_backend,
            gl_truncate=self.gl_truncate,
            gl_diameter=self.gl_diameter,
            mode=self.mode,
            outer_debias=self.outer_debias,
        )
        
        F_i, _ = dist.dual_sol()
        f = F_i.squeeze()
        
        # Calibrate
        n = len(f)
        num_points = max(n - 1, 1)
        calibrated = f * (1 + 1 / num_points) - f.mean()
        
        return b, tr_idx, va_idx, calibrated.cpu().numpy()
    
    def train_data_values(self, *args, **kwargs):
        """Parallel bootstrap computation."""
        import torch as _torch
        from torch.utils.data import Dataset, Subset

        def _ensure_tensor(data):
            """Convert Dataset/Subset objects to tensors."""
            if isinstance(data, (Dataset, Subset)):
                items = []
                for i in range(len(data)):
                    item = data[i]
                    # Handle tuple returns from datasets
                    if isinstance(item, (tuple, list)):
                        item = item[0]  # Take first element if it's a tuple
                    items.append(_torch.as_tensor(item, dtype=_torch.float32))
                return _torch.stack(items)
            return data

        # Prepare full data once
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        self.x_train_full = _ensure_tensor(x_train)
        self.x_valid_full = _ensure_tensor(x_valid)
        self.y_train_full = _ensure_tensor(self.y_train)
        self.y_valid_full = _ensure_tensor(self.y_valid)

        # Regression-aware path
        # Use DatasetDistance' mode handling rather than concatenating labels into features.

        # Prepare feature cost once
        self.feature_cost = self._make_feature_cost()

        N_tr = len(self.y_train_full)
        agg = np.zeros(N_tr, dtype=np.float64)
        cnt = np.zeros(N_tr, dtype=np.int32)
        
        if self.debug:
            print(f"[parallel-lava-oob] Using {self.n_jobs} cores for {self.n_bootstrap} bootstraps")
            print(f"[parallel-lava-oob] Sample sizes: train={self.train_sample_size or self.train_sample_frac*100}%, "
                  f"val={self.valid_sample_size or self.valid_sample_frac*100}%")
        
        # Parallel execution
        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            mp_context=mp.get_context('spawn')  # Better for CUDA
        ) as executor:
            # Submit all bootstrap tasks
            futures = []
            for b in range(self.n_bootstrap):
                future = executor.submit(self._compute_bootstrap_ot, b)
                futures.append(future)
            
            # Collect results
            for future in ProgressBar(iterable=futures, desc="Bootstraps"):
                b, tr_idx, va_idx, calibrated = future.result()
                
                # Aggregate
                np.add.at(agg, tr_idx, calibrated)
                np.add.at(cnt, tr_idx, 1)
                
                # Store indices if needed
                if hasattr(self, '_bootstrap_train_indices'):
                    self._bootstrap_train_indices.append(tr_idx)
                    self._bootstrap_valid_indices.append(va_idx)
        
        # Compute final values
        cnt = np.maximum(cnt, 1)
        mean_potential = agg / cnt
        self.data_values = -1.0 * mean_potential
        
        return self