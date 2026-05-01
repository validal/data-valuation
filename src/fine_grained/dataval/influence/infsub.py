from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin

from typing import Optional
import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset
from opendataval.dataval.api import DataEvaluator, ModelMixin
from multiprocessing import Pool, cpu_count
import warnings
import os
import math
import time

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False

# --- Module-level globals for parallel workers (avoid pickling instance state) ---
_PW_MODEL = None
_PW_X_TRAIN = None
_PW_Y_TRAIN = None
_PW_X_VALID = None
_PW_Y_VALID = None
_PW_NUM_POINTS = None
_PW_PROP = None
_PW_EVAL = None
_PW_FIT_ARGS = ()
_PW_FIT_KWARGS = {}
_PW_DEBUG = False
_PW_SUBSET_SIZE = None


def _resolve_subset_size(num_points: int, proportion: float, subset_size: Optional[int]) -> int:
    """Resolve final subset size.

    If `subset_size` is provided, it takes precedence over `proportion`.
    """
    if num_points <= 0:
        return 0
    if subset_size is not None:
        try:
            resolved = int(subset_size)
        except Exception as exc:
            raise TypeError("subset_size must be an int") from exc
        if resolved <= 0:
            raise ValueError("subset_size must be >= 1")
        if resolved > num_points:
            raise ValueError(
                f"subset_size={resolved} cannot exceed num_points={num_points}"
            )
        return resolved

    # Fall back to proportion
    try:
        p = float(proportion)
    except Exception as exc:
        raise TypeError("proportion must be a float") from exc
    if not (0.0 < p <= 1.0):
        raise ValueError("proportion must be in (0, 1]")
    resolved = int(round(p * num_points))
    resolved = max(1, min(num_points, resolved))
    return resolved


def _parallel_worker_init(pred_model, x_train, y_train, x_valid, y_valid,
                          num_points, proportion, evaluate_fn, fit_args, fit_kwargs, debug=False, subset_size=None):
    """Initializer run in each worker to set global context.

    This avoids pickling the full `ParallelizableInfluenceSubsample` instance, which
    may contain non-picklable objects (e.g., `wrapper` str-subclass evaluators).
    """
    global _PW_MODEL, _PW_X_TRAIN, _PW_Y_TRAIN, _PW_X_VALID, _PW_Y_VALID
    global _PW_NUM_POINTS, _PW_PROP, _PW_EVAL, _PW_FIT_ARGS, _PW_FIT_KWARGS
    _PW_MODEL = pred_model
    _PW_X_TRAIN = x_train
    _PW_Y_TRAIN = y_train
    _PW_X_VALID = x_valid
    _PW_Y_VALID = y_valid
    _PW_NUM_POINTS = num_points
    _PW_PROP = proportion
    # If evaluator is a `wrapper` (str subclass), unwrap to its function
    _PW_EVAL = getattr(evaluate_fn, "function", evaluate_fn)
    _PW_FIT_ARGS = fit_args
    _PW_FIT_KWARGS = fit_kwargs
    _PW_DEBUG = debug
    global _PW_SUBSET_SIZE
    _PW_SUBSET_SIZE = subset_size
    if _PW_DEBUG:
        try:
            print(f"[InfluenceSubsample Worker Init] pid={os.getpid()} num_points={_PW_NUM_POINTS} proportion={_PW_PROP} subset_size={_PW_SUBSET_SIZE}")
        except Exception:
            pass


def _parallel_train_single_model(seed):
    """Worker function: train a single model and return influence updates.

    Designed to be picklable and lightweight: only the `seed` is sent via the queue;
    all large objects are accessed from process-global context set by initializer.
    """
    # Local RNG per task
    local_rng = np.random.RandomState(seed)

    # Random subset indices
    subset_size = _resolve_subset_size(_PW_NUM_POINTS, _PW_PROP, _PW_SUBSET_SIZE)
    subset = local_rng.choice(_PW_NUM_POINTS, subset_size, replace=False)

    # Clone and fit model on the subset
    curr_model = _PW_MODEL.clone()
    curr_model.fit(
        Subset(_PW_X_TRAIN, indices=subset),
        Subset(_PW_Y_TRAIN, indices=subset),
        *_PW_FIT_ARGS,
        **_PW_FIT_KWARGS,
    )

    # Predict and evaluate on validation set
    y_valid_hat = curr_model.predict(_PW_X_VALID)
    curr_perf = _PW_EVAL(_PW_Y_VALID, y_valid_hat)

    # Prepare outputs
    influence_update = np.zeros(shape=(_PW_NUM_POINTS, 2))
    counts_update = np.zeros(shape=(_PW_NUM_POINTS, 2))

    included = (np.bincount(subset, minlength=_PW_NUM_POINTS) != 0).astype(int)
    rng = np.arange(_PW_NUM_POINTS)
    influence_update[rng, included] = curr_perf
    counts_update[rng, included] = 1

    if _PW_DEBUG:
        try:
            print(f"[InfluenceSubsample Worker] pid={os.getpid()} seed={seed} subset_size={subset.size} perf={curr_perf}")
        except Exception:
            pass

    return influence_update, counts_update


class InfluenceSubsample(DataEvaluator, ModelMixin):
    """Influence computed through subsamples implementation.

    Compute influence of each training example on for the validation dataset
    through closely-related subsampled influence.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    samples : int, optional
        Number of models to fit to take to find data values, by default 1000
    proportion : float, optional
        Proportion of data points to be in each sample, cardinality of each subset is
        :math:`(p)(num_points)`, by default 0.7 as specified by V. Feldman and C. Zhang
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 0.7,
        subset_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Subsample Data Valuation.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self._resolved_subset_size = _resolve_subset_size(
            self.num_points, self.proportion, self.subset_size
        )
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.influence_matrix = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Influence Subsample Data Valuator by sampling from subsets of
        :math:`(p)(num_points)` cardinality and computing the performance with the
        :math:`i` data point and without the :math:`i` data point. The form of sampling
        is similar to the shapely value when :math:`p` is :math:`0.5: (V. Feldman).
        Likewise, if we sample not from the subsets of a specific cardinality but the
        uniform across all subsets, it is similar to the Banzhaf value.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        for _ in tqdm.tqdm(range(self.num_models)):
            subset = self.random_state.choice(
                self.num_points, self._resolved_subset_size, replace=False
            )

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                *args,
                **kwargs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)
            curr_perf = self.evaluate(self.y_valid, y_valid_hat)

            included = (np.bincount(subset, minlength=self.num_points) != 0).astype(int)
            self.influence_matrix[range(self.num_points), included] += curr_perf
            self.sample_counts[range(self.num_points), included] += 1

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Influence Subsample data valuator. Finds
        the difference of average performance of all sets including data point minus
        not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.influence_matrix,
            self.sample_counts,
            out=np.zeros_like(self.influence_matrix),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point


class ParallelizableInfluenceSubsample(DataEvaluator, ModelMixin):
    """Influence computed through subsamples implementation.

    Compute influence of each training example on for the validation dataset
    through closely-related subsampled influence.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    samples : int, optional
        Number of models to fit to take to find data values, by default 1000
    proportion : float, optional
        Proportion of data points to be in each sample, cardinality of each subset is
        :math:`(p)(num_points)`, by default 0.7 as specified by V. Feldman and C. Zhang
    random_state : RandomState, optional
        Random initial state, by default None
    n_jobs : int, optional
        Number of parallel jobs to run. Defaults to -1 (use all available cores).
        Set to 1 for sequential processing.
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 0.7,
        subset_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        n_jobs: int = -1,
        debug: bool = False,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        detected = cpu_count()
        self.n_jobs = n_jobs if n_jobs != -1 else detected
        self.debug = debug
        warnings.warn(f"Using {self.n_jobs} parallel workers (cpu_count={detected})")

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Subsample Data Valuation.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self._resolved_subset_size = _resolve_subset_size(
            self.num_points, self.proportion, self.subset_size
        )
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.influence_matrix = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))
        if getattr(self, "debug", False):
            print(
                f"[ParallelizableInfluenceSubsample] input_data: num_points={self.num_points} subset_size={self._resolved_subset_size}"
            )
        return self

    def _train_single_model(self, seed, *fit_args, **fit_kwargs):
        """Train a single model on a random subset and return its influence."""
        # Create local random state based on seed
        local_rng = np.random.RandomState(seed)
        
        # Generate subset
        subset = local_rng.choice(self.num_points, self._resolved_subset_size, replace=False)
        
        # Clone and train model
        curr_model = self.pred_model.clone()
        curr_model.fit(
            Subset(self.x_train, indices=subset),
            Subset(self.y_train, indices=subset),
            *fit_args,
            **fit_kwargs,
        )
        
        # Evaluate on validation set
        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.evaluate(self.y_valid, y_valid_hat)
        
        # Create output arrays
        influence_update = np.zeros(shape=(self.num_points, 2))
        counts_update = np.zeros(shape=(self.num_points, 2))
        
        # Mark which points were included/excluded
        included = (np.bincount(subset, minlength=self.num_points) != 0).astype(int)
        influence_update[range(self.num_points), included] = curr_perf
        counts_update[range(self.num_points), included] = 1
        
        return influence_update, counts_update

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values in parallel.

        Trains the Influence Subsample Data Valuator by sampling from subsets of
        :math:`(p)(num_points)` cardinality and computing the performance with the
        :math:`i` data point and without the :math:`i` data point.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Generate seeds for reproducible randomness in parallel processes
        seeds = self.random_state.randint(0, 2**31 - 1, size=self.num_models)

        if self.n_jobs == 1:
            # Sequential processing (for debugging or small datasets)
            for seed in tqdm.tqdm(seeds, desc="Training models"):
                influence_update, counts_update = self._train_single_model(seed, *args, **kwargs)
                self.influence_matrix += influence_update
                self.sample_counts += counts_update
        else:
            # Parallel processing via module-level worker to avoid pickling `self`
            eval_fn = getattr(self.evaluate, "function", self.evaluate)
            init_args = (
                self.pred_model,
                self.x_train,
                self.y_train,
                self.x_valid,
                self.y_valid,
                self.num_points,
                self.proportion,
                eval_fn,
                args,
                kwargs,
                self.debug,
                self._resolved_subset_size,
            )
            if self.debug:
                print(f"[ParallelizableInfluenceSubsample] Spawning pool: n_jobs={self.n_jobs}, num_models={self.num_models}")
            with Pool(
                processes=self.n_jobs,
                initializer=_parallel_worker_init,
                initargs=init_args,
            ) as pool:
                # Stream aggregation to avoid holding all results in memory
                for influence_update, counts_update in tqdm.tqdm(
                    pool.imap_unordered(_parallel_train_single_model, seeds),
                    total=self.num_models,
                    desc=f"Training models ({self.n_jobs} workers)",
                ):
                    self.influence_matrix += influence_update
                    self.sample_counts += counts_update
            if self.debug:
                print(f"[ParallelizableInfluenceSubsample] Aggregation complete: counts_sum_included={int(self.sample_counts[:,1].sum())} counts_sum_excluded={int(self.sample_counts[:,0].sum())}")
        
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Influence Subsample data valuator. Finds
        the difference of average performance of all sets including data point minus
        not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.influence_matrix,
            self.sample_counts,
            out=np.zeros_like(self.influence_matrix),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point


class OptimizedInfluenceSubsample(DataEvaluator, ModelMixin):
    """Optimized, batched parallel Influence Subsample.

    Key optimizations:
    - Reduce workers for tiny tasks: default `n_jobs=min(16, cpu_count())`.
    - Batch processing: `batch_size` models per worker call (default 1000).
    - Efficient aggregation: update only included indices; derive excluded via totals.
    - Optional joblib backend and subset precomputation.
    - Streaming aggregation with progress, ETA, and models/sec.

    This class targets very large `num_models` with very small `subset_size`.
    """

    def __init__(
        self,
        num_models: int,
        proportion: float = 0.7,
        subset_size: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        n_jobs: int = -1,
        batch_size: int = 1000,
        use_joblib: bool = False,
        precompute_subsets: bool = False,
        debug: bool = False,
    ):
        self.num_models = int(num_models)
        self.proportion = float(proportion)
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        detected = cpu_count()
        # Use fewer workers for very small per-model tasks
        default_workers = min(16, detected)
        self.n_jobs = default_workers if n_jobs == -1 else max(1, int(n_jobs))
        self.batch_size = max(1, int(batch_size))
        self.use_joblib = bool(use_joblib and _HAS_JOBLIB)
        self.precompute_subsets = bool(precompute_subsets)
        self.debug = debug
        warnings.warn(
            f"OptimizedInfluenceSubsample: n_jobs={self.n_jobs} (cpu_count={detected}), "
            f"batch_size={self.batch_size}, joblib={self.use_joblib}, precompute_subsets={self.precompute_subsets}"
        )

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self._resolved_subset_size = _resolve_subset_size(
            self.num_points, self.proportion, self.subset_size
        )

        # Efficient aggregation structures
        self.sum_included = np.zeros(self.num_points, dtype=np.float64)
        self.count_included = np.zeros(self.num_points, dtype=np.int64)
        self.sum_all_perf = 0.0
        self.count_all = 0

        if self.debug:
            print(
                f"[OptimizedInfluenceSubsample] input_data: num_points={self.num_points} subset_size={self._resolved_subset_size}"
            )
            
        return self

    # --- Worker helpers ---
    @staticmethod
    def _batch_seed_tuple(seed: int, batch_size: int, subset_size: int):
        return (int(seed), int(batch_size), int(subset_size))

    @staticmethod
    def _process_batch_with_seeds(task):
        seed, batch_size, subset_size = task
        rng = np.random.RandomState(seed)

        perfs = np.empty(batch_size, dtype=np.float64)
        subsets = np.empty((batch_size, subset_size), dtype=np.int32)

        for j in range(batch_size):
            # Generate subset for this model
            subset = rng.choice(_PW_NUM_POINTS, subset_size, replace=False)
            subsets[j] = subset

            # Train and eval
            model = _PW_MODEL.clone()
            model.fit(Subset(_PW_X_TRAIN, indices=subset), Subset(_PW_Y_TRAIN, indices=subset), *_PW_FIT_ARGS, **_PW_FIT_KWARGS)
            yhat = model.predict(_PW_X_VALID)
            perfs[j] = _PW_EVAL(_PW_Y_VALID, yhat)

        if _PW_DEBUG:
            try:
                print(f"[Optimized Worker] pid={os.getpid()} batch_size={batch_size} subset_size={subset_size}")
            except Exception:
                pass
        return subsets, perfs

    @staticmethod
    def _process_batch_with_subsets(subsets):
        batch_size, subset_size = subsets.shape
        perfs = np.empty(batch_size, dtype=np.float64)
        for j in range(batch_size):
            subset = subsets[j]
            model = _PW_MODEL.clone()
            model.fit(Subset(_PW_X_TRAIN, indices=subset), Subset(_PW_Y_TRAIN, indices=subset), *_PW_FIT_ARGS, **_PW_FIT_KWARGS)
            yhat = model.predict(_PW_X_VALID)
            perfs[j] = _PW_EVAL(_PW_Y_VALID, yhat)

        if _PW_DEBUG:
            try:
                print(f"[Optimized Worker] pid={os.getpid()} batch_size={batch_size} subset_size={subset_size} (precomputed)")
            except Exception:
                pass
        return perfs

    def _aggregate_batch(self, subsets: np.ndarray, perfs: np.ndarray):
        # Update global totals
        self.sum_all_perf += float(perfs.sum())
        self.count_all += int(perfs.size)

        # Flatten indices and align values for vectorized accumulation
        flat_idx = subsets.reshape(-1)
        vals = np.repeat(perfs, subsets.shape[1])

        np.add.at(self.sum_included, flat_idx, vals)
        np.add.at(self.count_included, flat_idx, 1)

    def train_data_values(self, *args, **kwargs):
        # Prepare initializer globals
        eval_fn = getattr(self.evaluate, "function", self.evaluate)

        # Fit args need to be forwarded to workers via initializer
        init_args = (
            self.pred_model,
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            self.num_points,
            self.proportion,
            eval_fn,
            args,
            kwargs,
            self.debug,
            self._resolved_subset_size,
        )

        n_batches = math.ceil(self.num_models / self.batch_size)
        seeds = self.random_state.randint(0, 2**31 - 1, size=n_batches)

        start = time.perf_counter()
        processed = 0

        if self.precompute_subsets:
            # Precompute all subsets if memory allows
            try:
                all_subsets = np.empty((self.num_models, self._resolved_subset_size), dtype=np.int32)
                local_rng = np.random.RandomState(seeds[0])
                for i in range(self.num_models):
                    all_subsets[i] = local_rng.choice(self.num_points, self._resolved_subset_size, replace=False)
                if self.debug:
                    print(f"[Optimized] Precomputed subsets: shape={all_subsets.shape} bytes={all_subsets.nbytes}")
            except MemoryError:
                warnings.warn("Insufficient memory to precompute subsets; streaming instead.")
                self.precompute_subsets = False
                all_subsets = None
        else:
            all_subsets = None

        if self.use_joblib:
            if not _HAS_JOBLIB:
                warnings.warn("joblib not available; falling back to multiprocessing.Pool")
            else:
                # joblib path: prefer 'loky' processes; dispatch batches
                def _joblib_do_batch(batch_idx):
                    bs = self.batch_size if (batch_idx < n_batches - 1) else (self.num_models - batch_idx * self.batch_size)
                    if self.precompute_subsets:
                        start_i = batch_idx * self.batch_size
                        stop_i = start_i + bs
                        perfs = self._process_batch_with_subsets(all_subsets[start_i:stop_i])
                        return all_subsets[start_i:stop_i], perfs
                    else:
                        seed = int(seeds[batch_idx])
                        return self._process_batch_with_seeds((seed, bs, self._resolved_subset_size))

                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_joblib_do_batch)(b) for b in range(n_batches)
                )

                for subsets, perfs in tqdm.tqdm(results, total=len(results), desc=f"Optimized models (joblib, {self.n_jobs} workers)"):
                    self._aggregate_batch(subsets, perfs)
                    processed += perfs.size
                    if self.debug and processed % (self.batch_size * max(1, self.n_jobs)) == 0:
                        elapsed = time.perf_counter() - start
                        rate = processed / max(elapsed, 1e-9)
                        remaining = self.num_models - processed
                        eta = remaining / max(rate, 1e-9)
                        print(f"[Optimized] processed={processed}/{self.num_models} rate={rate:.1f} models/s ETA={eta/3600:.2f} h")
                return self

        # Default: multiprocessing.Pool with initializer; stream batches
        if self.debug:
            print(f"[Optimized] Spawning pool: n_jobs={self.n_jobs}, n_batches={n_batches}, batch_size={self.batch_size}")

        with Pool(
            processes=self.n_jobs,
            initializer=_parallel_worker_init,
            initargs=init_args,
            maxtasksperchild=1000,
        ) as pool:
            # Build tasks
            if self.precompute_subsets:
                # Send precomputed batch subsets to workers
                def batch_iter():
                    for b in range(n_batches):
                        start_i = b * self.batch_size
                        bs = self.batch_size if b < n_batches - 1 else (self.num_models - start_i)
                        yield all_subsets[start_i : start_i + bs]

                # Map and aggregate
                for perfs in tqdm.tqdm(
                    pool.imap_unordered(self._process_batch_with_subsets, batch_iter(), chunksize=1),
                    total=n_batches,
                    desc=f"Optimized models (pool, {self.n_jobs} workers)",
                ):
                    b = perfs.size
                    # Retrieve matching subsets window by processed offset
                    # Maintain a rolling pointer
                    # Simpler: recompute window from current processed
                    start_i = processed
                    subsets = all_subsets[start_i : start_i + b]
                    self._aggregate_batch(subsets, perfs)
                    processed += b
                    if self.debug and processed % (self.batch_size * max(1, self.n_jobs)) == 0:
                        elapsed = time.perf_counter() - start
                        rate = processed / max(elapsed, 1e-9)
                        remaining = self.num_models - processed
                        eta = remaining / max(rate, 1e-9)
                        print(f"[Optimized] processed={processed}/{self.num_models} rate={rate:.1f} models/s ETA={eta/3600:.2f} h")
            else:
                # Send seed batches; worker generates subsets
                def seed_tasks():
                    for b in range(n_batches):
                        bs = self.batch_size if b < n_batches - 1 else (self.num_models - b * self.batch_size)
                        yield (int(seeds[b]), int(bs), int(self._resolved_subset_size))

                for subsets, perfs in tqdm.tqdm(
                    pool.imap_unordered(self._process_batch_with_seeds, seed_tasks(), chunksize=1),
                    total=n_batches,
                    desc=f"Optimized models (pool, {self.n_jobs} workers)",
                ):
                    self._aggregate_batch(subsets, perfs)
                    processed += perfs.size
                    if self.debug and processed % (self.batch_size * max(1, self.n_jobs)) == 0:
                        elapsed = time.perf_counter() - start
                        rate = processed / max(elapsed, 1e-9)
                        remaining = self.num_models - processed
                        eta = remaining / max(rate, 1e-9)
                        print(f"[Optimized] processed={processed}/{self.num_models} rate={rate:.1f} models/s ETA={eta/3600:.2f} h")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        # Use derived excluded sums: sum_excluded[i] = sum_all - sum_included[i]
        # And counts: count_excluded[i] = count_all - count_included[i]
        sum_incl = self.sum_included
        cnt_incl = self.count_included
        sum_all = self.sum_all_perf
        cnt_all = self.count_all

        # Avoid division by zero
        msr_incl = np.divide(sum_incl, cnt_incl, out=np.zeros_like(sum_incl, dtype=np.float64), where=cnt_incl != 0)
        cnt_excl = cnt_all - cnt_incl
        sum_excl = sum_all - sum_incl
        msr_excl = np.divide(sum_excl, cnt_excl, out=np.zeros_like(sum_excl, dtype=np.float64), where=cnt_excl != 0)
        return msr_incl - msr_excl