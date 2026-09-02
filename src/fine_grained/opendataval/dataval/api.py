from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
import contextlib
import time
import sys
try:
    import resource  # Unix: ru_maxrss in kilobytes
except Exception:  # pragma: no cover
    resource = None
try:
    import psutil  # Optional: current RSS via psutil
except Exception:  # pragma: no cover
    psutil = None
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from opendataval.dataloader import DataFetcher
from opendataval.metrics import accuracy, neg_mse,neg_l2, recall,f1, balanced_acc
from opendataval.model import Model
from opendataval.util import ReprMixin

Self = TypeVar("Self", bound="DataEvaluator")


class DataEvaluator(ABC, ReprMixin):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    The following is an example of how the api would work:
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    Attributes
    ----------
    pred_model : Model
        Prediction model to find how much each training datum contributes towards it.
    data_values: np.array
        Cached data values, used by :py:mod:`opendataval.experiment.exper_methods`
    """

    Evaluators: ClassVar[dict[str, Self]] = {}

    def __init__(self, random_state: Optional[RandomState] = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def __init_subclass__(cls, *args, **kwargs):
        """Registers DataEvaluator types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Evaluators[cls.__name__.lower()] = cls

    def input_data(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: torch.Tensor,
        x_valid: Union[torch.Tensor, Dataset],
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for DataEvaluator.

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

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        return self

    def setup(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        exper_med=None,
    ):
        """Inputs model, metric and data into Data Evaluator.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        exper_med : ExperimentMediator, optional
            Reference to ExperimentMediator instance (used by some evaluators like TRAK)

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.input_fetcher(fetcher)

        if isinstance(self, ModelMixin):
            if metric is None:
                metric = accuracy if fetcher.one_hot else neg_mse
            self.input_model(pred_model).input_metric(metric)
        return self

    def train(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        *args,
        **kwargs,
    ):
        """Store and transform data, then train model to predict data values.

        Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        # Extract exper_med from kwargs for setup if present
        exper_med = kwargs.pop('exper_med', None)
        self.setup(fetcher, pred_model, metric, exper_med=exper_med)

        # Memory/time footprint tracking around training
        self._reset_peak_memory_stats()
        start_snapshot = self._memory_snapshot()
        t0 = time.perf_counter()
        self.train_data_values(*args, **kwargs)
        t1 = time.perf_counter()
        end_snapshot = self._memory_snapshot()
        self.memory_report = self._build_memory_report(start_snapshot, end_snapshot, t1 - t0)

        return self

    # train_and_evaluate method removed per user request; unified memory report is provided
    # by promoting memory_report to {"train": ..., "eval": ..., "combined": ...} after
    # evaluation via the data_values cached property.

    @abstractmethod
    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a trained Data Evaluator.
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """

    @cached_property
    def data_values(self) -> np.ndarray:
        """Cached data values with evaluation memory footprint.

        Captures CPU/GPU memory snapshots and elapsed time around
        `evaluate_data_values()` and stores a report in `self.memory_report_eval`.
        """
        # Memory/time footprint tracking around evaluation
        self._reset_peak_memory_stats()
        start_snapshot = self._memory_snapshot()
        t0 = time.perf_counter()
        values = self.evaluate_data_values()
        t1 = time.perf_counter()
        end_snapshot = self._memory_snapshot()
        eval_report = self._build_memory_report(start_snapshot, end_snapshot, t1 - t0)
        self.memory_report_eval = eval_report

        # If a train report exists, also produce a combined report and unify under memory_report
        train_report = getattr(self, "memory_report", None)
        if isinstance(train_report, dict) and "elapsed_seconds" in train_report and "train" not in train_report:
            combined = self._combine_reports(train_report, eval_report)
            # Promote to unified structure
            self.memory_report = {
                "train": train_report,
                "eval": eval_report,
                "combined": combined,
            }
        return values

    def _combine_reports(self, train_rep: dict, eval_rep: dict) -> dict:
        """Build a combined memory report from train and eval phase reports.

        Expects per-phase reports produced by _build_memory_report.
        """
        elapsed_total = 0.0
        try:
            elapsed_total = float(train_rep.get("elapsed_seconds", 0.0)) + float(eval_rep.get("elapsed_seconds", 0.0))
        except Exception:
            pass
        # CPU max RSS
        cpu_maxrss_start = train_rep.get("cpu_maxrss_kb") or train_rep.get("cpu_maxrss_kb_start")
        cpu_maxrss_end_candidates = []
        for rep in (train_rep, eval_rep):
            v = rep.get("cpu_maxrss_kb_end") or rep.get("cpu_maxrss_kb")
            if v is not None:
                cpu_maxrss_end_candidates.append(v)
        cpu_maxrss_end = max(cpu_maxrss_end_candidates) if cpu_maxrss_end_candidates else None

        cpu_rss_start = train_rep.get("cpu_rss_kb_start")
        cpu_rss_end = eval_rep.get("cpu_rss_kb_end", train_rep.get("cpu_rss_kb_end"))
        cpu_rss_delta = None
        try:
            if (cpu_rss_start is not None) and (cpu_rss_end is not None):
                cpu_rss_delta = int(cpu_rss_end) - int(cpu_rss_start)
        except Exception:
            pass

        # Phase peak across both: max of per-phase peaks or RSS delta
        phase_peaks = []
        for rep in (train_rep, eval_rep):
            v = rep.get("cpu_phase_peak_kb")
            if isinstance(v, int):
                phase_peaks.append(v)
        if isinstance(cpu_rss_delta, int):
            phase_peaks.append(cpu_rss_delta)
        cpu_phase_peak = max(phase_peaks) if phase_peaks else None

        # GPU fields
        cuda_device = eval_rep.get("cuda_device", train_rep.get("cuda_device"))
        gpu_alloc_end = eval_rep.get("gpu_allocated_bytes_end")
        gpu_res_end = eval_rep.get("gpu_reserved_bytes_end")
        def _max2(k: str):
            vals = [rep.get(k) for rep in (train_rep, eval_rep) if rep.get(k) is not None]
            return max(vals) if vals else None
        gpu_peak_alloc = _max2("gpu_peak_allocated_bytes")
        gpu_peak_res = _max2("gpu_peak_reserved_bytes")

        return {
            "elapsed_seconds": float(elapsed_total),
            "cpu_maxrss_kb_start": cpu_maxrss_start,
            "cpu_maxrss_kb_end": cpu_maxrss_end,
            "cpu_rss_kb_start": cpu_rss_start,
            "cpu_rss_kb_end": cpu_rss_end,
            "cpu_rss_delta_kb": cpu_rss_delta,
            "cpu_phase_peak_kb": cpu_phase_peak,
            "cuda_device": cuda_device,
            "gpu_allocated_bytes_end": gpu_alloc_end,
            "gpu_reserved_bytes_end": gpu_res_end,
            "gpu_peak_allocated_bytes": gpu_peak_alloc,
            "gpu_peak_reserved_bytes": gpu_peak_res,
        }

    def input_fetcher(self, fetcher: DataFetcher):
        """Input data from a DataFetcher object. Alternative way of adding data."""
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints

        # Fix label shape: convert [N, 1] to [N] for gradient-based evaluators
        # (InfluenceFunction, TRACIN, TRAK need 1D labels for cross_entropy)
        if self._should_squeeze_labels():
            if isinstance(y_train, torch.Tensor) and y_train.dim() > 1 and y_train.shape[1] == 1:
                y_train = y_train.squeeze(-1)
            if isinstance(y_valid, torch.Tensor) and y_valid.dim() > 1 and y_valid.shape[1] == 1:
                y_valid = y_valid.squeeze(-1)

        return self.input_data(x_train, y_train, x_valid, y_valid)

    def _should_squeeze_labels(self):
        """Check if this evaluator needs squeezed labels."""
        from opendataval.dataval.influence import InfluenceFunction, TracIn, TRAK
        return isinstance(self, (InfluenceFunction, TracIn, TRAK))

    # ------------------------------
    # Memory tracking helpers
    # ------------------------------
    def _reset_peak_memory_stats(self):
        """Reset GPU peak memory counters if CUDA is available."""
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _memory_snapshot(self) -> dict:
        """Capture a lightweight memory snapshot (CPU current+max RSS, GPU stats).

        Returns a dictionary with keys:
        - cpu_maxrss_kb: maximum resident set size (platform-dependent units; KB on Linux)
        - cpu_rss_kb: current resident set size (requires psutil)
        - gpu_allocated_bytes: current allocated bytes on active CUDA device
        - gpu_reserved_bytes: current reserved bytes on active CUDA device
        - gpu_max_allocated_bytes: peak allocated since last reset
        - gpu_max_reserved_bytes: peak reserved since last reset
        - cuda_device: active CUDA device index or None
        """
        snap = {
            "cpu_maxrss_kb": None,
            "cpu_rss_kb": None,
            "gpu_allocated_bytes": None,
            "gpu_reserved_bytes": None,
            "gpu_max_allocated_bytes": None,
            "gpu_max_reserved_bytes": None,
            "cuda_device": None,
        }
        # CPU max RSS
        if resource is not None and hasattr(resource, "getrusage"):
            try:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                snap["cpu_maxrss_kb"] = getattr(ru, "ru_maxrss", None)
            except Exception:
                pass
        # CPU current RSS
        if psutil is not None:
            try:
                p = psutil.Process()
                snap["cpu_rss_kb"] = int(p.memory_info().rss // 1024)
            except Exception:
                pass
        # GPU memory
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                snap["cuda_device"] = int(dev)
                torch.cuda.synchronize()
                snap["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated(dev))
                snap["gpu_reserved_bytes"] = int(torch.cuda.memory_reserved(dev))
                snap["gpu_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev))
                snap["gpu_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
            except Exception:
                pass
        return snap

    def _build_memory_report(self, start: dict, end: dict, elapsed_s: float) -> dict:
        """Construct a concise memory/time report for a phase (train or eval)."""
        report = {
            "elapsed_seconds": float(elapsed_s),
            "cpu_maxrss_kb_start": start.get("cpu_maxrss_kb"),
            "cpu_maxrss_kb_end": end.get("cpu_maxrss_kb"),
            "cpu_rss_kb_start": start.get("cpu_rss_kb"),
            "cpu_rss_kb_end": end.get("cpu_rss_kb"),
            "cpu_rss_delta_kb": None,
            "cpu_phase_peak_kb": None,
            "cuda_device": end.get("cuda_device", start.get("cuda_device")),
            "gpu_allocated_bytes_end": end.get("gpu_allocated_bytes"),
            "gpu_reserved_bytes_end": end.get("gpu_reserved_bytes"),
            "gpu_peak_allocated_bytes": end.get("gpu_max_allocated_bytes"),
            "gpu_peak_reserved_bytes": end.get("gpu_max_reserved_bytes"),
        }
        # Compute current RSS delta
        try:
            if (start.get("cpu_rss_kb") is not None) and (end.get("cpu_rss_kb") is not None):
                report["cpu_rss_delta_kb"] = int(end["cpu_rss_kb"]) - int(start["cpu_rss_kb"])
        except Exception:
            pass

        # Estimate CPU peak during phase: prefer increase in ru_maxrss; fallback to RSS delta
        try:
            ru_delta = None
            if (start.get("cpu_maxrss_kb") is not None) and (end.get("cpu_maxrss_kb") is not None):
                ru_delta = int(end["cpu_maxrss_kb"]) - int(start["cpu_maxrss_kb"])
            rss_delta = report.get("cpu_rss_delta_kb")
            candidates = [d for d in (ru_delta, rss_delta) if isinstance(d, int)]
            report["cpu_phase_peak_kb"] = max(0, max(candidates)) if candidates else None
        except Exception:
            pass
        return report


class ModelMixin:
    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor):
        """Evaluate performance of the specified metric between label and predictions.

        Moves input tensors to cpu because of certain bugs/errors that arise when the
        tensors are not on the same device

        Parameters
        ----------
        y : torch.Tensor
            Labels to be evaluate performance of predictions
        y_hat : torch.Tensor
            Predictions of labels

        Returns
        -------
        float
            Performance metric
        """
        return self.metric(y.cpu(), y_hat.cpu())

    def input_model(self, pred_model: Model):
        """Input the prediction model.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        """
        self.pred_model = pred_model.clone()
        return self

    def input_metric(
        self, pred_model: Optional[Model] = None, metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None
    ):
        """Input the evaluation metric (and optionally the prediction model).

        Can be called two ways:
        1. input_metric(metric) - sets metric on already-loaded model
        2. input_metric(pred_model, metric) - loads model and sets metric

        Parameters
        ----------
        pred_model : Model, optional
            Prediction model (if None, assumes already loaded)
        metric : Callable[[torch.Tensor, torch.Tensor], float], optional
            Evaluation function to determine prediction model performance

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        # Handle both calling conventions
        if metric is None and pred_model is not None:
            # Called as input_metric(metric) where pred_model is actually metric
            metric = pred_model
            pred_model = None

        if pred_model is not None:
            self.input_model(pred_model)

        if metric is not None:
            self.metric = metric

        return self


class ModelLessMixin:
    """Mixin for DataEvaluators without a prediction model and use embeddings.

    Using embeddings and then predictiong the data values has been used by
    Ruoxi Jia Group with their KNN Shapley and LAVA data evaluators.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Attributes
    ----------
    embedding_model : Model
        Embedding model used by model-less DataEvaluator to compute the data values for
        the embeddings and not the raw input.
    pred_model : Model
        The pred_model is unused for training, but to compare a series of models on
        the same algorithim, we compare against a shared prediction algorithim.
    """

    def embeddings(
        self, *tensors: tuple[Union[Dataset, torch.Tensor], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Returns Embeddings for the input tensors using batch processing

        Returns
        -------
        tuple[torch.Tensor, ...]
            Returns tupple of tensors equal to the number of tensors input
        """
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            import numpy as np
            from torch.utils.data import Dataset, Subset

            result = []
            batch_size = 1024  # H100-optimized batch size for faster embedding extraction

            for tensor in tensors:
                # Convert to numpy if needed
                if isinstance(tensor, torch.Tensor):
                    data_np = tensor.cpu().numpy()
                elif isinstance(tensor, (Dataset, Subset)):
                    # Convert Dataset/Subset to tensor by loading all samples
                    samples = []
                    for i in range(len(tensor)):
                        sample = tensor[i]
                        # Handle tuple (image, label) from Dataset
                        if isinstance(sample, tuple):
                            samples.append(sample[0])
                        else:
                            samples.append(sample)
                    data_tensor = torch.stack(samples) if isinstance(samples[0], torch.Tensor) else torch.tensor(samples, dtype=torch.float32)
                    data_np = data_tensor.cpu().numpy()
                else:
                    data_np = np.array(tensor)
                
                # Process in batches
                n_samples = len(data_np)
                embeddings_list = []
                
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    batch_data = data_np[batch_start:batch_end]
                    batch_embeddings = self.embedding_model.predict(batch_data)
                    embeddings_list.append(batch_embeddings)
                
                # Concatenate all batches
                if embeddings_list:
                    embeddings = np.vstack(embeddings_list)
                else:
                    embeddings = np.array([])
                
                # Convert back to tensor if original was tensor
                if isinstance(tensor, torch.Tensor):
                    embeddings = torch.from_numpy(embeddings).float()
                
                result.append(embeddings)
            
            return tuple(result)

        # No embedding is used
        return tensors
