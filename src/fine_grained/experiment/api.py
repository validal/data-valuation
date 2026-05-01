import math
import pathlib
import time
import warnings
from datetime import timedelta
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.utils import check_random_state

import numpy as np
import copy

from opendataval.dataloader import DataFetcher, mix_labels
from opendataval.dataval import DataEvaluator
from opendataval.experiment.util import filter_kwargs
from opendataval.metrics import Metrics
from opendataval.model import Model, ModelFactory


class ExperimentMediator:
    """Set up an experiment to compare a group of DataEvaluators.

    Attributes
    ----------
    timings : dict[str, timedelta]


    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher for the data set used for the experiment. All `exper_func` take a
        DataFetcher as an argument to have access to all data points and noisy indices.
    pred_model : Model
        Prediction model for the DataEvaluators
    train_kwargs : dict[str, Any], optional
        Training key word arguments for the prediction model, by default None
    metric_name : str | Metric | Callable[[Tensor, Tensor], float], optional
        Name of the performance metric used to evaluate the performance of the
        prediction model, by default accuracy
    output_dir: Union[str, pathlib.Path], optional
        Output directory of experiments
    raises_error: bool, optional
        Raises exception if one of the data evaluators fail, otherwise warns the user
        but continues computation. By default, False
    """

    def __init__(
        self,
        fetcher: DataFetcher,
        pred_model: Model,
        train_kwargs: Optional[dict[str, Any]] = None,
        metric_name: Optional[Union[str, Metrics, Callable]] = None,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        raises_error: bool = False,
    ):
        self.fetcher = fetcher
        self.pred_model = pred_model
        self.train_kwargs = {} if train_kwargs is None else train_kwargs

        if callable(metric_name):
            self.metric = metric_name
        elif metric_name is not None:
            self.metric = Metrics(metric_name)
        else:
            self.metric = Metrics.ACCURACY if self.fetcher.one_hot else Metrics.NEG_MSE
        self.data_evaluators = []

        if output_dir is not None:
            self.set_output_directory(output_dir)
        self.timings = {}
        self.raise_error = raises_error

    @classmethod
    def setup(
        cls,
        dataset_name: str,
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        force_download: bool = False,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise: Union[Callable[[DataFetcher], dict[str, Any]], str] = mix_labels,
        noise_kwargs: Optional[dict[str, Any]] = None,
        random_state: Optional[RandomState] = None,
        noise_random_state: Optional[RandomState] = None,
        pred_model: Optional[Model] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
        metric_name: Optional[Union[str, Metrics, Callable]] = None,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        raises_error: bool = False,
    ):
        """Create a DataFetcher from args and passes it into the init."""
        random_state = check_random_state(random_state)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs
        # Provide a separate RNG for noisification if given; otherwise use split RNG
        if noise_random_state is not None:
            noise_kwargs = {
                **noise_kwargs,
                "noise_random_state": check_random_state(noise_random_state),
            }
        else:
            # Explicitly forward the split RNG for noise when a separate one is not provided
            noise_kwargs = {
                **noise_kwargs,
                "noise_random_state": random_state,
            }

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise=add_noise,
            noise_kwargs=noise_kwargs,
        )

        return cls(
            fetcher=fetcher,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
            output_dir=output_dir,
            raises_error=raises_error,
        )

    @classmethod
    def model_factory_setup(
        cls,
        dataset_name: str,
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        force_download: bool = False,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise: Union[Callable[[DataFetcher], dict[str, Any]], str] = mix_labels,
        noise_kwargs: Optional[dict[str, Any]] = None,
        random_state: Optional[RandomState] = None,
        noise_random_state: Optional[RandomState] = None,
        model_name: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        train_kwargs: Optional[dict[str, Any]] = None,
        metric_name: Optional[Union[str, Metrics, Callable]] = None,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        raises_error: bool = False,
    ):
        """Set up ExperimentMediator from ModelFactory using an input string.

        Return a ExperimentMediator initialized with
        py:function`~opendataval.model.ModelFactory`

        Parameters
        ----------
        dataset_name : str
            Name of the data set, must be registered with
            :py:class:`~opendataval.dataloader.Register`
        cache_dir : Union[str, pathlib.Path], optional
            Directory of where to cache the loaded data, by default None which uses
            :py:attr:`Register.CACHE_DIR`
        force_download : bool, optional
            Forces download from source URL, by default False
        train_count : Union[int, float]
            Number/proportion training points
        valid_count : Union[int, float]
            Number/proportion validation points
        test_count : Union[int, float]
            Number/proportion test points
        add_noise : Callable
            If None, no changes are made. Takes as argument required arguments
            DataFetcher and adds noise to those the data points of DataFetcher as
            needed. Returns dict[str, np.ndarray] that has the updated np.ndarray in a
            dict to update the data loader with the following keys:

            - **"x_train"** -- Updated training covariates with noise, optional
            - **"y_train"** -- Updated training labels with noise, optional
            - **"x_valid"** -- Updated validation covariates with noise, optional
            - **"y_valid"** -- Updated validation labels with noise, optional
            - **"x_test"** -- Updated testing covariates with noise, optional
            - **"y_test"** -- Updated testing labels with noise, optional
            - **"noisy_train_indices"** -- Indices of training data set with noise
        noise_kwargs : dict[str, Any], optional
            Key word arguments passed to ``add_noise``, by default None
        random_state : RandomState, optional
            Random initial state, by default None
        model_name : str, optional
            Name of the preset model, check :py:func:`model_factory` for preset models,
            by default None
        device : torch.device, optional
            Tensor device for acceleration, by default torch.device("cpu")
        metric_name : str | Metric | Callable[[Tensor, Tensor], float], optional
            Name of the performance metric used to evaluate the performance of the
            prediction model, by default accuracy
        train_kwargs : dict[str, Any], optional
            Training key word arguments for the prediction model, by default None
        output_dir: Union[str, pathlib.Path]
            Output directory of experiments
        raises_error: bool, optional
            Raises exception if one of the data evaluators fail, otherwise warns the
            user but continues computation. By default, False

        Returns
        -------
        ExperimentMediator
            ExperimentMediator created from ModelFactory defaults
        """
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs
        # Provide a separate RNG for noisification if given; otherwise use split RNG
        if noise_random_state is not None:
            noise_kwargs = {
                **noise_kwargs,
                "noise_random_state": check_random_state(noise_random_state),
            }
        else:
            noise_kwargs = {
                **noise_kwargs,
                "noise_random_state": check_random_state(random_state),
            }

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise=add_noise,
            noise_kwargs=noise_kwargs,
        )

        pred_model = ModelFactory(
            model_name=model_name,
            fetcher=fetcher,
            device=device,
        )

        # Prints base line performance
        model = pred_model.clone()
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
        train_kwargs = {} if train_kwargs is None else train_kwargs
        
        logs = model.fit(x_train, y_train, **train_kwargs)
        if isinstance(logs, dict) and "epoch_losses" in logs:
            print("Epoch losses:", logs["epoch_losses"])
        else:
            print("logs",logs)

        if metric_name is None:
            metric = Metrics.ACCURACY if fetcher.one_hot else Metrics.NEG_MSE
        else:
            metric = Metrics(metric_name)
        perf = metric(y_test, model.predict(x_test).cpu())
        print(f"Base line model {metric_name=}: {perf=}")

        return cls(
            fetcher=fetcher,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
            output_dir=output_dir,
            raises_error=raises_error,
        )

    def compute_data_values(
        self, data_evaluators: list[DataEvaluator], *args, **kwargs
    ):
        """Computes the data values for the input data evaluators.

        Parameters
        ----------
        data_evaluators : list[DataEvaluator]
            List of DataEvaluators to be tested by `exper_func`
        """
        kwargs = {**kwargs, **self.train_kwargs}
        for data_val in data_evaluators:
            try:
                start_time = time.perf_counter()

                self.data_evaluators.append(
                    data_val.train(
                        self.fetcher, self.pred_model, self.metric, *args, **kwargs
                    )
                )

                end_time = time.perf_counter()
                delta = timedelta(seconds=end_time - start_time)

                self.timings[data_val] = delta

                print(f"Elapsed time {data_val!s}: {delta}")
            except Exception as ex:
                if self.raise_error:
                    raise ex

                warnings.warn(
                    f"""
                    An error occured during training, however training all evaluators
                    takes a long time, so we will be ignoring the evaluator:
                    {data_val!s} and proceeding.

                    The error is as follows: {ex!s}
                    """,
                    stacklevel=10,
                )

        self.num_data_eval = len(self.data_evaluators)
        return self

    
    def compute_data_values_groups(
    self,
    data_evaluators: list[DataEvaluator],
    K: int = 10,
    retrain: bool = True,
    save_output: bool = False,
    output_file: Optional[str] = None,
    min_group_size: int = 1,
    *args,
    **kwargs,
    ) -> pd.DataFrame:
        """Group-based summary of point-level data values."""
        import torch as _torch
        import traceback

        def _ensure_tensor(x, name="var"):
            """Convert any input into a torch tensor on CPU, with debug prints."""
            print(f"[DEBUG] _ensure_tensor({name}): type={type(x)}, "
                f"shape={getattr(x, 'shape', None)}")
            if isinstance(x, _torch.Tensor):
                return x.detach().cpu()
            return _torch.as_tensor(np.array(x)).cpu()

        kwargs = {**kwargs, **self.train_kwargs}
        results: list[dict[str, Any]] = []

        # Train evaluators first
        trained_evals = []
        for data_val in data_evaluators:
            try:
                start_time = time.perf_counter()
                trained = data_val.train(
                    self.fetcher, self.pred_model, self.metric, *args, **kwargs
                )
                trained_evals.append(trained)
                end_time = time.perf_counter()
                self.timings[data_val] = timedelta(seconds=end_time - start_time)
                print(f"Elapsed time {data_val!s}: {self.timings[data_val]}")
            except Exception as ex:
                if self.raise_error:
                    raise
                warnings.warn(
                    f"Evaluator {data_val!s} failed during training and will be skipped: {ex}",
                    stacklevel=10,
                )

        # Obtain valuations, rank & group
        for data_val in trained_evals:
            try:
                vals = None
                if hasattr(data_val, "data_values"):
                    vals = getattr(data_val, "data_values")
                elif hasattr(data_val, "get_data_values"):
                    vals = data_val.get_data_values(self.fetcher)
                else:
                    for fn in ("evaluate_data_values", "compute_data_values", "score_points"):
                        if hasattr(data_val, fn):
                            vals = getattr(data_val, fn)(
                                self.fetcher, self.pred_model, self.metric, *args, **kwargs
                            )
                            break
                if vals is None:
                    warnings.warn(
                        f"Could not obtain point-level values for evaluator {data_val!s}; skipping",
                        stacklevel=10,
                    )
                    continue
                vals = np.asarray(vals).reshape(-1)
                n_total = len(vals)
                print(f"[DEBUG] Evaluator {data_val}: obtained {n_total} data values")


                # Mapping to original indices
                orig_indices = None
                for attr in ("original_train_indices", "noisy_train_indices", "train_indices"):
                    oi = getattr(self.fetcher, attr, None)
                    if oi is not None and len(oi) == n_total:
                        orig_indices = np.asarray(oi)
                        break
                if orig_indices is None:
                    orig_indices = np.arange(n_total)

                order = np.argsort(-vals)  # high -> low
                sorted_vals = vals[order]
                sorted_orig = orig_indices[order]

                df_points = pd.DataFrame({
                    "index": sorted_orig,
                    "value": sorted_vals
                })

                # Name file per evaluator
                fname = f"data_values_{str(data_val)}.csv".replace("()", "").replace(" ", "_")
                save_path = (self.output_directory / fname) if hasattr(self, "output_directory") else pathlib.Path(fname)
                df_points.to_csv(save_path, index=False)
                groups_positions = np.array_split(np.arange(n_total), K)
                x_train, y_train, *rest = self.fetcher.datapoints

                for gid, pos in enumerate(groups_positions):
                    group_size = len(pos)
                    group_vals = sorted_vals[pos]

                    g_orig = sorted_orig[pos]
                    sum_values = float(np.sum(group_vals)) if group_size else 0.0
                    print(f"[DEBUG] Group {gid}: size={group_size}, sum_values={sum_values}, ")
                    
                    accuracy = None
                    if retrain and group_size >= min_group_size and hasattr(self, "pred_model"):
                        try:
                            try:
                                Xg = x_train[g_orig]
                                yg = y_train[g_orig]
                            except Exception:
                                Xg = x_train[pos]
                                yg = y_train[pos]

                            print(f"[DEBUG] Group {gid}: training subset shapes X={getattr(Xg,'shape',None)}, "
                                f"y={getattr(yg,'shape',None)}")

                            if len(Xg) >= min_group_size:
                                try:
                                    model = self.pred_model.clone()
                                except Exception:
                                    model = copy.deepcopy(self.pred_model)

                                try:
                                    _ = model.fit(Xg, yg, **self.train_kwargs)
                                except Exception as ex_fit:
                                    warnings.warn(
                                        f"Model training failed (evaluator={data_val}, group={gid}): {ex_fit}",
                                        stacklevel=10,
                                    )
                                else:
                                    # Pick validation/test set
                                    x_valid = getattr(self.fetcher, "x_valid", None)
                                    y_valid = getattr(self.fetcher, "y_valid", None)

                                    if x_valid is None or y_valid is None:
                                        try:
                                            *_, x_test, y_test = self.fetcher.datapoints
                                            x_valid, y_valid = x_test, y_test
                                            print(f"[DEBUG] Using test set for group {gid}")
                                        except Exception:
                                            x_valid, y_valid = x_train, y_train
                                            print(f"[DEBUG] Using training set for group {gid} (no test/valid found)")

                                    # 🔒 Convert validation data to tensors
                                    x_valid = _ensure_tensor(x_valid, name="x_valid")
                                    y_valid = _ensure_tensor(y_valid, name="y_valid")

                                    print(f"[DEBUG] Group {gid}: after ensure_tensor -> "
                                        f"x_valid type={type(x_valid)}, shape={x_valid.shape}")
                                    print(f"[DEBUG] Group {gid}: after ensure_tensor -> "
                                        f"y_valid type={type(y_valid)}, shape={y_valid.shape}")

                                    if x_valid is not None and y_valid is not None:
                                        try:
                                            preds = model.predict(x_valid)
                                            preds_t = _ensure_tensor(preds, name="preds")

                                            print(f"[DEBUG] Group {gid}: before argmax -> "
                                                f"preds_t.shape={preds_t.shape}, y_valid.shape={y_valid.shape}")

                                            # Handle labels safely
                                            if y_valid.ndim > 1 and y_valid.shape[1] > 1:
                                                print(f"[DEBUG] Group {gid}: applying argmax to y_valid, before={y_valid.shape}")
                                                y_valid = y_valid.argmax(dim=1)
                                                print(f"[DEBUG] Group {gid}: y_valid after argmax={y_valid.shape}")
                                            else:
                                                print(f"[DEBUG] Group {gid}: skipping argmax on y_valid, shape={y_valid.shape}")

                                            # Handle predictions safely
                                            if preds_t.ndim > 1 and preds_t.shape[1] > 1:
                                                print(f"[DEBUG] Group {gid}: applying argmax to preds_t, before={preds_t.shape}")
                                                preds_t = preds_t.argmax(dim=1)
                                                print(f"[DEBUG] Group {gid}: preds_t after argmax={preds_t.shape}")
                                            else:
                                                print(f"[DEBUG] Group {gid}: skipping argmax on preds_t, shape={preds_t.shape}")

                                            print(f"[DEBUG] Group {gid}: before metric -> preds_t.shape={preds_t.shape}, y_valid.shape={y_valid.shape}")

                                            accuracy = float(self.metric(y_valid, preds_t))

                                        except Exception as ex_eval:
                                            traceback.print_exc()
                                            warnings.warn(
                                                f"Evaluation failed (evaluator={data_val}, group={gid}): {ex_eval}",
                                                stacklevel=10,
                                            )
                        except Exception as ex_group:
                            warnings.warn(
                                f"Group retraining error (evaluator={data_val}, group={gid}): {ex_group}",
                                stacklevel=10,
                            )

                    results.append(
                        {
                            "method": str(data_val),
                            "group": int(gid),
                            "sum_values": sum_values,
                            "accuracy": accuracy,
                            "n_points": int(group_size),
                        }
                    )

            except Exception as ex_eval_loop:
                warnings.warn(
                    f"Failed processing evaluator {data_val!s}: {ex_eval_loop}",
                    stacklevel=10,
                )

        df = pd.DataFrame(results)
        if save_output:
            if output_file is None:
                output_file = "data_values_groups.csv"
            if hasattr(self, "output_directory"):
                df.to_csv(self.output_directory / output_file, index=False)
            else:
                warnings.warn(
                    "Output directory not set; grouped data values not saved",
                    stacklevel=10,
                )
        return df

 

    def plot(
        self,
        exper_func: Callable[..., dict[str, Any]],
        figure: Optional[Figure] = None,
        row: Optional[int] = None,
        col: int = 2,
        save_output: bool = False,
        **exper_kwargs,
    ) -> tuple[pd.DataFrame, Figure]:
        """Evaluate `exper_func` on each DataEvaluator and plots result in `fig`.

        Run an experiment on a list of pre-train DataEvaluators and their
        corresponding dataset and plots the result.

        Parameters
        ----------
        exper_func : Callable[..., dict[str, Any]]
            Experiment function, runs an experiment on a DataEvaluator and the data of
            the DataFetcher associated. Output must be a dict with results of the
            experiment. NOTE, the results must all be <= 1 dimensional but does not
            need to be the same length.
        fig : Figure, optional
            MatPlotLib Figure which each experiment result is plotted, by default None
        row : int, optional
            Number of rows of subplots in the plot, by default set to num_evaluators/col
        col : int, optional
            Number of columns of subplots in the plot, by default 2
        save_output : bool, optional
            Wether to save the outputs to ``self.output_dir``, by default False
        eval_kwargs : dict[str, Any], optional
            Additional key word arguments to be passed to the exper_func

        Returns
        -------
        tuple[pd.DataFrame, Figure]
            DataFrame containing the results for each DataEvaluator experiment.
            DataFrame is indexed: [DataEvaluator.DataEvaluator]

            Figure is a plotted version of the results dict.
        """
        if figure is None:
            figure = plt.figure(figsize=(15, 15))

        if not row:
            row = math.ceil(getattr(self, 'num_data_eval', len(self.data_evaluators)) / col) or 1

        data_eval_perf = {}
        filtered_kwargs = filter_kwargs(
            exper_func,
            train_kwargs=self.train_kwargs,
            metric=self.metric,
            model=self.pred_model,
            plot="placeholder",  # Place holder to confirm exper_func is plotable
            **exper_kwargs,
        )

        for i, data_val in enumerate(self.data_evaluators, start=1):
            if "plot" in filtered_kwargs:
                filtered_kwargs["plot"] = figure.add_subplot(row, col, i)
            try:
                eval_resp = exper_func(data_val, self.fetcher, **filtered_kwargs)
            except Exception as ex:
                warnings.warn(f"Experiment function failed for evaluator {data_val}: {ex}")
                continue

            data_eval_perf[str(data_val)] = eval_resp

        df_resp = pd.DataFrame.from_dict(data_eval_perf, "index")
        if not df_resp.empty:
            df_resp = df_resp.explode(list(df_resp.columns))

        if save_output:
            # Include method names as a column when saving so they're not lost
            if not df_resp.empty:
                df_save = df_resp.copy()
                df_save = df_save.reset_index()
                if 'index' in df_save.columns and df_save.columns[0] == 'index':
                    df_save = df_save.rename(columns={'index': 'method'})
            else:
                df_save = df_resp
            self.save_output(f"{exper_func.__name__}.csv", df_save)
        return df_resp, figure

    def evaluate(
        self,
        exper_func: Callable[..., dict[str, Any]],
        save_output: bool = False,
        **exper_kwargs,
    ) -> pd.DataFrame:
        """Evaluate `exper_func` on each DataEvaluator and return a DataFrame.

        This is a non-plotting counterpart to `plot`. It runs the provided
        experiment function for each trained data evaluator and aggregates the
        returned dictionaries into a single DataFrame.

        Parameters
        ----------
        exper_func : Callable[..., dict[str, Any]]
            Experiment function that accepts (evaluator, fetcher, ...) and returns
            a dictionary of results.
        save_output : bool, optional
            Whether to save the outputs to ``self.output_directory`` using the
            experiment function name as the CSV filename, by default False.
        exper_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to the experiment function.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results for each DataEvaluator experiment.
            DataFrame is indexed by the string form of each DataEvaluator and
            exploded so columns align with scalar values.
        """
        data_eval_perf: dict[str, dict[str, Any]] = {}

        filtered_kwargs = filter_kwargs(
            exper_func,
            train_kwargs=self.train_kwargs,
            metric=self.metric,
            model=self.pred_model,
            **exper_kwargs,
        )

        for data_val in self.data_evaluators:
            try:
                eval_resp = exper_func(data_val, self.fetcher, **filtered_kwargs)
            except Exception as ex:
                warnings.warn(f"Experiment function failed for evaluator {data_val}: {ex}")
                continue
            data_eval_perf[str(data_val)] = eval_resp

        df_resp = pd.DataFrame.from_dict(data_eval_perf, orient="index")
        if not df_resp.empty:
            df_resp = df_resp.explode(list(df_resp.columns))

        if save_output:
            # Include method names as a column when saving so they're not lost
            if not df_resp.empty:
                df_save = df_resp.copy()
                df_save = df_save.reset_index()
                if 'index' in df_save.columns and df_save.columns[0] == 'index':
                    df_save = df_save.rename(columns={'index': 'method'})
            else:
                df_save = df_resp
            self.save_output(f"{exper_func.__name__}.csv", df_save)
        return df_resp

    def train_and_evaluate(
        self, save_output: bool = False, output_file: Optional[str] = None
    ) -> dict[str, Optional[float]]:
        """Train the mediator's model on the training set and report metrics.

        Trains a fresh clone of ``self.pred_model`` (or a deep copy if clone is
        unavailable) on the fetcher's training set and evaluates the configured
        ``self.metric`` on the validation and test sets when available. Results
        are printed and optionally saved to ``self.output_directory`` via
        ``save_output``.

        Returns
        -------
        dict[str, Optional[float]]
            Dictionary with keys ``'valid'`` and ``'test'`` containing the
            computed metric values or ``None`` when a set is not available.
        """
        import copy
        import pandas as _pd

        # Instantiate a model to train
        try:
            model = self.pred_model.clone()
        except Exception:
            model = copy.deepcopy(self.pred_model)

        # Obtain datapoints
        x_train, y_train, *rest = self.fetcher.datapoints

        # Train
        logs = model.fit(x_train, y_train, **self.train_kwargs)
        if isinstance(logs, dict) and "epoch_losses" in logs:
            print("Epoch losses:", logs["epoch_losses"])
        else:
            print("logs", logs)

        # Gather validation/test sets
        x_valid = getattr(self.fetcher, "x_valid", None)
        y_valid = getattr(self.fetcher, "y_valid", None)
        x_test = getattr(self.fetcher, "x_test", None)
        y_test = getattr(self.fetcher, "y_test", None)

        # If not present, try to infer from the remaining datapoints
        try:
            if (x_valid is None or y_valid is None) and len(rest) >= 2:
                # Prefer the first pair in rest as valid
                x_valid = x_valid or rest[0]
                y_valid = y_valid or rest[1]
        except Exception:
            pass

        try:
            if (x_test is None or y_test is None) and len(rest) >= 2:
                # Prefer the last pair in rest as test
                x_test = x_test or rest[-2]
                y_test = y_test or rest[-1]
        except Exception:
            pass

        def _as_float(val):
            try:
                if hasattr(val, "item"):
                    return float(val.item())
                return float(val)
            except Exception:
                return None

        results: dict[str, Optional[float]] = {"valid": None, "test": None}

        if x_valid is not None and y_valid is not None:
            try:
                pred_valid = model.predict(x_valid)
                # Ensure tensors for metric functions (which expect torch.Tensor)
                import torch as _torch
                import numpy as _np

                def _to_torch(x, device=None):
                    if isinstance(x, _torch.Tensor):
                        return x.to(device) if device is not None else x.cpu()
                    # numpy -> tensor
                    if isinstance(x, _np.ndarray):
                        t = _torch.as_tensor(x)
                    else:
                        t = _torch.as_tensor(_np.array(x))
                    return t.to(device) if device is not None else t.cpu()

                # If prediction is tensor, use its device; else default to CPU
                pred_dev = None
                if isinstance(pred_valid, _torch.Tensor):
                    pred_dev = pred_valid.device
                pred_t = _to_torch(pred_valid, device=pred_dev)
                y_t = _to_torch(y_valid, device=pred_dev)

                perf_valid = self.metric(y_t, pred_t)
                perf_valid = _as_float(perf_valid)
                print(f"Validation metric ({self.metric}): {perf_valid}")
                results["valid"] = perf_valid
            except Exception as ex:
                print(f"Validation evaluation failed: {ex}")

        if x_test is not None and y_test is not None:
            try:
                pred_test = model.predict(x_test)
                import torch as _torch
                import numpy as _np

                def _to_torch(x, device=None):
                    if isinstance(x, _torch.Tensor):
                        return x.to(device) if device is not None else x.cpu()
                    if isinstance(x, _np.ndarray):
                        t = _torch.as_tensor(x)
                    else:
                        t = _torch.as_tensor(_np.array(x))
                    return t.to(device) if device is not None else t.cpu()

                pred_dev = None
                if isinstance(pred_test, _torch.Tensor):
                    pred_dev = pred_test.device
                pred_t = _to_torch(pred_test, device=pred_dev)
                y_t = _to_torch(y_test, device=pred_dev)

                perf_test = self.metric(y_t, pred_t)
                perf_test = _as_float(perf_test)
                print(f"Test metric ({self.metric}): {perf_test}")
                results["test"] = perf_test
            except Exception as ex:
                print(f"Test evaluation failed: {ex}")

        if save_output:
            df = _pd.DataFrame(
                [
                    {
                        "metric": str(self.metric),
                        "valid": results["valid"],
                        "test": results["test"],
                    }
                ]
            )
            if output_file is None:
                output_file = "model_performance.csv"
            self.save_output(output_file, df)

        return results

    def set_output_directory(self, output_directory: Union[str, pathlib.Path]):
        """Set directory to save output of experiment."""
        if isinstance(output_directory, str):
            output_directory = pathlib.Path(output_directory)
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self

    def save_output(self, file_name: str, df: pd.DataFrame):
        """Saves the output of the DataFrame to f"{self.output_directory}/{file_name}".

        Parameters
        ----------
        file_name : str
            Name of the file to save the DataFrame to.
        df : pd.DataFrame
            Output DataFrame from an experiment run by ExperimentMediator
        """
        if not hasattr(self, "output_directory"):
            warnings.warn("Output directory not set, output has not been saved")
            return
        # Avoid overwriting existing files: if exists, append incrementing suffix
        out_dir = self.output_directory
        out_dir.mkdir(parents=True, exist_ok=True)
        base = pathlib.Path(file_name)
        stem, suffix = base.stem, base.suffix or ".csv"
        candidate = out_dir / (stem + suffix)
        if candidate.exists():
            i = 1
            while True:
                new_name = out_dir / f"{stem}_{i}{suffix}"
                if not new_name.exists():
                    candidate = new_name
                    break
                i += 1
        df.to_csv(candidate, index=False)
        return candidate
