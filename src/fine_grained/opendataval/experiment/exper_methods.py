"""Experiments to test :py:class:`~opendataval.dataval.api.DataEvaluator`.

Experiments pass into :py:meth:`~opendataval.experiment.api.ExperimentMediator.evaluate`
and :py:meth:`~opendataval.experiment.api.ExperimentMediator.plot` evaluate performance
of one :py:class:`~opendataval.dataval.api.DataEvaluator` at a time.
"""
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.utils.data import Subset

from opendataval.dataloader import DataFetcher
from opendataval.dataval import DataEvaluator
from opendataval.experiment.util import bottom_k_percent_indices, f1_score, oned_twonn_clustering, recall_exact_label
from opendataval.metrics import Metrics
from opendataval.model import Model
from opendataval.util import get_name


def noisy_detection(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    indices: Optional[list[int]] = None,
) -> dict[str, float]:
    """Evaluate ability to identify noisy indices.

    Compute F1 score (of 2NN classifier) of the data evaluator
    on the noisy indices. KMeans labels are random, but because of the convexity of
    KMeans, the highest data point and lowest data point have different labels and
    belong to the most valuable/least valuable group. Thus, the least valuable group
    will be in one group and most valuable to zero for the F1 score.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices
    indices : list[int], optional
        Alternatively, pass in noisy indices instead of DataFetcher, by default None

    Returns
    -------
    dict[str, float]

        - **"kmeans_f1"** -- F1 score performance of a 1D KNN binary classifier
            of the data points. Classifies the lower data value data points as
            corrupted, and the higher value data points as correct.
    """
    import torch

    print("\n" + "="*80)
    print("[NOISY DETECTION] Starting evaluation...")
    print("="*80)

    # === INPUT CONVERSION LAYER ===
    # Safely convert all inputs to numpy arrays with correct dtypes

    # Convert data_values: can be numpy array, torch tensor (CPU/GPU)
    data_values_raw = evaluator.data_values
    print(f"\n[INPUT] data_values_raw:")
    print(f"  Type: {type(data_values_raw)}")
    if isinstance(data_values_raw, torch.Tensor):
        print(f"  Device: {data_values_raw.device}")
        print(f"  Dtype: {data_values_raw.dtype}")
        print(f"  Shape: {data_values_raw.shape}")
        data_values = data_values_raw.detach().cpu().numpy().astype(np.float64)
        print(f"  → Converted to numpy float64 on CPU")
    else:
        print(f"  Dtype: {getattr(data_values_raw, 'dtype', 'N/A')}")
        print(f"  Shape: {getattr(data_values_raw, 'shape', 'N/A')}")
        data_values = np.asarray(data_values_raw, dtype=np.float64)
        print(f"  → Converted to numpy float64")

    # Flatten to 1D
    data_values = data_values.flatten()
    print(f"  → Flattened to shape: {data_values.shape}")

    # Convert noisy_train_indices: can be list, numpy array, torch tensor (CPU/GPU)
    noisy_train_indices_raw = (
        fetcher.noisy_train_indices if isinstance(fetcher, DataFetcher) else indices
    )
    print(f"\n[INPUT] noisy_train_indices_raw:")
    print(f"  Type: {type(noisy_train_indices_raw)}")
    if isinstance(noisy_train_indices_raw, torch.Tensor):
        print(f"  Device: {noisy_train_indices_raw.device}")
        print(f"  Dtype: {noisy_train_indices_raw.dtype}")
        print(f"  Shape: {noisy_train_indices_raw.shape}")
        noisy_train_indices = noisy_train_indices_raw.detach().cpu().numpy().astype(np.int64)
        print(f"  → Converted to numpy int64 on CPU")
    else:
        print(f"  Dtype: {getattr(noisy_train_indices_raw, 'dtype', 'N/A')}")
        print(f"  Length: {len(noisy_train_indices_raw)}")
        noisy_train_indices = np.asarray(noisy_train_indices_raw, dtype=np.int64)
        print(f"  → Converted to numpy int64")

    # Ensure contiguity for sklearn operations
    data_values = np.ascontiguousarray(data_values)
    noisy_train_indices = np.ascontiguousarray(noisy_train_indices)
    print(f"\n[MEMORY] Ensured contiguity for sklearn operations")

    # === CLUSTERING ===
    print(f"\n[CLUSTERING] Running oned_twonn_clustering...")
    unvaluable, valuable = oned_twonn_clustering(data_values)

    # Ensure clustering output is numpy int64
    unvaluable = np.asarray(unvaluable, dtype=np.int64)
    valuable = np.asarray(valuable, dtype=np.int64)
    print(f"  → Unvaluable indices: {len(unvaluable)} (type: {type(unvaluable)}, dtype: {unvaluable.dtype})")
    print(f"  → Valuable indices: {len(valuable)} (type: {type(valuable)}, dtype: {valuable.dtype})")

    # === F1 SCORE COMPUTATION ===
    print(f"\n[F1 SCORE] Computing f1_score...")
    print(f"  unvaluable: {len(unvaluable)} samples, dtype={unvaluable.dtype}")
    print(f"  noisy_train_indices: {len(noisy_train_indices)} samples, dtype={noisy_train_indices.dtype}")
    f1_kmeans_label = f1_score(unvaluable, noisy_train_indices, len(data_values))
    print(f"  → F1 Score: {f1_kmeans_label:.4f}")

    # === BOTTOM K INDICES ===
    print(f"\n[BOTTOM K] Computing bottom_k_percent_indices...")
    low_k = bottom_k_percent_indices(data_values, len(noisy_train_indices))  # type: ignore
    print(f"  → Type before conversion: {type(low_k)}, dtype: {getattr(low_k, 'dtype', 'N/A')}")
    # Ensure low_k is numpy int64 (may come as float from utility function)
    low_k = np.asarray(low_k, dtype=np.int64)
    low_k = np.ascontiguousarray(low_k)
    print(f"  → Type after conversion: {type(low_k)}, dtype: {low_k.dtype}, shape: {low_k.shape}")

    # === RECALL COMPUTATION ===
    print(f"\n[RECALL] Computing recall_exact_label...")
    print(f"  low_k: {len(low_k)} samples, dtype={low_k.dtype}")
    print(f"  noisy_train_indices: {len(noisy_train_indices)} samples, dtype={noisy_train_indices.dtype}")
    recall_exact_label_value = recall_exact_label(low_k, noisy_train_indices)
    print(f"  → Exact Recall: {recall_exact_label_value:.4f}")

    # === CLUSTER STATISTICS ===
    print(f"\n[STATISTICS] Computing cluster statistics...")
    def cluster_stats(cluster_indices, vals):
        """Compute size and mean value for a cluster."""
        cluster_indices = np.asarray(cluster_indices, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        values = [vals[i] for i in cluster_indices]
        return len(values), sum(values) / len(values) if values else 0.0

    size_low, mean_low = cluster_stats(unvaluable, data_values)
    size_high, mean_high = cluster_stats(valuable, data_values)
    print(f"  Low cluster: size={size_low}, mean={mean_low:.6f}")
    print(f"  High cluster: size={size_high}, mean={mean_high:.6f}")

    result = {
        "method": evaluator.__class__.__name__,
        "kmeans_f1": f1_kmeans_label,
        "size_low": size_low,
        "mean_low": mean_low,
        "size_high": size_high,
        "mean_high": mean_high,
        "Exact_recall": recall_exact_label_value,
    }

    print("\n" + "="*80)
    print("[NOISY DETECTION] Evaluation complete!")
    print("="*80 + "\n")

    return result


def remove_high_low(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
    max_removal_fraction: float = 0.2,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, list[float]]:
    """Evaluate performance after removing high/low points determined by data valuator.

    Repeatedly removes ``percentile`` of most valuable/least valuable data points
    and computes the performance of the metric.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing training and testing data points, by default None
    model : Model, optional
        Model which performance will be evaluated, if not defined,
        uses evaluator's model to evaluate performance if evaluator uses a model
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
        - **"y_train"** Training labels
        - **"x_test"** Testing covariates
        - **"y_test"** Testing labels
    percentile : float, optional
        Percentile of data points to remove per iteration, by default 0.05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric : Metrics | Callable[[Tensor, Tensor], float], optional
        Name of DataEvaluator defined performance metric which is one of the defined
        metrics or a Callable[[Tensor, Tensor], float], by default accuracy
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None
    max_removal_fraction : float, optional
        Maximum fraction of data to remove, by default 0.5
    output_dir : str | Path, optional
        Directory to save class distribution CSV file to, by default None.
        If provided, saves "class_distribution_remove_high_low.csv" with class counts
        at each removal level for both high and low removal types, by default None

    Returns
    -------
    dict[str, list[float]]
        dict containing list of the performance of the DataEvaluator
        ``(i * percentile)`` valuable/most valuable data points are removed

        - **"axis"** -- Proportion of data values removed currently
        - **f"remove_least_influential_first_{metric}"** -- Performance of model
            after removing a proportion of the data points with the lowest data values
        - **"f"remove_most_influential_first_{metric}""** -- Performance of model
            after removing a proportion of the data points with the highest data values
    """
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    print(f"[DEBUG-REMOVEHIGHLOW] len(x_train)={len(x_train)} type={type(x_train)} "
          f"len(data_values)={len(evaluator.data_values)} fetcher_is_DataFetcher={isinstance(fetcher, DataFetcher)} "
          f"percentile={percentile} max_removal_fraction={max_removal_fraction}")

    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()

    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)  # Add at least 5/bin
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list, axis_values = [], [], []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    # Track class distributions if output directory is provided
    class_dist_records = []
    if output_dir is not None:
        # Get labels and determine number of classes
        if isinstance(fetcher, DataFetcher):
            y_train_labels = fetcher.y_train
        else:
            y_train_labels = data["y_train"]

        # Convert torch tensor to numpy if needed
        if isinstance(y_train_labels, torch.Tensor):
            y_train_labels = y_train_labels.detach().cpu().numpy()

        # Handle one-hot encoded labels
        if len(y_train_labels.shape) > 1:
            y_train_class_idx = np.argmax(y_train_labels, axis=1)
        else:
            y_train_class_idx = y_train_labels

        n_classes = len(np.unique(y_train_class_idx))

    for bin_index in range(0, num_points, num_period):
        if bin_index / num_points > max_removal_fraction:
            break

        axis_value = bin_index / num_points
        axis_values.append(axis_value)

        # Removing least valuable samples first (remove LOW values)
        most_valuable_indices = sorted_value_list[bin_index:]

        # Track class distribution for "low" removal
        if output_dir is not None:
            remaining_classes = y_train_class_idx[most_valuable_indices]
            class_counts = np.bincount(remaining_classes.astype(int), minlength=n_classes)
            record = {
                "method": str(evaluator),
                "axis": axis_value,
                "removal_type": "low",
            }
            for cls_idx, count in enumerate(class_counts):
                record[f"class_{cls_idx}_count"] = int(count)
            class_dist_records.append(record)

        # Fitting on valuable subset
        valuable_model = curr_model.clone()
        # Extract tensors from subsets to avoid compatibility issues
        x_subset = x_train[most_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[most_valuable_indices])
        y_subset = y_train[most_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[most_valuable_indices])
        valuable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        y_hat_valid = valuable_model.predict(x_test).to("cpu")
        valuable_score = metric(y_test.to("cpu"), y_hat_valid)
        valuable_list.append(valuable_score)

        # Removing most valuable samples first (remove HIGH values)
        least_valuable_indices = sorted_value_list[: num_points - bin_index]

        # Track class distribution for "high" removal
        if output_dir is not None:
            remaining_classes = y_train_class_idx[least_valuable_indices]
            class_counts = np.bincount(remaining_classes.astype(int), minlength=n_classes)
            record = {
                "method": str(evaluator),
                "axis": axis_value,
                "removal_type": "high",
            }
            for cls_idx, count in enumerate(class_counts):
                record[f"class_{cls_idx}_count"] = int(count)
            class_dist_records.append(record)

        # Fitting on unvaluable subset
        unvaluable_model = curr_model.clone()
        # Extract tensors from subsets to avoid compatibility issues
        x_subset = x_train[least_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[least_valuable_indices])
        y_subset = y_train[least_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[least_valuable_indices])
        unvaluable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        iy_hat_valid = unvaluable_model.predict(x_test).to("cpu")
        unvaluable_score = metric(y_test, iy_hat_valid)
        unvaluable_list.append(unvaluable_score)

    # Use the exact per-step fractions tracked during the loop (bin_index/num_points),
    # not a reconstruction via num_points // num_period, which only equals the true
    # step size when num_points is evenly divisible by num_period.
    x_axis = axis_values

    # Save class distributions to CSV
    if output_dir is not None and class_dist_records:
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "class_distribution_remove_high_low.csv"
        df_class_dist = pd.DataFrame(class_dist_records)

        # Append to existing CSV if it exists (for multiple evaluators)
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df_class_dist = pd.concat([df_existing, df_class_dist], ignore_index=True)
            print(f"\n[Class Distribution] Appending to: {csv_path}")
        else:
            print(f"\n[Class Distribution] Saved to: {csv_path}")

        df_class_dist.to_csv(csv_path, index=False)

    eval_results = {
        f"remove_least_influential_first_{get_name(metric)}": valuable_list,
        f"remove_most_influential_first_{get_name(metric)}": unvaluable_list,
        "axis": x_axis,
    }

    # Plot graphs
    if plot is not None:
        # Prediction performances after removing high or low values
        plot.plot(x_axis, valuable_list, "o-")
        plot.plot(x_axis, unvaluable_list, "x-")

        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(get_name(metric))
        plot.legend(["Removing low value data", "Removing high value data"])

        plot.set_title(str(evaluator))

    return eval_results


def remove_high_low_exact(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
    max_removal_fraction: float = 0.2,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, list[float]]:
    """Same as ``remove_high_low``, but the axis uses exact fixed percentage
    steps (0%, 5%, 10%, ... up to ``max_removal_fraction``) instead of
    deriving the step from ``num_points // round(num_points * percentile)``.
    That derivation silently distorts the axis whenever ``num_points`` isn't
    evenly divisible by the rounded step size (e.g. produces 1/19 steps
    instead of the intended 1/20 = 5% steps for a 39,739-row training set).

    Parameters are identical to ``remove_high_low``; see that function's
    docstring for details.
    """
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()

    num_points = len(x_train)
    num_steps = int(round(max_removal_fraction / percentile))
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list, axis_values = [], [], []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    # Track class distributions if output directory is provided
    class_dist_records = []
    if output_dir is not None:
        if isinstance(fetcher, DataFetcher):
            y_train_labels = fetcher.y_train
        else:
            y_train_labels = data["y_train"]

        if isinstance(y_train_labels, torch.Tensor):
            y_train_labels = y_train_labels.detach().cpu().numpy()

        if len(y_train_labels.shape) > 1:
            y_train_class_idx = np.argmax(y_train_labels, axis=1)
        else:
            y_train_class_idx = y_train_labels

        n_classes = len(np.unique(y_train_class_idx))

    for step_i in range(num_steps + 1):
        axis_value = step_i * percentile
        if axis_value > max_removal_fraction + 1e-9:
            break
        bin_index = round(axis_value * num_points)
        axis_values.append(axis_value)

        # Removing least valuable samples first (remove LOW values)
        most_valuable_indices = sorted_value_list[bin_index:]

        if output_dir is not None:
            remaining_classes = y_train_class_idx[most_valuable_indices]
            class_counts = np.bincount(remaining_classes.astype(int), minlength=n_classes)
            record = {
                "method": str(evaluator),
                "axis": axis_value,
                "removal_type": "low",
            }
            for cls_idx, count in enumerate(class_counts):
                record[f"class_{cls_idx}_count"] = int(count)
            class_dist_records.append(record)

        valuable_model = curr_model.clone()
        x_subset = x_train[most_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[most_valuable_indices])
        y_subset = y_train[most_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[most_valuable_indices])
        valuable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        y_hat_valid = valuable_model.predict(x_test).to("cpu")
        valuable_score = metric(y_test.to("cpu"), y_hat_valid)
        valuable_list.append(valuable_score)

        # Removing most valuable samples first (remove HIGH values)
        least_valuable_indices = sorted_value_list[: num_points - bin_index]

        if output_dir is not None:
            remaining_classes = y_train_class_idx[least_valuable_indices]
            class_counts = np.bincount(remaining_classes.astype(int), minlength=n_classes)
            record = {
                "method": str(evaluator),
                "axis": axis_value,
                "removal_type": "high",
            }
            for cls_idx, count in enumerate(class_counts):
                record[f"class_{cls_idx}_count"] = int(count)
            class_dist_records.append(record)

        unvaluable_model = curr_model.clone()
        x_subset = x_train[least_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[least_valuable_indices])
        y_subset = y_train[least_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[least_valuable_indices])
        unvaluable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        iy_hat_valid = unvaluable_model.predict(x_test).to("cpu")
        unvaluable_score = metric(y_test, iy_hat_valid)
        unvaluable_list.append(unvaluable_score)

    x_axis = axis_values

    if output_dir is not None and class_dist_records:
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "class_distribution_remove_high_low.csv"
        df_class_dist = pd.DataFrame(class_dist_records)

        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df_class_dist = pd.concat([df_existing, df_class_dist], ignore_index=True)
            print(f"\n[Class Distribution] Appending to: {csv_path}")
        else:
            print(f"\n[Class Distribution] Saved to: {csv_path}")

        df_class_dist.to_csv(csv_path, index=False)

    eval_results = {
        f"remove_least_influential_first_{get_name(metric)}": valuable_list,
        f"remove_most_influential_first_{get_name(metric)}": unvaluable_list,
        "axis": x_axis,
    }

    if plot is not None:
        plot.plot(x_axis, valuable_list, "o-")
        plot.plot(x_axis, unvaluable_list, "x-")

        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(get_name(metric))
        plot.legend(["Removing low value data", "Removing high value data"])

        plot.set_title(str(evaluator))

    return eval_results


def remove_high_low_with_logs(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()

    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)
    num_bins = int(num_points // num_period)
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list = [], []
    valuable_loss_logs, unvaluable_loss_logs = [], []

    # Ensure return_logs is included
    train_kwargs = train_kwargs.copy() if train_kwargs else {}
    train_kwargs["return_logs"] = True

    for bin_index in range(0, num_points, num_period):
        most_valuable_indices = sorted_value_list[bin_index:]

        # Fitting on valuable subset
        valuable_model = curr_model.clone()
        # Extract tensors from subsets to avoid compatibility issues
        x_subset = x_train[most_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[most_valuable_indices])
        y_subset = y_train[most_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[most_valuable_indices])
        logs_val = valuable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        y_hat_valid = valuable_model.predict(x_test).to("cpu")
        valuable_score = metric(y_test.to("cpu"), y_hat_valid)
        valuable_list.append(valuable_score)
        valuable_loss_logs.append(logs_val["epoch_losses"])
        print(f"[Valuable bin {bin_index}] Score: {valuable_score:.4f}, Losses: {logs_val['epoch_losses']}")

        # Removing most valuable samples first
        least_valuable_indices = sorted_value_list[: num_points - bin_index]

        # Fitting on unvaluable subset
        unvaluable_model = curr_model.clone()
        # Extract tensors from subsets to avoid compatibility issues
        x_subset = x_train[least_valuable_indices] if isinstance(x_train, torch.Tensor) else torch.tensor(x_train[least_valuable_indices])
        y_subset = y_train[least_valuable_indices] if isinstance(y_train, torch.Tensor) else torch.tensor(y_train[least_valuable_indices])
        logs_unval = unvaluable_model.fit(
            x_subset,
            y_subset,
            **train_kwargs,
        )
        iy_hat_valid = unvaluable_model.predict(x_test).to("cpu")
        unvaluable_score = metric(y_test.to("cpu"), iy_hat_valid)
        unvaluable_list.append(unvaluable_score)
        unvaluable_loss_logs.append(logs_unval["epoch_losses"])
        print(f"[Unvaluable bin {bin_index}] Score: {unvaluable_score:.4f}, Losses: {logs_unval['epoch_losses']}")

    x_axis = [i / num_bins for i in range(num_bins)]

    eval_results = {
        f"remove_least_influential_first_{get_name(metric)}": valuable_list,
        f"remove_most_influential_first_{get_name(metric)}": unvaluable_list,
        "axis": x_axis,
        "valuable_loss_logs": valuable_loss_logs,
        "unvaluable_loss_logs": unvaluable_loss_logs,
    }

    if plot is not None:
        plot.plot(x_axis, valuable_list[:num_bins], "o-")
        plot.plot(x_axis, unvaluable_list[:num_bins], "x-")
        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(get_name(metric))
        plot.legend(["Removing low value data", "Removing high value data"])
        plot.set_title(str(evaluator))

    return eval_results

def discover_corrupted_sample(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
) -> dict[str, list[float]]:
    """Evaluate discovery of noisy indices in low data value points.

    Repeatedly explores ``percentile`` of the data values and determines
    if within that total percentile, what proportion of the noisy indices are found.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices, by default None
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
    percentile : float, optional
        Percentile of data points to additionally search per iteration, by default .05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing list of the proportion of noisy indices found after exploring
        the ``(i * percentile)`` least valuable data points. If plot is not None,
        also returns optimal and random search performances as lists

        - **"axis"** -- Proportion of data values explored currently.
        - **"corrupt_found"** -- Proportion of corrupted data values found currently
        - **"optimal"** -- Optimal proportion of corrupted values found currently
            meaning if the inspected **only** contained corrupted samples until
            the number of corrupted samples are completely exhausted.
        - **"random"** -- Random proportion of corrupted samples found, meaning
            if the data points were explored randomly, we'd expect to find
            corrupted_samples in proportion to the number of corruption in the data set.
    """
    if isinstance(fetcher, DataFetcher):
        x_train, *_ = fetcher.datapoints
    else:
        x_train = data["x_train"]
    noisy_train_indices = fetcher.noisy_train_indices
    data_values = evaluator.data_values

    num_points = len(x_train)
    num_period = max(1, num_points // 20)  # Ensure 5% steps (20 periods)
    num_bins = 20  # 5% increments (0%, 5%, 10%, ..., 100%)

    sorted_value_list = np.argsort(data_values, kind="stable")  # Order descending
    noise_rate = len(noisy_train_indices) / len(data_values)

    # Output initialization
    found_rates = []

    # For each bin
    for bin_index in range(0, num_points + num_period, num_period):
        # from low to high data values
        found_rates.append(
            len(np.intersect1d(sorted_value_list[:bin_index], noisy_train_indices))
            / len(noisy_train_indices)
        )

    x_axis = [i / num_bins for i in range(len(found_rates))]
    eval_results = {"corrupt_found": found_rates, "axis": x_axis}

    # Plot corrupted label discovery graphs
    if plot is not None:
        # Corrupted label discovery results (dvrl, optimal, random)
        y_dv = found_rates[:num_bins]
        y_opt = [min((i / num_bins / noise_rate, 1.0)) for i in range(len(found_rates))]
        y_random = x_axis

        eval_results["optimal"] = y_opt
        eval_results["random"] = y_random

        plot.plot(x_axis, y_dv, "o-")
        plot.plot(x_axis, y_opt, "--")
        plot.plot(x_axis, y_random, ":")
        plot.set_xlabel("Prop of data inspected")
        plot.set_ylabel("Prop of discovered corrupted samples")
        plot.legend(["Evaluator", "Optimal", "Random"])

        plot.set_title(str(evaluator))

    # Returns True Positive Rate of corrupted label discovery
    return eval_results


def save_dataval(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    indices: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
):
    """Save the indices, labels, and respective data values of the DataEvaluator."""
    import torch

    train_indices = (
        fetcher.train_indices if isinstance(fetcher, DataFetcher) else indices
    )
    data_values = evaluator.data_values

    # Extract labels from fetcher
    labels = None
    if isinstance(fetcher, DataFetcher):
        y_train = fetcher.y_train

        # Convert torch tensor to numpy if needed
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        # Handle one-hot encoded labels
        if len(y_train.shape) > 1:
            labels = np.argmax(y_train, axis=1)
        else:
            labels = y_train

    # Build data dictionary
    data = {
        "indices": train_indices,
        "data_values": data_values,
    }

    # Add labels if available
    if labels is not None:
        data["labels"] = labels

    if output_path:
        # Create DataFrame with indices, labels, and data values
        df_dict = {
            "index": train_indices,
            "label": labels if labels is not None else [None] * len(train_indices),
            "data_value": data_values,
        }
        df = pd.DataFrame(df_dict)
        df.to_csv(output_path, index=False)

    return data

def save_datavalv2(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    indices: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
):
    train_indices = (
        fetcher.train_indices if isinstance(fetcher, DataFetcher) else indices
    )
    data_values = evaluator.evaluate_data_values()

    # Case for LOORemovalRanker with subset ranking
    if hasattr(evaluator, "removal_ranking"):
        removal_ranking = evaluator.removal_ranking
        filtered_indices = np.array(train_indices)[removal_ranking]
        filtered_values = np.array(data_values)[removal_ranking]
    else:
        filtered_indices = train_indices
        filtered_values = data_values

    data = {
        str(evaluator): {
            "indices": list(filtered_indices),
            "data_values": list(filtered_values)
        }
    }

    if output_path:
        import pandas as pd
        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.explode(list(df.columns))
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")

    return data




def increasing_bin_removal(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    bin_size: int = 1,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    """Evaluate accuracy after removing data points with data values above threshold.

    For each subplot, displays the proportion of the data set with data values less
    than the specified data value (x-axis) and the performance of the model when all
    data values greater than the specified data value is removed. This implementation
    was inspired by V. Feldman and C. Zhang in their paper [1] where the same principle
    was applied to memorization functions.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing training and valid data points, by default None
    model : Model, optional
        Model which performance will be evaluated, if not defined,
        uses evaluator's model to evaluate performance if evaluator uses a model
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
        - **"y_train"** Training labels
        - **"x_test"** Testing covariates
        - **"y_test"** Testing labels
    bin_size : float, optional
        We look at bins of equal size and find the data values cutoffs for the x-axis,
        by default 1
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric : Metrics | Callable[[Tensor, Tensor], float], optional
        Name of DataEvaluator defined performance metric which is one of the defined
        metrics or a Callable[[Tensor, Tensor], float], by default accuracy
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing the thresholds of data values examined, proportion of training
        data points removed, and performance after those data points were removed.

        - **"axis"** -- Thresholds of data values examined. For a given threshold,
            considers the subset of data points with data values below.
        - **"frac_datapoints_explored"** -- Proportion of data points with data values
            below the specified threshold
        - **f"{metric}_at_datavalues"** -- Performance metric when data values
            above the specified threshold are removed
    """
    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    num_points = len(data_values)

    # Starts with 10 data points
    bins_indices = [*range(5, num_points - 1, bin_size), num_points - 1]
    frac_datapoints_explored = [(i + 1) / num_points for i in bins_indices]

    sorted_indices = np.argsort(data_values)
    x_axis = data_values[sorted_indices[bins_indices]] / np.max(data_values)

    perf = []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_end in bins_indices:
        coalition = sorted_indices[:bin_end]

        new_model = curr_model.clone()
        new_model.fit(
            Subset(x_train, coalition),
            Subset(y_train, coalition),
            **train_kwargs,
        )
        y_hat = new_model.predict(x_test)
        perf.append(metric(y_hat, y_test))

    eval_results = {
        "frac_datapoints_explored": frac_datapoints_explored,
        f"{get_name(metric)}_at_datavalues": perf,
        "axis": x_axis,
    }

    if plot is not None:  # Removing everything above this threshold
        plot.plot(x_axis, perf)

        plot.set_xticks([])
        plot.set_ylabel(get_name(metric))
        plot.set_title(str(evaluator))

        divider = make_axes_locatable(plot)
        frac_inspected_plot = divider.append_axes("bottom", size="40%", pad="5%")

        frac_inspected_plot.fill_between(x_axis, frac_datapoints_explored)
        frac_inspected_plot.set_xlabel("Data Values Threshold")
        frac_inspected_plot.set_ylabel("Trainset Fraction")

    return eval_results

def get_co_contrib_matrix(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None
) -> dict[str, np.ndarray]:
    """Returns the co-contribution matrix if supported by the evaluator."""
    sampler = getattr(evaluator, "sampler", None)

    if sampler is None:
        raise ValueError("Evaluator has no sampler attached.")

    if hasattr(sampler, "co_contrib_matrix"):
        return {"co_contrib_matrix": sampler.co_contrib_matrix}

    raise ValueError("Sampler does not support co-contribution matrix tracking.")

def get_shapley_logs(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None
) -> dict[str, np.ndarray]:
    """Returns the Shapley logs if supported by the evaluator."""
    sampler = getattr(evaluator, "sampler", None)

    if sampler is None:
        raise ValueError("Evaluator has no sampler attached.")

    if hasattr(sampler, "marginal_increment_array_stack"):
        return {"marginal_increment_array_stack": sampler.contribution_trace}

    raise ValueError("Sampler does not support Shapley logs tracking.")

