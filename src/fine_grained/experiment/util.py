import inspect
from itertools import accumulate
from typing import Any, Callable, Sequence

import numpy as np


def filter_kwargs(func: Callable, **kwargs) -> dict[str, Any]:
    """Filters out non-arguments of a specific function out of kwargs.

    Parameters
    ----------
    func : Callable
        Function with a specified signature, whose kwargs can be extracted from kwargs
    kwargs : dict[str, Any]
        Key word arguments passed to the function

    Returns
    -------
    dict[str, Any]
        Key word arguments of func that are passed in as kwargs
    """
    params = inspect.signature(func).parameters.values()
    filter_keys = [p.name for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
    return {key: kwargs[key] for key in filter_keys if key in kwargs}


def oned_twonn_clustering(vals: Sequence[float]) -> tuple[Sequence[int], Sequence[int]]:
    """O(nlog(n)) sort, O(n) pass exact 2-NN clustering of 1 dimensional input data.

    References
    ----------
    .. [1] A. Grønlund, K. G. Larsen, A. Mathiasen, J. S. Nielsen, S. Schneider,
        and M. Song,
        Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D,
        arXiv.org, 2017. https://arxiv.org/abs/1701.07204.

    Parameters
    ----------
    vals : Sequence[float]
        Input floats which to cluster

    Returns
    -------
    tuple[Sequence[int], Sequence[int]]
        Indices of the data points in each cluster, because of the convexity of KMeans,
        the first sequence represents the lower value group and the second the higher
    """
    sid = np.argsort(vals, kind="stable")
    n = len(vals)

    psums = list(accumulate((vals[sid[i]] for i in range(n)), initial=0.0))
    psqsums = list(accumulate((vals[sid[i]] ** 2 for i in range(n)), initial=0.0))

    def cost(i: int, j: int):
        sij = psums[j + 1] - psums[i]
        uij = sij / (j - i + 1)
        return (uij**2) * (j - i + 1) + (psqsums[j + 1] - psqsums[i]) - 2 * uij * sij

    split = min((i for i in range(1, n)), key=lambda i: cost(0, i - 1) + cost(i, n - 1))
    return sid[range(0, split)], sid[range(split, n)]


def oned_threenn_clustering(vals: Sequence[float]) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """
    O(n^2) exact 3-cluster 1D clustering based on minimizing intra-cluster variance.

    Parameters
    ----------
    vals : Sequence[float]
        Input 1D data to be clustered.

    Returns
    -------
    tuple of three Sequences[int]
        Indices of data points in each cluster, ordered by increasing cluster mean.
    """
    vals = list(vals)
    sid = np.argsort(vals, kind="stable")
    n = len(vals)

    # Compute prefix sums and prefix squared sums
    psums = [0.0] + list(accumulate(vals[sid[i]] for i in range(n)))
    psqsums = [0.0] + list(accumulate(vals[sid[i]] ** 2 for i in range(n)))

    def cost(i: int, j: int) -> float:
        """Compute squared error for cluster from index i to j (inclusive)"""
        if i > j:
            return float('inf')
        s = psums[j + 1] - psums[i]
        ssq = psqsums[j + 1] - psqsums[i]
        count = j - i + 1
        mean = s / count
        return ssq - 2 * mean * s + count * mean ** 2

    # Try all (i, j) such that 0 < i < j < n
    best_split = None
    best_cost = float('inf')

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            total_cost = cost(0, i - 1) + cost(i, j - 1) + cost(j, n - 1)
            if total_cost < best_cost:
                best_cost = total_cost
                best_split = (i, j)

    i, j = best_split
    cluster1 = sid[0:i]
    cluster2 = sid[i:j]
    cluster3 = sid[j:n]

    # Sort clusters by mean
    means = [
        np.mean([vals[idx] for idx in cluster1]),
        np.mean([vals[idx] for idx in cluster2]),
        np.mean([vals[idx] for idx in cluster3]),
    ]
    clusters = [cluster1, cluster2, cluster3]
    ordered_clusters = [c for _, c in sorted(zip(means, clusters))]

    return tuple(ordered_clusters)


def f1_score(predicted: Sequence[float], actual: Sequence[float], total: int) -> float:
    """Computes the F1 score based on the indices of values found."""
    predicted_set, actual_set = set(predicted), set(actual)

    tp, fp, fn = 0, 0, 0
    for i in range(total):
        if i in predicted_set and i in actual_set:
            tp += 1
        elif i in predicted_set:
            fp += 1
        elif i in actual_set:
            fn += 1
    return 2 * tp / (2 * tp + fp + fn)

def bottom_k_percent_indices(values: Sequence[float], k_percent: float) -> list[int]:
    """Return indices of the lowest k% elements from values."""
    return list(np.argsort(values)[:k_percent])

def recall_exact_label(predicted_unvaluable: Sequence[int], actual_noisy: Sequence[int]) -> float:
    """Computes recall for noisy label detection."""
    predicted_set, actual_set = set(predicted_unvaluable), set(actual_noisy)
    tp = len(predicted_set & actual_set)
    fn = len(actual_set - predicted_set)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def precision_recall_f1(predicted, actual, total):
    """
    Computes precision, recall, and F1 score based on the indices of detected values.

    Parameters
    ----------
    predicted : Sequence[int]
        Indices of predicted 'positive' (noisy) samples.
    actual : Sequence[int]
        Indices of true 'positive' (noisy) samples.
    total : int
        Total number of data points.

    Returns
    -------
    tuple[float, float, float]
        (precision, recall, f1)
    """
    predicted_set, actual_set = set(predicted), set(actual)

    tp = len(predicted_set & actual_set)
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)

    # handle division-by-zero gracefully
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

