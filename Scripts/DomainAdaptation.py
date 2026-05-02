#!/usr/bin/env python3
"""
Domain Adaptation using Data Valuation
Complete SBATCH-ready experiment script
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# OpenDataVal imports
import opendataval.dataloader.datasets
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, LeaveOneOut, RandomEvaluator
)
from opendataval.experiment import ExperimentMediator
from opendataval.model.api import ClassifierSkLearnWrapper
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser("Domain Adaptation Data Valuation")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--method", type=str, default="InfluenceSubsample")
parser.add_argument("--job_id", type=int, default=1)
args = parser.parse_args()

SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id

np.random.seed(SEED)

print(f"SEED={SEED}, METHOD={METHOD}, JOB_ID={JOB_ID}")

# ============================================================
# Data preparation
# ============================================================

def prepare_source_target_data(Xs, Xt_val, Xt_test,
                               ys, yt_val, yt_test):

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    Xt_val = scaler.transform(Xt_val)
    Xt_test = scaler.transform(Xt_test)

    return Xs, Xt_val, Xt_test


def setup_fetcher(Xs, ys, Xt_val, yt_val, Xt_test, yt_test):

    def to_one_hot(y, k=7):
        return np.eye(k)[y.astype(int)]

    fetcher = DataFetcher.from_data_splits(
        x_train=Xs,
        y_train=to_one_hot(ys),
        x_valid=Xt_val,
        y_valid=to_one_hot(yt_val),
        x_test=Xt_test,
        y_test=to_one_hot(yt_test),
        one_hot=True,
        random_state=SEED
    )
    return fetcher


# ============================================================
# Evaluators
# ============================================================

def create_evaluators(method):
    if method == "InfluenceSubsample":
        proportions = [0.1, 0.2, 0.7]
        return [
            InfluenceSubsample(
                num_models=1000,
                proportion=p,
                random_state=SEED
            )
            for p in proportions
        ]
    if method == "AME":
        return [
            AME(num_models=20000, random_state=SEED)
        ]
    
    if method == "DataShapley":
        return [
            DataShapley(mc_epochs=135, random_state=SEED)
        ]
    if method == "DataBanzhaf":
        return [
            DataBanzhaf(num_models=80000, random_state=SEED)
        ]

    if method == "DVRL":
        seeds = range(10)
        rl_epochs_list = [1000, 3000, 5000]
        rl_batch_sizes = [32, 64, 128, 256, 512]

        return [
        DVRL(
            rl_epochs=epochs,
            rl_batch_size=batch_size,
            random_state=s
        )
        for s in seeds
        for epochs in rl_epochs_list
        for batch_size in rl_batch_sizes
    ]
    
    if method == "KNNShapley":
        return [KNNShapley(k_neighbors=5)]

    if method == "LAVA":
        return [
            LavaEvaluator(blur=0.05, lam_x=1, lam_y=0.1, debug=True)
        ]
    if method == "DataOob":
        proportions = [0.1, 0.2, 0.5, 0.7, 1.0]
        num_models_list = [10, 100, 1000]

        return [
            DataOob(
                num_models=num_models,
                proportion=p,
                random_state=SEED
            )
            for p in proportions
            for num_models in num_models_list
        ]
    if method == "LOO":
        return [LeaveOneOut()]  
    if method == "Random":
        return [RandomEvaluator(random_state=SEED)]
    if method == "ALL":
        return [
            RandomEvaluator(random_state=SEED),
            LeaveOneOut(),
            InfluenceSubsample(num_models=5000, proportion=0.2),
            KNNShapley(k_neighbors=50)
        ]

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Time / memory report
# ============================================================

def save_time_memory_report(exper_med, out_dir, method):

    rows = []

    for ev in exper_med.data_evaluators:
        name = str(ev)
        try:
            _ = ev.data_values
            rep = getattr(ev, "memory_report", {})
            train = rep.get("train", {})
            evalr = rep.get("eval", {})
            comb = rep.get("combined", {})

            rows.append({
                "method": name,
                "train_seconds": train.get("elapsed_seconds", 0),
                "eval_seconds": evalr.get("elapsed_seconds", 0),
                "combined_seconds": comb.get("elapsed_seconds", 0),
                "cpu_kb_train": train.get("cpu_phase_peak_kb", 0),
                "cpu_kb_eval": evalr.get("cpu_phase_peak_kb", 0),
                "gpu_bytes": comb.get("gpu_peak_allocated_bytes", 0),
                "status": "success"
            })
        except Exception as e:
            rows.append({"method": name, "status": f"failed: {str(e)[:80]}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"time_memory_{method}.csv"), index=False)


# ============================================================
# Custom evaluation: Weighted training on positive values
# ============================================================

def weighted_positive_training(exper_med, out_dir):
    """Train weighted LogisticRegression(s) using positively valued points for EACH evaluator.

    For every evaluator in `exper_med.data_evaluators`:
      - Read its `data_values`
      - Remove points with negative values
      - Train a weighted model on remaining points using values as sample weights
      - Evaluate accuracy on target validation and test sets
      - Append a summary row tagged with the evaluator name

    Saves a combined CSV: weighted_positive_training.csv with one row per evaluator.
    """

    # Access dataset splits
    x_train, y_train, x_valid, y_valid, x_test, y_test = exper_med.fetcher.datapoints

    # Convert one-hot labels to class indices if needed
    def to_class_indices(y):
        return np.argmax(y, axis=1) if y.ndim == 2 else y

    y_train_idx = to_class_indices(y_train)
    y_valid_idx = to_class_indices(y_valid)
    y_test_idx = to_class_indices(y_test)

    rows = []

    for ev in exper_med.data_evaluators:
        name = str(ev)
        try:
            dv = getattr(ev, "data_values", None)
            if dv is None:
                rows.append({
                    "evaluator": name,
                    "status": "no_data_values"
                })
                continue

            values = np.asarray(dv).reshape(-1)
            if values.shape[0] != x_train.shape[0]:
                rows.append({
                    "evaluator": name,
                    "status": "length_mismatch",
                    "values_len": int(values.shape[0]),
                    "train_len": int(x_train.shape[0])
                })
                continue

            # Filter: keep strictly positive values
            mask_pos = values > 0
            kept_count = int(mask_pos.sum())
            removed_count = int((~mask_pos).sum())

            if kept_count == 0:
                rows.append({
                    "evaluator": name,
                    "status": "no_positive_values",
                    "kept_count": 0,
                    "removed_count": removed_count
                })
                continue

            x_train_pos = x_train[mask_pos]
            y_train_pos = y_train_idx[mask_pos]
            sample_w = values[mask_pos]

            # Normalize weights to avoid extreme scales (strictly positive)
            w_min, w_max = float(sample_w.min()), float(sample_w.max())
            if w_max > w_min:
                sample_w = (sample_w - w_min) / (w_max - w_min) + 1e-8
            else:
                sample_w = np.ones_like(sample_w)

            # Train weighted LogisticRegression
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf.fit(x_train_pos, y_train_pos, sample_weight=sample_w)

            # Evaluate
            valid_acc = float(accuracy_score(y_valid_idx, clf.predict(x_valid)))
            test_acc = float(accuracy_score(y_test_idx, clf.predict(x_test)))

            rows.append({
                "evaluator": name,
                "evaluation": "WeightedPositiveTraining",
                "kept_count": kept_count,
                "removed_count": removed_count,
                "mean_positive_weight": float(sample_w.mean()),
                "valid_accuracy": valid_acc,
                "test_accuracy": test_acc,
                "status": "success"
            })
        except Exception as e:
            rows.append({
                "evaluator": name,
                "status": f"failed: {str(e)[:120]}"
            })

    if not rows:
        print("weighted_positive_training: No evaluators processed.")
        return

    summary = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "weighted_positive_training.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved weighted positive training summary to: {out_path}")


# ============================================================
# Main experiment
# ============================================================

def main():

    start = time.time()

    print("Loading datasets...")
    X_source = np.load("X_source.npy")
    y_source = np.load("y_source.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    Xs, Xt_val, Xt_test = prepare_source_target_data(
        X_source, X_val, X_test, y_source, y_val, y_test
    )

    fetcher = setup_fetcher(Xs, y_source, Xt_val, y_val, Xt_test, y_test)

    model = ClassifierSkLearnWrapper(LogisticRegression, fetcher.label_dim[0])

    exper_med = ExperimentMediator(fetcher, model, metric_name="accuracy")

    # Baseline
    data = exper_med.fetcher.datapoints
    m = exper_med.pred_model.clone()
    m.fit(data[0], data[1])
    baseline = exper_med.metric(m.predict(data[4]).cpu(), data[5])
    print(f"Baseline target accuracy: {baseline:.4f}")

    # Output directory
    out_dir =f"/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/domain adaptation/{METHOD}_SEED{SEED}_JOB{JOB_ID}"

    os.makedirs(out_dir, exist_ok=True)
    exper_med.set_output_directory(out_dir)

    evaluators = create_evaluators(METHOD)

    # Compute values
    exper_med.compute_data_values(evaluators)

    # Evaluations
    for func in [noisy_detection, discover_corrupted_sample, remove_high_low]:
        try:
            exper_med.evaluate(func, save_output=True)
        except Exception as e:
            print(f"Evaluation failed: {func.__name__}: {e}")

    # Custom evaluation: remove negatives, weight positives, train & report
    try:
        weighted_positive_training(exper_med, out_dir)
    except Exception as e:
        print(f"Custom evaluation failed: weighted_positive_training: {e}")

    # Save data values
    exper_med.evaluate(save_dataval, save_output=True)

    # Time/memory
    save_time_memory_report(exper_med, out_dir, METHOD)

    # Summary
    elapsed = time.time() - start
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Method: {METHOD}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Baseline accuracy: {baseline:.4f}\n")
        f.write(f"Total time: {elapsed/3600:.2f} hours\n")

    print("Experiment completed successfully.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
