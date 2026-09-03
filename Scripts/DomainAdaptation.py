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
import io
import re
import json
import contextlib
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle as sk_shuffle
import torch

# OpenDataVal imports
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, RandomEvaluator,
    InRunDataShapleyGhost, LoGRA
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.dataval.knnshap import KNNShapleyLSH
from opendataval.experiment import ExperimentMediator
from opendataval.model import ClassifierMLP
from opendataval.experiment.exper_methods import (
    remove_high_low, save_dataval
)

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser("Domain Adaptation Data Valuation")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--method", type=str, default="InfluenceSubsample")
parser.add_argument("--job_id", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=7, help="Number of classes (default: 7, HAM10000)")

# Method-specific hyperparameters (mirrors run_adult_dataval.py)
parser.add_argument("--proportion", type=float, default=None, help="Bootstrap proportion for DataOob")
parser.add_argument("--noise_rate", type=float, default=None, help="Label noise rate, if noisifying source labels")
parser.add_argument("--k_neighbors", type=int, default=None, help="K for KNNShapley (default: 5)")
parser.add_argument("--subset_size", type=int, default=None, help="Subset size for InfluenceSubsample")
parser.add_argument("--num_models", type=int, default=None, help="Number of models (AME, DataBanzhaf, BetaShapley, etc)")
parser.add_argument("--num_models_list", type=str, default=None, help="Comma-separated num_models for DataOob")
parser.add_argument("--mc_epochs", type=int, default=None, help="MC epochs for DataShapley")
parser.add_argument("--lam_y", type=float, default=None, help="Lambda Y for LAVA/SAVA")
parser.add_argument("--lambda_weight", type=float, default=None, help="Lambda weight for Kairos (default: 0.97)")
parser.add_argument("--lam_x", type=float, default=None, help="Lambda X for LAVA/SAVA")
parser.add_argument("--blur", type=float, default=None, help="Blur parameter for LAVA/SAVA")
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples for Kairos")
parser.add_argument("--rl_epochs", type=int, default=None, help="RL epochs for DVRL")
parser.add_argument("--rl_batch_size", type=int, default=None, help="RL batch size for DVRL")
parser.add_argument("--ak_k_neighbors", type=int, default=None, help="k_neighbors for AKShapley (default: 100)")
parser.add_argument("--ak_n_hash_table", type=int, default=None, help="n_hash_table for AKShapley (default: 100)")
parser.add_argument("--ak_eps", type=float, default=None, help="eps for AKShapley (default: 1/k_neighbors)")
parser.add_argument("--ak_alpha", type=float, default=None, help="alpha for AKShapley (default: 0.5)")
parser.add_argument("--ak_dist_rand", type=float, default=None, help="dist_rand for AKShapley (default: self-calibrated)")
parser.add_argument("--ak_t", type=float, default=None, help="t for AKShapley (default: self-calibrated)")
# List variants: sweep multiple values of one param within a single job/evaluator
# list, same pattern as --num_models_list for DataOob. k_neighbors stays singular
# (one per job/submission); alpha/n_hash_table/eps can each be swept together,
# producing a cross product of evaluators for that one k_neighbors value.
parser.add_argument("--ak_n_hash_table_list", type=str, default=None, help="Comma-separated n_hash_table values for AKShapley, run together in one job")
parser.add_argument("--ak_eps_list", type=str, default=None, help="Comma-separated eps values for AKShapley, run together in one job")
parser.add_argument("--ak_alpha_list", type=str, default=None, help="Comma-separated alpha values for AKShapley, run together in one job")

# MLP hyperparameters (mirrors run_adult_dataval.py)
parser.add_argument("--mlp_epochs", type=int, default=10, help="Number of epochs for MLP training (default: 10)")
parser.add_argument("--mlp_batch_size", type=int, default=64, help="Batch size for MLP training (default: 64)")
parser.add_argument("--mlp_lr", type=float, default=0.001, help="Learning rate for MLP training (default: 0.001)")
parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Hidden dimension for MLP (default: 64)")
parser.add_argument("--mlp_layers", type=int, default=2, help="Number of hidden layers for MLP (default: 2)")

# Tuning mode: grid search over MLP hyperparameters, no data valuation
parser.add_argument("--tune", action="store_true", help="Run MLP hyperparameter grid search instead of data valuation")
parser.add_argument("--tune_hidden_dims", type=str, default="16,32,64,128", help="Comma-separated hidden dims to try when --tune")
parser.add_argument("--tune_epochs", type=str, default="5,10,20", help="Comma-separated epoch counts to try when --tune")
parser.add_argument("--tune_batch_sizes", type=str, default="32,64,128", help="Comma-separated batch sizes to try when --tune")
parser.add_argument("--tune_lrs", type=str, default="0.01,0.001,0.0001", help="Comma-separated learning rates to try when --tune")

# Tuning mode 2: compute data values once (method set via --method), then grid search
# MLP hyperparameters directly on the filtered+weighted training (the actual downstream task).
parser.add_argument("--tune_filtered", action="store_true", help="Grid search MLP hyperparameters on the filtered+weighted training for --method, instead of the raw baseline")

args = parser.parse_args()

SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id
NUM_CLASSES = args.num_classes

PROPORTION = args.proportion
NOISE_RATE = args.noise_rate
K_NEIGHBORS = args.k_neighbors
SUBSET_SIZE = args.subset_size
NUM_MODELS = args.num_models
NUM_MODELS_LIST = (
    [int(x) for x in args.num_models_list.split(",")] if args.num_models_list else None
)
MC_EPOCHS = args.mc_epochs
LAM_Y = args.lam_y
LAMBDA_WEIGHT = args.lambda_weight
LAM_X = args.lam_x
BLUR = args.blur
NUM_SAMPLES = args.num_samples
RL_EPOCHS = args.rl_epochs
RL_BATCH_SIZE = args.rl_batch_size
AK_K_NEIGHBORS = args.ak_k_neighbors
AK_N_HASH_TABLE = args.ak_n_hash_table
AK_EPS = args.ak_eps
AK_ALPHA = args.ak_alpha
AK_DIST_RAND = args.ak_dist_rand
AK_T = args.ak_t
AK_N_HASH_TABLE_LIST = (
    [int(x) for x in args.ak_n_hash_table_list.split(",")] if args.ak_n_hash_table_list else None
)
AK_EPS_LIST = (
    [float(x) for x in args.ak_eps_list.split(",")] if args.ak_eps_list else None
)
AK_ALPHA_LIST = (
    [float(x) for x in args.ak_alpha_list.split(",")] if args.ak_alpha_list else None
)

MLP_EPOCHS = args.mlp_epochs
MLP_BATCH_SIZE = args.mlp_batch_size
MLP_LR = args.mlp_lr
MLP_HIDDEN_DIM = args.mlp_hidden_dim
MLP_LAYERS = args.mlp_layers

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"SEED={SEED}, METHOD={METHOD}, JOB_ID={JOB_ID}, NUM_CLASSES={NUM_CLASSES}")
print(f"MLP: layers={MLP_LAYERS}, hidden_dim={MLP_HIDDEN_DIM}, epochs={MLP_EPOCHS}, "
      f"batch_size={MLP_BATCH_SIZE}, lr={MLP_LR}")

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

    def to_one_hot(y, k=NUM_CLASSES):
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


def load_raw_data():
    print("Loading datasets...")
    X_source = np.load("X_source.npy")
    y_source = np.load("y_source.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    # Shuffle source rows (X/y kept paired) so ordering in the .npy files
    # (e.g. grouped by class) doesn't bias evaluators/training via row order.
    X_source, y_source = sk_shuffle(X_source, y_source, random_state=SEED)

    Xs, Xt_val, Xt_test = prepare_source_target_data(
        X_source, X_val, X_test, y_source, y_val, y_test
    )
    return Xs, y_source, Xt_val, y_val, Xt_test, y_test


def make_mlp(input_dim):
    return ClassifierMLP(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        layers=MLP_LAYERS,
        hidden_dim=MLP_HIDDEN_DIM,
    )


# ============================================================
# Evaluators
# ============================================================

def create_evaluators(method):
    if method == "InfluenceSubsample":
        n_models = NUM_MODELS if NUM_MODELS is not None else 1000
        # subset_size and proportion are equivalent (redundant) ways to size the
        # bootstrap subset -- pass only one. proportion takes precedence when set
        # explicitly; otherwise fall back to subset_size (or its default).
        if PROPORTION is not None:
            return [
                InfluenceSubsample(
                    num_models=n_models,
                    proportion=PROPORTION,
                    random_state=SEED
                )
            ]
        s_size = SUBSET_SIZE if SUBSET_SIZE is not None else 100
        return [
            InfluenceSubsample(
                num_models=n_models,
                subset_size=s_size,
                random_state=SEED
            )
        ]
    if method == "AME":
        num_models = NUM_MODELS if NUM_MODELS is not None else 20000
        return [
            AME(num_models=num_models, random_state=SEED)
        ]

    if method == "DataShapley":
        mc_epochs = MC_EPOCHS if MC_EPOCHS is not None else 135
        return [
            DataShapley(mc_epochs=mc_epochs, random_state=SEED)
        ]
    if method == "DataBanzhaf":
        num_models = NUM_MODELS if NUM_MODELS is not None else 80000
        return [
            DataBanzhaf(num_models=num_models, random_state=SEED)
        ]

    if method == "DVRL":
        rl_epochs = RL_EPOCHS if RL_EPOCHS is not None else 10000
        rl_batch_size = RL_BATCH_SIZE if RL_BATCH_SIZE is not None else 32
        return [
            DVRL(
                rl_epochs=rl_epoch,
                rl_batch_size=rl_batch_size,
                random_state=SEED
            )
            for rl_epoch in [1000, 2000, 3000]
        ] if RL_EPOCHS is None else [
            DVRL(rl_epochs=rl_epochs, rl_batch_size=rl_batch_size, random_state=SEED)
        ]

    if method == "KNNShapley":
        k_value = K_NEIGHBORS if K_NEIGHBORS is not None else 5
        return [KNNShapley(k_neighbors=k_value)]

    if method == "AKShapley":
        k_neighbors = AK_K_NEIGHBORS if AK_K_NEIGHBORS is not None else 100
        # eps controls K_star = max(k_neighbors, ceil(1/eps)) -- the number of
        # candidate neighbors the LSH search actually retrieves before the top
        # k_neighbors are used in the Shapley formula. eps < 1/k_neighbors gives
        # a wider approximation margin (more candidates than strictly needed,
        # better recall against LSH's lossiness); eps = 1/k_neighbors gives none.
        # When dist_rand/t are left None, contrast estimation runs and its
        # output array is sized independently of k_neighbors (by eps alone),
        # so k_neighbors > ceil(1/eps) can IndexError there -- passing dist_rand
        # and t explicitly (self-calibrated once, see AK_DIST_RAND/AK_T) skips
        # that step entirely and removes this fragility.
        eps_values = AK_EPS_LIST if AK_EPS_LIST is not None else (
            [AK_EPS] if AK_EPS is not None else [1.0 / k_neighbors]
        )
        n_hash_table_values = AK_N_HASH_TABLE_LIST if AK_N_HASH_TABLE_LIST is not None else (
            [AK_N_HASH_TABLE] if AK_N_HASH_TABLE is not None else [100]
        )
        alpha_values = AK_ALPHA_LIST if AK_ALPHA_LIST is not None else (
            [AK_ALPHA] if AK_ALPHA is not None else [0.5]
        )
        # dist_rand/t are dataset-specific calibration constants (average random
        # distance / LSH projection width). Leave as None (unless explicitly
        # overridden via --ak_dist_rand/--ak_t) so KNNShapleyLSH self-calibrates
        # them from this dataset's validation data, instead of reusing values
        # that were calibrated for the Adult dataset.
        dist_rand = AK_DIST_RAND
        t = AK_T
        return [
            KNNShapleyLSH(
                k_neighbors=k_neighbors,
                n_hash_table=n_hash_table,
                eps=eps,
                dist_rand=dist_rand,
                t=t,
                alpha=alpha,
                random_state=SEED
            )
            for alpha in alpha_values
            for n_hash_table in n_hash_table_values
            for eps in eps_values
        ]

    if method == "LAVA":
        lam_y = LAM_Y if LAM_Y is not None else 0.1
        lam_x = LAM_X if LAM_X is not None else 1.0
        blur = BLUR if BLUR is not None else 0.05
        return [
            LavaEvaluator(blur=blur, lam_x=lam_x, lam_y=lam_y, debug=True)
        ]

    if method == "SAVA":
        lam_y = LAM_Y if LAM_Y is not None else 5.0
        lam_x = LAM_X if LAM_X is not None else 1.0
        blur = BLUR if BLUR is not None else 0.05
        return [
            SavaEvaluator(
                lam_y=lam_y,
                lam_x=lam_x,
                blur=blur,
                batch_size=1024,
                debug=True
            )
        ]

    if method == "DataOob":
        proportions = [0.1, 0.2, 0.5, 0.7, 1.0] if PROPORTION is None else [PROPORTION]
        num_models_list = NUM_MODELS_LIST if NUM_MODELS_LIST is not None else [10, 100, 1000]
        return [
            DataOob(
                num_models=num_models,
                proportion=p,
                random_state=SEED
            )
            for p in proportions
            for num_models in num_models_list
        ]

    if method == "BetaShapley":
        num_models = NUM_MODELS if NUM_MODELS is not None else 10000
        return [BetaShapley(num_models=num_models, random_state=SEED)]

    if method == "KAIROS":
        lambda_weight = LAMBDA_WEIGHT if LAMBDA_WEIGHT is not None else 0.97
        num_samples = NUM_SAMPLES if NUM_SAMPLES is not None else 10000
        return [
            bKairos(
                lambda_weight=lambda_weight,
                unbiased=True,
                use_median_heuristic=True,
                num_samples=num_samples,
                batch_size=1024,
                random_state=SEED,
                debug=True
            )
        ]

    if method == "InRunDataShapleyGhost":
        epochs = MLP_EPOCHS
        batch_size = MLP_BATCH_SIZE
        lr = MLP_LR
        return [
            InRunDataShapleyGhost(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                random_state=SEED,
                verbose=True,
                scheduler_type="none",
                save_plots=True,
                plot_dir="debugs"
            )
        ]

    if method == "LoGRA":
        lora_configs = [
            ('none', 'none'),
            ('none', 'raw'),
            ('none', 'kfac'),
            ('none', 'ekfac'),
            ('pca', 'kfac'),
            ('pca', 'ekfac'),
            ('pca', 'raw'),
            ('random', 'kfac')]
        return [
            LoGRA(
                epochs=MLP_EPOCHS,
                batch_size=MLP_BATCH_SIZE,
                learning_rate=MLP_LR,
                lora=config[0],
                hessian=config[1],
                random_state=SEED,
                verbose=True
            )
            for config in lora_configs
        ]

    if method == "LOO":
        return [LeaveOneOut()]
    if method == "Random":
        return [RandomEvaluator(random_state=SEED)]
    if method == "ALL":
        return [
            RandomEvaluator(random_state=SEED)(),
            InfluenceSubsample(num_models=5000, proportion=0.2, random_state=SEED),
            KNNShapley(k_neighbors=5)
        ]

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Time / memory report
# ============================================================

def save_time_memory_report(exper_med, out_dir, method):
    """Report per-evaluator train/eval/combined time and (when available) memory.

    Several evaluators (KNNShapley, KNNShapleyLSH/AKShapley, SavaEvaluator, and
    our own KAIROS lambda-sweep loop) assign `self.data_values = ...` directly
    inside/around train_data_values(), which shadows the base DataEvaluator's
    `data_values` cached_property and permanently prevents it from ever running
    its wrapped eval-phase timing/memory instrumentation -- so `ev.memory_report`
    silently stays empty (all zeros) for exactly these evaluators. `exper_med`
    tracks `self.timings[data_val] = <timedelta>` unconditionally around every
    evaluator's `.train()` call in compute_data_values(), independent of that
    quirk, so use it as the authoritative combined/train time; memory_report is
    still used opportunistically for eval-phase split and peak memory, when it
    happens to be populated.
    """

    rows = []

    for ev in exper_med.data_evaluators:
        name = str(ev)
        try:
            _ = ev.data_values
            rep = getattr(ev, "memory_report", {})
            train = rep.get("train", {})
            evalr = rep.get("eval", {})
            comb = rep.get("combined", {})

            timing_delta = exper_med.timings.get(ev)
            train_seconds = timing_delta.total_seconds() if timing_delta is not None else train.get("elapsed_seconds", 0)
            combined_seconds = timing_delta.total_seconds() if timing_delta is not None else comb.get("elapsed_seconds", 0)

            rows.append({
                "method": name,
                "train_seconds": train_seconds,
                "eval_seconds": evalr.get("elapsed_seconds", 0),
                "combined_seconds": combined_seconds,
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
# Custom evaluation: Weighted training on positive values (MLP)
# ============================================================

def weighted_positive_training(exper_med, out_dir):
    """Train a weighted ClassifierMLP using positively valued points for EACH evaluator.

    For every evaluator in `exper_med.data_evaluators`:
      - Read its `data_values`
      - Remove points with negative values (filter out bad/noisy source points)
      - Train a fresh ClassifierMLP on remaining points, passing values as sample_weight
      - Evaluate accuracy on target validation and test sets
      - Append a summary row tagged with the evaluator name

    Saves a combined CSV: weighted_positive_training.csv with one row per evaluator.
    """

    x_train, y_train, x_valid, y_valid, x_test, y_test = exper_med.fetcher.datapoints
    input_dim = x_train.shape[1]

    def to_class_indices(y):
        y = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
        return np.argmax(y, axis=1) if y.ndim == 2 else y

    y_train_idx = to_class_indices(y_train)
    y_valid_idx = to_class_indices(y_valid)
    y_test_idx = to_class_indices(y_test)

    rows = []
    epoch_losses_by_evaluator = {}

    for ev in exper_med.data_evaluators:
        name = str(ev)
        try:
            dv = getattr(ev, "data_values", None)
            if dv is None:
                rows.append({"evaluator": name, "status": "no_data_values"})
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

            # Initial (pre-filter) value range -- sanity check on the raw data_values
            # the evaluator produced, before any filtering/weighting logic touches them.
            raw_value_min = float(values.min())
            raw_value_max = float(values.max())

            # Filter: keep values > 0, drop values <= 0 (bad/noisy source points)
            mask_pos = values > 0
            kept_count = int(mask_pos.sum())
            removed_count = int((~mask_pos).sum())

            if kept_count == 0:
                rows.append({
                    "evaluator": name,
                    "status": "no_positive_values",
                    "raw_value_min": raw_value_min,
                    "raw_value_max": raw_value_max,
                    "kept_count": 0,
                    "removed_count": removed_count,
                })
                continue

            x_train_pos = x_train[mask_pos]
            y_train_pos_idx = y_train_idx[mask_pos]
            y_train_pos_oh = np.eye(NUM_CLASSES)[y_train_pos_idx]
            sample_w = values[mask_pos]

            # Normalize weights to avoid extreme scales (strictly positive), then
            # rescale to mean 1. ClassifierMLP.fit computes
            # loss = (per_sample_loss * weight).mean() -- a scaled mean, not a
            # weighted average -- so leaving weights in [0, 1] would shrink the
            # effective loss/gradient magnitude by whatever the mean weight is
            # (e.g. 30x smaller), silently acting like a much smaller LR than
            # tuned. Rescaling to mean 1 preserves relative importance between
            # points while keeping the overall gradient scale matched to the
            # LR/epochs tuned on unweighted data.
            w_min, w_max = float(sample_w.min()), float(sample_w.max())
            if w_max > w_min:
                sample_w = (sample_w - w_min) / (w_max - w_min) + 1e-8
            else:
                sample_w = np.ones_like(sample_w)
            sample_w = sample_w / sample_w.mean()

            x_train_pos_t = torch.as_tensor(x_train_pos, dtype=torch.float32)
            y_train_pos_t = torch.as_tensor(y_train_pos_oh, dtype=torch.float32)
            sample_w_t = torch.as_tensor(sample_w, dtype=torch.float32)

            # Capture fit()'s printed "Epoch N/M: loss=X..." lines (without
            # modifying the library) so we can verify the weighted model
            # actually trained -- i.e. loss decreased -- rather than assuming it.
            clf = make_mlp(input_dim)
            fit_log = io.StringIO()
            with contextlib.redirect_stdout(fit_log):
                clf.fit(
                    x_train_pos_t,
                    y_train_pos_t,
                    sample_weight=sample_w_t,
                    batch_size=MLP_BATCH_SIZE,
                    epochs=MLP_EPOCHS,
                    lr=MLP_LR,
                    print_loss=True,
                )
            print(fit_log.getvalue(), end="")

            epoch_losses = [float(m) for m in re.findall(r"loss=([\d.]+)", fit_log.getvalue())]
            first_epoch_loss = epoch_losses[0] if epoch_losses else None
            last_epoch_loss = epoch_losses[-1] if epoch_losses else None
            loss_decreased = (
                (last_epoch_loss < first_epoch_loss) if epoch_losses and len(epoch_losses) > 1 else None
            )
            epoch_losses_by_evaluator[name] = {
                "epoch_losses": epoch_losses,
                "first_epoch_loss": first_epoch_loss,
                "last_epoch_loss": last_epoch_loss,
                "loss_decreased": loss_decreased,
            }

            x_valid_t = torch.as_tensor(x_valid, dtype=torch.float32) if not torch.is_tensor(x_valid) else x_valid
            x_test_t = torch.as_tensor(x_test, dtype=torch.float32) if not torch.is_tensor(x_test) else x_test

            valid_pred = clf.predict(x_valid_t)
            test_pred = clf.predict(x_test_t)
            valid_pred = valid_pred.cpu().numpy() if hasattr(valid_pred, "cpu") else np.asarray(valid_pred)
            test_pred = test_pred.cpu().numpy() if hasattr(test_pred, "cpu") else np.asarray(test_pred)

            valid_acc = float(accuracy_score(y_valid_idx, np.argmax(valid_pred, axis=1)))
            test_acc = float(accuracy_score(y_test_idx, np.argmax(test_pred, axis=1)))

            rows.append({
                "evaluator": name,
                "evaluation": "WeightedPositiveTraining",
                "raw_value_min": raw_value_min,
                "raw_value_max": raw_value_max,
                "kept_count": kept_count,
                "removed_count": removed_count,
                "mean_positive_weight": float(sample_w.mean()),
                "first_epoch_loss": first_epoch_loss,
                "last_epoch_loss": last_epoch_loss,
                "loss_decreased": loss_decreased,
                "valid_accuracy": valid_acc,
                "test_accuracy": test_acc,
                "status": "success" if loss_decreased is not False else "success_loss_did_not_decrease"
            })
        except Exception as e:
            rows.append({
                "evaluator": name,
                "status": f"failed: {str(e)[:120]}"
            })

    if not rows:
        print("weighted_positive_training: No evaluators processed.")
        return

    full = pd.DataFrame(rows)
    full_path = os.path.join(out_dir, "weighted_positive_training_full.csv")
    full.to_csv(full_path, index=False)
    print(f"Saved full weighted positive training details to: {full_path}")

    summary_cols = [c for c in ["evaluator", "kept_count", "removed_count",
                                 "valid_accuracy", "test_accuracy", "status"] if c in full.columns]
    summary = full[summary_cols]
    summary_path = os.path.join(out_dir, "weighted_positive_training_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved weighted positive training summary to: {summary_path}")

    losses_path = os.path.join(out_dir, "weighted_positive_training_epoch_losses.json")
    with open(losses_path, "w") as f:
        json.dump(epoch_losses_by_evaluator, f, indent=2)
    print(f"Saved per-epoch loss trajectories to: {losses_path}")


# ============================================================
# MLP hyperparameter tuning (grid search, no data valuation)
# ============================================================

def run_tuning():
    """Grid search over MLP hidden_dim/epochs/batch_size/lr on source->target accuracy."""

    Xs, ys, Xt_val, yt_val, Xt_test, yt_test = load_raw_data()
    input_dim = Xs.shape[1]

    x_train_t = torch.as_tensor(Xs, dtype=torch.float32)
    y_train_t = torch.as_tensor(np.eye(NUM_CLASSES)[ys.astype(int)], dtype=torch.float32)
    x_valid_t = torch.as_tensor(Xt_val, dtype=torch.float32)
    x_test_t = torch.as_tensor(Xt_test, dtype=torch.float32)

    hidden_dims = [int(x) for x in args.tune_hidden_dims.split(",")]
    epochs_list = [int(x) for x in args.tune_epochs.split(",")]
    batch_sizes = [int(x) for x in args.tune_batch_sizes.split(",")]
    lrs = [float(x) for x in args.tune_lrs.split(",")]

    print(f"Tuning grid: hidden_dims={hidden_dims}, epochs={epochs_list}, "
          f"batch_sizes={batch_sizes}, lrs={lrs}")
    print(f"Total configs: {len(hidden_dims) * len(epochs_list) * len(batch_sizes) * len(lrs)}")

    rows = []
    for hidden_dim in hidden_dims:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for lr in lrs:
                    try:
                        torch.manual_seed(SEED)
                        model = ClassifierMLP(
                            input_dim=input_dim,
                            num_classes=NUM_CLASSES,
                            layers=MLP_LAYERS,
                            hidden_dim=hidden_dim,
                        )
                        t0 = time.time()
                        model.fit(
                            x_train_t, y_train_t,
                            batch_size=batch_size,
                            epochs=epochs,
                            lr=lr,
                            print_loss=False,
                        )
                        elapsed = time.time() - t0

                        valid_pred = model.predict(x_valid_t)
                        test_pred = model.predict(x_test_t)
                        valid_pred = valid_pred.cpu().numpy() if hasattr(valid_pred, "cpu") else np.asarray(valid_pred)
                        test_pred = test_pred.cpu().numpy() if hasattr(test_pred, "cpu") else np.asarray(test_pred)

                        valid_acc = float(accuracy_score(yt_val, np.argmax(valid_pred, axis=1)))
                        test_acc = float(accuracy_score(yt_test, np.argmax(test_pred, axis=1)))

                        rows.append({
                            "hidden_dim": hidden_dim,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "valid_accuracy": valid_acc,
                            "test_accuracy": test_acc,
                            "train_seconds": elapsed,
                            "status": "success"
                        })
                        print(f"  hidden_dim={hidden_dim} epochs={epochs} batch_size={batch_size} "
                              f"lr={lr}: valid_acc={valid_acc:.4f} test_acc={test_acc:.4f}")
                    except Exception as e:
                        rows.append({
                            "hidden_dim": hidden_dim,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "status": f"failed: {str(e)[:120]}"
                        })

    out_dir = os.path.join(RESULTS_DIR, "tuning", f"SEED{SEED}_JOB{JOB_ID}")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "mlp_tuning_results.csv")
    df.to_csv(out_path, index=False)

    if (df["status"] == "success").any():
        best = df[df["status"] == "success"].sort_values("valid_accuracy", ascending=False).iloc[0]
        print("\nBest config by validation accuracy:")
        print(best)

    print(f"\nTuning results saved to: {out_path}")


# ============================================================
# MLP hyperparameter tuning on the filtered+weighted task
# ============================================================

def run_tuning_filtered():
    """Compute data values once for --method (model-free for methods like KNNShapley),
    then grid search MLP hidden_dim/epochs/batch_size/lr directly on the filtered+weighted
    training task, so the tuning target matches what weighted_positive_training actually does.
    """

    Xs, ys, Xt_val, yt_val, Xt_test, yt_test = load_raw_data()
    fetcher = setup_fetcher(Xs, ys, Xt_val, yt_val, Xt_test, yt_test)
    input_dim = fetcher.x_train.shape[1]

    print(f"\nComputing data values once with method={METHOD} (model-free for KNNShapley)...")
    dummy_model = make_mlp(input_dim)
    exper_med = ExperimentMediator(fetcher, dummy_model, metric_name="accuracy",
                                    train_kwargs={"epochs": 1, "batch_size": 32, "lr": 0.001})
    evaluators = create_evaluators(METHOD)
    if len(evaluators) != 1:
        raise ValueError(
            f"--tune_filtered expects exactly one evaluator config for --method {METHOD}, "
            f"got {len(evaluators)}. Pass specific hyperparameters (e.g. --k_neighbors) to narrow it down."
        )
    exper_med.compute_data_values(evaluators)
    ev = evaluators[0]
    ev_name = str(ev)

    x_train, y_train, x_valid, y_valid, x_test, y_test = exper_med.fetcher.datapoints

    def to_class_indices(y):
        y = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
        return np.argmax(y, axis=1) if y.ndim == 2 else y

    y_train_idx = to_class_indices(y_train)
    y_valid_idx = to_class_indices(y_valid)
    y_test_idx = to_class_indices(y_test)

    values = np.asarray(ev.data_values).reshape(-1)
    mask_pos = values > 0
    kept_count = int(mask_pos.sum())
    removed_count = int((~mask_pos).sum())
    print(f"Filtered training set: kept {kept_count}, removed {removed_count} (negative-value points)")

    x_train_pos = x_train[mask_pos]
    y_train_pos_idx = y_train_idx[mask_pos]
    y_train_pos_oh = np.eye(NUM_CLASSES)[y_train_pos_idx]
    sample_w = values[mask_pos]

    # See weighted_positive_training's comment: rescale to mean 1 so the loss
    # scale (loss * weight).mean() matches what unweighted training expects,
    # keeping the tuned LR/epochs meaningful.
    w_min, w_max = float(sample_w.min()), float(sample_w.max())
    if w_max > w_min:
        sample_w = (sample_w - w_min) / (w_max - w_min) + 1e-8
    else:
        sample_w = np.ones_like(sample_w)
    sample_w = sample_w / sample_w.mean()

    x_train_pos_t = torch.as_tensor(x_train_pos, dtype=torch.float32)
    y_train_pos_t = torch.as_tensor(y_train_pos_oh, dtype=torch.float32)
    sample_w_t = torch.as_tensor(sample_w, dtype=torch.float32)
    x_valid_t = torch.as_tensor(x_valid, dtype=torch.float32) if not torch.is_tensor(x_valid) else x_valid
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32) if not torch.is_tensor(x_test) else x_test

    hidden_dims = [int(x) for x in args.tune_hidden_dims.split(",")]
    epochs_list = [int(x) for x in args.tune_epochs.split(",")]
    batch_sizes = [int(x) for x in args.tune_batch_sizes.split(",")]
    lrs = [float(x) for x in args.tune_lrs.split(",")]

    print(f"Tuning grid on filtered+weighted task: hidden_dims={hidden_dims}, epochs={epochs_list}, "
          f"batch_sizes={batch_sizes}, lrs={lrs}")
    print(f"Total configs: {len(hidden_dims) * len(epochs_list) * len(batch_sizes) * len(lrs)}")

    rows = []
    for hidden_dim in hidden_dims:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for lr in lrs:
                    try:
                        torch.manual_seed(SEED)
                        model = ClassifierMLP(
                            input_dim=input_dim,
                            num_classes=NUM_CLASSES,
                            layers=MLP_LAYERS,
                            hidden_dim=hidden_dim,
                        )
                        t0 = time.time()
                        model.fit(
                            x_train_pos_t, y_train_pos_t,
                            sample_weight=sample_w_t,
                            batch_size=batch_size,
                            epochs=epochs,
                            lr=lr,
                            print_loss=False,
                        )
                        elapsed = time.time() - t0

                        valid_pred = model.predict(x_valid_t)
                        test_pred = model.predict(x_test_t)
                        valid_pred = valid_pred.cpu().numpy() if hasattr(valid_pred, "cpu") else np.asarray(valid_pred)
                        test_pred = test_pred.cpu().numpy() if hasattr(test_pred, "cpu") else np.asarray(test_pred)

                        valid_acc = float(accuracy_score(y_valid_idx, np.argmax(valid_pred, axis=1)))
                        test_acc = float(accuracy_score(y_test_idx, np.argmax(test_pred, axis=1)))

                        rows.append({
                            "hidden_dim": hidden_dim,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "valid_accuracy": valid_acc,
                            "test_accuracy": test_acc,
                            "train_seconds": elapsed,
                            "status": "success"
                        })
                        print(f"  hidden_dim={hidden_dim} epochs={epochs} batch_size={batch_size} "
                              f"lr={lr}: valid_acc={valid_acc:.4f} test_acc={test_acc:.4f}")
                    except Exception as e:
                        rows.append({
                            "hidden_dim": hidden_dim,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "status": f"failed: {str(e)[:120]}"
                        })

    out_dir = os.path.join(RESULTS_DIR, f"tuning_filtered_{METHOD}", f"SEED{SEED}_JOB{JOB_ID}")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "mlp_tuning_filtered_results.csv")
    df.to_csv(out_path, index=False)

    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Method: {ev_name}\n")
        f.write(f"Kept: {kept_count}, Removed: {removed_count}\n")

    if (df["status"] == "success").any():
        ok = df[df["status"] == "success"]
        best_valid = ok.sort_values("valid_accuracy", ascending=False).iloc[0]
        best_test = ok.sort_values("test_accuracy", ascending=False).iloc[0]
        print("\nBest config by validation accuracy:")
        print(best_valid)
        print("\nBest config by test accuracy:")
        print(best_test)

    print(f"\nFiltered tuning results saved to: {out_path}")


# ============================================================
# Main experiment
# ============================================================

def main():

    start = time.time()

    Xs, ys, Xt_val, yt_val, Xt_test, yt_test = load_raw_data()

    fetcher = setup_fetcher(Xs, ys, Xt_val, yt_val, Xt_test, yt_test)

    input_dim = fetcher.x_train.shape[1]
    model = make_mlp(input_dim)
    train_kwargs = {
        "epochs": MLP_EPOCHS,
        "batch_size": MLP_BATCH_SIZE,
        "lr": MLP_LR,
    }

    exper_med = ExperimentMediator(fetcher, model, metric_name="accuracy", train_kwargs=train_kwargs)

    # Baseline
    data = exper_med.fetcher.datapoints
    m = exper_med.pred_model.clone()
    m.fit(data[0], data[1], **exper_med.train_kwargs)
    baseline = exper_med.metric(m.predict(data[4]).cpu(), data[5])
    print(f"Baseline target accuracy: {baseline:.4f}")

    # Output directory
    out_dir = os.path.join(RESULTS_DIR, METHOD, f"SEED{SEED}_JOB{JOB_ID}")

    os.makedirs(out_dir, exist_ok=True)
    exper_med.set_output_directory(out_dir)

    evaluators = create_evaluators(METHOD)

    # Compute values
    exper_med.compute_data_values(evaluators)

    if METHOD == "KAIROS":
        # KAIROS builds an expensive kernel once; evaluate_data_values(lambda_weight=lam)
        # reuses that kernel to re-score points for a new lambda without recomputing it,
        # so we can sweep lambda cheaply (same trick as run_adult_dataval.py).
        LAMBDA_VALUES = [0, 0.5, 0.8, 0.9, 0.97, 1.0]
        evaluator = evaluators[0]
        base_output_dir = out_dir

        print(f"\nRunning lambda sweep for KAIROS: {LAMBDA_VALUES}")
        for lam in LAMBDA_VALUES:
            lambda_dir = os.path.join(base_output_dir, f"LAMBDA_{lam}")
            os.makedirs(lambda_dir, exist_ok=True)
            exper_med.set_output_directory(lambda_dir)

            print(f"\n[Lambda={lam}] Re-scoring data values (kernel reused)...")
            evaluator.data_values = evaluator.evaluate_data_values(lambda_weight=lam)

            try:
                exper_med.evaluate(remove_high_low, max_removal_fraction=0.5, save_output=True)
            except Exception as e:
                print(f"[Lambda={lam}] Evaluation failed: remove_high_low: {e}")

            try:
                weighted_positive_training(exper_med, lambda_dir)
            except Exception as e:
                print(f"[Lambda={lam}] Custom evaluation failed: weighted_positive_training: {e}")

            try:
                exper_med.evaluate(save_dataval, save_output=True)
            except Exception as e:
                print(f"[Lambda={lam}] save_dataval failed: {e}")

        exper_med.set_output_directory(base_output_dir)
    else:
        # Evaluations
        # noisy_detection and discover_corrupted_sample assume synthetically
        # injected label noise (fetcher.noisy_train_indices), which doesn't apply
        # to domain adaptation (no injected noise), so only remove_high_low runs.
        try:
            exper_med.evaluate(remove_high_low, max_removal_fraction=0.5, save_output=True)
        except Exception as e:
            print(f"Evaluation failed: remove_high_low: {e}")

        # Custom evaluation: remove negatives, weight positives, train MLP & report
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
        f.write(f"MLP: layers={MLP_LAYERS}, hidden_dim={MLP_HIDDEN_DIM}, epochs={MLP_EPOCHS}, "
                f"batch_size={MLP_BATCH_SIZE}, lr={MLP_LR}\n")
        f.write(f"Total time: {elapsed/3600:.2f} hours\n")

    print("Experiment completed successfully.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    if args.tune_filtered:
        run_tuning_filtered()
    elif args.tune:
        run_tuning()
    else:
        main()
