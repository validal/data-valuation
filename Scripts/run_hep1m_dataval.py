# run_hepmass_dataval.py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional, Union, List, Dict
import json
import os
import sys
from pathlib import Path
import argparse
import time
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

# Ensure dataset registry (in case of fresh session)
import opendataval.dataloader.datasets  # triggers @Register decorators
from opendataval.dataval.kairos.bkairos import bKairos

from opendataval.dataloader import mix_labels, add_gauss_noise, DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, LeaveOneOut, RandomEvaluator,
    InRunDataShapleyGhost, LoGRA, Kairos
)
from opendataval.dataval.influence import LossEvaluator
from opendataval.dataval.knnshap import KNNShapleyLSH
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.model import ClassifierMLP
import torch
from sklearn.utils import shuffle
# GhostSuite and LogIX imports
try:
    from ghostEngines import GradDotProdEngine
    print("✓ GhostSuite import OK")
except ImportError:
    print("⚠ GhostSuite not available (optional for InRunDataShapleyGhost)")

try:
    import logix
    print("✓ LogIX import OK")
except ImportError:
    print("⚠ LogIX not available (optional for LoGRA)")

# =============================================================================
# REFERENCE: How to instantiate all available evaluators with their parameters
# =============================================================================
#
# # Combine multiple evaluators (do NOT overwrite the list)
# evaluators = []
#
# # KNNShapley - k-nearest neighbors Shapley value
# evaluators.append(KNNShapley(k_neighbors=10))
#
# # KNNShapleyLSH - LSH variant of KNN Shapley
# evaluators.append(KNNShapleyLSH(k_neighbors=10, n_hash_table=20, eps=0.01,
#                                  dist_rand=7.2352, t=2.2510, alpha=0.5,
#                                  random_state=42))
#
# # DataOob - Out-of-bag data valuation
# evaluators.append(DataOob(num_models=100, proportion=0.8, random_state=42))
#
# # DataShapley - Monte Carlo Shapley
# evaluators.append(DataShapley(mc_epochs=5000, min_cardinality=5,
#                               cache_name="shapley_cache", random_state=42))
#
# # BetaShapley - Beta distribution-based Shapley
# evaluators.append(BetaShapley(num_models=1000, random_state=42))
#
# # DataBanzhaf - Banzhaf value approximation
# evaluators.append(DataBanzhaf(num_models=1000, random_state=42))
#
# # LeaveOneOut - Exact leave-one-out valuation
# evaluators.append(LeaveOneOut())
#
# # RandomEvaluator - Random baseline
# evaluators.append(RandomEvaluator(random_state=42))
#
# # InfluenceSubsample - Influence function with subsampling
# evaluators.append(InfluenceSubsample(num_models=10000, proportion=0.5,
#                                      random_state=42))
#
# # AME - Advantage Model-based Evaluation
# evaluators.append(AME(num_models=100000, random_state=42))
#
# # DVRL - Data Valuation using Reinforcement Learning
# evaluators.append(DVRL(rl_epochs=1000, rl_batch_size=32, random_state=42))
#
# # LossEvaluator - Simple Loss-Based Data Valuation Baseline
# # Trains one model to convergence, values each point as negative loss
# # High-value points = low loss (clean samples)
# # Low-value points = high loss (noisy/hard samples)
# evaluators.append(LossEvaluator(epochs=10, batch_size=32, learning_rate=0.01,
#                                  verbose=True))
#
# # LavaEvaluator - Label-aware Shapley with feature importance
# evaluators.append(LavaEvaluator(blur=0.05, lam_x=1.0, lam_y=1.0,
#                                 debug=True, random_state=42))
#
# # SavaEvaluator - Shapley with Wasserstein distance
# evaluators.append(SavaEvaluator(batch_size=1024, lam_x=1.0, lam_y=1.0,
#                                 p=2, blur=0.05, mode="cls", debug=True,
#                                 random_state=42))
#
# # Now compute data values with all evaluators
# print('Computing data values...')
# exper_med = exper_med.compute_data_values(data_evaluators=evaluators)
# print('✓ Computation complete')
#
# # Run evaluation on noisy detection
# results = exper_med.evaluate(noisy_detection, save_output=True)
# print('\n[Results] Noisy Detection F1-Scores:')
# print(results)
# =============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run HEPMASS data valuation experiment')
parser.add_argument('--seed', type=int, default=42, help='Random seed for the experiment')
parser.add_argument('--method', type=str, required=True,
                   choices=['DataOob', 'AME', 'DataBanzhaf', 'DataShapley',
                           'InfluenceSubsample', 'LOO_Random', 'KNNShapley', 'AKShapley',
                           'DVRL', 'BetaShapley', 'LAVA', 'SAVA', 'InRunDataShapleyGhost',
                           'LoGRA', 'Kairos', 'LossEvaluator', 'ALL'],
                   help='Method to run (or ALL for all methods)')
parser.add_argument('--job_id', type=int, default=1, help='Job ID for naming outputs')

# DataOob parameters
parser.add_argument('--oob_num_models', type=int, default=None,
                   help='DataOob: num_models (overrides default range if set)')
parser.add_argument('--oob_proportion', type=float, default=None,
                   help='DataOob: proportion (overrides default range if set)')

# AKShapley parameters (baseline / best config: k=1000, eps=0.0001, alpha=0.5, n_hash_table=100)
parser.add_argument('--akshapley_k', type=int, default=None,
                   help='AKShapley: k_neighbors (default: 1000)')
parser.add_argument('--akshapley_eps', type=float, default=None,
                   help='AKShapley: eps parameter (default: 0.0001)')
parser.add_argument('--akshapley_alpha', type=float, default=None,
                   help='AKShapley: alpha parameter (default: 0.5)')
parser.add_argument('--akshapley_n_hash', type=int, default=None,
                   help='AKShapley: n_hash_table (default: 100)')

# DVRL parameters
parser.add_argument('--dvrl_epochs', type=int, default=None,
                   help='DVRL: rl_epochs (overrides default range if set)')
parser.add_argument('--dvrl_batch_size', type=int, default=None,
                   help='DVRL: rl_batch_size (overrides default range if set)')

# InRunDataShapleyGhost parameters
parser.add_argument('--inrun_epochs', type=int, default=None,
                   help='InRunDataShapleyGhost: epochs (default: 5)')
parser.add_argument('--inrun_batch_size', type=int, default=None,
                   help='InRunDataShapleyGhost: batch_size (overrides default range if set)')

# LoGRA parameters
parser.add_argument('--logra_lora', type=str, default=None,
                   help='LoGRA: lora method (none/pca, overrides default range if set)')
parser.add_argument('--logra_hessian', type=str, default=None,
                   help='LoGRA: hessian method (kfac/raw/none, overrides default range if set)')
parser.add_argument('--logra_batch_size', type=int, default=None,
                   help='LoGRA: batch_size (default: 1024)')

# SAVA parameters
parser.add_argument('--sava_batch_size', type=int, default=None,
                   help='SAVA: batch_size (default: 1024)')
parser.add_argument('--sava_lam_y', type=float, default=None,
                   help='SAVA: lam_y weight (overrides default range if set)')

args = parser.parse_args()

# Set seeds
SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id

print(f"Running experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")


def set_global_seeds(seed):
    """
    Set all global random seeds for reproducibility.

    CRITICAL: Call this BEFORE creating any models!
    Models initialize weights using RNG, so this must be called first.
    """
    print(f"\n[Seeding] Setting global random seeds to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seeding] ✓ Global seeds configured for reproducibility")


def load_and_prepare_hepmass():
    """Load HEPMASS datasets and prepare them for the experiment."""
    print("Loading HEPMASS datasets...")

    # Use absolute paths for data files
    data_dir = "/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/Hep1M_SHF/Data"

    # Load datasets from CSV files
    train_set = pd.read_csv(os.path.join(data_dir, 'hepmass_train_1M.csv'))
    valid_set = pd.read_csv(os.path.join(data_dir, 'hepmass_valid_10K.csv'))
    test_set = pd.read_csv(os.path.join(data_dir, 'hepmass_test_50K.csv'))
    
    # Identify label column (first column based on your code)
    label_col = train_set.columns[0]
    print(f"Using label column: '{label_col}'")
    
    # Extract features and labels
    feature_cols = [col for col in train_set.columns if col != label_col]
    
    X_train = train_set[feature_cols]
    Y_train = train_set[label_col]
    
    X_valid = valid_set[feature_cols]
    Y_valid = valid_set[label_col]
    
    X_test = test_set[feature_cols]
    Y_test = test_set[label_col]
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"X_test:  {X_test.shape}")
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    
    # Fit scaler on training data, transform all sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to numpy arrays with float32 for efficiency
    X_train_np = X_train_scaled.astype(np.float32)
    X_valid_np = X_valid_scaled.astype(np.float32)
    X_test_np = X_test_scaled.astype(np.float32)
    
    # Convert labels to integers
    y_train = Y_train.values.astype(int)
    y_valid = Y_valid.values.astype(int)
    y_test = Y_test.values.astype(int)
    
    print(f"\nOriginal label distributions:")
    print(f"Train - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
    print(f"Valid - Class 0: {(y_valid == 0).sum()}, Class 1: {(y_valid == 1).sum()}")
    print(f"Test  - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")
    
    # One-hot encoding function
    def to_one_hot_numpy(labels, num_classes=2):
        """Convert 1D label array to one-hot encoded array"""
        return np.eye(num_classes)[labels]
    
    # One-hot encode for OpenDataVal compatibility
    y_train_onehot = to_one_hot_numpy(y_train)
    y_valid_onehot = to_one_hot_numpy(y_valid)
    y_test_onehot = to_one_hot_numpy(y_test)
    
    print(f"\nOne-hot encoded label shapes:")
    print(f"y_train_onehot: {y_train_onehot.shape}")
    print(f"y_valid_onehot: {y_valid_onehot.shape}")
    print(f"y_test_onehot:  {y_test_onehot.shape}")
    
    return X_train_np, y_train_onehot, X_valid_np, y_valid_onehot, X_test_np, y_test_onehot


def create_experiment_mediator():
    """Create and configure the ExperimentMediator for HEPMASS.

    CRITICAL: Global seeds are set at the beginning to ensure model
    weight initialization is reproducible.
    """
    print("Creating HEPMASS experiment mediator...")

    # ✅ STEP 1: Set global seeds BEFORE loading data or creating any models
    set_global_seeds(SEED)

    # ✅ STEP 2: Load and prepare data (uses seeded RNG)
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_prepare_hepmass()
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    num_classes = y_train.shape[1]

    # ✅ STEP 3: Create the fetcher with seed
    fetcher = DataFetcher.from_data_splits(
        x_train=X_train,
        y_train=y_train,
        x_valid=X_valid,
        y_valid=y_valid,
        x_test=X_test,
        y_test=y_test,
        one_hot=True,
        random_state=SEED
    )

    # Add noise to labels
    noise_rate = 0.2
    print(f"\nAdding noise to labels with rate: {noise_rate}")
    fetcher = fetcher.noisify(mix_labels, noise_rate=noise_rate, rng=SEED)
    # ---- DEBUG: is the noise balanced across classes? ----
    noisy_idx = np.asarray(fetcher.noisy_train_indices)
    yt = np.asarray(fetcher.y_train)
    yt_lab = yt.argmax(1) if yt.ndim > 1 else yt      # current (post-noise) labels

    print(f"[noise debug] seed={SEED}, total noisy = {len(noisy_idx)}")

    # 1) Overall class balance of the (noisy) training labels
    print("[noise debug] class counts in train labels:",
          np.bincount(yt_lab, minlength=num_classes).tolist())

    # 2) How many noisy samples fall in each (current) class
    noisy_by_class = np.bincount(yt_lab[noisy_idx], minlength=num_classes)
    print("[noise debug] noisy samples per class:", noisy_by_class.tolist())

    # 3) Per-class noise RATE = noisy-in-class / total-in-class
    total_by_class = np.bincount(yt_lab, minlength=num_classes)
    per_class_rate = noisy_by_class / np.maximum(total_by_class, 1)
    print("[noise debug] per-class noise rate:",
          [f"{r:.3f}" for r in per_class_rate])
    # ---- end debug ----

    # ✅ STEP 4: Create prediction model AFTER seeding
    # The model's weight initialization uses RNG and respects the seed set above
    metric_name = "accuracy"
    input_dim = X_train.shape[1]
    print(f"\nCreating prediction model: ClassifierMLP with input_dim={input_dim}, num_classes={num_classes}")
    pred_model = ClassifierMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        layers=2,
        hidden_dim=100,
    )
    print(f"[Model] ✓ Model created with seeded weight initialization")

    train_kwargs = {
        "epochs": 5,
        "batch_size": 1024,
        "lr": 0.001,
    }

    # Create ExperimentMediator with proper train_kwargs
    exper_med = ExperimentMediator(
        fetcher=fetcher,
        pred_model=pred_model,
        metric_name=metric_name,
        train_kwargs=train_kwargs
    )

    return exper_med


def create_method_evaluators(method_name, output_dir=None):
    """Create evaluators for a specific method."""
    print(f"Creating evaluators for method: {method_name}")

    
    if method_name == "DataOob":
        # Use provided parameters or defaults
        if args.oob_num_models is not None:
            MODEL_SIZES = [args.oob_num_models]
        else:
            MODEL_SIZES = [1,5,10,100]

        if args.oob_proportion is not None:
            PROPORTIONS = [args.oob_proportion]
        else:
            PROPORTIONS = [0.1,0.5,1.0]

        evaluators = [
            DataOob(num_models=m, proportion=p, random_state=s)
            for m in MODEL_SIZES
            for p in PROPORTIONS
            for s in [SEED]
            ]
        
    elif method_name == "LOO_Random":
        evaluators = [
            RandomEvaluator(random_state=s) for s in [SEED]
        ]

    elif method_name == "KNNShapley":
        evaluators = [] 
        for rs in [SEED]:
            for k in [1,5,10, 100, 500, 1000]:
                evaluators.append(KNNShapley(k_neighbors=k, random_state=rs))
    
    elif method_name == "AKShapley":
        # Baseline / best config: k=1000, eps=0.0001, alpha=0.5, n_hash_table=100
        # (dist_rand=7.3622, t=2.399 held fixed -- dataset-specific constants, not swept)
        k = args.akshapley_k if args.akshapley_k is not None else 1000
        eps = args.akshapley_eps if args.akshapley_eps is not None else 0.0001
        alpha = args.akshapley_alpha if args.akshapley_alpha is not None else 0.5
        n_hash = args.akshapley_n_hash if args.akshapley_n_hash is not None else 100
        evaluators = [KNNShapleyLSH(
            k_neighbors=k,
            dist_rand=7.3622,
            n_hash_table=n_hash,
            eps=eps,
            alpha=alpha,
            t=2.399,
            random_state=SEED
        )
        for _ in [1]
        ]
    
    elif method_name == "DVRL":
        # Use provided parameters or defaults
        if args.dvrl_epochs is not None:
            MODEL_SIZES = [args.dvrl_epochs]
        else:
            MODEL_SIZES = [1000,2000,3000]

        if args.dvrl_batch_size is not None:
            BATCH_SIZES = [args.dvrl_batch_size]
        else:
            BATCH_SIZES = [32,128,512,1024,2000]

        # Chunk size for the post-training final-weights scoring pass, kept
        # independent of rl_batch_size. With n_train=1M and a small
        # rl_batch_size (e.g. 64), leaving extraction_batch_size unset makes
        # it default to rl_batch_size, so the final pass loops N_train/64 ~=
        # 15625 times accumulating via torch.cat -- ~30GB peak memory
        # observed. A larger fixed value keeps that pass cheap regardless of
        # rl_batch_size.
        DVRL_EXTRACTION_BATCH_SIZE = 1024

        evaluators = []
        for rl_epochs in MODEL_SIZES:
            for batch_size in BATCH_SIZES:
                for random_state in [SEED]:
                    evaluators.append(
                        DVRL(
                            rl_epochs=rl_epochs,
                            rl_batch_size=batch_size,
                            random_state=random_state,
                            extraction_batch_size=DVRL_EXTRACTION_BATCH_SIZE,
                        )
                    )
                    
    elif method_name == "LAVA":
        LAM_Y = [1, 5, 10, 50, 100]
        evaluators = [
            LavaEvaluator(blur=0.05, debug=True, lam_x=1.0, lam_y=lam_y, random_state=s)
            for lam_y in LAM_Y
            for s in [SEED]
        ]
    elif method_name == "SAVA":
        batch_size = args.sava_batch_size if args.sava_batch_size is not None else 1024

        if args.sava_lam_y is not None:
            lam_y_values = [args.sava_lam_y]
        else:
            lam_y_values = [1,5,10,50,100]

        evaluators = [
            SavaEvaluator(
                batch_size=batch_size,
                lam_x=1.0,
                lam_y=lam_y,
                p=2,
                blur=0.05,
                mode="cls",
                debug=True,
                random_state=SEED,
                stratified_batches=True
            )
            for lam_y in lam_y_values
        ]    
    elif method_name == "InRunDataShapleyGhost":
        EPOCHS = args.inrun_epochs if args.inrun_epochs is not None else 5
        LR = 0.001
        plots_dir = output_dir if output_dir else "."

        if args.inrun_batch_size is not None:
            BATCH_SIZES = [args.inrun_batch_size]
        else:
            BATCH_SIZES = [32,512,1024,10000,100000]

        evaluators = [
            InRunDataShapleyGhost(
                epochs=EPOCHS,
                batch_size=batch_size,
                learning_rate=LR,
                random_state=s,
                verbose=True,
                save_plots=True,
                plot_dir=plots_dir
            )
            for s in [SEED]
            for batch_size in BATCH_SIZES
        ]

    elif method_name == "LoGRA":
        EPOCHS = 5
        BATCH_SIZE = args.logra_batch_size if args.logra_batch_size is not None else 1024
        LR = 0.001

        if args.logra_lora is not None:
            loras = [args.logra_lora]
        else:
            loras = ['none', 'pca']

        if args.logra_hessian is not None:
            hessians = [args.logra_hessian]
        else:
            hessians = ['ekfac','kfac', 'raw','none']

        evaluators = [
            LoGRA(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                lora=lora,
                hessian=hessian,
                random_state=s,
                verbose=True,
                cleanup=True,
                log_loader_batch_size=65536
            )
            for s in [SEED]
            for lora in loras
            for hessian in hessians
        ]

    elif method_name == "Kairos":
        # ONE evaluator - kernels computed once, lambda tuned later
        evaluators = [
            bKairos(
                lambda_weight=0.97,              # ✅ Default (will test multiple lambdas)
                unbiased=True,                   # ✅ Unbiased computation
                use_median_heuristic=True,       # ✅ Auto-estimate σ
                num_samples=10000,               # ✅ Samples for σ estimation
                batch_size=1024,                 # ✅ EXTRA: Memory-efficient batching
                random_state=SEED,               # ✅ Random seed
                debug=True                       # ✅ Debug output
            )
        ]

    elif method_name == "LossEvaluator":
        EPOCH = 5
        BATCH_SIZE = 1024
        evaluators = [
            LossEvaluator(
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                learning_rate=0.001,
                verbose=True
            )
            for s in [SEED]
        ]

    elif method_name == "ALL":
        # Combine all methods (for testing)
        evaluators = []
        for m in ["LOO_Random"]:
            evaluators.extend(create_method_evaluators(m, output_dir))

    else:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"Created {len(evaluators)} evaluators for {method_name}")
    return evaluators


def save_time_memory_report(exper_med, output_dir, method_name):
    """Save time and memory report for all evaluators."""
    print(f"\nCreating time/memory report for {method_name}...")
    
    rows = []
    successful_evaluators = 0
    
    for idx, ev in enumerate(exper_med.data_evaluators):
        name = str(ev)
        
        try:
            # Try to get data values to ensure evaluation happened
            _ = ev.data_values
            
            # Get memory report if it exists
            rep = getattr(ev, "memory_report", None)
            
            if rep and isinstance(rep, dict):
                train = rep.get("train", rep)
                evalr = rep.get("eval", {})
                comb = rep.get("combined", {})
                
                rows.append({
                    "method": name,
                    "train_seconds": train.get("elapsed_seconds", 0),
                    "eval_seconds": evalr.get("elapsed_seconds", 0),
                    "combined_seconds": comb.get("elapsed_seconds", 0),
                    "cpu_peak_kb_train": train.get("cpu_phase_peak_kb", 0),
                    "cpu_peak_kb_eval": evalr.get("cpu_phase_peak_kb", 0),
                    "gpu_peak_alloc_bytes": comb.get("gpu_peak_allocated_bytes", 0)
                        or evalr.get("gpu_peak_allocated_bytes", 0)
                        or train.get("gpu_peak_allocated_bytes", 0),
                    "status": "success"
                })
                successful_evaluators += 1
            else:
                rows.append({
                    "method": name,
                    "train_seconds": 0,
                    "eval_seconds": 0,
                    "combined_seconds": 0,
                    "cpu_peak_kb_train": 0,
                    "cpu_peak_kb_eval": 0,
                    "gpu_peak_alloc_bytes": 0,
                    "status": "success (no memory report)"
                })
                successful_evaluators += 1
                
        except Exception as e:
            error_msg = str(e)[:100]
            rows.append({
                "method": name,
                "train_seconds": 0,
                "eval_seconds": 0,
                "combined_seconds": 0,
                "cpu_peak_kb_train": 0,
                "cpu_peak_kb_eval": 0,
                "gpu_peak_alloc_bytes": 0,
                "status": f"failed: {error_msg}"
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(exper_med.data_evaluators)}] {name[:50]}...")
    
    # Create DataFrame and save
    if rows:
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"time_memory_{method_name}.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Time/memory report saved to: {output_path}")
        print(f"Total evaluators: {len(rows)}")
        print(f"Successful: {successful_evaluators}")
        print(f"Failed: {len(rows) - successful_evaluators}")
    
    return rows


def run_method_experiment(method_name):
    """Run experiment for a specific method."""
    print("=" * 70)
    print(f"HEPMASS Data Valuation Experiment - {method_name}")
    print("=" * 70)

    # Change to a temp directory with more space for LogIX
    if method_name == "LoGRA":
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="logra_", dir="/tmp")
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        print(f"Changed working directory to: {temp_dir}")

    start_time = time.time()
    
    # Create experiment mediator
    exper_med = create_experiment_mediator()

    # Create output directory with method name
    base_remote_dir = (
        f"/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/Hep1M_SHF/results"
    )
    output_dir = os.path.join(base_remote_dir, method_name, f'SEED{SEED}_JOB{JOB_ID}')

    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create method-specific evaluators
    all_evaluators = create_method_evaluators(method_name, output_dir)
    
    # Compute data values
    print(f"\nComputing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    
    try:
        exper_med = exper_med.compute_data_values(data_evaluators=all_evaluators)
        print(f"✓ {method_name} computation completed successfully")
    except Exception as e:
        print(f"✗ {method_name} computation failed: {e}")
        # Continue to save partial results
    
    # Run evaluations
    print(f"\nRunning evaluations for {method_name}...")

    # Special handling for Kairos: evaluate each lambda (kernel reuse)
    if method_name == "Kairos":
        import time as time_module
        LAMBDA_VALUES = [0, 0.5, 0.8,0.9, 0.95, 0.97, 0.99, 1]
        evaluator = all_evaluators[0]
        base_output_dir = output_dir
        lambda_timing_results = []

        for lam in LAMBDA_VALUES:
            # Create lambda-specific output directory
            lambda_dir = os.path.join(base_output_dir, f"LAMBDA_{lam}")
            os.makedirs(lambda_dir, exist_ok=True)
            exper_med.set_output_directory(lambda_dir)

            lambda_start_time = time_module.time()

            # Update data values for this lambda (NO kernel recomputation)
            evaluator.data_values = evaluator.evaluate_data_values(lambda_weight=lam)

            print(f"\n[Lambda={lam}] Running evaluations...")

            # Run all evaluation functions
            evaluation_functions = [
                (noisy_detection, "Noisy detection"),
                (discover_corrupted_sample, "Corrupted sample discovery"),
                (remove_high_low, "Remove high/low")
            ]

            for eval_func, eval_name in evaluation_functions:
                try:
                    exper_med.evaluate(eval_func, save_output=True)
                    print(f"  ✓ {eval_name}")
                except Exception as e:
                    print(f"  ✗ {eval_name} failed: {str(e)[:80]}")

            # Save data values
            try:
                exper_med.evaluate(save_dataval, save_output=True)
                print(f"  ✓ Data values saved")
            except Exception as e:
                print(f"  ✗ Data values save failed: {str(e)[:80]}")

            # Record timing for this lambda
            lambda_time = time_module.time() - lambda_start_time
            lambda_timing_results.append({
                'lambda': lam,
                'time_seconds': lambda_time
            })

        # Save lambda timing summary
        lambda_timing_df = pd.DataFrame(lambda_timing_results)
        timing_csv_path = os.path.join(base_output_dir, "lambda_timing_summary.csv")
        lambda_timing_df.to_csv(timing_csv_path, index=False)
        print(f"\n✓ Lambda timing saved to: {timing_csv_path}")

    else:
        # Standard evaluation for other methods
        evaluation_functions = [
            (noisy_detection, "Noisy detection"),
            (discover_corrupted_sample, "Corrupted sample discovery"),
            (remove_high_low, "Remove high/low")
        ]

        for eval_func, eval_name in evaluation_functions:
            try:
                exper_med.evaluate(eval_func, save_output=True)
                print(f"  ✓ {eval_name} completed")
            except Exception as e:
                print(f"  ✗ {eval_name} failed: {e}")

        # Save data values
        print(f"\nSaving data values for {method_name}...")
        try:
            values = exper_med.evaluate(save_dataval, save_output=True)
            print(f"  ✓ Data values saved")
        except Exception as e:
            print(f"  ✗ Data values save failed: {e}")
    
    # Save time/memory report
    save_time_memory_report(exper_med, output_dir, method_name)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save final summary
    summary_path = os.path.join(output_dir, f"summary_{method_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"HEPMASS Data Valuation - {method_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Job ID: {JOB_ID}\n")
        f.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Total Evaluators: {len(all_evaluators)}\n")
    
    print(f"\n{'='*70}")
    print(f"{method_name} EXPERIMENT COMPLETED!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

    # Restore original working directory and clean up temp for LoGRA
    if method_name == "LoGRA":
        try:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")

    return exper_med


def generate_shell_scripts():
    """Generate shell scripts (.sh) that will create and submit SLURM jobs."""
    print("=" * 70)
    print("GENERATING SHELL SCRIPTS (.sh) FOR SLURM JOB CREATION")
    print("=" * 70)
    
    # Define all methods
    methods = [
        'DataOob', 'AME', 'DataBanzhaf', 'DataShapley',
        'InfluenceSubsample', 'LOO_Random', 'KNNShapley', 'AKShapley',
        'DVRL', 'BetaShapley', 'LAVA', 'SAVA', 'InRunDataShapleyGhost',
        'LoGRA', 'Kairos', 'LossEvaluator'
    ]
    
    # Define paths
    base_dir = "/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/Hep1M_SHF"
    scripts_dir = os.path.join(base_dir, "scripts")
    logs_dir = os.path.join(base_dir, "logs")
    
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Logs directory: {logs_dir}")
    
    # Define seed ranges for different methods
    seed_ranges = {
        'DataOob': list(range(40, 50)),
        'AME': list(range(40, 50)),
        'DataBanzhaf': list(range(40, 50)),
        'DataShapley': list(range(40, 50)),
        'InfluenceSubsample': list(range(40, 50)),
        'LOO_Random': list(range(40, 50)),
        'KNNShapley': list(range(40, 50)),
        'AKShapley': list(range(40, 50)),
        'DVRL': list(range(40, 50)),
        'BetaShapley': list(range(40, 50)),
        'LAVA': list(range(40, 50)),
        'SAVA': list(range(40, 50)),
        'InRunDataShapleyGhost': list(range(40, 50)),
        'LoGRA': list(range(40, 50)),
        'Kairos': list(range(40, 50)),
        'LossEvaluator': list(range(40, 50))
    }
    
    # Create master submission script
    master_script_path = os.path.join(scripts_dir, "submit_all_methods.sh")
    with open(master_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Master script to submit all HEPMASS data valuation jobs\n\n")
        f.write(f"SCRIPTS_DIR=\"{scripts_dir}\"\n\n")
        f.write("echo \"Submitting all method scripts...\"\n\n")
        f.write("for script in ${SCRIPTS_DIR}/run_hepmass_*.sh; do\n")
        f.write("    if [ -f \"$script\" ]; then\n")
        f.write("        echo \"Submitting: $script\"\n")
        f.write("        bash \"$script\"\n")
        f.write("        sleep 0.5\n")
        f.write("    fi\n")
        f.write("done\n\n")
        f.write("echo \"All jobs submitted!\"\n")
    
    os.chmod(master_script_path, 0o755)
    print(f"Created master script: {master_script_path}")
    
    # Generate individual shell scripts for each method
    for method in methods:
        seeds = seed_ranges.get(method, list(range(40, 50)))
        print(f"\nGenerating shell script for {method} with seeds {seeds[0]}-{seeds[-1]}")
        
        # Create shell script that will generate and submit SLURM scripts
        shell_script_path = os.path.join(scripts_dir, f"run_hepmass_{method}.sh")
        
        with open(shell_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Shell script to generate and submit SLURM jobs for {method}\n")
            f.write(f"# This script creates .slurm files and submits them\n\n")
            
            f.write(f"BASE_DIR=\"{base_dir}\"\n")
            f.write(f"LOGS_BASE=\"{logs_dir}\"\n")
            f.write(f"METHOD=\"{method}\"\n\n")
            
            f.write("# Loop over seeds\n")
            for seed in seeds:
                # Create seed-specific log directory
                f.write(f"SEED={seed}\n")
                f.write(f"mkdir -p \"${{LOGS_BASE}}/seed${{SEED}}\"\n\n")
                
                # Determine environment setup
                if method in ['InRunDataShapleyGhost', 'LoGRA', 'Kairos']:
                    env_setup = """# Activate Python 3.12 virtual environment (GhostSuite)
source ~/ghost_env/.venv/bin/activate
python --version
"""
                else:
                    env_setup = """# Activate Python 3.9 conda environment
conda activate py39_env
python --version
"""
                
                # Create SLURM script
                f.write(f"cat > ${{BASE_DIR}}/scripts/run_hepmass_{method}_seed${{SEED}}.slurm << SLURM_SCRIPT\n")
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks=1\n")
                f.write("#SBATCH --cpus-per-task=56\n")
                f.write(f"#SBATCH --output=${{LOGS_BASE}}/seed${{SEED}}/run_hepmass_{method}_%j_seed${{SEED}}.log\n")
                f.write(f"#SBATCH --error=${{LOGS_BASE}}/seed${{SEED}}/run_hepmass_{method}_%j_seed${{SEED}}.err\n")
                f.write("#SBATCH --time=36:00:00\n")
                f.write(f"#SBATCH --job-name=hepmass_{method}_seed${{SEED}}\n\n")
                
                f.write("# ---------------------------------\n")
                f.write("# Environment setup\n")
                f.write("# ---------------------------------\n")
                f.write("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
                f.write("export PYTHONUNBUFFERED=1\n\n")
                
                f.write("source ~/.bashrc\n\n")
                f.write("# Deactivate any existing environment\n")
                f.write("conda deactivate 2>/dev/null || true\n")
                f.write(env_setup)
                f.write("\n")
                
                f.write("# ---------------------------------\n")
                f.write("# Move to script directory\n")
                f.write("# ---------------------------------\n")
                f.write(f"cd \"${{BASE_DIR}}\"\n\n")
                
                f.write("# ---------------------------------\n")
                f.write("# Run Python script for specific method\n")
                f.write("# ---------------------------------\n")
                f.write(f"echo \"Starting HEPMASS 1M experiment for method: {method}\"\n")
                f.write("echo \"Job ID: $SLURM_JOB_ID\"\n")
                f.write("echo \"Start time: $(date)\"\n")
                f.write("echo \"Seed: $SEED\"\n\n")
                
                f.write(f"python ${{BASE_DIR}}/run_hepmass_dataval.py --seed ${{SEED}} --method {method} --job_id $SLURM_JOB_ID\n\n")
                
                f.write("# ---------------------------------\n")
                f.write("# Completion message\n")
                f.write("# ---------------------------------\n")
                f.write(f"echo \"Job completed for method: {method}\"\n")
                f.write("echo \"End time: $(date)\"\n")
                f.write("echo \"Job ID: $SLURM_JOB_ID completed successfully\"\n\n")
                
                f.write("# Clean up environment\n")
                f.write("conda deactivate 2>/dev/null || true\n")
                f.write("SLURM_SCRIPT\n\n")
                
                # Submit the SLURM script
                f.write(f"# Submit the job\n")
                f.write(f"sbatch ${{BASE_DIR}}/scripts/run_hepmass_{method}_seed${{SEED}}.slurm\n\n")
                f.write("# Optional: small delay to avoid overwhelming the scheduler\n")
                f.write("sleep 0.5\n\n")
            
            f.write(f"echo \"All jobs submitted for {method} with seeds {seeds[0]}-{seeds[-1]}\"\n")
        
        os.chmod(shell_script_path, 0o755)
        print(f"  Created: {shell_script_path}")
    
    print("\n" + "=" * 70)
    print("SHELL SCRIPT GENERATION COMPLETE!")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Logs directory: {logs_dir}")
    print("\nTo run all methods, execute:")
    print(f"  bash {master_script_path}")
    print("\nOr to run a specific method:")
    print(f"  bash {scripts_dir}/run_hepmass_METHODNAME.sh")
    print("=" * 70)


if __name__ == "__main__":
    if METHOD == "ALL":
        generate_shell_scripts()
    else:
        # Run specific method
        try:
            exper_med = run_method_experiment(METHOD)
        except KeyboardInterrupt:
            print(f"\nExperiment for {METHOD} interrupted by user.")
        except Exception as e:
            print(f"\nFatal error in {METHOD} experiment: {e}")
            import traceback
            traceback.print_exc()