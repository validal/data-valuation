# run_hepmass_dataval.py
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
from sklearn.preprocessing import StandardScaler

# Ensure dataset registry (in case of fresh session)

from opendataval.dataloader import DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, RandomEvaluator,
    InRunDataShapleyGhost, LoGRA)
from opendataval.dataval.knnshap import KNNShapleyLSH
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.model import ClassifierMLP
import torch

# GhostSuite and LogIX imports
try:
    print("✓ GhostSuite import OK")
except ImportError:
    print("⚠ GhostSuite not available (optional for InRunDataShapleyGhost)")

try:
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
# # Trains one model to convergence, values each point as negative loss
# # High-value points = low loss (clean samples)
# # Low-value points = high loss (noisy/hard samples)
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
                           'InfluenceSubsample', 'KNNShapley','AKShapley',
                           'DVRL', 'BetaShapley', 'LAVA','SAVA', 'InRunDataShapleyGhost',
                           'LoGRA', 'Kairos', 'ALL'],
                   help='Method to run (or ALL for all methods)')
parser.add_argument(
    "--lam_y",
    type=float,
    default=None,
    help="Label distance weight for SAVA. If not set, use all values [1, 5, 10, 50, 100]."
)
parser.add_argument(
    "--subset_size",
    type=int,
    default=500,
    help="Subset size for InfluenceSubsample"
)
parser.add_argument(
    "--num_models",
    type=int,
    default=100000,
    help="Number of models for InfluenceSubsample, AME, and other methods"
)
parser.add_argument('--dvrl_epochs_list', type=str, default=None,
                   help='DVRL: comma-separated rl_epochs values (default: 5000)')
parser.add_argument('--dvrl_batch_size', type=int, default=None,
                   help='DVRL: rl_batch_size (default: 64)')
args = parser.parse_args()

# Set seeds
SEED = args.seed
METHOD = args.method
LAM_Y = args.lam_y
SUBSET_SIZE = args.subset_size
NUM_MODELS = args.num_models
DVRL_EPOCHS_LIST = [int(x) for x in args.dvrl_epochs_list.split(",")] if args.dvrl_epochs_list else None
DVRL_BATCH_SIZE = args.dvrl_batch_size

print(f"Running experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")

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

    # Define data directory with absolute path
    DATA_DIR = "/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/Hep1K/Data"

    # Load datasets from CSV files
    train_set = pd.read_csv(os.path.join(DATA_DIR, 'hepmass_train_1000.csv'))
    valid_set = pd.read_csv(os.path.join(DATA_DIR, 'hepmass_valid_200.csv'))
    test_set = pd.read_csv(os.path.join(DATA_DIR, 'hepmass_test_500.csv'))
    
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
        "batch_size": 32,
        "lr": 0.001,
    }

    # Create ExperimentMediator with proper train_kwargs
    exper_med = ExperimentMediator(
        fetcher=fetcher,
        pred_model=pred_model,
        metric_name=metric_name,
        train_kwargs=train_kwargs
    )

    # Test baseline performance
    print("\nTesting baseline performance...")
    data = exper_med.fetcher.datapoints
    model = exper_med.pred_model.clone()
    model.fit(data[0], data[1], **exper_med.train_kwargs)
    y_pred = model.predict(data[4])
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    baseline = exper_med.metric(y_pred, data[5])
    print(f"Baseline metric ({metric_name}): {baseline:.4f}")

    return exper_med

def create_method_evaluators(method_name):
    """Create evaluators for a specific method."""
    print(f"Creating evaluators for method: {method_name}")

    # Your original model sizes
    MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
    
    if method_name == "DataOob":
        PROPORTIONS = [0.5]
        MODEL_SIZES = [10]
        evaluators = [
            DataOob(num_models=m, proportion=p, random_state=s)
            for m in MODEL_SIZES
            for p in PROPORTIONS
            for s in [SEED]
            ]
        
    elif method_name == "AME":
        MODEL_SIZES = [100000]
        evaluators = []
        for m in MODEL_SIZES:
            for s in [1]:
                if m <= 1:
                    continue
                ame = AME(num_models=m, random_state=s)
                evaluators.append(ame)
                
    elif method_name == "DataBanzhaf":
        MODEL_SIZES = [1000,10000]

        evaluators = [
            DataBanzhaf(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in [SEED]
        ]
        
    elif method_name == "DataShapley":
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs, 
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_run{run_idx}_SEED_{SEED}_HEPMASS1k",
                random_state=run_idx
            )
            for mc_epochs in [700]
            for run_idx in [SEED]  # 1..10
        ]
        
    elif method_name == "InfluenceSubsample":

        INFLUENCE_PROPORTIONS = [0.1,0.2,0.5, 0.7]
        MODEL_SIZES = [1000]
        evaluators = []

        for p in [INFLUENCE_PROPORTIONS[3]]:  # Use only the first proportion for testing
            for s in [SEED]:  # 1..10
                evaluators.append(
                    InfluenceSubsample(
                        num_models=NUM_MODELS,
                        #proportion=0.7,
                        subset_size=SUBSET_SIZE,
                        # random_state offset by +1 so this evaluator's own subsample
                        # RNG doesn't reuse the exact seed passed to noisify()'s rng=SEED.
                        # With matching (seed, N, K) both draw identical first samples
                        # via RandomState(seed).choice(N, K, replace=False) -- see
                        # mix_labels() in opendataval/dataloader/noisify.py -- which
                        # silently leaks the noisy-label indices into model 0's subset
                        # whenever subset_size happens to equal the noise count.
                        random_state=s + 1,
                        verbose=True
                    )

                )
            
    elif method_name == "KNNShapley":
        evaluators = []
        for rs in [SEED]:
            for k in [10]:
                evaluators.append(KNNShapley(k_neighbors=k, random_state=rs, debug=True))
            #evaluators.append(KNNShapley(k_neighbors=10, random_state=rs))
            #evaluators.append(KNNShapleyLSH(k_neighbors=10, n_hash_table=20, eps=0.01,dist_rand=7.2352 ,t=2.2510,alpha=0.5, random_state=rs))
    elif method_name == "AKShapley":
        evaluators = []
        for k in [10]:
            for eps in [0.01]:
                for alpha in [0.1]:
                    for n_hash_table in [10]:
                        evaluators.append(
                            KNNShapleyLSH(
                                k_neighbors=k,
                                n_hash_table=n_hash_table,
                                eps=eps,
                                dist_rand=7.2352,
                                t=2.2510,
                                alpha=alpha,
                                random_state=SEED,
                            )
                        )
    
    elif method_name == "DVRL":
        BATCH_SIZES = [DVRL_BATCH_SIZE] if DVRL_BATCH_SIZE is not None else [64]
        MODEL_SIZES = DVRL_EPOCHS_LIST if DVRL_EPOCHS_LIST is not None else [5000]

        evaluators = []
        for rl_epochs in MODEL_SIZES:
            for batch_size in BATCH_SIZES:
                for random_state in [SEED]:  # 1 to 10
                    evaluators.append(
                        DVRL(
                            rl_epochs=rl_epochs,
                            rl_batch_size=batch_size,
                            random_state=random_state
                        )
                    )
                    
    elif method_name == "BetaShapley":
        MODEL_SIZES = [100, 500, 1000]
        evaluators = [
            BetaShapley(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in [SEED]
        ]
        
    elif method_name == "LAVA":
        LAM_Y = [1, 5, 10, 50, 100]
        evaluators = [
            LavaEvaluator(blur=0.05, debug=True, lam_x=1.0, lam_y=lam_y, random_state=s)
            for lam_y in LAM_Y
            for s in [SEED]
        ]
    elif method_name == "SAVA":
        lam_y_values = [1, 5, 10, 50, 100]
     

        evaluators = [
            SavaEvaluator(
                batch_size=1024,           # points per train/val batch
                lam_x=1.0,                # feature distance weight
                lam_y=lam_y,               # label distance weight (varied)
                p=2,                      # p-Wasserstein order
                blur=0.05,                # GeomLoss entropic scale
                mode="cls",               # "cls" or "reg"
                debug=True,
                random_state=SEED,
                stratified_batches=True
            )
            for lam_y in lam_y_values
        ]    
    elif method_name == "InRunDataShapleyGhost":
        # In-Run Shapley with different configurations and random states
        EPOCHS = 5
        BATCH_SIZES = [128]

        LR = 0.001
        evaluators = [
            InRunDataShapleyGhost(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                random_state=s,
                verbose=True,
                # pct_start = lr_peak_epoch / epochs; default lr_peak_epoch=5 with
                # EPOCHS=5 gives pct_start=1.0, a zero-length OneCycleLR anneal
                # phase that ZeroDivisionErrors on the last step when total steps
                # is small (n_train=1000, batch_size=128 -> only 7 batches/epoch
                # x 5 = 35 steps). lr_peak_epoch=1 keeps pct_start=0.2, normal.
                lr_peak_epoch=1,
            )
            for s in [SEED]
            for BATCH_SIZE in BATCH_SIZES
        ]

    elif method_name == "LoGRA":
        EPOCHS = 5
        BATCH_SIZE = 32
        LR = 0.001
        # LoGRA with different Hessian and LoRA approximation combinations + SEEDS
        LORA_CONFIGS = [
            ('none', 'none'),
            ('none', 'raw'),
            ('none', 'kfac'),
            ('pca', 'kfac'),
            ('pca', 'ekfac'),
            ('random', 'kfac')]
        evaluators = [
            LoGRA(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                lora=config[0],
                hessian=config[1],
                random_state=s,
                verbose=True
            )
            #for lora, hess in LORA_CONFIGS
            for s in [SEED]  # Use first 3 SEEDS
            for config in LORA_CONFIGS
        ]

    elif method_name == "Kairos":
        # Kairos with different configurations and random states
        LAMDA = [0, 0.5, 0.8, 0.9, 0.97, 1.0]

        evaluators = [
            bKairos(
                num_samples=1000,
                lambda_weight=lambda_weight,
                use_median_heuristic=True,
                random_state=s,
                debug=True
            )
            for lambda_weight in LAMDA
            for s in [SEED]  # Use first 3 SEEDS
        ]
        # evaluators = [
        #     bKairos(
        #         lambda_weight=lambda_weight,
        #         sigma_feature=3.0,
        #         batch_size=1024,           # points per train/val batch
        #         random_state=s,
        #         debug=True
        #     )
        #     for lambda_weight in [0.97]
        #     for s in [SEED]  # Use first 3 SEEDS
        # ]

    elif method_name == "ALL":
        # Combine all methods (for testing)
        evaluators = []
        for m in []:  # Start with LOO_Random for testing
            evaluators.extend(create_method_evaluators(m))

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
    
    start_time = time.time()
    
    # Create experiment mediator
    exper_med = create_experiment_mediator()

    # Create output directory with method name
    base_remote_dir = (
        f"/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/Hep1K/results"
    )
    #output_dir = os.path.join(base_remote_dir, f'{method_name}_SEED{SEED}')
    output_dir = os.path.join(base_remote_dir, method_name, f'SEED{SEED}')

    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create method-specific evaluators
    all_evaluators = create_method_evaluators(method_name)
    
    # Compute data values
    print(f"\nComputing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    
    try:
        import resource as _resource
        _ru_before = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _t_before = time.perf_counter()
        exper_med = exper_med.compute_data_values(data_evaluators=all_evaluators)
        _t_after = time.perf_counter()
        _ru_after = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        print(f"[OUTER-INSTRUMENT] compute_data_values() wall time: {_t_after - _t_before:.3f}s")
        print(f"[OUTER-INSTRUMENT] ru_maxrss before: {_ru_before} KB, after: {_ru_after} KB, delta: {_ru_after - _ru_before} KB")
        print(f"✓ {method_name} computation completed successfully")
    except Exception as e:
        print(f"✗ {method_name} computation failed: {e}")
        # Continue to save partial results
    
    # Run evaluations
    print(f"\nRunning evaluations for {method_name}...")
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
        f.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Total Evaluators: {len(all_evaluators)}\n")
    
    print(f"\n{'='*70}")
    print(f"{method_name} EXPERIMENT COMPLETED!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return exper_med

if __name__ == "__main__":
    try:
        exper_med = run_method_experiment(METHOD)
    except KeyboardInterrupt:
        print(f"\nExperiment for {METHOD} interrupted by user.")
    except Exception as e:
        print(f"\nFatal error in {METHOD} experiment: {e}")
        import traceback
        traceback.print_exc()
