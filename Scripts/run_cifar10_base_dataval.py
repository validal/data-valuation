# run_cifar10_dataval.py

from random import seed
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import time
import torch

# Ensure dataset registry

from opendataval.dataloader import mix_labels, DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, RandomEvaluator, InRunDataShapleyGhost, LoGRA
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods    import (
    discover_corrupted_sample,
    noisy_detection,
    remove_high_low,
    save_dataval
)
from opendataval.dataval.knnshap import KNNShapleyLSH

# ============================
# Argument parsing
# ============================
parser = argparse.ArgumentParser(description="Run CIFAR-10 data valuation experiment")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=[
        "DataOob", "AME", "DataBanzhaf", "DataShapley",
        "InfluenceSubsample", "KNNShapley", "AKShapley", "KAIROS",
        "DVRL", "BetaShapley", "LAVA", "SAVA", "InRunDataShapleyGhost", "LoGRA", "ALL"
    ],
)
parser.add_argument("--job_id", type=int, default=1)
parser.add_argument(
    "--proportion",
    type=float,
    default=None,
    help="Bootstrap proportion for DataOob (e.g. 1.0, 0.7, 0.5, 0.2)"
)
parser.add_argument(
    "--noise_rate",
    type=float,
    default=0.2,
    help="Label noise rate (default: 0.2)"
)
parser.add_argument(
    "--k_neighbors",
    type=int,
    default=100,
    help="K value for KNNShapley (default: 100)"
)
parser.add_argument(
    "--subset_size",
    type=int,
    default=None,
    help="Subset size for InfluenceSubsample"
)
parser.add_argument(
    "--num_models",
    type=int,
    default=None,
    help="Number of models (AME, DataBanzhaf, BetaShapley, etc)"
)
parser.add_argument(
    "--mc_epochs",
    type=int,
    default=None,
    help="MC epochs for DataShapley (default: 1000)"
)
parser.add_argument(
    "--lam_y",
    type=float,
    default=None,
    help="Lambda Y for LAVA/SAVA (LAVA default: 100.0, SAVA default: 5.0)"
)
parser.add_argument(
    "--lambda_weight",
    type=float,
    default=None,
    help="Lambda weight for Kairos (default: 1.0)"
)
parser.add_argument(
    "--lam_x",
    type=float,
    default=None,
    help="Lambda X for LAVA/SAVA (default: 1.0)"
)
parser.add_argument(
    "--blur",
    type=float,
    default=None,
    help="Blur parameter for LAVA/SAVA (default: 0.05)"
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Number of samples for Kairos (default: 10000)"
)
parser.add_argument(
    "--rl_epochs",
    type=int,
    default=None,
    help="RL epochs for DVRL (default: 10000)"
)
parser.add_argument(
    "--rl_batch_size",
    type=int,
    default=None,
    help="RL batch size for DVRL (default: 32)"
)
parser.add_argument(
    "--dvrl_epochs_list",
    type=str,
    default=None,
    help="Comma-separated rl_epochs values for DVRL, crossed with rl_batch_size "
         "in one job (e.g. 5000,10000). Overrides the default [1000,2000,3000] sweep."
)
parser.add_argument(
    "--inrun_batch_sizes",
    type=str,
    default=None,
    help="Comma-separated batch sizes for InRunDataShapleyGhost, run together "
         "in one job (e.g. 1024,4096,10000,20000). Overrides the default "
         "[32,64,128,256,512] sweep."
)
parser.add_argument(
    "--ak_k_neighbors",
    type=int,
    default=None,
    help="k_neighbors for AKShapley (default: 100)"
)
parser.add_argument(
    "--ak_n_hash_table",
    type=int,
    default=None,
    help="n_hash_table for AKShapley (default: 100)"
)
parser.add_argument(
    "--ak_eps",
    type=float,
    default=None,
    help="eps for AKShapley (default: 0.001)"
)
parser.add_argument(
    "--ak_alpha",
    type=float,
    default=None,
    help="alpha for AKShapley (default: 0.5)"
)
parser.add_argument(
    "--ak_dist_rand",
    type=float,
    default=None,
    help="dist_rand for AKShapley (default: 31.9286)"
)
parser.add_argument(
    "--ak_t",
    type=float,
    default=None,
    help="t for AKShapley (default: 2.2280)"
)
args = parser.parse_args()

SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id
PROPORTION = args.proportion
NOISE_RATE = args.noise_rate
K_NEIGHBORS = args.k_neighbors
SUBSET_SIZE = args.subset_size
NUM_MODELS = args.num_models
MC_EPOCHS = args.mc_epochs
LAM_Y = args.lam_y
LAMBDA_WEIGHT = args.lambda_weight
LAM_X = args.lam_x
BLUR = args.blur
NUM_SAMPLES = args.num_samples
RL_EPOCHS = args.rl_epochs
RL_BATCH_SIZE = args.rl_batch_size
DVRL_EPOCHS_LIST = (
    [int(x) for x in args.dvrl_epochs_list.split(",")] if args.dvrl_epochs_list else None
)
INRUN_BATCH_SIZES = (
    [int(x) for x in args.inrun_batch_sizes.split(",")] if args.inrun_batch_sizes else None
)
AK_K_NEIGHBORS = args.ak_k_neighbors
AK_N_HASH_TABLE = args.ak_n_hash_table
AK_EPS = args.ak_eps
AK_ALPHA = args.ak_alpha
AK_DIST_RAND = args.ak_dist_rand
AK_T = args.ak_t

print("Running CIFAR-10 experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")
print(f"  - NOISE_RATE: {NOISE_RATE}")
print(f"  - K_NEIGHBORS: {K_NEIGHBORS}")

# ============================================================
# SEEDING (for reproducibility)
# ============================================================

def set_global_seeds(seed=42):
    """Set all global random seeds for reproducibility."""
    print(f"\n[Seeding] Setting global random seeds to {seed}")
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# EXPERIMENT MEDIATOR CREATION
# ============================================================

def create_experiment_mediator(cache_dir="./data_files/", seed=42):
    """Create experiment mediator for CIFAR-10 with ResNet18 embeddings (512-dim).

    Input: x_train shape (40000, 512) from registered cifar10-embedding-resnet18 dataset.
    """

    print("\n" + "="*60)
    print("CREATING EXPERIMENT MEDIATOR (ResNet18 512-dim embeddings)")
    print("="*60)
        # Load the registered embedding dataset
    print("\nLoading registered dataset 'cifar10-embedding-resnet18'...")
    # Fixed split seed so train/valid/test indices never change across --seed runs;
    # label noise (below) still varies with SEED.
    SPLIT_SEED = 42
    fetcher = DataFetcher(
        "cifar10-embedding-resnet18",
        cache_dir=cache_dir,
        random_state=SPLIT_SEED
    )

    print(f"Total samples loaded: {fetcher.num_points}")

    # Split deterministically with fixed seed (40K/10K/10K)
    print("\nSplitting with deterministic counts (seed={})...".format(seed))
    fetcher.split_dataset_by_count(40000, 10000, 10000)

    # Add label noise
    print(f"\nAdding {NOISE_RATE*100}% label noise...")
    fetcher = fetcher.noisify(mix_labels, noise_rate=NOISE_RATE, rng=SEED)

    # Prediction model: ClassifierMLP with auto-detected input dimension
    input_dim = fetcher.x_train.shape[1]  # Auto-detect from embeddings (512-dim for ResNet18)
    print(f"\nCreating ClassifierMLP with parameters:")
    print(f"  input_dim: {input_dim} (auto-detected from embeddings)")
    print(f"  num_classes: 10")
    print(f"  layers: 2")
    print(f"  hidden_dim: 100")

    from opendataval.model import ClassifierMLP

    pred_model = ClassifierMLP(
        input_dim=input_dim,
        num_classes=10,
        layers=2,
        hidden_dim=100,
    )

    # Training kwargs from tuning
    train_kwargs = {
        "epochs": 5,
        "batch_size": 32,
        "lr": 0.001,
    }

    print(f"\nTraining kwargs from tuning:")
    print(f"  epochs: {train_kwargs['epochs']}")
    print(f"  batch_size: {train_kwargs['batch_size']}")
    print(f"  lr: {train_kwargs['lr']}")

    exper_med = ExperimentMediator(
        fetcher,
        pred_model,
        metric_name="accuracy",
        train_kwargs=train_kwargs
    )

    # Test baseline performance
    print("\nTesting baseline performance...")
    data = exper_med.fetcher.datapoints
    model = exper_med.pred_model.clone()
    model.fit(data[0], data[1], **exper_med.train_kwargs)
    y_pred = model.predict(data[4])
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu()
    if isinstance(y_pred, np.ndarray):
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = np.argmax(y_pred.numpy(), axis=1)
    y_true_class = np.argmax(data[5], axis=1)
    accuracy = y_pred_class == y_true_class
    if hasattr(accuracy, 'numpy'):
        accuracy = accuracy.numpy()
    baseline = np.mean(accuracy)
    print(f"Baseline accuracy: {baseline:.4f}")

    return exper_med


# ============================================================
# EVALUATOR CREATION (for scalability study)
# ============================================================

def create_method_evaluators(method_name, output_dir=None):
    """Create evaluators for a specific method."""
    print(f"\nCreating evaluators for method: {method_name}")
    
    if method_name == "DataOob":
        # Test different bootstrap proportions for scalability study
        proportions = [0.1,0.5,1.0] if PROPORTION is None else [PROPORTION]
        num_models = NUM_MODELS if NUM_MODELS is not None else 0
        evaluators = [
            DataOob(num_models=num_models, proportion=p, random_state=SEED)
            for p in proportions
        ]

    elif method_name == "AME":
        num_models = NUM_MODELS if NUM_MODELS is not None else 5000
        evaluators = [AME(num_models=num_models, random_state=SEED)]
                
    elif method_name == "DataBanzhaf":
        num_models = NUM_MODELS if NUM_MODELS is not None else 20000
        evaluators = [DataBanzhaf(num_models=num_models, random_state=SEED)]
        
    elif method_name == "DataShapley":
        mc_epochs = MC_EPOCHS if MC_EPOCHS is not None else 1000
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs,
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_SEED{SEED}",
                random_state=SEED
            )
        ]
        
    elif method_name == "InfluenceSubsample":
        n_models = NUM_MODELS if NUM_MODELS is not None else 700000
        s_size = SUBSET_SIZE if SUBSET_SIZE is not None else 100
        evaluators = [
            InfluenceSubsample(
                num_models=n_models,
                subset_size=s_size,
                # +1 so this evaluator's own subsample RNG doesn't reuse the
                # exact seed passed to noisify()'s rng=SEED (see Hep1K/Adult
                # debug: RandomState(seed).choice(N,K,replace=False) is
                # deterministic, so matching seeds can silently correlate --
                # confirmed here since subset_size=8000 == noise_count=8000).
                random_state=SEED + 1
            )
        ]

    elif method_name == "KAIROS":
        # ONE evaluator - kernels computed once, lambda tuned later
        lambda_weight = LAMBDA_WEIGHT if LAMBDA_WEIGHT is not None else 0.97
        num_samples = NUM_SAMPLES if NUM_SAMPLES is not None else 10000
        evaluators = [
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
    elif method_name == "KNNShapley":
        # Regular KNNShapley (exact neighbors)
        k_value = K_NEIGHBORS if K_NEIGHBORS is not None else 100
        print(f"\nCreating KNNShapley evaluator with k={k_value}")
        evaluators = [KNNShapley(k_neighbors=k_value)]

    elif method_name == "AKShapley":
        # KNNShapley with LSH (Locality Sensitive Hashing - approximate neighbors)
        k_neighbors = AK_K_NEIGHBORS if AK_K_NEIGHBORS is not None else 100
        n_hash_table = AK_N_HASH_TABLE if AK_N_HASH_TABLE is not None else 100
        eps = AK_EPS if AK_EPS is not None else 0.001
        alpha = AK_ALPHA if AK_ALPHA is not None else 0.5
        dist_rand = AK_DIST_RAND if AK_DIST_RAND is not None else 31.9286
        t = AK_T if AK_T is not None else 2.2280

        evaluators = []
        print(f"\nCreating AKShapley (KNNShapleyLSH) with k={k_neighbors}, n_hash_table={n_hash_table}, eps={eps}, alpha={alpha}, dist_rand={dist_rand}, t={t}")

        evaluator = KNNShapleyLSH(
            k_neighbors=k_neighbors,
            n_hash_table=n_hash_table,
            eps=eps,
            dist_rand=dist_rand,
            t=t,
            alpha=alpha,
            random_state=SEED
        )
        evaluators.append(evaluator)
        
    elif method_name == "DVRL":
        batch_sizes = [RL_BATCH_SIZE] if RL_BATCH_SIZE is not None else [32, 64, 128, 256, 512]
        epochs_values = DVRL_EPOCHS_LIST if DVRL_EPOCHS_LIST is not None else [1000, 2000, 3000]
        evaluators = [
            DVRL(
                rl_epochs=rl_epochs,
                rl_batch_size=rl_batch_size,
                random_state=SEED
            )
            for rl_batch_size in batch_sizes
            for rl_epochs in epochs_values
        ]
                    
    elif method_name == "BetaShapley":
        num_models = NUM_MODELS if NUM_MODELS is not None else 10000
        evaluators = [BetaShapley(num_models=num_models, random_state=SEED)]
        
    elif method_name == "LAVA":
        lam_y = LAM_Y if LAM_Y is not None else 1.0
        lam_x = LAM_X if LAM_X is not None else 1.0
        blur = BLUR if BLUR is not None else 0.05
        evaluators = [
            LavaEvaluator(
                lam_y=lam_y,
                lam_x=lam_x,
                blur=blur,
                debug=True
            )
        ]
    elif method_name == "SAVA":
        lam_y = LAM_Y if LAM_Y is not None else 5.0
        lam_x = LAM_X if LAM_X is not None else 1.0
        blur = BLUR if BLUR is not None else 0.05
        evaluators = [
            SavaEvaluator(
                lam_y=lam_y,
                lam_x=lam_x,
                blur=blur,
                batch_size=1024,
                debug=True
            )
        ]

    elif method_name == "InRunDataShapleyGhost":
        EPOCHS = 5
        BATCH_SIZES = INRUN_BATCH_SIZES if INRUN_BATCH_SIZES is not None else [32, 64, 128, 256, 512]
        LR = 0.001
        # Debug CSVs are named only by epochs/batch_size/lr (no seed), so they
        # must live under this run's own output_dir to avoid seeds overwriting
        # each other's debug files.
        plot_dir = os.path.join(output_dir, "debugs") if output_dir else "debugs"
        evaluators = [
            InRunDataShapleyGhost(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                random_state=SEED,
                verbose=True,
                # OneCycleLR (the default) divides by zero on a ragged last
                # batch. scheduler_type="none" disables the LR scheduler
                # entirely, so the optimizer uses a fixed learning_rate.
                scheduler_type="none",
                save_plots=True,
                plot_dir=plot_dir
            )
            for BATCH_SIZE in BATCH_SIZES
        ]

    elif method_name == "LoGRA":
        EPOCHS = 5
        BATCH_SIZE = 32
        LR = 0.001
        LORA_CONFIGS = [
            ('none', 'none'),
            ('none', 'raw'),
            ('none', 'kfac'),
            ('none', 'ekfac'),
            ('pca', 'kfac'),
            ('pca', 'ekfac'),
            ('pca', 'raw'),
            ('random', 'kfac')]
        evaluators = [
            LoGRA(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                lora=config[0],
                hessian=config[1],
                random_state=SEED,
                verbose=True
            )
            for config in LORA_CONFIGS
        ]

    elif method_name == "ALL":
        evaluators = []
        for m in ["DataOob", "KNNShapley"]:
            evaluators.extend(create_method_evaluators(m, output_dir))
            
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    print(f"Created {len(evaluators)} evaluators for {method_name}")
    return evaluators


# ============================================================
# TIME/MEMORY REPORT
# ============================================================

def save_time_memory_report(exper_med, output_dir, method_name):
    """Save time and memory report for all evaluators."""
    print(f"\nCreating time/memory report for {method_name}...")
    
    rows = []
    successful_evaluators = 0
    
    for idx, ev in enumerate(exper_med.data_evaluators):
        name = str(ev)
        
        try:
            _ = ev.data_values
            
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
                    "gpu_peak_alloc_bytes": comb.get("gpu_peak_allocated_bytes", 0),
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
    
    if rows:
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"time_memory_{method_name}.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Time/memory report saved to: {output_path}")
        print(f"Total evaluators: {len(rows)}")
        print(f"Successful: {successful_evaluators}")
        print(f"Failed: {len(rows) - successful_evaluators}")
    
    return rows


# ============================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================

def run_method_experiment(method_name):
    """Run data valuation experiment on CIFAR-10 with ResNet18 embeddings (512-dim).

    Data flow:
    1. Load cifar10-embedding-resnet18 dataset (cached after first run)
    2. Apply deterministic split: 40K train, 10K valid, 10K test (seed=42)
    3. Add label noise (20% by default)
    4. Compute data values using specified method
    """
    print("=" * 70)
    print(f"CIFAR-10 Data Valuation - ResNet9 Embeddings (128-dim) - {method_name}")
    print("=" * 70)

    # Set reproducibility seeds
    set_global_seeds(SEED)

    start_time = time.time()
  

    # Create experiment mediator
    print("\n3. Creating experiment mediator...")
    exper_med = create_experiment_mediator(
        cache_dir="./data_files/", seed=SEED
    )
    
    # Create output directory (matching DogFish structure)
    base_results_dir = "Results"
    output_dir = os.path.join(base_results_dir, method_name, f'SEED{SEED}_JOB{JOB_ID}')
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")

    # Create method-specific evaluators
    print(f"\n4. Creating evaluators for {method_name}...")
    all_evaluators = create_method_evaluators(method_name, output_dir)

    # Compute data values
    print(f"\n5. Computing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    try:
        exper_med = exper_med.compute_data_values(data_evaluators=all_evaluators)
        print(f"✓ {method_name} computation completed successfully")
    except Exception as e:
        print(f"✗ {method_name} computation failed: {e}")
        # Continue to save partial results

    # Special handling for KAIROS: evaluate each lambda (kernel reuse)
    if method_name == "KAIROS":
        import time as time_module
        LAMBDA_VALUES = [0, 0.5, 0.8, 0.9, 0.97, 1.0]
        evaluator = all_evaluators[0]
        base_output_dir = output_dir
        lambda_timing_results = []

        print(f"\n6. Running lambda tuning for KAIROS...")
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
        print(f"\n6. Running evaluations for {method_name}...")
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
        print(f"\n7. Saving data values for {method_name}...")
        try:
            values = exper_med.evaluate(save_dataval, save_output=True)
            print(f"  ✓ Data values saved")
        except Exception as e:
            print(f"  ✗ Data values save failed: {e}")
    
    # Save time/memory report
    print(f"\n8. Generating time/memory report...")
    save_time_memory_report(exper_med, output_dir, method_name)

    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Save final summary
    summary_path = os.path.join(output_dir, f"summary_{method_name}.txt")
    data_source = "ResNet18 embeddings (generated)"
    feature_dim = "512-dim"
    with open(summary_path, 'w') as f:
        f.write(f"CIFAR-10 Data Valuation - {method_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Job ID: {JOB_ID}\n")
        f.write(f"Noise Rate: {NOISE_RATE}\n")
        f.write(f"Data: Train/Valid/Test = 40000/10000/10000\n")
        f.write(f"Data Source: {data_source}\n")
        f.write(f"Features: {feature_dim}\n")
        f.write(f"Split Method: deterministic (fixed seed={SEED})\n")
        f.write(f"Classes: 10 (CIFAR-10)\n\n")
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: ClassifierMLP (2-layer)\n")
        f.write(f"Input Dim: 512\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Evaluators: {len(all_evaluators)}\n")
        f.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
        f.write(f"Output Directory: {output_dir}\n")

    print(f"\n{'='*70}")
    print(f"{method_name} EXPERIMENT COMPLETED!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*70}")
    
    return exper_med


# ============================================================
# JOB SCRIPT GENERATION FOR CLUSTER
# ============================================================

def run_all_methods():
    """Create job scripts for all methods to run on cluster."""
    print("Creating job scripts for all methods...")
    
    methods = ['DataOob', 'AME', 'DataBanzhaf', 'DataShapley',
               'InfluenceSubsample', 'KNNShapley', 'KAIROS',
               'DVRL', 'BetaShapley', 'LAVA']
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Master script
    master_script = """#!/bin/bash
# Master script to submit all CIFAR-10 data valuation jobs

METHODS=("DataOob" "AME" "DataBanzhaf" "DataShapley" "InfluenceSubsample" 
         "KNNShapley" "DVRL" "BetaShapley" "LAVA")

for method in "${METHODS[@]}"; do
    echo "Submitting job for method: $method"
    sbatch run_cifar10_${method}.sh
    sleep 2
done

echo "All jobs submitted!"
"""
    
    with open("submit_all_cifar10_methods.sh", "w") as f:
        f.write(master_script)
    
    os.chmod("submit_all_cifar10_methods.sh", 0o755)
    print("Created master script: submit_all_cifar10_methods.sh")
    
    # Individual job scripts
    for method in methods:
        job_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --output=logs/run_cifar10_{method}_%j.log
#SBATCH --error=logs/run_cifar10_{method}_%j.err
#SBATCH --time=36:00:00
#SBATCH --job-name=cifar10_{method}

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.bashrc
conda deactivate 2>/dev/null || true
source ~/ghost_env/.venv/bin/activate

# Move to script directory
cd /srv/lustre01/project/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Fine_grained_valuation/Revision/CIFAR10

# Create logs directory
mkdir -p logs

# Run experiment
echo "Starting CIFAR-10 experiment for method: {method}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python run_cifar10_dataval.py --seed 42 --method {method} --job_id $SLURM_JOB_ID --noise_rate 0.2

echo "Job completed for method: {method}"
echo "End time: $(date)"
"""
        
        script_filename = f"run_cifar10_{method}.sh"
        with open(script_filename, "w") as f:
            f.write(job_script)
        
        os.chmod(script_filename, 0o755)
        print(f"Created job script: {script_filename}")
    
    print("\nTo run all methods, execute:")
    print("  ./submit_all_cifar10_methods.sh")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CIFAR-10 DATA VALUATION WITH RESNET9 EMBEDDINGS (128-dim)")
    print("="*70)
    print(f"\n📊 Dataset: cifar10-embedding-resnet9 (registered)")
    print(f"   - Data: 60K samples → Split 40K/10K/10K (deterministic, seed=42)")
    print(f"   - Embeddings: 128-dim ResNet9 features")
    print(f"   - Caching: Automatic (fast on subsequent runs)")
    print(f"\n⚠ IMPORTANT: Before running methods, ensure hyperparameters are tuned!")
    print(f"  Run: cd ModelTuning && ./run_tuning.sh")
    print(f"  This will create: ModelTuning/best_config.json")
    print(f"\nConfiguring experiment with:")
    print(f"  - Seed: {SEED}")
    print(f"  - Method: {METHOD}")
    print(f"  - Noise Rate: {NOISE_RATE}")
    print(f"  - Job ID: {JOB_ID}")
    print(f"\nUsage:")
    print(f"  python run_cifar10_dataval.py --seed 42 --method KNNShapley")
    print("\n" + "="*70 + "\n")

    if METHOD == "ALL":
        run_all_methods()
    else:
        try:
            exper_med = run_method_experiment(METHOD)
        except KeyboardInterrupt:
            print(f"\nExperiment for {METHOD} interrupted by user.")
        except Exception as e:
            print(f"\nFatal error in {METHOD} experiment: {e}")
            import traceback
            traceback.print_exc()