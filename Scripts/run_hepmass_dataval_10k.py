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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure dataset registry (in case of fresh session)
import opendataval.dataloader.datasets  # triggers @Register decorators

from opendataval.dataloader import mix_labels, add_gauss_noise, DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, LeaveOneOut, RandomEvaluator
)
from opendataval.dataval.knnshap import KNNShapleyLSH
from opendataval.dataval.lava import SavaEvaluator   
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)
from opendataval.model.api import ClassifierSkLearnWrapper

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run HEPMASS data valuation experiment')
parser.add_argument('--seed', type=int, default=42, help='Random seed for the experiment')
parser.add_argument('--method', type=str, required=True, 
                   choices=['DataOob', 'AME', 'DataBanzhaf', 'DataShapley', 
                           'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
                           'DVRL', 'BetaShapley', 'LAVA','SAVA', 'ALL'],
                   help='Method to run (or ALL for all methods)')
parser.add_argument('--job_id', type=int, default=1, help='Job ID for naming outputs')
parser.add_argument('--k_neighbors', type=int, default=10, help='Number of nearest neighbors for KNN-based methods')
parser.add_argument('--n_hash_table', type=int, default=100, help='Number of hash tables for LSH-based methods')
parser.add_argument('--eps', type=float, default=0.01, help='Epsilon parameter for LSH-based methods')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for LSH-based methods')
args = parser.parse_args()

# Set seeds
SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id
K_NEIGHBORS = args.k_neighbors
N_HASH_TABLE = args.n_hash_table
EPS = args.eps
ALPHA = args.alpha

print(f"Running experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")


def load_and_prepare_hepmass():
    """Load HEPMASS datasets and prepare them for the experiment."""
    print("Loading HEPMASS datasets...")
    
    # Load datasets from CSV files
    train_set = pd.read_csv('hepmass_train_10000.csv')
    valid_set = pd.read_csv('hepmass_valid_2000.csv')
    test_set = pd.read_csv('hepmass_test_5000.csv')
    
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
    """Create and configure the ExperimentMediator for HEPMASS."""
    print("Creating HEPMASS experiment mediator...")
    
    # Load and prepare data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_prepare_hepmass()
    
    # Create the fetcher from your splits
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
    
    # Create prediction model
    metric_name = "accuracy"
    
    # Use ClassifierSkLearnWrapper with LogisticRegression
    pred_model = ClassifierSkLearnWrapper(LogisticRegression, fetcher.label_dim[0])
    
    # Create ExperimentMediator
    exper_med = ExperimentMediator(fetcher, pred_model, metric_name=metric_name)
    
    # Test baseline performance
    print("\nTesting baseline performance...")
    data = exper_med.fetcher.datapoints
    model = exper_med.pred_model.clone()
    model.fit(data[0], data[1], **exper_med.train_kwargs)
    y_pred = model.predict(data[4]).cpu()
    baseline = exper_med.metric(y_pred, data[5])
    print(f"Baseline metric ({metric_name}): {baseline:.4f}")
    
    return exper_med


def create_method_evaluators(method_name):
    """Create evaluators for a specific method."""
    print(f"Creating evaluators for method: {method_name}")
    
    # Your original model sizes
    MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000]#, 2000, 3000, 5000]
    LARGE_MODEL_SIZES = [8000,10000,15000,20000,30000, 50000]

    SEEDS = list(range(1, 11))  # 1..10
    
    if method_name == "DataOob":
        MODEL_SIZES = [10]
        PROPORTIONS = [1.0, 0.7, 0.5, 0.2,0.1]
        evaluators = [
            DataOob(num_models=m, proportion=p, random_state=s)
            for m in MODEL_SIZES
            for p in PROPORTIONS
            for s in SEEDS
        ]
        
    elif method_name == "AME":
        #MODEL_SIZES = [8000,10000,15000,20000]
        #MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000]# 2000, 3000, 5000]
        MODEL_SIZES = [2000, 3000]
        MODEL_SIZES = [5000]

        evaluators = []
        for m in [250000]:
            for s in [1]:
                if m <= 1:
                    continue
                ame = AME(num_models=m, random_state=s)
                evaluators.append(ame)
                
    elif method_name == "DataBanzhaf":
        #MODEL_SIZES = [8000,10000,15000,20000]
        #MODEL_SIZES = [30000,50000]
        MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]


        evaluators = [
            DataBanzhaf(num_models=m, random_state=s)
            for m in [1000000]
            for s in [1]
        ]
        
    elif method_name == "DataShapley":
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs, 
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_run{run_idx}_SEED_{SEED}_HEPMASS10K_exc",
                random_state=run_idx
            )
            for mc_epochs in [1000]
            for run_idx in [1]  # 1..10
        ]
        
    elif method_name == "InfluenceSubsample":
        # MODEL_SIZES = [8000,10000,15000,20000,30000]
        # INFLUENCE_PROPORTIONS = [0.1,0.2, 0.5, 0.7, 0.9]
        # evaluators = [
        #     InfluenceSubsample(num_models=m, proportion=p, random_state=s)
        #     for m in MODEL_SIZES
        #     for p in INFLUENCE_PROPORTIONS
        #     for s in SEEDS
        # ]

        SMALL_MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
        LARGE_MODEL_SIZES = [8000,10000,15000,20000,30000, 50000]

        INFLUENCE_PROPORTIONS = [0.01,0.02,0.03,0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        evaluators = []

        for p in [0.1]:
            for m in [5000000]:
                for s in [1]:
                    evaluators.append(
                        InfluenceSubsample(
                            num_models=m,
                            proportion=p,
                            random_state=s
                        )
                    )
            
    elif method_name == "LOO_Random":
        evaluators = [
            LeaveOneOut(),  # One time
        ] + [
            RandomEvaluator(random_state=s) for s in SEEDS  # 1..10
        ]
        
    elif method_name == "KNNShapley":
        print(f"Creating KNNShapleyLSH evaluators with k_neighbors={K_NEIGHBORS}, n_hash_table={N_HASH_TABLE}, eps={EPS}, alpha={ALPHA}")
        evaluators = [KNNShapleyLSH(k_neighbors=K_NEIGHBORS, n_hash_table=N_HASH_TABLE, eps=EPS, alpha=ALPHA,
                                    dist_rand=7.3182, t=2.2470, random_state=seed)
                    for seed in SEEDS]
        
    elif method_name == "DVRL":
        BATCH_SIZES = [32, 64, 128, 256, 512]
        rl_epochs = [8000,10000,15000,20000]
        evaluators = []
        for rl_epochs in [1000,2000]:
            for batch_size in BATCH_SIZES:
                for random_state in SEEDS:  # 1 to 10
                    evaluators.append(
                        DVRL(
                            rl_epochs=rl_epochs,
                            rl_batch_size=batch_size,
                            random_state=random_state
                        )
                    )
                    
    elif method_name == "BetaShapley":
        evaluators = [
            BetaShapley(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in SEEDS
        ]
    
    elif method_name == "SAVA":
        evaluators = [SavaEvaluator(batch_size=1024, lam_x=1, lam_y=1, debug=True,random_state=rs,stratified_batches=True) for rs in SEEDS]
    elif method_name == "LAVA":
        # Fixed blur and lam_x for main sweeps
        blur_fixed = 0.05
        
        evaluators = []
        
        # ===========================================
        # 1. Label weight sweep (lam_x=1, vary lam_y from 0 to 100)
        # ===========================================
        lam_y_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        for lam_y in lam_y_values:
            evaluators.append(
                LavaEvaluator(
                    blur=blur_fixed, 
                    debug=True, 
                    lam_x=1.0, 
                    lam_y=lam_y
                )
            )
        
        # ===========================================
        # 2. Inverse sweep (lam_y=1, vary lam_x from 0 to 100)
        # ===========================================
        lam_x_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        for lam_x in lam_x_values:
            evaluators.append(
                LavaEvaluator(
                    blur=blur_fixed,
                    debug=True,
                    lam_x=lam_x,
                    lam_y=1.0
                )
            )
        
        # ===========================================
        # 3. Blur sweep (speed/accuracy trade-off)
        # ===========================================
        blur_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        for blur_val in blur_values:
            evaluators.append(
                LavaEvaluator(
                    blur=blur_val,
                    debug=True,
                    lam_x=1.0,
                    lam_y=1.0
                )
            )
        
        # ===========================================
        # 4. Higher-order cost and entropy regularization
        # ===========================================
        # Different p-norms
        for p in [1, 2, 3, 4]:
            evaluators.append(
                LavaEvaluator(
                    blur=0.05,
                    debug=True,
                    lam_x=1.0,
                    lam_y=1.0,
                    p=p
                )
            )
        
        # Entropy regularization
        for entreg in [0.01, 0.05, 0.10, 0.20]:
            evaluators.append(
                LavaEvaluator(
                    blur=0.05,
                    debug=True,
                    lam_x=1.0,
                    lam_y=1.0,
                    entreg=entreg
                )
            )
        
        # Combined p and entreg
        evaluators.append(
            LavaEvaluator(
                blur=0.05,
                debug=True,
                lam_x=1.0,
                lam_y=1.0,
                p=4,
                entreg=0.05
            )
        )
        
        evaluators.append(
            LavaEvaluator(
                blur=0.05,
                debug=True,
                lam_x=1.0,
                lam_y=1.0,
                p=1,
                entreg=0.10
            )
        )
        
        
    elif method_name == "ALL":
        # Combine all methods (for testing)
        evaluators = []
        for m in ["LOO_Random"]:  # Start with LOO_Random for testing
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
        f"/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Hepmass10K"
    )
    output_dir = os.path.join(base_remote_dir, f'{method_name}_SEED{SEED}_JOB{JOB_ID}')
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create method-specific evaluators
    all_evaluators = create_method_evaluators(method_name)
    
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
    
    return exper_med


def run_all_methods():
    """Run all methods as separate jobs (this function just creates job scripts)."""
    print("Creating job scripts for all methods...")
    
    methods = ['DataOob', 'AME', 'DataBanzhaf', 'DataShapley', 
               'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
               'DVRL', 'BetaShapley', 'LAVA']
    
    # Create a master script to submit all jobs
    master_script = """#!/bin/bash
# Master script to submit all HEPMASS data valuation jobs

METHODS=("DataOob" "AME" "DataBanzhaf" "DataShapley" "InfluenceSubsample" 
         "LOO_Random" "KNNShapley" "DVRL" "BetaShapley" "LAVA")

for method in "${METHODS[@]}"; do
    echo "Submitting job for method: $method"
    sbatch run_hepmass_${method}.sh
    sleep 2
done

echo "All jobs submitted!"
"""
    
    with open("submit_all_methods.sh", "w") as f:
        f.write(master_script)
    
    os.chmod("submit_all_methods.sh", 0o755)
    print("Created master script: submit_all_methods.sh")
    
    # Create individual job scripts for each method
    for method in methods:
        job_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run_hepmass_{method}_%j.log
#SBATCH --error=logs/run_hepmass_{method}_%j.err
#SBATCH --time=36:00:00
#SBATCH --job-name=hepmass_{method}

# ---------------------------------
# Environment setup
# ---------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.bashrc
conda activate py39_env

# ---------------------------------
# Move to script directory
# ---------------------------------
cd "/home/mehdi.touil/ondemand/Experimental evaluation/Hepmass/"

# ---------------------------------
# Create logs directory
# ---------------------------------
mkdir -p logs

# ---------------------------------
# Run Python script for specific method
# ---------------------------------
echo "Starting HEPMASS experiment for method: {method}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python run_hepmass_dataval.py --seed 42 --method {method} --job_id $SLURM_JOB_ID

# ---------------------------------
# Completion message
# ---------------------------------
echo "Job completed for method: {method}"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID completed successfully"
"""
        
        script_filename = f"run_hepmass_{method}.sh"
        with open(script_filename, "w") as f:
            f.write(job_script)
        
        os.chmod(script_filename, 0o755)
        print(f"Created job script: {script_filename}")
    
    print("\nTo run all methods, execute:")
    print("  ./submit_all_methods.sh")
    print("\nOr to run a specific method:")
    print("  sbatch run_hepmass_METHODNAME.sh")


if __name__ == "__main__":
    if METHOD == "ALL":
        run_all_methods()
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