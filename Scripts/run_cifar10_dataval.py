# run_cifar10_dataval.py

from random import seed

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Ensure dataset registry
import opendataval.dataloader.datasets

from opendataval.dataloader import mix_labels, DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, ParallelLavaOOBEvaluator,
    LeaveOneOut, RandomEvaluator
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample,
    noisy_detection,
    remove_high_low,
    save_dataval
)
from opendataval.model.api import ClassifierSkLearnWrapper
from opendataval.dataval.knnshap import KNNShapleyLSH

# ============================
# Argument parsing
# ============================
parser = argparse.ArgumentParser(description="Run CIFAR-10 data valuation experiment")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_seed", type=int, default=1)

parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=[
        "DataOob", "AME", "DataBanzhaf", "DataShapley",
        "InfluenceSubsample", "LOO_Random", "KNNShapley",
        "DVRL", "BetaShapley", "LAVA", "SAVA","ALL"
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
    help="Number of models for InfluenceSubsample"
)
args = parser.parse_args()

SEED = args.seed
RUN_SEED = args.run_seed
METHOD = args.method
JOB_ID = args.job_id
PROPORTION = args.proportion
NOISE_RATE = args.noise_rate
K_NEIGHBORS = args.k_neighbors
SUBSET_SIZE = args.subset_size
NUM_MODELS = args.num_models

print("Running CIFAR-10 experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")
print(f"  - NOISE_RATE: {NOISE_RATE}")
print(f"  - K_NEIGHBORS: {K_NEIGHBORS}")

# ============================================================
# DATA LOADING (Assuming embeddings are pre-loaded)
# ============================================================

def load_cifar10_embeddings(embedding_path="cifar10_embeddings.npz"):
    """Load pre-computed CIFAR-10 ResNet-18 embeddings.

    Expected keys (produced by cifar10-embeddings.py):
        train_embeddings  (40000, 512)  float32
        train_labels      (40000,)      int64
        val_embeddings    (10000, 512)  float32   ← model was trained on this split
        val_labels        (10000,)      int64
        test_embeddings   (10000, 512)  float32
        test_labels       (10000,)      int64
    """
    print("="*60)
    print("LOADING CIFAR-10 RESNET-18 EMBEDDINGS")
    print("="*60)

    with np.load(embedding_path) as f:
        x_train = f['train_embeddings']   # (40000, 512)
        y_train = f['train_labels']        # (40000,)  integers
        x_valid = f['val_embeddings']      # (10000, 512)
        y_valid = f['val_labels']          # (10000,)  integers
        x_test  = f['test_embeddings']     # (10000, 512)
        y_test  = f['test_labels']         # (10000,)  integers

    print(f"\nLoaded splits:")
    print(f"  train : {x_train.shape}  labels {y_train.shape}")
    print(f"  val   : {x_valid.shape}  labels {y_valid.shape}")
    print(f"  test  : {x_test.shape}   labels {y_test.shape}")

    # Scale features (fit on train only)
    print("\nScaling features...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_valid = scaler.transform(x_valid).astype(np.float32)
    x_test  = scaler.transform(x_test).astype(np.float32)

    # One-hot encode labels  (10 classes)
    def to_onehot(labels, n=10):
        return np.eye(n, dtype=np.float32)[labels]

    y_train = to_onehot(y_train)
    y_valid = to_onehot(y_valid)
    y_test  = to_onehot(y_test)

    print(f"  y_train one-hot: {y_train.shape}")
    return x_train, y_train, x_valid, y_valid, x_test, y_test


# ============================================================
# EXPERIMENT MEDIATOR CREATION
# ============================================================

def create_experiment_mediator(x_train, y_train, x_valid, y_valid, x_test, y_test):
    """Create experiment mediator for CIFAR-10 data."""
    
    print("\n" + "="*60)
    print("CREATING EXPERIMENT MEDIATOR")
    print("="*60)
    
    fetcher = DataFetcher.from_data_splits(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
        one_hot=True,  # Labels are already one-hot encoded
        random_state=SEED,
    )
    
    # Add label noise
    print(f"\nAdding {NOISE_RATE*100}% label noise...")
    fetcher = fetcher.noisify(mix_labels, noise_rate=NOISE_RATE, rng=SEED)
    
    # Prediction model (Logistic Regression for embeddings)
    pred_model = ClassifierSkLearnWrapper(
        LogisticRegression, 
        fetcher.label_dim[0]
    )
    
    exper_med = ExperimentMediator(
        fetcher,
        pred_model,
        metric_name="accuracy",
        train_kwargs={}
    )
    
    # Test baseline performance
    print("\nTesting baseline performance...")
    data = exper_med.fetcher.datapoints
    model = exper_med.pred_model.clone()
    model.fit(data[0], data[1], **exper_med.train_kwargs)
    y_pred = model.predict(data[4]).cpu()
    baseline = exper_med.metric(y_pred, data[5])
    print(f"Baseline accuracy: {baseline:.4f}")
    
    return exper_med


# ============================================================
# EVALUATOR CREATION (for scalability study)
# ============================================================

def create_method_evaluators(method_name):
    """Create evaluators for a specific method."""
    print(f"\nCreating evaluators for method: {method_name}")
    
    if method_name == "DataOob":
        # Test different bootstrap proportions for scalability study
        proportions = [0.1, 0.2, 0.5, 0.7, 1.0] if PROPORTION is None else [PROPORTION]
        model_sizes = [50,100]
        evaluators = [
            DataOob(num_models=m, proportion=p, random_state=seed)
            for m in model_sizes
            for p in proportions
            for seed in [1,2,3,4,5,6,7,8,9,10]
        ]

    elif method_name == "AME":
        model_sizes = [5000]
        evaluators = [
            AME(num_models=m, random_state=SEED)
            for m in model_sizes
        ]
                
    elif method_name == "DataBanzhaf":
        model_sizes = [20000]

        evaluators = [
            DataBanzhaf(num_models=m, random_state=RUN_SEED)
            for m in model_sizes
        ]
        
    elif method_name == "DataShapley":
        mc_epochs_list = [1000]
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs, 
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_SEED{SEED}",
                random_state=SEED
            )
            for mc_epochs in mc_epochs_list
        ]
        
    elif method_name == "InfluenceSubsample":
        n_models = NUM_MODELS if NUM_MODELS is not None else 700000
        s_size = SUBSET_SIZE if SUBSET_SIZE is not None else 100
        evaluators = [
            InfluenceSubsample(
                num_models=n_models,
                subset_size=s_size,
                random_state=RUN_SEED
            )
        ]

    elif method_name == "LOO_Random":
        evaluators = [RandomEvaluator(random_state=seed) for seed in [1,2,3,4,5,6,7,8,9,10]]
        
    elif method_name == "KNNShapley":
        # k_values = [50,100]
        # k_values 
        # evaluators = [
        #     KNNShapley(k_neighbors=k)
        #     for k in k_values
        # ]
        evaluators = []
        RS = 10
        print("\nCreating KNNShapleyLSH evaluator with fixed parameters for scalability study",RS)
        evaluator = KNNShapleyLSH(
            k_neighbors=100, 
            n_hash_table=100,
            eps=0.001,
            dist_rand=31.9286,
            t=2.2280,
            alpha=0.5,
            random_state=RS 
        )
        evaluators.append(evaluator)
        
    elif method_name == "DVRL":
        batch_sizes = [32, 64, 128, 256, 512]
        rl_epochs_list = [10000]
        evaluators = [
            DVRL(
                rl_epochs=rl_epochs,
                rl_batch_size=batch_size,
                random_state=seed
            )
            for rl_epochs in rl_epochs_list
            for batch_size in batch_sizes
            for seed in [1,2,3,4,5,6,7,8,9,10]
        ]
                    
    elif method_name == "BetaShapley":
        model_sizes = [1000, 2000, 5000, 10000]
        evaluators = [
            BetaShapley(num_models=m, random_state=SEED)
            for m in model_sizes
        ]
        
    elif method_name == "LAVA":
        evaluators = [
            LavaEvaluator(
                lam_y=100.0,
                lam_x=1.0,
                blur=0.05,
                debug=True
            )
        ]
    elif method_name == "SAVA":
        evaluators = [
            SavaEvaluator(
                lam_y=5.0,
                lam_x=1.0,
                blur=0.05,
                batch_size=1024,
                debug=True
            )
        ]
        
    elif method_name == "ALL":
        evaluators = []
        for m in ["DataOob", "KNNShapley", "LOO_Random"]:
            evaluators.extend(create_method_evaluators(m))
            
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
    """Run experiment for a specific method."""
    print("=" * 70)
    print(f"CIFAR-10 Data Valuation Experiment - {method_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load data
    print("\n1. Loading CIFAR-10 embeddings...")
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_cifar10_embeddings()
    
    # Create experiment mediator
    print("\n2. Creating experiment mediator...")
    exper_med = create_experiment_mediator(
        x_train, y_train, x_valid, y_valid, x_test, y_test
    )
    
    # Create output directory
    output_dir = f'CIFAR10_Results/{method_name}_SEED{SEED}_NOISE{NOISE_RATE}_JOB{JOB_ID}'
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create method-specific evaluators
    all_evaluators = create_method_evaluators(method_name)
    
    # Compute data values
    print(f"\n3. Computing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    
    try:
        exper_med = exper_med.compute_data_values(data_evaluators=all_evaluators)
        print(f"✓ {method_name} computation completed successfully")
    except Exception as e:
        print(f"✗ {method_name} computation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run evaluations
    print(f"\n4. Running evaluations for {method_name}...")
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
    print(f"\n5. Saving data values for {method_name}...")
    try:
        exper_med.evaluate(save_dataval, save_output=True)
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
        f.write(f"CIFAR-10 Data Valuation - {method_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Job ID: {JOB_ID}\n")
        f.write(f"Noise Rate: {NOISE_RATE}\n")
        f.write(f"Train/Valid/Test: 40000/10000/10000\n")
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


# ============================================================
# JOB SCRIPT GENERATION FOR CLUSTER
# ============================================================

def run_all_methods():
    """Create job scripts for all methods to run on cluster."""
    print("Creating job scripts for all methods...")
    
    methods = ['DataOob', 'AME', 'DataBanzhaf', 'DataShapley', 
               'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
               'DVRL', 'BetaShapley', 'LAVA']
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Master script
    master_script = """#!/bin/bash
# Master script to submit all CIFAR-10 data valuation jobs

METHODS=("DataOob" "AME" "DataBanzhaf" "DataShapley" "InfluenceSubsample" 
         "LOO_Random" "KNNShapley" "DVRL" "BetaShapley" "LAVA")

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
conda activate py39_env

# Move to script directory
cd /path/to/your/experiment/directory

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