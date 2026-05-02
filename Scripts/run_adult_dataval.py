# run_adult_dataval.py

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

# Ensure dataset registry
import opendataval.dataloader.datasets

from opendataval.dataloader import mix_labels, DataFetcher
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator,ParallelLavaOOBEvaluator,
    LeaveOneOut, RandomEvaluator
)
from opendataval.dataval.lava import SavaEvaluator
from opendataval.dataval.knnshap import KNNShapleyLSH

from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample,
    noisy_detection,
    remove_high_low,
    save_dataval
)
from opendataval.model.api import ClassifierSkLearnWrapper


# ============================
# Argument parsing
# ============================
parser = argparse.ArgumentParser(description="Run Adult data valuation experiment")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=[
        "DataOob", "AME", "DataBanzhaf", "DataShapley",
        "InfluenceSubsample", "LOO_Random", "KNNShapley",
        "DVRL", "BetaShapley", "LAVA","SAVA", "ALL"
    ],
)
parser.add_argument(
    "--rs",
    type=int,
    default=None,
    help="Random-state / seed to use for SavaEvaluator (if set, runs only this rs)"
)
parser.add_argument(
    "--proportion",
    type=float,
    default=None,
    help="Bootstrap proportion for DataOob (e.g. 1.0, 0.7, 0.5, 0.2). If not set, use all proportions."
)
parser.add_argument(
    "--noise_rate",
    type=float,
    default=0.2,
    help="Label noise rate (default: 0.2)"
)
args = parser.parse_args()

SEED = args.seed
METHOD = args.method
PROPORTION = args.proportion
NOISE_RATE = args.noise_rate
RS = args.rs


print("Running Adult dataset experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - NOISE_RATE: {NOISE_RATE}")


# ============================================================
# Adult DATASET CONFIGURATION
# ============================================================

def create_experiment_mediator():
    """Create experiment mediator for Adult dataset with specified split."""
    print("Creating Adult dataset experiment mediator...")
    
    # Adult dataset configuration
    dataset_name = "adult"
    
    # Calculate split sizes based on 48K total
    # Train: 30000, Valid: 5000, Test: 10000 (sums to 45000, leaving 3000 unused from full Adult)
    # Or use exactly: T=28235, V=5647, T_s=14118 (sum=48000) from earlier calculation
    # Let's use the exact calculated values:
    #train_count, valid_count, test_count = 28235, 5647, 14118
    train_count, valid_count, test_count = 30000, 5000, 13842

    
    print(f"Dataset split: Train={train_count}, Valid={valid_count}, Test={test_count}")
    print(f"Total samples: {train_count + valid_count + test_count}")
    
    noise_kwargs = {'noise_rate': NOISE_RATE}
    
    # Use KNN classifier as specified
    model_name = "sklogreg"
    metric_name = "accuracy"  # Changed from "accuracy" to "roc_auc" based on metrics discussion
    
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        cache_dir="../data_files/",  
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels, 
        noise_kwargs=noise_kwargs,
        train_kwargs={},
        model_name=model_name,
        metric_name=metric_name,
        random_state=SEED
    )
    
    print("Testing baseline performance...")
    data = exper_med.fetcher.datapoints
    model = exper_med.pred_model.clone()
    model.fit(data[0], data[1], **exper_med.train_kwargs)
    y_pred = model.predict(data[4]).cpu()
    baseline = exper_med.metric(y_pred, data[5])
    print(f"Baseline {metric_name}: {baseline:.4f}")
    
    return exper_med


def create_method_evaluators(method_name):
    """Create evaluators for a specific method."""
    print(f"Creating evaluators for method: {method_name}")
    
    # Adjusted model sizes for Adult dataset (smaller than DogFish)
    MODEL_SIZES = [1, 2, 5, 10, 50, 100]
    
    if method_name == "DataOob":
    
        evaluators = [
            DataOob(num_models=m, proportion=0.1, random_state=SEED)
            for m in MODEL_SIZES
        ]

    elif method_name == "AME":
        MODEL_SIZES = [10000, 15000, 20000]  # Larger for AME
        evaluators = []
        for m in MODEL_SIZES:
            for s in [1]:
                if m <= 1:
                    continue
                ame = AME(num_models=m, random_state=s)
                evaluators.append(ame)
                
    elif method_name == "DataBanzhaf":
        MODEL_SIZES = [350000]  # Larger for DataBanzhaf
        evaluators = [
            DataBanzhaf(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in [1]
        ]
        
    elif method_name == "DataShapley":
        mc_epochs_list = [1000]  # Fewer epochs for efficiency
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs, 
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_run{run_idx}_SEED_{SEED}_ADULT",
                random_state=run_idx
            )
            for mc_epochs in mc_epochs_list
            for run_idx in range(1)  # Fewer runs
        ]
        
    elif method_name == "InfluenceSubsample":
        evaluators = []
        for m in [100000,300000,500000]:
            for s in [SEED]:
                evaluators.append(
                    InfluenceSubsample(
                        num_models=m,
                        subset_size=100,
                        random_state=s
                    )
                )

    elif method_name == "LOO_Random":
        evaluators = [
            LeaveOneOut(),  # One time
        ] + [
            #RandomEvaluator(random_state=s) for s in range(1, 11)  # 1..10
        ]
        
    elif method_name == "KNNShapley":
        # # Adjust k values for Adult dataset size
        # k_values = [10, 100, 5000, 15000]
        # evaluators = [KNNShapley(k_neighbors=k) for k in k_values]
        evaluators = []
        evaluator = KNNShapleyLSH(
            k_neighbors=10, 
            n_hash_table=100,
            eps=0.01, 
            alpha=0.5,
            random_state=10 
        )
        evaluators.append(evaluator)
        
    elif method_name == "DVRL":
        BATCH_SIZES = [32, 64, 128, 256, 512]
        evaluators = []
        for seed in SEEDS:
            for rl_epochs in [2000,3000,5000]:  # Fewer epochs for efficiency
                for batch_size in BATCH_SIZES:
                        evaluators.append(
                            DVRL(
                                rl_epochs=rl_epochs,
                                rl_batch_size=batch_size,
                                random_state=seed
                            )
                        )
                    
    elif method_name == "BetaShapley":
        MODEL_SIZES = [1000, 2000, 5000]
        evaluators = [
            BetaShapley(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in range(1, 4)  # Fewer seeds
        ]
        
    elif method_name == "LAVA": 
        evaluators = [ LavaEvaluator(lam_y=5,debug=True,blur = 0.05)]
    elif method_name == "SAVA":
        # Create SavaEvaluator(s) for specified rs seeds. If RS provided, use single seed.
        rs_list = [RS] if RS is not None else [1]
        evaluators = [
            SavaEvaluator(
                batch_size=1024,
                lam_x=1.0,
                lam_y=1.0,
                p=2,
                blur=0.05,
                mode="cls",
                debug=True,
                stratified_batches=True,
                random_state=rs
            )
            for rs in rs_list
        ]
        
        
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
    print(f"Adult Data Valuation Experiment - {method_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create experiment mediator
    exper_med = create_experiment_mediator()
    
    # Create output directory with method name
    output_dir = f'Adult_Results/{method_name}_SEED{SEED}_NOISE{NOISE_RATE}'
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
        import traceback
        traceback.print_exc()
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
            import traceback
            traceback.print_exc()
    
    # Save data values
    print(f"\nSaving data values for {method_name}...")
    try:
        values = exper_med.evaluate(save_dataval, save_output=True)
        print(f"  ✓ Data values saved")
    except Exception as e:
        print(f"  ✗ Data values save failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save time/memory report
    save_time_memory_report(exper_med, output_dir, method_name)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save final summary
    summary_path = os.path.join(output_dir, f"summary_{method_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Adult Data Valuation - {method_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Noise Rate: {NOISE_RATE}\n")
        f.write(f"Train/Valid/Test: 28235/5647/14118\n")
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
    methods = ['DataOob', 'AME', 'DataBanzhaf', 'DataShapley',
               'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
               'DVRL', 'BetaShapley', 'LAVA']
    for method in methods:
        print(f"\n{'='*60}\nRunning method: {method}\n{'='*60}")
        try:
            run_method_experiment(method)
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()


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