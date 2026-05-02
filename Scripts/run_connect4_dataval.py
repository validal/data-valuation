# run_connect4_dataval.py

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

# Ensure dataset registry
import opendataval.dataloader.datasets

from opendataval.dataloader import mix_labels
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
from opendataval.dataval.knnshap import KNNShapleyLSH


# ============================
# Argument parsing
# ============================
parser = argparse.ArgumentParser(description="Run Connect4 data valuation experiment")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=[
        "DataOob", "AME", "DataBanzhaf", "DataShapley",
        "InfluenceSubsample", "LOO_Random", "KNNShapley",
        "DVRL", "BetaShapley", "LAVA","SAVA","ALL"
    ],
)
parser.add_argument(
    "--rs",
    type=int,
    default=None,
    help="Random-state / seed to use for SavaEvaluator (if set, runs only this rs)"
)
parser.add_argument("--job_id", type=int, default=1)
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
JOB_ID = args.job_id
PROPORTION = args.proportion
NOISE_RATE = args.noise_rate
RS = args.rs

print("Running Connect4 dataset experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")
print(f"  - NOISE_RATE: {NOISE_RATE}")


# ============================================================
# Connect4 DATASET CONFIGURATION
# ============================================================

def create_experiment_mediator():
    """Create experiment mediator for Connect4 dataset with 60/20/20 split."""
    print("Creating Connect4 dataset experiment mediator...")

    dataset_name = "connect4"

    # Approximate counts based on full dataset (~67557): 60/20/20
    train_count = 40000
    valid_count = 13778
    test_count  = 13779


    print(f"Dataset split: Train={train_count}, Valid={valid_count}, Test={test_count}")
    print(f"Total samples: {train_count + valid_count + test_count}")

    noise_kwargs = {'noise_rate': NOISE_RATE}

    model_name = "sklogreg"
    metric_name = "accuracy"

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
    print(f"Creating evaluators for method: {method_name}")

    MODEL_SIZES = [1, 2, 5, 10, 50, 100]

    PROPORTIONS = [0.1, 0.2, 0.5, 0.7, 1.0]
    NUM_MODELS_LIST = [1, 5, 10, 100]
    SEATS = range(1, 11)  # 1 → 10

    evaluators = []

    if method_name == "DataOob":
        for proportion in PROPORTIONS:
            for num_models in NUM_MODELS_LIST:
                for seat in SEATS:
                    evaluators.append(
                        DataOob(
                            num_models=num_models,
                            proportion=proportion,
                            random_state=SEED + seat  # ensures different randomness per seat
                        )
                    )


    elif method_name == "AME":
        MODEL_SIZES = [10000, 15000, 20000]
        evaluators = []
        for m in MODEL_SIZES:
            for s in [1]:
                if m <= 1:
                    continue
                evaluators.append(AME(num_models=m, random_state=s))

    elif method_name == "DataBanzhaf":
        MODEL_SIZES = [180000]
        evaluators = [
            DataBanzhaf(num_models=m, random_state=s)
            for m in MODEL_SIZES
            for s in [1]
        ]

    elif method_name == "DataShapley":
        mc_epochs_list = [1000]
        evaluators = [
            DataShapley(
                mc_epochs=mc_epochs,
                min_cardinality=5,
                cache_name=f"shapley_mc{mc_epochs}_run{run_idx}_SEED_{SEED}_CONNECT4",
                random_state=run_idx
            )
            for mc_epochs in mc_epochs_list
            for run_idx in range(1)
        ]

    elif method_name == "InfluenceSubsample":
        evaluators = []
        for m in [10000]:
            for s in [5]:
                evaluators.append(
                    InfluenceSubsample(
                        num_models=m,
                        subset_size=4000,
                        random_state=s
                    )
                )

    elif method_name == "LOO_Random":
        evaluators = [
            LeaveOneOut()
        ] 

    elif method_name == "KNNShapley":
        print("RS for KNNShapleyLSH:", RS)
        evaluators = []
        evaluator = KNNShapleyLSH(
            k_neighbors=10, 
            n_hash_table=100,
            eps=0.01, 
            alpha=0.5,
            dist_rand=14.8177,
            t=2.0620,
            random_state=RS 
        )
        evaluators.append(evaluator)
        #k_values = [5000,10000]
        #evaluators = [KNNShapley(k_neighbors=k) for k in k_values]

    elif method_name == "DVRL":
        BATCH_SIZES = [32, 64, 128, 256, 512]
        evaluators = []
        for seed in range(1, 11):
            for rl_epochs in [2000, 3000, 5000]:
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
            for s in range(1, 4)
        ]

    elif method_name == "LAVA":
        evaluators = [ LavaEvaluator(lam_y=1,debug=True,blur = 0.05)]
    elif method_name == "SAVA":
        rs_list = [RS] if RS is not None else [1,2,3,4,5]
        evaluators = [
            SavaEvaluator(
                batch_size=1024,
                lam_x=1.0,
                lam_y=10.0,
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
        evaluators = []
        for m in ["LOO_Random"]:
            evaluators.extend(create_method_evaluators(m))

    else:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"Created {len(evaluators)} evaluators for {method_name}")
    return evaluators


def save_time_memory_report(exper_med, output_dir, method_name):
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
    print("=" * 70)
    print(f"Connect4 Data Valuation Experiment - {method_name}")
    print("=" * 70)

    start_time = time.time()
    exper_med = create_experiment_mediator()

    output_dir = f'Connect4_Results/{method_name}_SEED{SEED}_NOISE{NOISE_RATE}_JOB{JOB_ID}'
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")

    all_evaluators = create_method_evaluators(method_name)

    print(f"\nComputing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    try:
        exper_med = exper_med.compute_data_values(data_evaluators=all_evaluators)
        print(f"✓ {method_name} computation completed successfully")
    except Exception as e:
        print(f"✗ {method_name} computation failed: {e}")
        import traceback
        traceback.print_exc()

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

    print(f"\nSaving data values for {method_name}...")
    try:
        _ = exper_med.evaluate(save_dataval, save_output=True)
        print(f"  ✓ Data values saved")
    except Exception as e:
        print(f"  ✗ Data values save failed: {e}")
        import traceback
        traceback.print_exc()

    save_time_memory_report(exper_med, output_dir, method_name)

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    summary_path = os.path.join(output_dir, f"summary_{method_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Connect4 Data Valuation - {method_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Job ID: {JOB_ID}\n")
        f.write(f"Noise Rate: {NOISE_RATE}\n")
        f.write(f"Train/Valid/Test: 40534/13511/13512\n")
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
    print("Creating job scripts for all methods...")
    methods = ['DataOob', 'AME', 'DataBanzhaf', 'DataShapley',
               'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
               'DVRL', 'BetaShapley', 'LAVA']

    master_script = """#!/bin/bash
# Master script to submit all Connect4 data valuation jobs

METHODS=("DataOob" "AME" "DataBanzhaf" "DataShapley" "InfluenceSubsample" 
         "LOO_Random" "KNNShapley" "DVRL" "BetaShapley" "LAVA")

for method in "${METHODS[@]}"; do
    echo "Submitting job for method: $method"
    sbatch run_connect4_${method}.sh
    sleep 2
done

echo "All jobs submitted!"
"""

    with open("submit_all_methods.sh", "w") as f:
        f.write(master_script)
    os.chmod("submit_all_methods.sh", 0o755)
    print("Created master script: submit_all_methods.sh")

    for method in methods:
        job_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --output=logs/run_connect4_{method}_%j.log
#SBATCH --error=logs/run_connect4_{method}_%j.err
#SBATCH --time=36:00:00
#SBATCH --job-name=connect4_{method}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source ~/.bashrc
conda activate py39_env

cd "/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Datasets_Results/Connect4"

mkdir -p logs

# Ensure data exists
python prepare_connect4_data.py

echo "Starting Connect4 experiment for method: {method}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python run_connect4_dataval.py --seed 42 --method {method} --job_id $SLURM_JOB_ID --noise_rate 0.2

echo "Job completed for method: {method}"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID completed successfully"
"""
        script_filename = f"run_connect4_{method}.sh"
        with open(script_filename, "w") as f:
            f.write(job_script)
        os.chmod(script_filename, 0o755)
        print(f"Created job script: {script_filename}")

    print("\nTo run all methods, execute:")
    print("  ./submit_all_methods.sh")
    print("\nOr to run a specific method:")
    print("  sbatch run_connect4_METHODNAME.sh")


if __name__ == "__main__":
    if METHOD == "ALL":
        run_all_methods()
    else:
        try:
            _ = run_method_experiment(METHOD)
        except KeyboardInterrupt:
            print(f"\nExperiment for {METHOD} interrupted by user.")
        except Exception as e:
            print(f"\nFatal error in {METHOD} experiment: {e}")
            import traceback
            traceback.print_exc()
