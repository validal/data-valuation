# run_hepmass_dataval.py
from multiprocessing.util import debug
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
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,ParallelizableDataOob,ParallelizableDataOob,ParallelizableInfluenceSubsample,OptimizedInfluenceSubsample,OptimizedDataOob,
    InfluenceSubsample, KNNShapley, LavaEvaluator, LeaveOneOut, RandomEvaluator,BatchwiseLavaEvaluator, LavaOOBEvaluator,ParallelLavaOOBEvaluator)
from opendataval.dataval.knnshap import KNNShapleyLSH
from opendataval.dataval.lava import SavaEvaluator
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)
from opendataval.model.api import ClassifierSkLearnWrapper
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def get_system_memory():
    """Get system memory usage."""
    memory = psutil.virtual_memory()
    return {
        'total_mb': memory.total / 1024 / 1024,
        'available_mb': memory.available / 1024 / 1024,
        'used_mb': memory.used / 1024 / 1024,
        'percent': memory.percent
    }

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run HEPMASS data valuation experiment')
parser.add_argument('--seed', type=int, default=42, help='Random seed for the experiment')
parser.add_argument('--method', type=str, required=True, 
                   choices=['DataOob', 'AME', 'DataBanzhaf', 'DataShapley', 
                           'InfluenceSubsample', 'LOO_Random', 'KNNShapley',
                           'DVRL', 'BetaShapley', 'LAVA', 'ALL', 'SAVA'],
                   help='Method to run (or ALL for all methods)')
parser.add_argument('--job_id', type=int, default=1, help='Job ID for naming outputs')
args = parser.parse_args()

# Set seeds
SEED = args.seed
METHOD = args.method
JOB_ID = args.job_id

print(f"Running experiment with:")
print(f"  - SEED: {SEED}")
print(f"  - METHOD: {METHOD}")
print(f"  - JOB_ID: {JOB_ID}")

def compute_hepmass_distance_matrix(x_train, x_valid, batch_size=512, cache_file=None):
    """Compute distance matrix for HEPMASS dataset with timing and memory display."""
    import time
    start_time = time.time()
    
    n = len(x_train)
    m = len(x_valid)
    
    print(f"Computing distance matrix: {n} train × {m} validation")
    print(f"Batch size: {batch_size}")
    print(f"Total pairs to compute: {n:,} × {m:,} = {n*m:,}")
    
    # Display initial memory
    initial_mem = get_memory_usage()
    system_mem = get_system_memory()
    print(f"\nInitial memory usage:")
    print(f"  Process RSS: {initial_mem['rss_mb']:.1f} MB, VMS: {initial_mem['vms_mb']:.1f} MB")
    print(f"  System: {system_mem['used_mb']:.1f}/{system_mem['total_mb']:.1f} MB ({system_mem['percent']:.1f}%) used")
    
    # Check cache
    if cache_file and os.path.exists(cache_file):
        print(f"\nLoading cached distance matrix from {cache_file}")
        try:
            load_start = time.time()
            dist_matrix = torch.load(cache_file)
            load_time = time.time() - load_start
            
            if dist_matrix.shape == (n, m):
                loaded_mem = get_memory_usage()
                mem_increase = loaded_mem['rss_mb'] - initial_mem['rss_mb']
                size_gb = dist_matrix.numel() * dist_matrix.element_size() / (1024**3)
                
                print(f"✓ Loaded cached matrix in {load_time:.2f} seconds")
                print(f"  Matrix size: {size_gb:.2f} GB")
                print(f"  Memory increase: {mem_increase:.1f} MB")
                print(f"  Current RSS: {loaded_mem['rss_mb']:.1f} MB")
                return dist_matrix
            else:
                print(f"Cache shape mismatch: {dist_matrix.shape} != ({n}, {m})")
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    # Convert to tensors
    convert_start = time.time()
    print(f"\nConverting data to tensors...")
    
    x_train_t = torch.as_tensor(np.array(x_train))
    x_valid_t = torch.as_tensor(np.array(x_valid))
    
    # Flatten
    x_train_view = x_train_t.view(n, -1)
    x_valid_view = x_valid_t.view(m, -1)
    
    convert_time = time.time() - convert_start
    convert_mem = get_memory_usage()
    print(f"✓ Tensor conversion: {convert_time:.2f} seconds")
    print(f"  Memory after conversion: {convert_mem['rss_mb']:.1f} MB (+{convert_mem['rss_mb'] - initial_mem['rss_mb']:.1f} MB)")
    
    # Compute number of batches
    n_batches_train = (n + batch_size - 1) // batch_size
    n_batches_valid = (m + batch_size - 1) // batch_size
    total_operations = n_batches_train * n_batches_valid
    
    print(f"\nProcessing {n_batches_train} train batches × {n_batches_valid} valid batches = {total_operations} operations")
    
    # Compute in batches
    dist_list = []
    batch_times = []
    total_compute_time = 0
    max_batch_memory = 0
    
    print(f"\nStarting batch computation...")
    
    for i, x_train_batch in enumerate(DataLoader(x_train_view, batch_size)):
        batch_start = time.time()
        batch_mem_start = get_memory_usage()
        
        if i % 5 == 0 or i == n_batches_train - 1:
            elapsed = time.time() - start_time
            progress = (i * n_batches_valid) / total_operations * 100
            current_mem = get_memory_usage()
            system_mem = get_system_memory()
            
            print(f"  Batch {i+1}/{n_batches_train} - Progress: {progress:.1f}%")
            print(f"    Elapsed: {elapsed:.1f}s - RSS: {current_mem['rss_mb']:.1f} MB")
            print(f"    System: {system_mem['used_mb']:.1f}/{system_mem['total_mb']:.1f} MB ({system_mem['percent']:.1f}%)")
        
        dist_row = []
        for x_val_batch in DataLoader(x_valid_view, batch_size):
            d = torch.cdist(x_train_batch, x_val_batch)
            dist_row.append(d)
        
        row_cat = torch.cat(dist_row, dim=1) if dist_row else torch.empty(0)
        dist_list.append(row_cat)
        
        batch_time = time.time() - batch_start
        batch_mem_end = get_memory_usage()
        batch_mem_usage = batch_mem_end['rss_mb'] - batch_mem_start['rss_mb']
        max_batch_memory = max(max_batch_memory, batch_mem_usage)
        
        batch_times.append(batch_time)
        total_compute_time += batch_time
        
        if i % 10 == 0 and i > 0:
            avg_batch_time = np.mean(batch_times[-10:])
            remaining_batches = n_batches_train - i - 1
            est_remaining = avg_batch_time * remaining_batches
            print(f"    Avg batch: {avg_batch_time:.2f}s - Est remaining: {est_remaining/60:.1f}min")
            print(f"    Peak batch memory: {max_batch_memory:.1f} MB")
    
    # Concatenate all batches
    concat_start = time.time()
    concat_mem_start = get_memory_usage()
    print(f"\nConcatenating {len(dist_list)} batches into final matrix...")
    
    dist_matrix = torch.cat(dist_list, dim=0)
    
    concat_time = time.time() - concat_start
    concat_mem = get_memory_usage()
    concat_mem_increase = concat_mem['rss_mb'] - concat_mem_start['rss_mb']
    
    total_time = time.time() - start_time
    final_mem = get_memory_usage()
    total_mem_increase = final_mem['rss_mb'] - initial_mem['rss_mb']
    
    # Calculate expected matrix size
    size_gb = dist_matrix.numel() * dist_matrix.element_size() / (1024**3)
    
    # Display timing and memory summary
    print("\n" + "="*70)
    print("DISTANCE MATRIX COMPUTATION SUMMARY")
    print("="*70)
    print(f"MATRIX INFO:")
    print(f"  Shape: {tuple(dist_matrix.shape)}")
    print(f"  Size: {size_gb:.3f} GB")
    print(f"  Dtype: {dist_matrix.dtype}")
    print(f"  Device: {dist_matrix.device}")
    
    print(f"\nTIMING:")
    print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"    - Tensor conversion: {convert_time:.2f}s ({convert_time/total_time*100:.1f}%)")
    print(f"    - Batch computation: {total_compute_time:.2f}s ({total_compute_time/total_time*100:.1f}%)")
    print(f"    - Concatenation: {concat_time:.2f}s ({concat_time/total_time*100:.1f}%)")
    print(f"  Average batch time: {np.mean(batch_times):.2f}s")
    print(f"  Throughput: {n*m/total_time:,.0f} pairs/second")
    
    print(f"\nMEMORY:")
    print(f"  Initial RSS: {initial_mem['rss_mb']:.1f} MB")
    print(f"  Final RSS: {final_mem['rss_mb']:.1f} MB")
    print(f"  Total increase: {total_mem_increase:.1f} MB ({total_mem_increase/1024:.2f} GB)")
    print(f"  Concatenation increase: {concat_mem_increase:.1f} MB")
    print(f"  Peak batch memory: {max_batch_memory:.1f} MB")
    print(f"  Expected matrix size: {size_gb:.3f} GB")
    
    # Check if memory usage matches expected
    if total_mem_increase / 1024 < size_gb * 0.8:
        print(f"  ⚠️  Memory increase ({total_mem_increase/1024:.3f} GB) less than expected matrix size ({size_gb:.3f} GB)")
        print(f"     This might indicate memory is being swapped or tensors are not fully loaded")
    
    # Save cache
    if cache_file:
        save_start = time.time()
        print(f"\nSaving to cache: {cache_file}")
        torch.save(dist_matrix, cache_file)
        save_time = time.time() - save_start
        print(f"✓ Saved in {save_time:.2f} seconds")
    
    system_final = get_system_memory()
    print(f"\nSYSTEM MEMORY:")
    print(f"  Used: {system_final['used_mb']:.1f}/{system_final['total_mb']:.1f} MB ({system_final['percent']:.1f}%)")
    print("="*70)
    
    return dist_matrix

def load_and_prepare_hepmass():
    """Load HEPMASS datasets and prepare them for the experiment."""
    print("Loading HEPMASS datasets...")
    
    # Load datasets from CSV files
    train_set = pd.read_csv('hepmass_train_7M.csv').drop(columns=["Unnamed: 0"])
    valid_set = pd.read_csv('hepmass_valid_1.4M.csv')
    test_set = pd.read_csv('hepmass_test_2M.csv')
    
    # Identify label column (first column based on your code)
    label_col = train_set.columns[0]
    print(f"Using label column: '{label_col}'")
    
    # Extract features and labels
    feature_cols = [col for col in train_set.columns if col != label_col]
    
    X_train = train_set[feature_cols]
    Y_train = train_set[label_col]
    
    X_valid = valid_set[feature_cols]
    Y_valid = valid_set[label_col]

    N_PER_CLASS = 5000  # 5000 total, balanced

    idx_class0 = np.where(Y_valid.values == 0)[0]
    idx_class1 = np.where(Y_valid.values == 1)[0]

    rng = np.random.default_rng(42)
    chosen = np.concatenate([
        rng.choice(idx_class0, N_PER_CLASS, replace=False),
        rng.choice(idx_class1, N_PER_CLASS, replace=False)
    ])
    rng.shuffle(chosen)

    X_valid = X_valid.iloc[chosen]
    Y_valid = Y_valid.iloc[chosen]
    X_test = test_set[feature_cols]
    Y_test = test_set[label_col]

    N_PER_CLASS = 25000  # 25000 total, balanced test
    idx_class0 = np.where(Y_test.values == 0)[0]
    idx_class1 = np.where(Y_test.values == 1)[0]
    chosen = np.concatenate([
        rng.choice(idx_class0, N_PER_CLASS, replace=False),
        rng.choice(idx_class1, N_PER_CLASS, replace=False)
    ])
    rng.shuffle(chosen)
    X_test = X_test.iloc[chosen]
    Y_test = Y_test.iloc[chosen]


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


def create_method_evaluators(method_name,exper_med):
    """Create evaluators for a specific method."""
    print(f"Creating evaluators for method: {method_name}")
    
    # Your original model sizes
    MODEL_SIZES = [1, 2, 5, 10, 50, 100,1000]# 500, 1000, 2000, 3000, 5000]
    #LARGE_MODEL_SIZES = [8000,10000,15000,20000,30000, 50000]

    SEEDS = list(range(1, 11))  # 1..10
    
    if method_name == "DataOob":
        PROPORTIONS = [1.0, 0.7, 0.5, 0.2,0.1]
        
        evaluators = [
            DataOob(num_models=m, proportion=p, random_state=s)
            for m in [10]
            for p in [0.1]
            for s in [7, 8, 9, 10]
        ]
        
    elif method_name == "AME":
        #MODEL_SIZES = [8000,10000,15000,20000]
        #MODEL_SIZES = [1, 2, 5, 10, 50, 100, 500, 1000]# 2000, 3000, 5000]
        MODEL_SIZES = [2000, 3000]
        MODEL_SIZES = [5000]

        evaluators = []
        for m in [1000000]:
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
                cache_name=f"shapley_mc{mc_epochs}_run{run_idx}_SEED_{SEED}_HEPMASS100K",
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

        for p in [int(1e-4/7)]:
            for m in [int(7e7)]:
                for s in [1]:
                    evaluators.append(
                        OptimizedInfluenceSubsample(
                            num_models=m,
                            proportion=p,
                            random_state=s,
                            n_jobs=56,
                            debug=True
                        )
                    )
            
    elif method_name == "LOO_Random":
        evaluators = [
            #LeaveOneOut(),  # One time
        ] + [
            RandomEvaluator(random_state=s) for s in SEEDS  # 1..10
        ]

        

    elif method_name == "KNNShapley":
        # evaluator = KNNShapleyLSH(k_neighbors=100, dist_rand=7.3622, n_hash_table=200, eps=0.0001, alpha=0.5, t=2.399)
        RS = 1
        evaluator = KNNShapleyLSH(k_neighbors=100, dist_rand=7.3622, n_hash_table=100, eps=0.00005, alpha=0.5, t=2.399,random_state=RS)
        print(f"Created KNNShapleyLSH evaluator with k={evaluator.k_neighbors}, n_hash_table={evaluator.n_hash_table},random_state={RS}")
        evaluators = [evaluator]        

    elif method_name == "OLDKNNShapley":
        k_values = [
            1,          # Pure nearest neighbor
            10,         # Small local neighborhood
            50,         # Small neighborhood
            100,        # Moderate neighborhood  
            500,        # Medium neighborhood
            1000,       # 1% of previous / 0.1% of new
            5000,       # 5% of previous / 0.5% of new
            10000,      # 10% of previous / 1% of new
            20000,      # 20% of previous
            50000,      # 50% of previous / 5% of new
            100000,     # Whole dataset of previous / 10% of new
            250000,     # 25% of new dataset
            500000,     # 50% of new dataset
            1000000     # Whole dataset (k = N)
        ]
        #k_values = [1, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000]
        #k_values = [10,50, 100,500,1000,2000, 5000, 10000]

        
        
        # Create distance matrix cache file
        dist_cache_file = f"hepmass_dist_matrix_SEED{SEED}_10M.pt"
        
        # We'll compute distance matrix before creating evaluators
        # Get the data from experiment mediator
        data = exper_med.fetcher.datapoints
        x_train, y_train = data[0], data[1]
        x_valid, y_valid = data[2], data[3]
        
        # Compute distance matrix ONCE
        print("\n" + "="*60)
        print("COMPUTING DISTANCE MATRIX (One-time for all k values)")
        print("="*60)
        dist_matrix = compute_hepmass_distance_matrix(
            x_train, x_valid, 
            batch_size=512,
            cache_file=dist_cache_file
        )
        
        # Now create evaluators with the precomputed distance matrix
        evaluators = []
        for k in k_values:
            evaluator = KNNShapley(
                k_neighbors=k,
                batch_size=512,
                debug=True
            )
            
            # Set the distance matrix
            evaluator.set_distance_matrix(dist_matrix)
            
            evaluators.append(evaluator)
        
        print(f"Created {len(evaluators)} KNNShapley evaluators with shared distance matrix")


    elif method_name == "DVRL":
        # BATCH_SIZES = [32, 64, 128, 256, 512]
        # Running = [100,1000,5000,10000]
        evaluators = []
        for rl_epochs in [10000]:
            for batch_size in [512]:
                for random_state in [1]:  # 1 to 10
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
            evaluators = []
            evaluator = SavaEvaluator(
            batch_size=1024,           # points per train/val batch
            lam_x=1.0,                # feature distance weight
            lam_y=10.0,                # label distance weight
            p=2,                      # p-Wasserstein order
            blur=0.05,                # GeomLoss entropic scale
            mode="cls",               # "cls" or "reg"
            debug=True,
            random_state=10
        )    
            evaluators.append(evaluator) 
    
    elif method_name == "LAVA":
        # from itertools import product

        # # ---- LAVA hyperparameter grid ----
        # # ---- Extensive LAVA hyperparameter grid ----
        # lava_grid = {
        #     # Label regularization strength
        #     "lam_y": [
        #         0,        # no label regularization (edge case)
        #         0.1, 0.5,
        #         1,
        #         5,
        #         10,
        #         20,
        #         50,
        #         100,
        #     ],

        #     # Bootstrap stability / variance reduction
        #     "n_bootstrap": [
        #         10,     # very noisy, fast (debug)
        #         25,
        #         50,
        #         100,
        #         200,
        #         300,
        #         500,    # very stable, slow
        #     ],

        #     # Subsampling of training data for OT
        #     "sample_frac": [
        #         0.001,   # very cheap, noisy OT
        #         0.003,
        #         0.005,
        #         0.01,    # expensive, high-quality OT
        #     ]
        # }

        evaluators = []

        # for lam_y, n_bootstrap, sample_frac in product(
        #     lava_grid["lam_y"],
        #     lava_grid["n_bootstrap"],
        #     lava_grid["sample_frac"]
        # ):
        evaluators.append(
    ParallelLavaOOBEvaluator(
        lam_y=10,
        n_bootstrap=5000,
        train_sample_size=10000,
        valid_sample_size=2000,
        debug=True
    )
    )   
        #evaluators = []

        #evaluators.append(BatchwiseLavaEvaluator(blur = 0.05 , debug = True, lam_y = 10, lam_x = 1, train_batch_size= 512,val_batch_size= 512))
        #evaluators.append(ParallelLavaOOBEvaluator(lam_y=10,n_bootstrap=500,train_sample_frac=0.05,valid_sample_frac=0.05,debug=True))

        
        # ===========================================
        # 1. Label weight sweep (lam_x=1, vary lam_y from 0 to 100)
        # ===========================================
        # lam_y_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        # for lam_y in lam_y_values:
        #     evaluators.append(
        #         LavaEvaluator(
        #             blur=blur_fixed, 
        #             debug=True, 
        #             lam_x=1.0, 
        #             lam_y=lam_y
        #         )
        #     )
        
        # # ===========================================
        # # 2. Inverse sweep (lam_y=1, vary lam_x from 0 to 100)
        # # ===========================================
        # lam_x_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        # for lam_x in lam_x_values:
        #     evaluators.append(
        #         LavaEvaluator(
        #             blur=blur_fixed,
        #             debug=True,
        #             lam_x=lam_x,
        #             lam_y=1.0
        #         )
        #     )
        
        # # ===========================================
        # # 3. Blur sweep (speed/accuracy trade-off)
        # # ===========================================
        # blur_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        # for blur_val in blur_values:
        #     evaluators.append(
        #         LavaEvaluator(
        #             blur=blur_val,
        #             debug=True,
        #             lam_x=1.0,
        #             lam_y=1.0
        #         )
        #     )
        
        # # ===========================================
        # # 4. Higher-order cost and entropy regularization
        # # ===========================================
        # # Different p-norms
        # for p in [1, 2, 3, 4]:
        #     evaluators.append(
        #         LavaEvaluator(
        #             blur=0.05,
        #             debug=True,
        #             lam_x=1.0,
        #             lam_y=1.0,
        #             p=p
        #         )
        #     )
        
        # # Entropy regularization
        # for entreg in [0.01, 0.05, 0.10, 0.20]:
        #     evaluators.append(
        #         LavaEvaluator(
        #             blur=0.05,
        #             debug=True,
        #             lam_x=1.0,
        #             lam_y=1.0,
        #             entreg=entreg
        #         )
        #     )
        
        # # Combined p and entreg
        # evaluators.append(
        #     LavaEvaluator(
        #         blur=0.05,
        #         debug=True,
        #         lam_x=1.0,
        #         lam_y=1.0,
        #         p=4,
        #         entreg=0.05
        #     )
        # )
        
        # evaluators.append(
        #     LavaEvaluator(
        #         blur=0.05,
        #         debug=True,
        #         lam_x=1.0,
        #         lam_y=1.0,
        #         p=1,
        #         entreg=0.10
        #     )
        # )
        
        
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
    output_dir = f'HEPMASS_Results10M/{method_name}_SEED{SEED}_JOB{JOB_ID}'
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create method-specific evaluators
    all_evaluators = create_method_evaluators(method_name, exper_med)
    
    # Compute data values
    print(f"\nComputing data values for {method_name} ({len(all_evaluators)} evaluators)...")
    print(f"\nComputing data values for {all_evaluators[0]})...")

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
#SBATCH --cpus-per-task=56
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
cd "/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/HepMass10M"

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

python run_hepmass_dataval_10M.py --seed 42 --method {method} --job_id $SLURM_JOB_ID

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