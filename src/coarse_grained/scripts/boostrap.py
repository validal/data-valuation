import sys
sys.path.insert(0, '.')

import os
import argparse
import time
import random
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset, Subset
import traceback

# Import model definitions and training utilities
from model.resnet import BasicBlock, Bottleneck, ResNet
from model.cnn import CNN
from utils import load_dataset_cls, train, test

# Import baseline modules
from baselines import OT
from baselines import RV
from baselines import DAVINZ
from baselines.ntk import compute_ntk_score_batched_permute, compute_ntk_score_batched
from baselines.mmd import rbf_mmd2
import resource


# ===============================
# Helper Functions
# ===============================
def parse_seed_range(seed_range_str):
    """Parse seed range string (e.g., '0-10') into list of seeds."""
    if '-' in seed_range_str:
        start, end = seed_range_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in seed_range_str:
        return [int(s.strip()) for s in seed_range_str.split(',')]
    else:
        return [int(seed_range_str)]


def generate_bootstraps(args, train_data, train_labels, output_dir, bootstrap_seeds, bootstrap_size):
    """Generate bootstrap samples from training data with reproducible seeds and variable sizes."""
    bootstrap_dir = output_dir / 'bootstraps'
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_summary = {}
    
    for seed in bootstrap_seeds:
        rng = np.random.RandomState(seed)
        
        # Randomly sample size between 1000 and bootstrap_size (default 10000)
        min_size = 10000
        max_size = bootstrap_size
        sample_size = rng.randint(min_size, max_size + 1)
        
        # Sample with replacement from train set
        train_size = len(train_data)
        sample_size = min(sample_size, train_size)
        bootstrap_indices = rng.choice(train_size, size=sample_size, replace=True)
        
        # Get actual indices from Subset if needed
        if isinstance(train_data, Subset):
            actual_indices = [train_data.indices[i] for i in bootstrap_indices]
        else:
            actual_indices = list(bootstrap_indices)
        
        # Count class distribution (handle arbitrary number of classes)
        bootstrap_labels = None
        try:
            if torch.is_tensor(train_labels):
                lab_arr = train_labels.cpu().numpy()
                bootstrap_labels = lab_arr[actual_indices]
            else:
                bootstrap_labels = np.asarray(train_labels)[actual_indices]
        except Exception:
            # fallback if indexing fails
            bootstrap_labels = np.asarray([train_labels[i] for i in actual_indices])
        unique, counts = np.unique(bootstrap_labels, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
        
        # Save bootstrap
        bootstrap_path = bootstrap_dir / f"bootstrap_seed{seed}_size{sample_size}.pt"
        torch.save({
            'indices': actual_indices,
            'seed': seed,
            'size': sample_size
        }, bootstrap_path)
        
        # Save metadata
        metadata_path = bootstrap_dir / f"bootstrap_seed{seed}_metadata.json"
        metadata = {
            'seed': seed,
            'size': sample_size,
            'dataset': args.dataset,
            'class_distribution': class_counts,
            'sampling_strategy': 'with_replacement_variable_size',
            'size_range': [min_size, max_size],
            'timestamp': time.time()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        metadata_summary[seed] = metadata
        print(f"[INFO] Bootstrap {seed}: size={sample_size} (range: {min_size}-{max_size}), classes={class_counts}")
    
    print(f"[INFO] Generated {len(bootstrap_seeds)} bootstraps with variable sizes in {bootstrap_dir}")
    return metadata_summary


def iter_bootstraps(args, output_dir, train_inputs, train_labels, batch_size=128):
    """
    Yield (seed, size, indices, loader) for each available bootstrap sample.
    """
    bootstrap_seeds = parse_seed_range(args.bootstrap_seeds)
    bootstrap_dir = output_dir / 'bootstraps'

    if not bootstrap_dir.exists() or not any(bootstrap_dir.glob("bootstrap_seed*_size*.pt")):
        raise FileNotFoundError(f"No bootstrap files found in directory: {bootstrap_dir}")

    for seed in bootstrap_seeds:
        bootstrap_files = list(bootstrap_dir.glob(f"bootstrap_seed{seed}_size*.pt"))
        if not bootstrap_files:
            print(f"[WARNING] Bootstrap {seed} not found, skipping.")
            continue

        bootstrap_path = bootstrap_files[0]
        bootstrap_data = torch.load(bootstrap_path)
        indices = bootstrap_data['indices']
        size = bootstrap_data.get('size', len(indices))
        bootstrap_dataset = Subset(TensorDataset(train_inputs, train_labels), indices)
        bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=batch_size, shuffle=False)

        yield seed, size, indices, bootstrap_loader


def train_bootstrap_model(args, model_class, bootstrap_indices, full_train_data, val_data, test_data,
                         lr, momentum, weight_decay, num_epochs, batch_size, device, 
                         output_dir, bootstrap_seed):
    """Train model on bootstrap sample, evaluate on validation and test sets."""
    print(f"\n{'='*60}")
    print(f"[INFO] TRAINING BOOTSTRAP MODEL")
    print(f"[INFO] Bootstrap Seed: {bootstrap_seed}")
    print(f"[INFO] Train on: bootstrap set ({len(bootstrap_indices)} samples)")
    print(f"[INFO] Evaluate on: validation set and test set")
    print(f"{'='*60}\n")
    
    # Create model instance
    model = model_class.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create bootstrap dataset
    bootstrap_data = Subset(full_train_data, bootstrap_indices)
    
    # DataLoaders
    loaders = {
        'train': DataLoader(bootstrap_data, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(val_data, batch_size=batch_size, shuffle=False),
        'test':  DataLoader(test_data, batch_size=batch_size, shuffle=False)
    }
    
    # Setup CSV logging (store under train_bootstraps subfolder)
    train_bs_dir = output_dir / 'train_bootstraps'
    train_bs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = train_bs_dir / f"{args.dataset.lower()}_bootstrap_seed{bootstrap_seed}_training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'elapsed_time'])
    
    print(f"[INFO] Training {model.__class__.__name__} for {num_epochs} epochs (LR={lr})...")
    print(f"[INFO] Training log: {csv_path}")
    start_time = time.time()
    
    # Train on bootstrap
    train(model, loaders, loss_fn, optimizer, device, num_epochs=num_epochs)
    
    total_time = time.time() - start_time
    print(f"[INFO] Bootstrap model training finished in {total_time:.2f} s")
    csv_file.close()
    
    # Evaluate on validation and test sets
    model.eval()
    val_acc, val_loss = test(model, loaders, loss_fn, device, dataloader_key='val')
    test_acc, test_loss = test(model, loaders, loss_fn, device, dataloader_key='test')
    print(f"[INFO] Bootstrap Model - Val accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")
    print(f"[INFO] Bootstrap Model - Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    
    # Append final metrics to CSV
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['final', 'N/A', val_acc, val_loss, test_acc, test_loss, total_time])
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_name = f"{args.dataset.lower()}_{model.__class__.__name__.lower()}_bootstrap_seed{bootstrap_seed}.pt"
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    
    metrics = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'epochs': num_epochs,
        'lr': lr,
        'seed': args.seed,
        'bootstrap_seed': bootstrap_seed,
        'dataset': args.dataset,
        'model_arch': model.__class__.__name__,
        'model_type': 'bootstrap_model',
        'trained_on': f'bootstrap_{bootstrap_seed}',
        'bootstrap_size': len(bootstrap_indices),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }
    
    torch.save(metrics, checkpoint_path)
    print(f"[INFO] Bootstrap model saved to {checkpoint_path}")
    # Also copy checkpoint to outputs for easy access
    try:
        import shutil
        dest = train_bs_dir / checkpoint_name
        shutil.copyfile(checkpoint_path, str(dest))
    except Exception:
        pass
    
    return model, checkpoint_path, metrics




# ===============================
# Training Methods
# ===============================
def train_base_model(args, model_class, loaders, train_data, val_data, test_data, 
                     lr, momentum, weight_decay, num_epochs, batch_size, device, output_dir):
    """
    Train base model on train set, evaluate on validation and test sets.
    Logs training progress to CSV.
    
    Args:
        output_dir: Directory to save training logs and checkpoints
    
    Returns:
        model, checkpoint_path, metrics_dict
    """
    print(f"\n{'='*60}")
    print(f"[INFO] TRAINING MODEL 1: Base Model")
    print(f"[INFO] Train on: train set ({len(train_data)} samples)")
    print(f"[INFO] Evaluate on: validation + test sets")
    print(f"{'='*60}\n")
    
    # Create model instance and use all GPUs if available
    model = model_class.to(device)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using only GPU {args.gpu} (no DataParallel).")
        # DataParallel disabled: model will use only the specified GPU
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup CSV logging (store under base_valid_model subfolder)
    base_dir = output_dir / 'base_valid_model'
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"{args.dataset.lower()}_base_model_seed{args.seed}_training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'elapsed_time'])
    
    print(f"[INFO] Training {model.__class__.__name__} for {num_epochs} epochs (LR={lr})...")
    print(f"[INFO] Training log: {csv_path}")
    start_time = time.time()
    
    # Train on train set with periodic evaluation
    train(model, loaders, loss_fn, optimizer, device, num_epochs=num_epochs)
    
    total_time = time.time() - start_time
    print(f"[INFO] Model 1 training finished in {total_time:.2f} s")
    csv_file.close()
    
    # Evaluate on validation and test sets
    model.eval()
    val_acc, val_loss = test(model, loaders, loss_fn, device, dataloader_key='val')
    test_acc, test_loss = test(model, loaders, loss_fn, device, dataloader_key='test')
    print(f"[INFO] Model 1 - Val accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")
    print(f"[INFO] Model 1 - Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    
    # Append final metrics to CSV
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['final', 'N/A', val_acc, val_loss, test_acc, test_loss, total_time])
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_name = f"{args.dataset.lower()}_{model.__class__.__name__.lower()}_base_model_seed{args.seed}.pt"
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    
    metrics = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'epochs': num_epochs,
        'lr': lr,
        'seed': args.seed,
        'dataset': args.dataset,
        'model_arch': model.__class__.__name__,
        'model_type': 'base_model',
        'trained_on': 'train_set',
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }
    
    torch.save(metrics, checkpoint_path)
    print(f"[INFO] Model 1 saved to {checkpoint_path}")
    # Also copy checkpoint into outputs base_valid_model folder
    try:
        import shutil
        dest = base_dir / checkpoint_name
        shutil.copyfile(checkpoint_path, str(dest))
    except Exception:
        pass
    
    return model, checkpoint_path, metrics


def train_valid_model(args, model_class, train_data, val_data, test_data,
                      lr, momentum, weight_decay, num_epochs, batch_size, device, output_dir):
    """
    Train feature extractor model on VALIDATION SET.
    Evaluate on TRAIN SET and TEST SET.
    Logs training progress to CSV.
    
    Args:
        output_dir: Directory to save training logs and checkpoints
    
    Returns:
        model, checkpoint_path, metrics_dict
    """
    print(f"\n{'='*60}")
    print(f"[INFO] TRAINING MODEL 2: Feature Extractor")
    print(f"[INFO] Train on: VALIDATION SET ({len(val_data)} samples)")
    print(f"[INFO] Evaluate on: TRAIN SET + TEST SET")
    print(f"{'='*60}\n")
    
    # Create model instance
    model = model_class.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create loaders with validation set as training set
    loaders = {
        'train': DataLoader(val_data, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(test_data, batch_size=batch_size, shuffle=False),
        'test':  DataLoader(train_data, batch_size=batch_size, shuffle=False)
    }
    
    # Setup CSV logging (store under base_valid_model subfolder)
    base_dir = output_dir / 'base_valid_model'
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"{args.dataset.lower()}_feature_extractor_seed{args.seed}_training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'test_accuracy', 'test_loss', 'train_accuracy', 'train_loss_eval', 'elapsed_time'])
    
    print(f"[INFO] Training {model.__class__.__name__} for {num_epochs} epochs (LR={lr})...")
    print(f"[INFO] Training log: {csv_path}")
    start_time = time.time()
    
    # Train on validation set
    train(model, loaders, loss_fn, optimizer, device, num_epochs=num_epochs)
    
    total_time = time.time() - start_time
    print(f"[INFO] Model 2 training finished in {total_time:.2f} s")
    csv_file.close()
    
    # Evaluate on test and train sets
    model.eval()
    test_acc, test_loss = test(model, loaders, loss_fn, device, dataloader_key='val')  # test set
    train_acc, train_loss = test(model, loaders, loss_fn, device, dataloader_key='test')  # train set
    print(f"[INFO] Model 2 - Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    print(f"[INFO] Model 2 - Train accuracy: {train_acc:.4f}, loss: {train_loss:.4f}")
    
    # Append final metrics to CSV
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['final', 'N/A', test_acc, test_loss, train_acc, train_loss, total_time])
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_name = f"{args.dataset.lower()}_{model.__class__.__name__.lower()}_feature_extractor_seed{args.seed}.pt"
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    
    metrics = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'epochs': num_epochs,
        'lr': lr,
        'seed': args.seed,
        'dataset': args.dataset,
        'model_arch': model.__class__.__name__,
        'model_type': 'feature_extractor',
        'trained_on': 'validation_set',
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }
    
    torch.save(metrics, checkpoint_path)
    print(f"[INFO] Model 2 saved to {checkpoint_path}")
    # Also copy checkpoint into outputs base_valid_model folder
    try:
        import shutil
        dest = base_dir / checkpoint_name
        shutil.copyfile(checkpoint_path, str(dest))
    except Exception:
        pass
    
    return model, checkpoint_path, metrics


# ===============================
# Main Function
# ===============================
def main():
    parser = argparse.ArgumentParser(description='Bootstrap Correlation Experiment')
    
    # Hardware and basic settings
    parser.add_argument('--gpu', type=str, default='0', help='GPU device index')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR_10', 'CIFAR_100', 'MNIST', 'TINYIMAGENET_100'],
                        help='Dataset to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save logs and metrics')
    
    # Operation flags
    parser.add_argument('--train_base', action='store_true', help='Train base model')
    parser.add_argument('--train_valid', action='store_true', help='Train feature extractor')
    parser.add_argument('--generate_bootstraps', action='store_true', help='Generate bootstrap samples')
    parser.add_argument('--train_bootstrap', action='store_true', help='Train models on bootstraps')
    parser.add_argument('--compute_ot', action='store_true', help='Compute OT distances')
    parser.add_argument('--compute_volume', action='store_true', help='Compute volume baseline metrics')
    parser.add_argument('--compute_davinz', action='store_true', help='Compute davinz baseline metrics')
    parser.add_argument('--tune_ntk', action='store_true', help='Run NTK tuning (n_batch and n_permute)')
    # RV tuning
    parser.add_argument('--rv_tuning', action='store_true', help='Run RV omega/alpha grid tuning')
    parser.add_argument('--rv_repeats', type=int, default=5, help='Number of independent repeats per RV config')
    # DaVinz batch tuning
    parser.add_argument('--davinz_batch_tuning', action='store_true', help='Run DaVinz n_batch grid tuning')
    parser.add_argument('--davinz_batch_values', type=str, default='10,20,50,100,250,500,1000',
                        help='Comma-separated n_batch values to try for DaVinz')
    parser.add_argument('--davinz_batch_repeats', type=int, default=5, help='Number of independent seeds/repeats per n_batch')
    
    # Bootstrap parameters
    parser.add_argument('--bootstrap_seeds', type=str, default='0-5',
                        help='Bootstrap seed range (e.g., "0-10" or "0,1,2,3")')
    parser.add_argument('--bootstrap_size', type=int, default=10000,
                        help='Maximum size of each bootstrap sample')
    
    # OT parameters
    parser.add_argument('--feature_extractor_path', type=str, default=None,
                        help='Path to feature extractor (for CIFAR-10)')
    parser.add_argument('--tinyimagenet_url', type=str, default=None,
                        help='Optional URL to download Tiny ImageNet archive if not available locally')
    parser.add_argument('--lambda_x', type=float, default=1.0, help='OT lambda_x parameter')
    parser.add_argument('--lambda_y', type=float, default=1.0, help='OT lambda_y parameter')
    parser.add_argument('--entreg', type=float, default=1e-1, help='OT entropic regularization')
    parser.add_argument('--use_test_as_val', action='store_true', help='Use the dataset test set as the validation set')
    parser.add_argument('--ot_repeats', type=int, default=1, help='Number of repeated OT runs per bootstrap seed')
    
    args = parser.parse_args()
    
    # Default: show help if no flags
    operations = [args.train_base, args.train_valid, args.generate_bootstraps,
                  args.train_bootstrap, args.compute_ot, args.compute_volume,
                  args.compute_davinz, args.tune_ntk, args.rv_tuning,
                  args.davinz_batch_tuning]
    if not any(operations):
        parser.print_help()
        return

    # ===============================
    # Environment setup
    # ===============================
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # Parse requested GPUs as physical device indices (e.g. '0' or '0,1,2')
    gpu_list = [int(x) for x in str(args.gpu).split(',') if x.strip() != ''] if getattr(args, 'gpu', None) is not None else []
    print(f"[INFO] Requested GPU(s): {gpu_list}")
    # Do not overwrite CUDA_VISIBLE_DEVICES here; use torch.cuda.set_device with physical indices
    device = torch.device('cpu')
    if torch.cuda.is_available() and gpu_list:
        try:
            # If a single GPU requested, set it as the active device
            torch.cuda.set_device(int(gpu_list[0]))
            device = torch.device(f'cuda:{int(gpu_list[0])}')
        except Exception as e:
            print(f"[WARN] Failed to set initial CUDA device: {e}")
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] Using device: {device}")

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Create dataset-specific subfolders under outputs (e.g., outputs/CIFAR_10)
    dataset_out = output_dir / args.dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    # subfolders: bootstraps, train_bootstraps, base_valid_model, OT, RV, davinz
    (dataset_out / 'bootstraps').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'train_bootstraps').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'base_valid_model').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'OT').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'RV').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'davinz').mkdir(parents=True, exist_ok=True)

    # Use dataset-specific output dir for subsequent writes
    output_dir = dataset_out

    # ===============================
    # Data loading with fixed split
    # ===============================
    # Choose sensible trim limits per dataset; None means keep all available
    trim_val = None
    if args.dataset == 'CIFAR_10':
        trim_val = 50000
    elif args.dataset == 'CIFAR_100':
        trim_val = 50000
    elif args.dataset == 'MNIST':
        trim_val = 60000
    elif args.dataset == 'TINYIMAGENET_100':
        trim_val = None


    try:
        ds_res = load_dataset_cls(
            args.dataset,
            trim_dataset=trim_val,
            num_parties=None
        )
        # load_dataset_cls returns either 6 values (train, test, dims, num_keep)
        # or 9 values (train, val, test, dims, n_train, n_valid). Handle both.
        if isinstance(ds_res, tuple) or isinstance(ds_res, list):
                # If loader returns 9 values we have an explicit validation split
                if len(ds_res) == 9:
                    print(f"[INFO] Dataset loader returned 9 values: using explicit validation split")
                    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, dims, n_train, n_valid = ds_res
                    validation_provided = True
                    validation_source = 'validation_split'
                elif len(ds_res) == 6:
                    print(f"[INFO] Dataset loader returned 6 values: mapping test set to validation set")
                    train_inputs, train_labels, test_inputs, test_labels, dims, num_to_keep = ds_res
                    # map test -> val for scripts that expect a validation split
                    val_inputs = test_inputs
                    val_labels = test_labels
                    n_train = len(train_inputs)
                    n_valid = len(val_inputs)
                    validation_provided = False
                    validation_source = 'test_mapped_to_val'
                else:
                    raise ValueError(f"Unexpected return length from load_dataset_cls: {len(ds_res)}")
        else:
            raise RuntimeError("load_dataset_cls did not return expected tuple/list")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Please ensure the dataset is correctly placed or downloaded, then rerun.")
        return


    # Ensure correct dtypes and shape
    if not torch.is_tensor(train_inputs):
        train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    else:
        train_inputs = train_inputs.to(dtype=torch.float32)

    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    else:
        train_labels = train_labels.to(dtype=torch.long)

    if not torch.is_tensor(val_inputs):
        val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    else:
        val_inputs = val_inputs.to(dtype=torch.float32)

    if not torch.is_tensor(val_labels):
        val_labels = torch.tensor(val_labels, dtype=torch.long)
    else:
        val_labels = val_labels.to(dtype=torch.long)

    # Ensure test inputs/labels (if present) are converted to torch tensors
    # NumPy arrays have a `.size` attribute that's an int (not callable),
    # which causes TensorDataset to error when it calls `.size(0)`.
    if 'test_inputs' in locals() and 'test_labels' in locals():
        if not torch.is_tensor(test_inputs):
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
        else:
            test_inputs = test_inputs.to(dtype=torch.float32)

        if not torch.is_tensor(test_labels):
            test_labels = torch.tensor(test_labels, dtype=torch.long)
        else:
            test_labels = test_labels.to(dtype=torch.long)

    # Reshape to NCHW
    train_inputs = train_inputs.view(-1, *dims)
    val_inputs = val_inputs.view(-1, *dims)
    # Reshape test inputs too when available
    if 'test_inputs' in locals():
        try:
            test_inputs = test_inputs.view(-1, *dims)
        except Exception:
            # If view fails, leave as-is; downstream code will catch shape errors
            pass

    # Optionally use the test set as the validation set (explicit flag)
    if getattr(args, 'use_test_as_val', False):
        if 'test_inputs' in locals() and 'test_labels' in locals():
            try:
                # ensure tensors
                if not torch.is_tensor(test_inputs):
                    test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
                if not torch.is_tensor(test_labels):
                    test_labels = torch.tensor(test_labels, dtype=torch.long)
                # reshape
                test_inputs = test_inputs.view(-1, *dims)
            except Exception:
                pass
            val_inputs = test_inputs
            val_labels = test_labels
            n_valid = len(val_inputs) if hasattr(val_inputs, '__len__') else int(val_inputs.size(0))
            print(f"[INFO] Using test set as validation set: val size={n_valid}")

    # Debug: report whether a validation split was provided or mapped/forced
    try:
        # If variables set above exist, prefer explicit flags
        v_provided = validation_provided if 'validation_provided' in locals() else False
        v_source = validation_source if 'validation_source' in locals() else ('test_forced' if getattr(args, 'use_test_as_val', False) else 'unknown')
    except Exception:
        v_provided = False
        v_source = 'unknown'
    print(f"[DEBUG] Validation provided: {v_provided}; validation_source={v_source}")

    print(f"[INFO] Original train size: {len(train_inputs)}, val size: {len(val_inputs)}")

    # Save class distribution CSVs for train, valid, test
    import csv
    def save_class_distribution(labels, filename):
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'count'])
            for u, c in zip(unique, counts):
                writer.writerow([int(u), int(c)])

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    save_class_distribution(train_labels, out_dir / 'train_class_distribution.csv')
    save_class_distribution(val_labels, out_dir / 'valid_class_distribution.csv')
    if 'test_inputs' in locals() and 'test_labels' in locals():
        save_class_distribution(test_labels, out_dir / 'test_class_distribution.csv')


    # Use original valid folder as validation set for Tiny Image Net
    num_classes = int(len(torch.unique(train_labels).cpu().numpy())) if torch.is_tensor(train_labels) else int(len(np.unique(train_labels)))
    train_data = TensorDataset(train_inputs, train_labels)
    val_data   = TensorDataset(val_inputs, val_labels)
    if args.dataset == 'TINYIMAGENET_100':
        test_data = None  # Not used for Tiny ImageNet-100
    else:
        test_data = TensorDataset(test_inputs, test_labels)

    print(f"[INFO] Train size: {len(train_data)}, Val size: {len(val_data)}")

    # ===============================
    # Bootstrap generation
    # ===============================
    if args.generate_bootstraps:
        bootstrap_seeds = parse_seed_range(args.bootstrap_seeds)
        print(f"\n[INFO] Generating {len(bootstrap_seeds)} bootstraps...")
        generate_bootstraps(args, train_data, train_labels, output_dir, 
                           bootstrap_seeds, args.bootstrap_size)
        return

    # ===============================
    # Bootstrap training
    # ===============================
    if args.train_bootstrap:
        # Hyperparameters
        lr = 0.01
        momentum = 0.9
        weight_decay = 5e-4
        # use similar schedule to CIFAR for image tasks
        num_epochs = 300 if args.dataset in ('CIFAR_10', 'TINYIMAGENET_100') else 100
        batch_size = 128
        
        bootstrap_results = []
        for bootstrap_seed, _, bootstrap_indices, _ in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=batch_size
        ):
            # Create model class
            # use inferred num_classes from data
            if args.dataset in ('CIFAR_10', 'CIFAR_100', 'TINYIMAGENET_100'):
                # use ResNet50 (Bottleneck blocks)
                #model_class = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes)
                model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=100)
            elif args.dataset == 'MNIST':
                model_class = CNN(in_channels=1, num_classes=num_classes)

            model, checkpoint_path, metrics = train_bootstrap_model(
                args, model, bootstrap_indices,
                TensorDataset(train_inputs, train_labels), val_data, test_data,
                lr, momentum, weight_decay, num_epochs, batch_size, device,
                output_dir, bootstrap_seed
            )
            bootstrap_results.append(metrics)

        # Write all results to a single CSV
            gpu_id = getattr(args, 'gpu', 0)
            results_dir = output_dir / 'train_bootstraps'
            results_dir.mkdir(parents=True, exist_ok=True)
            csv_path = results_dir / f"{args.dataset.lower()}_bootstrap_results_gpu{gpu_id}.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'bootstrap_seed', 'bootstrap_size', 'val_accuracy', 'val_loss',
                'test_accuracy', 'test_loss', 'epochs', 'lr', 'model_arch', 'checkpoint_path'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in bootstrap_results:
                writer.writerow({
                    'bootstrap_seed': m.get('bootstrap_seed'),
                    'bootstrap_size': m.get('bootstrap_size'),
                    'val_accuracy': m.get('val_accuracy'),
                    'val_loss': m.get('val_loss'),
                    'test_accuracy': m.get('test_accuracy'),
                    'test_loss': m.get('test_loss'),
                    'epochs': m.get('epochs'),
                    'lr': m.get('lr'),
                    'model_arch': m.get('model_arch'),
                    'checkpoint_path': checkpoint_path
                })
        print(f"[INFO] All bootstrap results saved to {csv_path}")
        print("[INFO] Bootstrap training completed.")
        return

    # ===============================
    # OT Distance Computation
    # ===============================
    if args.compute_ot:
        print("\n" + "="*60)
        print("COMPUTING OT DISTANCES")
        print("="*60)
        
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        
        # Determine feature extractor path
        feature_extractor_path = args.feature_extractor_path
        # Prefer a known checkpoint path when available for CIFAR-10
        if args.dataset == 'CIFAR_10' and feature_extractor_path is None:
            preferred = Path('/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Total_Data_Value/checkpoints/cifar_10_resnet_feature_extractor_seed0.pt')
            if preferred.exists():
                feature_extractor_path = preferred
            else:
                feature_extractor_path = Path(args.save_dir) / f"{args.dataset.lower()}_resnet_feature_extractor_seed0.pt"
        if args.dataset == 'MNIST' and feature_extractor_path is None:
            feature_extractor_path = Path(args.save_dir) / f"{args.dataset.lower()}_cnn_feature_extractor_seed0.pt"
        
        
        results = []

        repeats = max(1, int(getattr(args, 'ot_repeats', 1)))
        for seed, size, _, bootstrap_loader in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=128
        ):
            # select GPU for this bootstrap (round-robin over requested physical GPUs)
            if 'gpu_list' in locals() and gpu_list and torch.cuda.is_available():
                chosen = gpu_list[int(seed) % len(gpu_list)]
                try:
                    torch.cuda.set_device(int(chosen))
                    device = torch.device(f'cuda:{int(chosen)}')
                except Exception:
                    pass
            print(f"\n[INFO] Computing OT distance for bootstrap seed {seed} (size {size}) on device {device}...")
            
            try:
                # Debug: show which feature extractor path will be used and if it exists
                try:
                    fe_path = feature_extractor_path
                    fe_exists = Path(fe_path).exists() if fe_path is not None else False
                except Exception:
                    fe_exists = False
                print(f"[DEBUG-OT] feature_extractor_path={feature_extractor_path} exists={fe_exists}")

                # Run OT multiple times (repeats) with different seeds to quantify run-to-run variability
                for rep in range(repeats):
                    # reseed globally for reproducibility between repeats
                    try:
                        set_seed(args.seed + seed * 100 + rep)
                    except Exception:
                        # set_seed defined in main; if unavailable, fallback to numpy/torch seeding
                        import random as _rand
                        _rand.seed(args.seed + seed * 100 + rep)
                        np.random.seed(args.seed + seed * 100 + rep)
                        torch.manual_seed(args.seed + seed * 100 + rep)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(args.seed + seed * 100 + rep)

                    ot_result = OT.compute_ot_distance(
                        bootstrap_loader, val_loader,
                        dataset=args.dataset,
                        feature_extractor_path=feature_extractor_path,
                        device=device,
                        lambda_x=args.lambda_x,
                        lambda_y=args.lambda_y,
                        entreg=args.entreg
                    )

                    # Add seed/repeat info
                    ot_result['seed'] = seed
                    ot_result['repeat'] = rep
                    ot_result['bootstrap_size'] = size
                    results.append(ot_result)
                # Debug: print timing details for OT including feature extraction
                timing = ot_result.get('timing', {}) if isinstance(ot_result, dict) else {}
                print(f"  Distance: {ot_result.get('distance', float('nan')):.6f}")
                print(f"  OT computation time: {timing.get('ot_computation', np.nan):.4f}s")
                print(f"  OT timing keys: {list(timing.keys())}")
                print(f"  Feature extraction time: {timing.get('feature_extraction', timing.get('feature_extraction_estimate', np.nan))}")
                
            except Exception as e:
                print(f"[ERROR] Failed for seed {seed}: {e}")
                traceback.print_exc()
        
        # Create an OT output folder suffixed with lambda parameters and save results there
        lx_s = str(args.lambda_x).replace('.', 'p')
        ly_s = str(args.lambda_y).replace('.', 'p')
        ot_out = output_dir / f"OT_lx{lx_s}_ly{ly_s}"
        ot_out.mkdir(parents=True, exist_ok=True)

        # Save results via existing helper into the parameterized OT folder
        OT.save_ot_results(results, args.dataset, ot_out)

        # Print timing summary
        if results:
            OT.print_timing_summary(results)

        # Also write an aggregated CSV with standardized columns under the parameterized OT folder
        ot_csv = ot_out / f"ot_results_{args.dataset.lower()}_lx{lx_s}_ly{ly_s}.csv"
        import csv as _csv
        headers = [
            'seed', 'repeat', 'bootstrap_size', 'val_size', 'dataset',
            'feature_extraction_time_s', 'feature_extraction_mem_bytes',
            'ot_time_s', 'ot_mem_bytes', 'total_time_s', 'total_mem_bytes',
            'ot_distance', 'gpu_id', 'timestamp'
        ]
        with open(ot_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                timing = r.get('timing', {}) if isinstance(r, dict) else {}
                mem = r.get('mem', {}) if isinstance(r, dict) else {}
                row = {
                    'seed': r.get('seed'),
                    'repeat': r.get('repeat', 0),
                    'bootstrap_size': r.get('bootstrap_size'),
                    'val_size': int(r.get('val_size', timing.get('val_size', 0))) if r.get('val_size', None) is not None else None,
                    'dataset': r.get('dataset', args.dataset),
                    'feature_extraction_time_s': timing.get('feature_extraction', timing.get('feature_extraction_estimate', np.nan)),
                    'feature_extraction_mem_bytes': mem.get('feature_extraction', np.nan),
                    'ot_time_s': timing.get('ot_computation', timing.get('ot_time', np.nan)),
                    'ot_mem_bytes': mem.get('ot', np.nan),
                    'total_time_s': timing.get('total', np.nan) if timing.get('total', None) is not None else (timing.get('feature_extraction', 0) + timing.get('ot_computation', 0)),
                    'total_mem_bytes': mem.get('total', np.nan),
                    'ot_distance': r.get('distance', r.get('ot_distance', np.nan)),
                    'gpu_id': args.gpu,
                    'timestamp': r.get('timestamp', time.time())
                }
                writer.writerow(row)

        print(f"[INFO] OT aggregated CSV written to {ot_csv}")

        return

    # ===============================
    # Volume Baseline Computation
    # ===============================
    if args.compute_volume:
        print("\n" + "="*60)
        print("COMPUTING RV BASELINE METRICS")
        print("="*60)

        feature_extractor_path = args.feature_extractor_path
        # Prefer a known checkpoint path when available for CIFAR-10
        if args.dataset == 'CIFAR_10' and feature_extractor_path is None:
            preferred = Path('/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Total_Data_Value/checkpoints/cifar_10_resnet_feature_extractor_seed0.pt')
            if preferred.exists():
                feature_extractor_path = preferred
            else:
                feature_extractor_path = Path(args.save_dir) / f"{args.dataset.lower()}_resnet_feature_extractor_seed0.pt"
        if args.dataset == 'MNIST' and feature_extractor_path is None:  
            feature_extractor_path = Path(args.save_dir) / f"{args.dataset.lower()}_cnn_feature_extractor_seed0.pt"
        results = []
        for seed, size, _, bootstrap_loader in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=128
        ):
            # select GPU for this bootstrap (round-robin over requested physical GPUs)
            if 'gpu_list' in locals() and gpu_list and torch.cuda.is_available():
                chosen = gpu_list[int(seed) % len(gpu_list)]
                try:
                    torch.cuda.set_device(int(chosen))
                    device = torch.device(f'cuda:{int(chosen)}')
                except Exception:
                    pass
            print(f"\n[INFO] Computing RV for bootstrap seed {seed} (size {size}) on device {device}...")
            # Debug: show which feature extractor path will be used and if it exists
            try:
                fe_path = feature_extractor_path
                fe_exists = Path(fe_path).exists() if fe_path is not None else False
            except Exception:
                fe_exists = False
            print(f"[DEBUG-RV] feature_extractor_path={feature_extractor_path} exists={fe_exists}")

            rv_result = RV.compute_rv_metric(
                bootstrap_loader,
                dataset=args.dataset,
                device=device,
                feature_extractor_path=feature_extractor_path
            )

            # Debug: print timing details for RV including feature extraction
            rt = rv_result.get('timing', {}) if isinstance(rv_result, dict) else {}
            print(f"  RV timing keys: {list(rt.keys())}")
            print(f"  Feature extraction time: {rt.get('feature_extraction', rt.get('feature_extraction_estimate', np.nan))}")
            rv_result['seed'] = seed
            rv_result['bootstrap_size'] = size
            results.append(rv_result)

        RV.save_rv_results(results, args.dataset, output_dir)

        # Also write aggregated CSV under outputs/RV with standardized columns
        rv_out = output_dir / 'RV'
        rv_out.mkdir(parents=True, exist_ok=True)
        rv_csv = rv_out / f"rv_results_{args.dataset.lower()}.csv"
        import csv as _csv
        headers = [
            'seed', 'bootstrap_size', 'val_size', 'dataset',
            'feature_extraction_time_s', 'feature_extraction_mem_bytes',
            'rv_time_s', 'rv_mem_bytes', 'total_time_s', 'total_mem_bytes',
            'volume', 'gpu_id', 'timestamp'
        ]
        with open(rv_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                timing = r.get('timing', {}) if isinstance(r, dict) else {}
                mem = r.get('mem', {}) if isinstance(r, dict) else {}
                row = {
                    'seed': r.get('seed'),
                    'bootstrap_size': r.get('bootstrap_size'),
                    'val_size': int(r.get('val_size', timing.get('val_size', 0))) if r.get('val_size', None) is not None else None,
                    'dataset': r.get('dataset', args.dataset),
                    'feature_extraction_time_s': timing.get('feature_extraction', timing.get('feature_extraction_estimate', np.nan)),
                    'feature_extraction_mem_bytes': mem.get('feature_extraction', np.nan),
                    'rv_time_s': timing.get('rv_computation', timing.get('volume_time', np.nan)),
                    'rv_mem_bytes': mem.get('rv', np.nan),
                    'total_time_s': timing.get('total', np.nan) if timing.get('total', None) is not None else (timing.get('feature_extraction', 0) + timing.get('rv_computation', 0)),
                    'total_mem_bytes': mem.get('total', np.nan),
                    'volume': r.get('volume', np.nan),
                    'gpu_id': args.gpu,
                    'timestamp': r.get('timestamp', time.time())
                }
                writer.writerow(row)

        print(f"[INFO] RV aggregated CSV written to {rv_csv}")
        print(f"\n[INFO] RV computation completed for {len(results)} bootstraps")
        return

    # ===============================
    # RV Tuning (omega x alpha grid)
    # ===============================
    if args.rv_tuning:
        print("\n" + "="*60)
        print("RUNNING RV TUNING (omega x alpha grid)")
        print("="*60)

        feature_extractor_path = args.feature_extractor_path
        if args.dataset == 'CIFAR_10' and feature_extractor_path is None:
            preferred = Path('/home/mehdi.touil/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/touil-lustre/Total_Data_Value/checkpoints/cifar_10_resnet_feature_extractor_seed0.pt')
            feature_extractor_path = preferred if preferred.exists() else Path(args.save_dir) / f"{args.dataset.lower()}_resnet_feature_extractor_seed0.pt"
        if args.dataset == 'MNIST' and feature_extractor_path is None:
            feature_extractor_path = Path(args.save_dir) / f"{args.dataset.lower()}_cnn_feature_extractor_seed0.pt"

        # Grid: omega x alpha_multiplier
        # alpha_internal = 1 / (alpha_multiplier * n)
        omega_values = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
        alpha_configs = [
            ('1n',   1),
            ('10n',  10),
            ('100n', 100),
        ]
        n_rv_repeats = max(1, int(args.rv_repeats))

        tuning_out = output_dir / 'RV_tuning'
        tuning_out.mkdir(parents=True, exist_ok=True)

        import csv as _csv

        summary_csv = tuning_out / f"rv_tuning_summary_{args.dataset.lower()}.csv"
        summary_header = ['config', 'omega', 'alpha_multiplier', 'alpha_label',
                          'rep', 'seed', 'bootstrap_size',
                          'log_volume', 'log_robust_volume',
                          'rv_time_s', 'total_time_s', 'timestamp']
        with open(summary_csv, 'w', newline='') as f:
            _csv.DictWriter(f, fieldnames=summary_header).writeheader()

        for omega in omega_values:
            omega_str = str(omega).replace('.', 'p')
            for alpha_label, alpha_multiplier in alpha_configs:
                config_name = f"omega{omega_str}_alpha1_{alpha_label}"
                config_dir = tuning_out / 'RV' / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[RV TUNE] Config: {config_name}  "
                      f"(omega={omega}, alpha_multiplier={alpha_multiplier} -> alpha_internal=1/({alpha_multiplier}*n))")

                for rep in range(1, n_rv_repeats + 1):
                    rep_dir = config_dir / f"seed{rep}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    set_seed(args.seed + rep * 1000)

                    rep_csv = rep_dir / 'rv_results.csv'
                    rep_header = ['bootstrap_seed', 'bootstrap_size',
                                  'log_volume', 'log_robust_volume',
                                  'rv_time_s', 'feature_extraction_time_s', 'total_time_s', 'timestamp']
                    with open(rep_csv, 'w', newline='') as f:
                        _csv.DictWriter(f, fieldnames=rep_header).writeheader()

                    print(f"  [rep {rep}/{n_rv_repeats}]")
                    for seed, size, _, bootstrap_loader in iter_bootstraps(
                        args, output_dir, train_inputs, train_labels, batch_size=128
                    ):
                        try:
                            res = RV.compute_rv_metric(
                                bootstrap_loader,
                                dataset=args.dataset,
                                device=device,
                                feature_extractor_path=feature_extractor_path,
                                max_samples=10000,
                                omega=omega,
                                alpha=alpha_multiplier
                            )
                            timing = res.get('timing', {})
                            ts = time.time()
                            row = {
                                'bootstrap_seed': seed,
                                'bootstrap_size': size,
                                'log_volume': res.get('log_volume', ''),
                                'log_robust_volume': res.get('log_robust_volume', ''),
                                'rv_time_s': timing.get('rv_computation', ''),
                                'feature_extraction_time_s': timing.get('feature_extraction', ''),
                                'total_time_s': timing.get('total', ''),
                                'timestamp': ts
                            }
                            with open(rep_csv, 'a', newline='') as f:
                                _csv.DictWriter(f, fieldnames=rep_header).writerow(row)

                            summary_row = {
                                'config': config_name,
                                'omega': omega,
                                'alpha_multiplier': alpha_multiplier,
                                'alpha_label': alpha_label,
                                'rep': rep,
                                'seed': seed,
                                'bootstrap_size': size,
                                'log_volume': res.get('log_volume', ''),
                                'log_robust_volume': res.get('log_robust_volume', ''),
                                'rv_time_s': timing.get('rv_computation', ''),
                                'total_time_s': timing.get('total', ''),
                                'timestamp': ts
                            }
                            with open(summary_csv, 'a', newline='') as f:
                                _csv.DictWriter(f, fieldnames=summary_header).writerow(summary_row)

                        except Exception as e:
                            print(f"[ERROR] RV tuning failed config={config_name} rep={rep} seed={seed}: {e}")
                            import traceback
                            traceback.print_exc()

                    print(f"  [rep {rep}] done -> {rep_csv}")

        print(f"\n[INFO] RV tuning complete. Results under {tuning_out}")
        print(f"[INFO] Aggregated summary: {summary_csv}")
        return

    # ===============================
    # davinz Baseline Computation
    # ===============================
    if args.compute_davinz:
        print("\n" + "="*60)
        print("COMPUTING davinz BASELINE METRICS")
        print("="*60)

        results = []
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        # Initialize or load an untrained model once. Create-and-save on first run,
        # load from disk on subsequent runs so initial weights remain consistent.
        os.makedirs(args.save_dir, exist_ok=True)
        if args.dataset == 'CIFAR_10':
            # Use ResNet-18 (BasicBlock) for CIFAR-10 to match ResNet-18 behaviour
            _model_template = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes)
            default_n_batch = 100
        elif args.dataset == 'TINYIMAGENET_100':
            _model_template = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes)
            default_n_batch = 100
        elif args.dataset == 'MNIST':
            _model_template = CNN(in_channels=1, num_classes=num_classes)
            default_n_batch = 1
        else:
            _model_template = None
            default_n_batch = 1

        init_path = None
        if _model_template is not None:
            init_path = Path(args.save_dir) / f"{args.dataset.lower()}_{_model_template.__class__.__name__.lower()}_init_seed{args.seed}.pt"

        model = None
        if init_path is not None and init_path.exists():
            try:
                model = torch.load(str(init_path)).to(device)
                print(f"[INFO] Loaded initial model from {init_path}")
            except Exception as e:
                print(f"[WARN] Failed to load initial model ({init_path}): {e}")
                model = _model_template.to(device) if _model_template is not None else None
                try:
                    torch.save(model, str(init_path))
                    print(f"[INFO] Saved recreated initial model to {init_path}")
                except Exception:
                    pass
        else:
            if _model_template is not None:
                model = _model_template.to(device)
                try:
                    if init_path is not None:
                        torch.save(model, str(init_path))
                        print(f"[INFO] Created and saved initial model at {init_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save initial model at {init_path}: {e}")

                # Additional debug: model summary for the initialized template
                try:
                    param_count = sum(p.numel() for p in model.parameters())
                except Exception:
                    param_count = None
                try:
                    conv1_shape = tuple(model.conv1.weight.shape) if hasattr(model, 'conv1') and hasattr(model.conv1, 'weight') else None
                except Exception:
                    conv1_shape = None
                ah_loaded = False
                try:
                    import sys
                    ah_loaded = 'autograd_hacks' in sys.modules
                except Exception:
                    try:
                        ah_loaded = 'autograd_hacks' in globals()
                    except Exception:
                        ah_loaded = False
                print(f"[DEBUG-MODEL] init model class={model.__class__.__name__}, params={param_count}, conv1_shape={conv1_shape}, autograd_hacks_loaded={ah_loaded}")

        n_batch = default_n_batch

        # If requested, run DaVinz batch-size tuning grid and exit
        if getattr(args, 'davinz_batch_tuning', False):
            print("\n[INFO] Running DaVinz batch-size tuning")
            batch_values = [int(x.strip()) for x in str(getattr(args, 'davinz_batch_values', '10,20,50,100,250,500,1000')).split(',') if x.strip()]
            repeats = int(getattr(args, 'davinz_batch_repeats', 5))

            tuning_out = output_dir / 'Davinz_batchtuning'
            tuning_out.mkdir(parents=True, exist_ok=True)

            # Use validation loader once
            val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

            seed_list = parse_seed_range(getattr(args, 'bootstrap_seeds', '0-0'))

            # Desired CSV header per user request
            header = [
                'bootstrap_size', 'dataset', 'timing', 'rep', 'mem', 'mmd_raw_time',
                'n_batch', 'status', 'n_permute', 'davinz_score', 'seed', 'error',
                'mmd_raw_mem', 'bootstrap_labels_hash', 'maxsamples_used', 'val_size',
                'mmd', 'ntk', 'bootstrap_labels_sample', 'mmd_raw', 'bootstrap_labels_unique'
            ]

            for rep_idx in range(repeats):
                rep = rep_idx + 1
                rep_model_seed = int(args.seed) + rep_idx * 1000
                print(f"\n[Davinz Tune] Repeat {rep}/{repeats} (model_seed={rep_model_seed})")

                # Initialize fresh model for this repeat
                try:
                    set_seed(rep_model_seed)
                    if _model_template is not None:
                        model_rep = type(_model_template)(*getattr(_model_template, 'args', [])) if False else None
                        # instantiate by class directly
                        if args.dataset == 'CIFAR_10':
                            model_rep = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes).to(device)
                        elif args.dataset == 'TINYIMAGENET_100':
                            model_rep = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes).to(device)
                        elif args.dataset == 'CIFAR_100':
                            model_rep = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes).to(device)
                        elif args.dataset == 'MNIST':
                            model_rep = CNN(in_channels=1, num_classes=num_classes).to(device)
                        else:
                            model_rep = None
                    else:
                        model_rep = None
                except Exception as e:
                    print(f"[ERROR] Failed to initialize repeat model (rep={rep}): {e}")
                    traceback.print_exc()
                    model_rep = None

                # For each n_batch, create rep folder and run all bootstrap seeds using model_rep
                for nb in batch_values:
                    nb_parent = tuning_out / f'n_batch_{nb}'
                    nb_dir = nb_parent / f'rep_{rep}'
                    nb_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[Davinz Tune] n_batch={nb} rep={rep} -> outputs: {nb_dir}")

                    out_csv = nb_dir / 'davinz_results.csv'
                    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
                    if write_header:
                        with open(out_csv, 'w', newline='') as _f:
                            w = csv.writer(_f)
                            w.writerow(header)

                    # save model info
                    try:
                        with open(nb_dir / 'model_info.json', 'w') as mf:
                            json.dump({'rep': rep, 'model_seed': rep_model_seed}, mf, indent=2)
                    except Exception:
                        pass

                    results_rows = []
                    skipped_config = False

                    for seed in seed_list:
                        # find bootstrap file
                        bootstrap_files = list((output_dir / 'bootstraps').glob(f"bootstrap_seed{seed}_size*.pt"))
                        if not bootstrap_files:
                            miss = {k: '' for k in header}
                            miss['seed'] = seed
                            miss['status'] = 'missing_bootstrap'
                            with open(out_csv, 'a', newline='') as _f:
                                w = csv.writer(_f)
                                w.writerow([miss.get(h, '') for h in header])
                            print(f"[WARN] Missing bootstrap {seed} -> logged in {out_csv}")
                            continue

                        bp = torch.load(str(bootstrap_files[0]))
                        indices = bp.get('indices', [])
                        bsize = int(bp.get('size', len(indices)))
                        bootstrap_dataset = Subset(TensorDataset(train_inputs, train_labels), indices)
                        bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=128, shuffle=False)

                        try:
                            # ensure model on device
                            if model_rep is not None:
                                try:
                                    model_rep.to(device)
                                except Exception:
                                    pass

                            davinz_result = DAVINZ.compute_davinz(
                                bootstrap_loader,
                                val_loader,
                                dataset=args.dataset,
                                device=device,
                                model=model_rep,
                                #diagonal_I_mag=1e-6,
                                n_batch=nb)

                            davinz_result['seed'] = seed
                            davinz_result['bootstrap_size'] = bsize
                            davinz_result['rep'] = rep
                            davinz_result['n_batch'] = nb
                            davinz_result['status'] = 'ok'
                            davinz_result['error'] = ''

                            # prepare ordered row
                            row = {k: '' for k in header}
                            timing = davinz_result.get('timing', {}) if isinstance(davinz_result, dict) else {}
                            mem = davinz_result.get('mem', {}) if isinstance(davinz_result, dict) else {}
                            row.update({
                                'bootstrap_size': davinz_result.get('bootstrap_size'),
                                'dataset': args.dataset,
                                'timing': json.dumps(timing),
                                'rep': rep,
                                'mem': json.dumps(mem),
                                'mmd_raw_time': davinz_result.get('mmd_raw_time'),
                                'n_batch': nb,
                                'status': 'ok',
                                'n_permute': davinz_result.get('n_permute'),
                                'davinz_score': davinz_result.get('davinz_score'),
                                'seed': seed,
                                'error': '',
                                'mmd_raw_mem': davinz_result.get('mmd_raw_mem'),
                                'bootstrap_labels_hash': davinz_result.get('bootstrap_labels_hash'),
                                'maxsamples_used': davinz_result.get('maxsamples_used'),
                                'val_size': davinz_result.get('val_size', timing.get('val_size', '')),
                                'mmd': davinz_result.get('mmd'),
                                'ntk': davinz_result.get('ntk'),
                                'bootstrap_labels_sample': json.dumps(davinz_result.get('bootstrap_labels_sample', '')),
                                'mmd_raw': davinz_result.get('mmd_raw'),
                                'bootstrap_labels_unique': json.dumps(davinz_result.get('bootstrap_labels_unique', ''))
                            })

                            with open(out_csv, 'a', newline='') as _f:
                                w = csv.writer(_f)
                                w.writerow([row.get(h, '') for h in header])

                            results_rows.append(davinz_result)

                        except RuntimeError as e:
                            msg = str(e).lower()
                            errrow = {k: '' for k in header}
                            errrow['seed'] = seed
                            errrow['rep'] = rep
                            errrow['status'] = 'failed_runtime'
                            errrow['error'] = str(e)
                            with open(out_csv, 'a', newline='') as _f:
                                w = csv.writer(_f)
                                w.writerow([errrow.get(h, '') for h in header])
                            print(f"[ERROR] RuntimeError for n_batch={nb} seed={seed} rep={rep}: {e}")
                            traceback.print_exc()
                            if 'out of memory' in msg:
                                # mark parent dir to indicate config skipped
                                try:
                                    (nb_parent / 'SKIPPED_MEMORY_ERROR').write_text('true')
                                except Exception:
                                    pass
                                skipped_config = True
                                break
                            else:
                                continue
                        except Exception as e:
                            errrow = {k: '' for k in header}
                            errrow['seed'] = seed
                            errrow['rep'] = rep
                            errrow['status'] = 'failed_exception'
                            errrow['error'] = str(e)
                            with open(out_csv, 'a', newline='') as _f:
                                w = csv.writer(_f)
                                w.writerow([errrow.get(h, '') for h in header])
                            print(f"[ERROR] Exception for n_batch={nb} seed={seed} rep={rep}: {e}")
                            traceback.print_exc()
                            continue

                    # per-rep per-n_batch summary
                    try:
                        mmds = [r.get('mmd') for r in results_rows if r.get('mmd') is not None]
                        ntks = [r.get('ntk') for r in results_rows if r.get('ntk') is not None and r.get('ntk') != 0]
                        mean_mmd = float(np.mean(mmds)) if mmds else 0.0
                        mean_ntk = float(np.mean(ntks)) if ntks else 0.0
                        kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0
                        summary = {'rep': rep, 'n_batch': nb, 'mean_mmd': mean_mmd, 'mean_ntk': mean_ntk, 'kappa': kappa, 'n_completed': len(results_rows)}
                        with open(nb_dir / 'summary.json', 'w') as sf:
                            json.dump(summary, sf, indent=2)
                    except Exception:
                        pass

                # end nb loop

            # After all repeats, restructure outputs into tiered folders and
            # produce JSON metadata + flat CSVs per n_batch/rep, a summary per n_batch,
            # and a combined aggregated CSV across all n_batch and repeats.
            aggregated_rows = []
            # Do not write experiment-level files at tuning root; keep outputs inside each rep folder
            print(f"[INFO] DaVinz tuning: per-rep outputs available under {tuning_out}/*/rep_*/")

            for nb in batch_values:
                nb_parent = tuning_out / f'n_batch_{nb}'
                nb_parent.mkdir(parents=True, exist_ok=True)
                rep_dirs = sorted([d for d in nb_parent.iterdir() if d.is_dir() and d.name.startswith('rep_')])

                # collect per-rep summaries for cross-repeat stats
                per_rep_stats = []

                for rd in rep_dirs:
                    # Read the raw per-rep CSV that we wrote during the runs
                    in_csv = rd / 'davinz_results.csv'
                    flat_rows = []
                    timing_map = {}
                    bootstrap_meta_map = {}

                    if in_csv.exists():
                        try:
                            with open(in_csv, 'r', newline='') as rf:
                                rdr = csv.DictReader(rf)
                                for row in rdr:
                                    # parse essential fields
                                    seed = row.get('seed')
                                    try:
                                        seed_i = int(seed) if seed not in (None, '', 'None') else None
                                    except Exception:
                                        seed_i = None

                                    bsize = row.get('bootstrap_size')
                                    try:
                                        bsize_i = int(bsize) if bsize not in (None, '', 'None') else None
                                    except Exception:
                                        bsize_i = None

                                    mmd = row.get('mmd')
                                    ntk = row.get('ntk')
                                    davinz_score = row.get('davinz_score')
                                    status = row.get('status', '')
                                    val_size = row.get('val_size', '')
                                    timestamp = row.get('timestamp') or time.time()

                                    # try to coerce numerics
                                    try:
                                        mmd_f = float(mmd) if mmd not in (None, '', 'None') else None
                                    except Exception:
                                        mmd_f = None
                                    try:
                                        ntk_f = float(ntk) if ntk not in (None, '', 'None') else None
                                    except Exception:
                                        ntk_f = None

                                    flat = {
                                        'n_batch': nb,
                                        'rep': rd.name.replace('rep_', ''),
                                        'seed': seed_i,
                                        'bootstrap_size': bsize_i,
                                        'mmd': mmd_f,
                                        'ntk': ntk_f,
                                        'davinz_score': float(davinz_score) if davinz_score not in (None, '', 'None') else None,
                                        'status': status,
                                        'val_size': int(val_size) if val_size not in (None, '', 'None') else None,
                                        'timestamp': timestamp
                                    }
                                    flat_rows.append(flat)
                                    aggregated_rows.append(flat)

                                    # parse timing & mem JSON fields if present
                                    try:
                                        timing = json.loads(row.get('timing') or '{}')
                                    except Exception:
                                        timing = {}
                                    try:
                                        mem = json.loads(row.get('mem') or '{}')
                                    except Exception:
                                        mem = {}

                                    timing_map[str(seed)] = {
                                        'mmd_time_s': timing.get('mmd_time_s', timing.get('mmd_raw_time', timing.get('mmd_time', None))),
                                        'ntk_time_s': timing.get('ntk_time_s', timing.get('ntk_time', None)),
                                        'total_time_s': timing.get('total', None),
                                        'mmd_mem_bytes': mem.get('mmd', mem.get('mmd_raw', None)),
                                        'ntk_mem_bytes': mem.get('ntk', None),
                                        'total_mem_bytes': mem.get('total', None)
                                    }

                                    # bootstrap metadata: try to load the originally saved bootstrap metadata JSON
                                    try:
                                        if seed_i is not None:
                                            meta_path = output_dir / 'bootstraps' / f'bootstrap_seed{seed_i}_metadata.json'
                                            meta = json.load(open(meta_path)) if meta_path.exists() else {}
                                        else:
                                            meta = {}
                                    except Exception:
                                        meta = {}

                                    # attach labels info from the CSV fields if present
                                    try:
                                        labels_sample = json.loads(row.get('bootstrap_labels_sample') or '[]')
                                    except Exception:
                                        labels_sample = []
                                    try:
                                        labels_unique = json.loads(row.get('bootstrap_labels_unique') or '[]')
                                    except Exception:
                                        labels_unique = []
                                    labels_hash = row.get('bootstrap_labels_hash')

                                    bootstrap_meta_map[str(seed)] = {
                                        'class_distribution': meta.get('class_distribution', {}),
                                        'labels_hash': labels_hash,
                                        'labels_sample': labels_sample,
                                        'labels_unique': labels_unique,
                                        'bootstrap_metadata': meta
                                    }

                        except Exception:
                            # skip malformed per-rep CSV
                            continue

                    # If davinz_score fields are missing in the per-rep CSV, compute
                    # a per-rep kappa = mean(mmd)/mean(ntk) and fill davinz_score = -(kappa*ntk + mmd)
                    try:
                        # collect numeric mmd/ntk for this rep
                        rep_mmds = [fr.get('mmd') for fr in flat_rows if fr.get('mmd') is not None]
                        rep_ntks = [fr.get('ntk') for fr in flat_rows if fr.get('ntk') is not None and fr.get('ntk') != 0]
                        mean_mmd = float(np.mean(rep_mmds)) if rep_mmds else 0.0
                        mean_ntk = float(np.mean(rep_ntks)) if rep_ntks else 0.0
                        rep_kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0
                        # Update flat_rows davinz_score when missing
                        for fr in flat_rows:
                            if fr.get('davinz_score') in (None, '', 'None'):
                                m = fr.get('mmd')
                                n = fr.get('ntk')
                                try:
                                    if m is not None and n is not None:
                                        fr['davinz_score'] = float(-(rep_kappa * n + m))
                                    else:
                                        fr['davinz_score'] = None
                                except Exception:
                                    fr['davinz_score'] = None
                        # also update aggregated_rows entries that reference these same seed/rep
                        for ar in aggregated_rows:
                            if ar.get('rep') == rd.name.replace('rep_', ''):
                                if ar.get('davinz_score') in (None, '', 'None'):
                                    # find matching seed in flat_rows
                                    match = next((x for x in flat_rows if x.get('seed') == ar.get('seed')), None)
                                    if match is not None:
                                        ar['davinz_score'] = match.get('davinz_score')
                    except Exception:
                        pass

                    # write per-rep files into rd
                    try:
                        # results_flat.csv
                        rfpath = rd / 'results_flat.csv'
                        with open(rfpath, 'w', newline='') as wf:
                            fieldnames = ['n_batch', 'rep', 'seed', 'bootstrap_size', 'mmd', 'ntk', 'davinz_score', 'status', 'val_size', 'timestamp']
                            wr = csv.DictWriter(wf, fieldnames=fieldnames)
                            wr.writeheader()
                            for rr in flat_rows:
                                wr.writerow(rr)
                    except Exception:
                        pass

                    try:
                        with open(rd / 'timing_metrics.json', 'w') as tf:
                            json.dump(timing_map, tf, indent=2)
                    except Exception:
                        pass

                    try:
                        with open(rd / 'bootstrap_metadata.json', 'w') as bm:
                            json.dump(bootstrap_meta_map, bm, indent=2)
                    except Exception:
                        pass

                    # ensure model_info.json exists (was written earlier); if not, create minimal
                    try:
                        mi = rd / 'model_info.json'
                        if not mi.exists():
                            json.dump({'rep': rd.name.replace('rep_', ''), 'note': 'model_info not present'}, open(mi, 'w'), indent=2)
                    except Exception:
                        pass

                    # try to capture per-rep summary if present (summary.json)
                    try:
                        sj = rd / 'summary.json'
                        if sj.exists():
                            js = json.load(open(sj))
                            per_rep_stats.append(js)
                    except Exception:
                        pass

                # end rep loop

                # compute cross-repeat statistics for this n_batch
                if True:
                    kappas = [s.get('kappa') for s in per_rep_stats if s.get('kappa') is not None]
                    mm = [s.get('mean_mmd') for s in per_rep_stats if s.get('mean_mmd') is not None]
                    nn = [s.get('mean_ntk') for s in per_rep_stats if s.get('mean_ntk') is not None]

                    stats = {
                        'n_batch': nb,
                        'kappa_mean': float(np.mean(kappas)) if kappas else None,
                        'kappa_std': float(np.std(kappas, ddof=0)) if kappas else None,
                        'kappa_min': float(np.min(kappas)) if kappas else None,
                        'kappa_max': float(np.max(kappas)) if kappas else None,
                        'mmd_mean': float(np.mean(mm)) if mm else None,
                        'mmd_std': float(np.std(mm, ddof=0)) if mm else None,
                        'ntk_mean': float(np.mean(nn)) if nn else None,
                        'ntk_std': float(np.std(nn, ddof=0)) if nn else None,
                        'n_repeats': len(per_rep_stats)
                    }
                    # intentionally skip writing cross-repeat summaries to keep tuning outputs
                    # scoped to per-rep folders only
                    pass

            print("\n[INFO] DaVinz batch-size tuning completed.")
            return

        for seed, size, _, bootstrap_loader in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=128
        ):
            # select GPU for this bootstrap (round-robin over requested physical GPUs)
            if 'gpu_list' in locals() and gpu_list and torch.cuda.is_available():
                chosen = gpu_list[int(seed) % len(gpu_list)]
                try:
                    torch.cuda.set_device(int(chosen))
                    device = torch.device(f'cuda:{int(chosen)}')
                except Exception:
                    pass
            print(f"\n[INFO] Computing davinz for bootstrap seed {seed} (size {size}) on device {device}...")
            # Ensure the preloaded initial model is on the current device
            if model is not None:
                try:
                    model = model.to(device)
                except Exception:
                    pass
       
            davinz_result = DAVINZ.compute_davinz(
                bootstrap_loader,
                val_loader,
                dataset=args.dataset,
                device=device,
                model=model,
                diagonal_I_mag=1e-6,
                n_batch=n_batch)

            davinz_result['seed'] = seed
            davinz_result['bootstrap_size'] = size
            if np.isnan(davinz_result['ntk']):
                raise ValueError(f"NTK value is NaN for seed {seed}")
            results.append(davinz_result)

            # Debug: print raw-MMD values immediately for diagnosis
            try:
                print(f"[DEBUG] After compute_davinz seed={seed} mmd_raw={davinz_result.get('mmd_raw')} mmd_raw_time={davinz_result.get('mmd_raw_time')} mmd_raw_mem={davinz_result.get('mmd_raw_mem')}")
            except Exception:
                pass

            # Append raw-MMD row immediately so file is populated during runs
            try:
                raw_csv = output_dir / 'davinz' / 'mmd_raw_Features.csv'
                raw_csv.parent.mkdir(parents=True, exist_ok=True)
                header = ['seed', 'bootstrap_size', 'mmd_raw', 'mmd_raw_time_s', 'mmd_raw_mem_bytes', 'mmd_raw_feature_dim', 'timestamp']
                write_header = not raw_csv.exists() or raw_csv.stat().st_size == 0
                with open(raw_csv, 'a', newline='') as _f:
                    _writer = csv.writer(_f)
                    if write_header:
                        _writer.writerow(header)
                    _writer.writerow([
                        float(davinz_result.get('seed') or 0),
                        int(davinz_result.get('bootstrap_size') or 0),
                        float(davinz_result.get('mmd_raw') or 0.0),
                        float(davinz_result.get('mmd_raw_time') or 0.0),
                        int(davinz_result.get('mmd_raw_mem') or 0),
                        int(davinz_result.get('mmd_raw_feature_dim') or 0),
                        time.time()
                    ])
                print(f"[INFO] Raw MMD written for seed {seed}: {davinz_result.get('mmd_raw')}")
            except Exception as e:
                print(f"[ERROR] Failed to write raw MMD for seed {seed}: {e}")
                traceback.print_exc()

            # Concise info: print only bootstrap seed and key metrics (MMD and NTK)
            mmd_val = davinz_result.get('mmd') or davinz_result.get('mmd_value') or davinz_result.get('mmd_score')
            ntk_val = davinz_result.get('ntk') or davinz_result.get('ntk_score') or davinz_result.get('ntk_value')
            print(f"[INFO] davinz seed {seed} - MMD: {mmd_val}, NTK: {ntk_val}")

        # Save davinz results (existing helper will compute global kappa and write CSV)
        DAVINZ.save_davinz_results(results, args.dataset, output_dir)

        # Also write raw-MMD per-bootstrap CSV for later analysis
        try:
            raw_csv = output_dir / 'davinz' / 'mmd_raw_Features.csv'
            raw_csv.parent.mkdir(parents=True, exist_ok=True)
            header = ['seed', 'bootstrap_size', 'mmd_raw', 'mmd_raw_time_s', 'mmd_raw_mem_bytes', 'mmd_raw_feature_dim', 'timestamp']
            write_header = not raw_csv.exists() or raw_csv.stat().st_size == 0
            with open(raw_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                for r in results:
                    writer.writerow([
                        float(r.get('seed') or 0),
                        int(r.get('bootstrap_size') or 0),
                        float(r.get('mmd_raw') or 0.0),
                        float(r.get('mmd_raw_time') or 0.0),
                        int(r.get('mmd_raw_mem') or 0),
                        int(r.get('mmd_raw_feature_dim') or 0),
                        time.time()
                    ])
            print(f"[INFO] Saved raw MMD CSV to {raw_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to write aggregated raw MMD CSV: {e}")
            traceback.print_exc()

        # Also write an aggregated standardized CSV under outputs/davinz
        davinz_out = output_dir / 'davinz'
        davinz_out.mkdir(parents=True, exist_ok=True)
        dav_csv = davinz_out / f"davinz_results_{args.dataset.lower()}.csv"
        import csv as _csv
        # compute mean mmd and mean ntk for kappa
        mmds = [r.get('mmd') for r in results if r.get('mmd') is not None]
        ntks = [r.get('ntk') for r in results if r.get('ntk') is not None and r.get('ntk') != 0]
        mean_mmd = float(np.mean(mmds)) if mmds else 0.0
        mean_ntk = float(np.mean(ntks)) if ntks else 0.0
        kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0

        # Flatten timing/memory and expand label samples into separate columns.
        # Determine max number of sample labels across results to build columns.
        max_samples = 0
        for rr in results:
            s = rr.get('bootstrap_labels_sample') if isinstance(rr, dict) else None
            if s is None:
                continue
            try:
                ln = len(s)
            except Exception:
                ln = 0
            if ln > max_samples:
                max_samples = ln

        # Base columns start with mmd, ntk, davinz_score as requested
        sample_cols = [f'bootstrap_labels_sample_{i}' for i in range(max_samples)]
        headers = [
            'mmd', 'ntk', 'davinz_score',
            'mmd_time_s', 'ntk_time_s', 'total_time_s',
            'mmd_mem_bytes', 'ntk_mem_bytes', 'total_mem_bytes',
            'mmd_raw', 'mmd_raw_time_s', 'mmd_raw_mem_bytes',
            'mmd_feature_dim', 'mmd_raw_feature_dim',
            'n_batch', 'n_permute', 'status', 'rep', 'seed', 'error',
            'bootstrap_size', 'dataset', 'bootstrap_labels_hash', 'maxsamples_used', 'val_size', 'bootstrap_labels_unique'
        ] + sample_cols + ['timestamp']

        with open(dav_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                timing = r.get('timing', {}) if isinstance(r, dict) else {}
                mem = r.get('mem', {}) if isinstance(r, dict) else {}
                this_mmd = r.get('mmd', r.get('mmd_value', np.nan))
                this_ntk = r.get('ntk', r.get('ntk_value', np.nan))
                davinz_score = -(kappa * this_ntk + this_mmd) if (this_ntk is not None and this_mmd is not None) else r.get('davinz_score', np.nan)

                # expand sample list
                samples = r.get('bootstrap_labels_sample') if isinstance(r, dict) else []
                if samples is None:
                    samples = []

                sample_vals = {}
                for i in range(max_samples):
                    sample_vals[f'bootstrap_labels_sample_{i}'] = samples[i] if i < len(samples) else ''

                row = {
                    'mmd': this_mmd,
                    'ntk': this_ntk,
                    'davinz_score': davinz_score,
                    'mmd_time_s': timing.get('mmd_time', np.nan),
                    'ntk_time_s': timing.get('ntk_time', np.nan),
                    'total_time_s': timing.get('total', np.nan) if timing.get('total', None) is not None else (timing.get('mmd_time', 0) + timing.get('ntk_time', 0)),
                    'mmd_mem_bytes': mem.get('mmd', np.nan),
                    'ntk_mem_bytes': mem.get('ntk', np.nan),
                    'total_mem_bytes': mem.get('total', np.nan),
                    'mmd_raw': r.get('mmd_raw', np.nan),
                    'mmd_raw_time_s': r.get('mmd_raw_time', np.nan),
                    'mmd_raw_mem_bytes': r.get('mmd_raw_mem', np.nan),
                    'mmd_feature_dim': int(r.get('mmd_feature_dim') or 0),
                    'mmd_raw_feature_dim': int(r.get('mmd_raw_feature_dim') or 0),
                    'n_batch': r.get('n_batch', r.get('n_batch_used', np.nan)),
                    'n_permute': r.get('n_permute', r.get('n_permute_used', np.nan)),
                    'status': r.get('status', 'ok'),
                    'rep': r.get('rep', ''),
                    'seed': r.get('seed'),
                    'error': r.get('error', ''),
                    'bootstrap_size': r.get('bootstrap_size'),
                    'dataset': r.get('dataset', args.dataset),
                    'bootstrap_labels_hash': r.get('bootstrap_labels_hash', ''),
                    'maxsamples_used': r.get('maxsamples_used', ''),
                    'val_size': int(r.get('val_size', timing.get('val_size', 0))) if r.get('val_size', None) is not None else None,
                    'bootstrap_labels_unique': r.get('bootstrap_labels_unique', ''),
                    'timestamp': r.get('timestamp', time.time())
                }
                row.update(sample_vals)
                writer.writerow(row)

        print(f"[INFO] DaVinz aggregated CSV written to {dav_csv}")
        print(f"\n[INFO] davinz computation completed for {len(results)} bootstraps")
        return

    # ===============================
    # NTK TUNING
    # ===============================
    if args.tune_ntk:
        print("\n" + "="*60)
        print("RUNNING NTK TUNING")
        print("="*60)

        tuning_out = output_dir / 'davinz_tuning'
        tuning_out.mkdir(parents=True, exist_ok=True)

        # Set and display GPU for tuning based on --gpu
        try:
            requested_gpu = str(args.gpu).split(',')[0]
            print(f"[NTK TUNE] Requested GPU(s) via --gpu: {args.gpu}")
            if torch.cuda.is_available():
                # Try to set CUDA device to the requested GPU index
                try:
                    torch.cuda.set_device(int(requested_gpu))
                except Exception:
                    # If setting by global index fails, fall back to using CUDA_VISIBLE_DEVICES mapping
                    pass
                cur_dev = torch.cuda.current_device()
                try:
                    dev_name = torch.cuda.get_device_name(cur_dev)
                except Exception:
                    dev_name = 'Unknown CUDA device'
                device = torch.device('cuda')
                print(f"[NTK TUNE] Using torch cuda device index {cur_dev}: {dev_name}")
            else:
                device = torch.device('cpu')
                print("[NTK TUNE] CUDA not available, using CPU for tuning")
        except Exception as e:
            print(f"[NTK TUNE] Warning: failed to set/display GPU info: {e}")

        # Grid for n_batch tuning
        batch_values = [20, 50, 100, 250, 500, 1000]

        batch_csv = tuning_out / f"batch_tuning_{args.dataset.lower()}.csv"
        import csv as _csv
        # Write header
        with open(batch_csv, 'w', newline='') as bf:
            writer = _csv.DictWriter(bf, fieldnames=[
                'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
            ])
            writer.writeheader()

        # Use val loader once
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        for n_batch in batch_values:
            print(f"\n[NTK TUNE] Testing n_batch={n_batch}")
            results_rows = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_labels, batch_size=128
            ):
                print(f"  Seed {seed} (size {size})...")
                # Collect tensors (use DAVINZ helper)
                bx, by = DAVINZ._collect_inputs_labels(bootstrap_loader, device, max_samples=10000)
                vx, _ = DAVINZ._collect_inputs_labels(val_loader, device, max_samples=10000)

                # compute MMD once
                mmd_start = time.time()
                sigma = DAVINZ._estimate_sigma(bx[: min(1000, bx.size(0))], vx[: min(1000, vx.size(0))])
                mmd_squared = rbf_mmd2(bx.reshape(bx.size(0), -1), vx.reshape(vx.size(0), -1), sigma=sigma)
                mmd = float(torch.sqrt(mmd_squared).item())
                mmd_time = time.time() - mmd_start

                # Prepare memory tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                ntk_start = time.time()
                if args.dataset == 'CIFAR_10':
                    ntk_model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10).to(device)
                    ntk_model_source = 'constructed_cifar10_resnet_basic'
                elif args.dataset == 'TINYIMAGENET_100':
                    ntk_model = ResNet(Bottleneck, [3,4,6,3], in_channels=3, num_classes=num_classes).to(device)
                    ntk_model_source = 'constructed_tinyimagenet_resnet50'
                else:
                    ntk_model = CNN(in_channels=1, num_classes=num_classes).to(device)
                    ntk_model_source = 'constructed_cnn'

                # Debug: report model used for NTK
                try:
                    param_count = sum(p.numel() for p in ntk_model.parameters())
                except Exception:
                    param_count = None
                try:
                    conv1_shape = tuple(ntk_model.conv1.weight.shape) if hasattr(ntk_model, 'conv1') and hasattr(ntk_model.conv1, 'weight') else None
                except Exception:
                    conv1_shape = None
                ah_loaded = False
                try:
                    import sys
                    ah_loaded = 'autograd_hacks' in sys.modules
                except Exception:
                    try:
                        ah_loaded = 'autograd_hacks' in globals()
                    except Exception:
                        ah_loaded = False
                print(f"[DEBUG-NTK] Using model for NTK: source={ntk_model_source}, class={ntk_model.__class__.__name__}, params={param_count}, conv1_shape={conv1_shape}, autograd_hacks_loaded={ah_loaded}")

                ntk_val, _ = compute_ntk_score_batched(
                    ntk_model,
                    bx, by.long(), mode='cls', n_batch=n_batch, use_hack=True, diagonal_I_mag=1e-6)
                ntk_time = time.time() - ntk_start

                if torch.cuda.is_available():
                    ntk_mem = torch.cuda.max_memory_allocated()
                else:
                    ntk_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem_before

                row = {
                    'n_batch': n_batch,
                    'seed': seed,
                    'bootstrap_size': size,
                    'mmd': mmd,
                    'mmd_time': mmd_time,
                    'ntk': float(ntk_val),
                    'ntk_time': ntk_time,
                    'ntk_mem_bytes': int(ntk_mem),
                    'val_size': int(vx.size(0)),
                    'dataset': args.dataset
                }
                results_rows.append(row)

            # compute kappa for this n_batch and append davinz score
            mmds = [r['mmd'] for r in results_rows]
            ntks = [r['ntk'] for r in results_rows if r['ntk'] not in (0, None)]
            mean_mmd = float(np.mean(mmds)) if mmds else 0.0
            mean_ntk = float(np.mean(ntks)) if ntks else 0.0
            kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0

            # Write rows with davinz
            with open(batch_csv, 'a', newline='') as bf:
                writer = _csv.DictWriter(bf, fieldnames=[
                    'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                    'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
                ])
                for r in results_rows:
                    writer.writerow(r)

            print(f"[NTK TUNE] Completed n_batch={n_batch}: mean_mmd={mean_mmd:.6f}, mean_ntk={mean_ntk:.6f}, kappa={kappa:.6f}")

        # Permute tuning: fix n_batch=100
        permute_values = [1, 2, 5, 10, 20, 50, 100,500]
        fixed_batch = 100
        permute_csv = tuning_out / f"permute_tuning_{args.dataset.lower()}.csv"
        with open(permute_csv, 'w', newline='') as pf:
            writer = _csv.DictWriter(pf, fieldnames=[
                'n_permute', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
            ])
            writer.writeheader()

        for n_permute in permute_values:
            print(f"\n[NTK TUNE] Testing n_permute={n_permute} (n_batch={fixed_batch})")
            results_rows = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_labels, batch_size=128
            ):
                bx, by = DAVINZ._collect_inputs_labels(bootstrap_loader, device, max_samples=10000)
                vx, _ = DAVINZ._collect_inputs_labels(val_loader, device, max_samples=10000)

                mmd_start = time.time()
                sigma = DAVINZ._estimate_sigma(bx[: min(1000, bx.size(0))], vx[: min(1000, vx.size(0))])
                mmd_squared = rbf_mmd2(bx.reshape(bx.size(0), -1), vx.reshape(vx.size(0), -1), sigma=sigma)
                mmd = float(torch.sqrt(mmd_squared).item())
                mmd_time = time.time() - mmd_start

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                ntk_start = time.time()
                if args.dataset == 'CIFAR_10':
                    ntk_model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10).to(device)
                    ntk_model_source = 'constructed_cifar10_resnet_basic'
                elif args.dataset == 'TINYIMAGENET_100':
                    ntk_model = ResNet(Bottleneck, [3,4,6,3], in_channels=3, num_classes=num_classes).to(device)
                    ntk_model_source = 'constructed_tinyimagenet_resnet50'
                else:
                    ntk_model = CNN(in_channels=1, num_classes=num_classes).to(device)
                    ntk_model_source = 'constructed_cnn'

                # Debug: report model used for NTK
                try:
                    param_count = sum(p.numel() for p in ntk_model.parameters())
                except Exception:
                    param_count = None
                try:
                    conv1_shape = tuple(ntk_model.conv1.weight.shape) if hasattr(ntk_model, 'conv1') and hasattr(ntk_model.conv1, 'weight') else None
                except Exception:
                    conv1_shape = None
                ah_loaded = False
                try:
                    import sys
                    ah_loaded = 'autograd_hacks' in sys.modules
                except Exception:
                    try:
                        ah_loaded = 'autograd_hacks' in globals()
                    except Exception:
                        ah_loaded = False
                print(f"[DEBUG-NTK] Using model for NTK (permute): source={ntk_model_source}, class={ntk_model.__class__.__name__}, params={param_count}, conv1_shape={conv1_shape}, autograd_hacks_loaded={ah_loaded}")

                ntk_val, _ = compute_ntk_score_batched_permute(
                    ntk_model,
                    bx, by.long(), mode='cls', n_batch=fixed_batch, n_permute=n_permute, use_hack=True, diagonal_I_mag=1e-6)
                ntk_time = time.time() - ntk_start

                if torch.cuda.is_available():
                    ntk_mem = torch.cuda.max_memory_allocated()
                else:
                    ntk_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem_before

                row = {
                    'n_permute': n_permute,
                    'seed': seed,
                    'bootstrap_size': size,
                    'mmd': mmd,
                    'mmd_time': mmd_time,
                    'ntk': float(ntk_val),
                    'ntk_time': ntk_time,
                    'ntk_mem_bytes': int(ntk_mem),
                    'val_size': int(vx.size(0)),
                    'dataset': args.dataset
                }
                results_rows.append(row)

            # compute kappa and write rows
            mmds = [r['mmd'] for r in results_rows]
            ntks = [r['ntk'] for r in results_rows if r['ntk'] not in (0, None)]
            mean_mmd = float(np.mean(mmds)) if mmds else 0.0
            mean_ntk = float(np.mean(ntks)) if ntks else 0.0
            kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0

            with open(permute_csv, 'a', newline='') as pf:
                writer = _csv.DictWriter(pf, fieldnames=[
                    'n_permute', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                    'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
                ])
                for r in results_rows:
                    writer.writerow(r)

            print(f"[NTK TUNE] Completed n_permute={n_permute}: mean_mmd={mean_mmd:.6f}, mean_ntk={mean_ntk:.6f}, kappa={kappa:.6f}")

        print(f"\n[NTK TUNING] Results saved in {tuning_out}")
        return

    # ===============================
    # Train base/feature extractor models
    # ===============================
    if args.train_base or args.train_valid:
        # Hyperparameters
        lr = 0.01
        momentum = 0.9
        weight_decay = 5e-4
        num_epochs = 300 if args.dataset in ('CIFAR_10', 'CIFAR_100', 'TINYIMAGENET_100') else 200
        batch_size = 128

        # Create model instance
        if args.dataset == 'CIFAR_10':
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes).to(device)
        elif args.dataset in ('CIFAR_100', 'TINYIMAGENET_100'):
            model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes).to(device)
        elif args.dataset == 'MNIST':
            model = CNN(in_channels=1, num_classes=num_classes).to(device)

        # DataLoaders
        loaders = {
            'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
            'val':   DataLoader(val_data,   batch_size=batch_size, shuffle=False),
            'test':  DataLoader(test_data,  batch_size=batch_size, shuffle=False)
        }

        # Train models
        if args.train_base:
            train_base_model(args, model, loaders, train_data, val_data, test_data,
                            lr, momentum, weight_decay, num_epochs, batch_size, device, output_dir)

        if args.train_valid:
            if args.dataset == 'CIFAR_10':
                model_class_fe = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes)
            elif args.dataset in ('CIFAR_100', 'TINYIMAGENET_100'):
                model_class_fe = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3, num_classes=num_classes)
            elif args.dataset == 'MNIST':
                model_class_fe = CNN(in_channels=1, num_classes=num_classes)

            train_valid_model(args, model_class_fe, train_data, val_data, test_data,
                             lr, momentum, weight_decay, num_epochs, batch_size, device, output_dir)

    print("\n[INFO] All operations completed.")


if __name__ == '__main__':
    main()