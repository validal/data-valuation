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
from model.tabular_mlp import TabularMLP
from utils import load_dataset_cls, train, test

# Import baseline modules (needed for compute flags)
from baselines import OT, RV, DAVINZ
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
        min_size = 1000
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
        if torch.is_tensor(train_labels):
            all_labels = train_labels.cpu().numpy()
        else:
            all_labels = np.array(train_labels)
        bootstrap_labels = all_labels[actual_indices]
        unique, counts = np.unique(bootstrap_labels, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        
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

    if not bootstrap_dir.exists():
        raise FileNotFoundError(f"Bootstrap directory not found: {bootstrap_dir}")

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

        # Debug: inspect first batch of bootstrap_loader
        try:
            b0 = next(iter(bootstrap_loader))
            if isinstance(b0, (list, tuple)) and len(b0) >= 2:
                xb0, yb0 = b0[0], b0[1]
                print(f"[DEBUG-BOOT-{seed}] first batch x.shape={tuple(xb0.shape)}, y.shape={tuple(yb0.shape)}, y_unique={torch.unique(yb0).tolist()}")
            else:
                xb0 = b0
                print(f"[DEBUG-BOOT-{seed}] first batch x.shape={tuple(xb0.shape)}")
        except Exception as e:
            print(f"[DEBUG-BOOT-{seed}] failed to inspect batch: {e}")
        yield seed, size, indices, bootstrap_loader


def train_bootstrap_model(args, model, bootstrap_indices, full_train_data, val_data, test_data,
                         lr, momentum, weight_decay, num_epochs, batch_size, device, 
                         output_dir, bootstrap_seed):
    """Train model on bootstrap sample, evaluate on validation and test sets."""
    print(f"\n{'='*60}")
    print(f"[INFO] TRAINING BOOTSTRAP MODEL")
    print(f"[INFO] Bootstrap Seed: {bootstrap_seed}")
    print(f"[INFO] Train on: bootstrap set ({len(bootstrap_indices)} samples)")
    print(f"[INFO] Evaluate on: validation set and test set")
    print(f"{'='*60}\n")
    
    # Move model to device
    model = model.to(device)
    
    # Choose optimizer depending on dataset
    if args.dataset in ['ADULT', 'HIGGS']:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Create bootstrap dataset
    bootstrap_data = Subset(full_train_data, bootstrap_indices)
    
    # DataLoaders
    loaders = {
        'train': DataLoader(bootstrap_data, batch_size=batch_size, shuffle=True, drop_last=True),
        'val':   DataLoader(val_data, batch_size=batch_size, shuffle=False) if len(val_data) > 0 else None,
        'test':  DataLoader(test_data, batch_size=batch_size, shuffle=False)
    }
    
    # Setup CSV logging
    train_bs_dir = output_dir / 'train_bootstraps'
    train_bs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = train_bs_dir / f"{args.dataset.lower()}_bootstrap_seed{bootstrap_seed}_training_log.csv"
    
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'elapsed_time'])
    
    print(f"[INFO] Training {model.__class__.__name__} for {num_epochs} epochs (LR={lr})...")
    print(f"[INFO] Training log: {csv_path}")
    start_time = time.time()
    
    # Train on bootstrap
    train(model, loaders, loss_fn, optimizer, device, num_epochs=num_epochs)
    
    total_time = time.time() - start_time
    print(f"[INFO] Bootstrap model training finished in {total_time:.2f} s")
    
    # Evaluate on validation and test sets
    model.eval()
    val_acc, val_loss = 0.0, 0.0
    if loaders['val'] is not None:
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
# Main Function
# ===============================
def main():
    parser = argparse.ArgumentParser(description='Bootstrap Correlation Experiment for Tabular Datasets (HIGGS/ADULT)')
    
    # Hardware and basic settings
    parser.add_argument('--gpu', type=str, default='0', help='GPU device index')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, required=True, choices=['HIGGS', 'ADULT', 'COVTYPE'],
                        help='Dataset to use (tabular)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save logs and metrics')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of the training set to use for validation (stratified)')
    
    # Operation flags
    parser.add_argument('--generate_bootstraps', action='store_true', 
                        help='Generate bootstrap samples')
    parser.add_argument('--train_bootstrap', action='store_true', 
                        help='Train models on bootstraps')
    parser.add_argument('--compute_ot', action='store_true', help='Compute OT distances')
    parser.add_argument('--compute_davinz', action='store_true', help='Compute DaVinz baseline metrics')
    parser.add_argument('--compute_volume', action='store_true', help='Compute RV / volume baseline metrics')
    parser.add_argument('--rv_tuning', action='store_true', help='Run RV omega/alpha/max_samples grid tuning')
    parser.add_argument('--rv_repeats', type=int, default=5, help='Number of independent repeats per RV config')
    parser.add_argument('--tune_ntk', action='store_true', help='Run NTK tuning (n_batch and n_permute)')
    # DaVinz batch tuning
    parser.add_argument('--davinz_batch_tuning', action='store_true', help='Run DaVinz n_batch grid tuning')
    parser.add_argument('--davinz_batch_values', type=str, default='10,20,50,100,250,500,1000',
                        help='Comma-separated n_batch values to try for DaVinz')
    parser.add_argument('--davinz_batch_repeats', type=int, default=5, help='Number of independent seeds/repeats per n_batch')
    # Robustness testing (label/feature noise)
    parser.add_argument('--robustness', action='store_true', help='Run robustness noise sweep tests')
    parser.add_argument('--noise_levels', type=str, default='0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated noise levels for robustness tests')
    parser.add_argument('--robustness_seeds', type=str, default='0', help='Seed or range for robustness runs (e.g. 0-2)')
    parser.add_argument('--base_size', type=int, default=1000, help='Base party size for robustness scenarios')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max samples to use in baselines during robustness')
    parser.add_argument('--val_seed', type=int, default=0, help='Seed used for fixed validation split during robustness')
    
    # Bootstrap parameters
    parser.add_argument('--bootstrap_seeds', type=str, default='0-5',
                        help='Bootstrap seed range (e.g., "0-10" or "0,1,2,3")')
    parser.add_argument('--bootstrap_size', type=int, default=10000,
                        help='Maximum size of each bootstrap sample')
    # OT / baseline params (optional)
    parser.add_argument('--feature_extractor_path', type=str, default=None,
                        help='Path to feature extractor (if needed by baselines)')
    parser.add_argument('--lambda_x', type=float, default=1.0, help='OT lambda_x parameter')
    parser.add_argument('--lambda_y', type=float, default=1.0, help='OT lambda_y parameter')
    parser.add_argument('--entreg', type=float, default=1e-1, help='OT entropic regularization')
    parser.add_argument('--ot_repeats', type=int, default=1, help='Number of independent OT runs per bootstrap seed (each saved in its own rep subfolder)')
    parser.add_argument('--n_batch', type=int, default=1, help='DaVinz NTK n_batch parameter')
    parser.add_argument('--n_permute', type=int, default=1, help='DaVinz NTK n_permute parameter')
    
    args = parser.parse_args()
    
    # Default: show help if no flags
    operations = [args.generate_bootstraps, args.train_bootstrap,
                  args.compute_ot, args.compute_davinz, args.compute_volume,
                  args.robustness, args.tune_ntk, args.davinz_batch_tuning,
                  args.rv_tuning]
    if not any(operations):
        parser.print_help()
        return

    # ===============================
    # Environment setup
    # ===============================
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    gpu_list = [int(x) for x in str(args.gpu).split(',') if x.strip() != ''] if args.gpu else []
    print(f"[INFO] Requested GPU(s): {gpu_list}")
    
    device = torch.device('cpu')
    if torch.cuda.is_available() and gpu_list:
        try:
            torch.cuda.set_device(int(gpu_list[0]))
            device = torch.device(f'cuda:{int(gpu_list[0])}')
        except Exception as e:
            print(f"[WARN] Failed to set CUDA device: {e}")
            device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # Create dataset-specific subfolders
    dataset_out = output_dir / args.dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / 'bootstraps').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'train_bootstraps').mkdir(parents=True, exist_ok=True)

    # Use dataset-specific output dir for subsequent writes
    output_dir = dataset_out

    # ===============================
    # Data loading for Tabular dataset
    # ===============================
    print(f"[INFO] Loading {args.dataset} dataset...")
    train_inputs, train_labels, test_inputs_tensor, test_labels, dims, _ = load_dataset_cls(
        args.dataset,
        trim_dataset=None,  # Use full dataset
        num_parties=10
    )

    # Convert to tensors
    if not torch.is_tensor(train_inputs):
        train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    if not torch.is_tensor(test_inputs_tensor):
        test_inputs_tensor = torch.tensor(test_inputs_tensor, dtype=torch.float32)
    if not torch.is_tensor(test_labels):
        test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Ensure correct shapes
    if len(train_inputs.shape) == 1:
        train_inputs = train_inputs.unsqueeze(1)
    if len(test_inputs_tensor.shape) == 1:
        test_inputs_tensor = test_inputs_tensor.unsqueeze(1)

    print(f"[INFO] Train inputs shape: {train_inputs.shape}")
    print(f"[INFO] Train labels shape: {train_labels.shape}")
    print(f"[INFO] Test inputs shape: {test_inputs_tensor.shape}")
    print(f"[INFO] Test labels shape: {test_labels.shape}")

    # Create global stratified splits to ensure test and validation sizes when dataset is large
    from sklearn.model_selection import train_test_split

    desired_test = 10000
    desired_val = 5000

    try:
        # Work with numpy arrays for sklearn
        X_all = np.concatenate([train_inputs.numpy(), test_inputs_tensor.numpy()], axis=0)
        y_all = np.concatenate([train_labels.numpy(), test_labels.numpy()], axis=0)
        total_n = len(y_all)

        if total_n >= (desired_test + desired_val + 1):
            # Split off the test set first (absolute number)
            X_rem, X_test_split, y_rem, y_test_split = train_test_split(
                X_all, y_all, test_size=desired_test, random_state=args.seed, stratify=y_all
            )

            # Then split remaining into train and validation (absolute number)
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_rem, y_rem, test_size=desired_val, random_state=args.seed, stratify=y_rem
            )

            # Convert back to tensors
            train_inputs = torch.tensor(X_train_split, dtype=torch.float32)
            train_labels = torch.tensor(y_train_split, dtype=torch.long)
            val_inputs = torch.tensor(X_val_split, dtype=torch.float32)
            val_labels = torch.tensor(y_val_split, dtype=torch.long)
            test_inputs_tensor = torch.tensor(X_test_split, dtype=torch.float32)
            test_labels = torch.tensor(y_test_split, dtype=torch.long)

            # Create TensorDatasets
            train_data = TensorDataset(train_inputs, train_labels)
            val_data = TensorDataset(val_inputs, val_labels)
            test_data = TensorDataset(test_inputs_tensor, test_labels)

            print(f"[INFO] {args.dataset} global split applied: total={total_n}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        else:
            # Not enough samples for fixed splits; fall back to splitting training set by fraction
            print(f"[WARN] Not enough samples ({total_n}) for desired fixed splits; falling back to train/val fraction split ({args.val_frac}).")
            total_indices = np.arange(len(train_inputs))
            try:
                train_idx, val_idx = train_test_split(
                    total_indices,
                    test_size=args.val_frac,
                    random_state=args.seed,
                    stratify=train_labels.numpy()
                )
            except Exception:
                train_idx, val_idx = train_test_split(
                    total_indices,
                    test_size=args.val_frac,
                    random_state=args.seed
                )

            train_data = Subset(TensorDataset(train_inputs, train_labels), train_idx)
            val_data = Subset(TensorDataset(train_inputs, train_labels), val_idx)
            test_data = TensorDataset(test_inputs_tensor, test_labels)

            print(f"[INFO] Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    except Exception as e:
        print(f"[WARN] Failed to perform global splits: {e}. Falling back to fraction split on training set.")
        total_indices = np.arange(len(train_inputs))
        try:
            train_idx, val_idx = train_test_split(
                total_indices,
                test_size=args.val_frac,
                random_state=args.seed,
                stratify=train_labels.numpy()
            )
        except Exception:
            train_idx, val_idx = train_test_split(
                total_indices,
                test_size=args.val_frac,
                random_state=args.seed
            )

        train_data = Subset(TensorDataset(train_inputs, train_labels), train_idx)
        val_data = Subset(TensorDataset(train_inputs, train_labels), val_idx)
        test_data = TensorDataset(test_inputs_tensor, test_labels)

        print(f"[INFO] Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    print(f"[INFO] Feature dimension: {train_inputs.shape[1]}")

    # Model factory for tabular datasets
    input_dim = train_inputs.shape[1]
    num_classes = len(torch.unique(train_labels))
    
    def create_model():
        if args.dataset == 'HIGGS' or args.dataset == 'COVTYPE':
            hidden = (512, 256)
            dropout = 0.2
        else:
            hidden = (256, 128)
            dropout = 0.3
        return TabularMLP(
            input_dim=input_dim,
            hidden_dims=hidden,
            num_classes=num_classes,
            dropout=dropout
        )

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
    # OT / Baseline computations
    # ===============================
    if args.compute_ot or args.compute_davinz or args.compute_volume:
        # Prepare val loader
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    if args.compute_ot:
        print("\n" + "="*60)
        print("COMPUTING OT DISTANCES")
        print("="*60)

        lx_s = str(args.lambda_x).replace('.', 'p')
        ly_s = str(args.lambda_y).replace('.', 'p')
        ot_base = output_dir / f"OT_lx{lx_s}_ly{ly_s}"
        ot_base.mkdir(parents=True, exist_ok=True)

        n_ot_repeats = max(1, int(args.ot_repeats))

        for rep in range(1, n_ot_repeats + 1):
            print(f"\n[OT] === Repeat {rep}/{n_ot_repeats} ===")
            rep_dir = ot_base / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            # reseed for this repeat
            set_seed(args.seed + rep * 1000)

            results = []
            for seed, size, indices, _ in iter_bootstraps(
                args, output_dir, train_inputs, train_labels, batch_size=128
            ):
                print(f"[INFO] OT rep={rep} - bootstrap seed {seed} (size={size})")
                try:
                    # Build sanitized bootstrap loader
                    try:
                        boot_X = train_inputs[indices].float()
                        boot_y = train_labels[indices].long()
                        if boot_y.dim() == 0:
                            boot_y = boot_y.unsqueeze(0)
                        try:
                            uniq, counts = torch.unique(boot_y, return_counts=True)
                            counts = counts.cpu().numpy()
                            max_count = int(counts.max())
                            min_count = int(counts.min())
                            imbalance_ratio = float(max_count) / max(1, min_count)
                        except Exception:
                            uniq = None; counts = None; imbalance_ratio = 1.0
                        per_class_cap = 5000; total_cap = 10000
                        if counts is not None and (max_count > per_class_cap or imbalance_ratio > 10 or len(boot_y) > total_cap):
                            boot_indices_array = np.arange(len(boot_y))
                            selected_idx = []
                            rng = np.random.RandomState(seed)
                            for lbl in uniq.cpu().numpy().tolist():
                                lbl = int(lbl)
                                class_idx = boot_indices_array[(boot_y.cpu().numpy() == lbl)]
                                k = min(len(class_idx), per_class_cap, total_cap)
                                chosen = rng.choice(class_idx, size=k, replace=False) if len(class_idx) > k else class_idx
                                selected_idx.extend(chosen.tolist())
                            if len(selected_idx) > total_cap:
                                selected_idx = list(rng.choice(np.array(selected_idx), size=total_cap, replace=False))
                            selected_idx = np.array(selected_idx, dtype=int)
                            boot_X = boot_X[selected_idx]
                            boot_y = boot_y[selected_idx]
                        boot_loader_sanitized = DataLoader(TensorDataset(boot_X, boot_y), batch_size=128, shuffle=False)
                    except Exception as e:
                        print(f"[WARN] Failed to build sanitized bootstrap loader for seed {seed}: {e}")
                        boot_loader_sanitized = None

                    # Build sanitized val loader
                    try:
                        val_tensors = val_data.tensors
                        val_X, val_y = val_tensors[0].float(), val_tensors[1].long()
                        if val_y.dim() == 0:
                            val_y = val_y.unsqueeze(0)
                        if len(val_y) > 10000:
                            rngv = np.random.RandomState(args.seed + rep)
                            sel = rngv.choice(np.arange(len(val_y)), size=10000, replace=False)
                            val_X = val_X[sel]; val_y = val_y[sel]
                        val_loader_sanitized = DataLoader(TensorDataset(val_X, val_y), batch_size=128, shuffle=False)
                    except Exception:
                        val_loader_sanitized = None

                    def build_balanced_loader(Xtensor, ytensor, rng_seed, per_class=2000, total=5000):
                        labels_np = ytensor.cpu().numpy()
                        uniq_lbls = np.unique(labels_np)
                        rng2 = np.random.RandomState(rng_seed)
                        sel_idx = []
                        for l in uniq_lbls:
                            idxs = np.where(labels_np == l)[0]
                            k = min(len(idxs), per_class)
                            sel_idx.extend((rng2.choice(idxs, size=k, replace=False) if len(idxs) > k else idxs).tolist())
                        if len(sel_idx) > total:
                            sel_idx = list(rng2.choice(np.array(sel_idx), size=total, replace=False))
                        sel_idx = np.array(sel_idx, dtype=int)
                        return DataLoader(TensorDataset(Xtensor[sel_idx].float(), ytensor[sel_idx].long()), batch_size=128, shuffle=False)

                    try:
                        res = OT.compute_ot_distance(
                            boot_loader_sanitized if boot_loader_sanitized is not None else DataLoader(Subset(TensorDataset(train_inputs, train_labels), indices), batch_size=128),
                            val_loader_sanitized if val_loader_sanitized is not None else val_loader,
                            dataset=args.dataset,
                            feature_extractor_path=args.feature_extractor_path,
                            device=device,
                            lambda_x=args.lambda_x,
                            lambda_y=args.lambda_y,
                            entreg=args.entreg
                        )
                    except Exception as e_ot:
                        print(f"[WARN] OT failed seed={seed} rep={rep}: {e_ot}. Retrying with balanced downsample.")
                        traceback.print_exc()
                        try:
                            Xb = boot_loader_sanitized.dataset.tensors[0] if boot_loader_sanitized else train_inputs[indices]
                            yb = boot_loader_sanitized.dataset.tensors[1] if boot_loader_sanitized else train_labels[indices]
                            Xv = val_loader_sanitized.dataset.tensors[0] if val_loader_sanitized else val_data.tensors[0]
                            yv = val_loader_sanitized.dataset.tensors[1] if val_loader_sanitized else val_data.tensors[1]
                            res = OT.compute_ot_distance(
                                build_balanced_loader(Xb, yb, seed),
                                build_balanced_loader(Xv, yv, args.seed + rep),
                                dataset=args.dataset,
                                feature_extractor_path=args.feature_extractor_path,
                                device=device,
                                lambda_x=args.lambda_x,
                                lambda_y=args.lambda_y,
                                entreg=max(args.entreg, 1e-1)
                            )
                        except Exception as e_retry:
                            print(f"[ERROR] OT retry failed seed={seed} rep={rep}: {e_retry}")
                            traceback.print_exc()
                            res = {'error': str(e_retry)}

                    res['seed'] = seed
                    res['rep'] = rep
                    res['timestamp'] = time.time()
                    results.append(res)

                except Exception as e:
                    print(f"[ERROR] OT computation failed seed={seed} rep={rep}: {e}")
                    traceback.print_exc()

            # Save this rep's results into its own subfolder
            if results:
                OT.save_ot_results(results, args.dataset, rep_dir)
                OT.print_timing_summary(results)
                # Write flat CSV for this rep
                import csv as _csv
                rep_csv = rep_dir / f"ot_results_{args.dataset.lower()}_lx{lx_s}_ly{ly_s}_rep{rep}.csv"
                headers = ['seed', 'rep', 'bootstrap_size', 'ot_distance', 'dataset',
                           'feature_extraction_time_s', 'ot_time_s', 'total_time_s', 'timestamp']
                with open(rep_csv, 'w', newline='') as f:
                    writer = _csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for r in results:
                        timing = r.get('timing', {})
                        writer.writerow({
                            'seed': r.get('seed'),
                            'rep': r.get('rep', rep),
                            'bootstrap_size': r.get('bootstrap_size', ''),
                            'ot_distance': r.get('distance', r.get('ot_distance', '')),
                            'dataset': args.dataset,
                            'feature_extraction_time_s': timing.get('feature_extraction', ''),
                            'ot_time_s': timing.get('ot_computation', ''),
                            'total_time_s': timing.get('total', ''),
                            'timestamp': r.get('timestamp', '')
                        })
                print(f"[INFO] OT rep={rep} CSV saved to {rep_csv}")

        print(f"[INFO] OT tuning complete: {n_ot_repeats} repeats saved under {ot_base}")

    if args.compute_volume:
        print("\n" + "="*60)
        print("COMPUTING RV / VOLUME METRICS")
        print("="*60)
        results = []
        for seed, size, _, bootstrap_loader in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=128
        ):
            print(f"[INFO] RV - processing bootstrap seed {seed} (size={size})")
            try:
                res = RV.compute_rv_metric(
                    bootstrap_loader,
                    dataset=args.dataset,
                    device=device,
                    feature_extractor_path=args.feature_extractor_path,
                    max_samples=10000
                )
                res['seed'] = seed
                res['timestamp'] = time.time()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] RV computation failed for seed {seed}: {e}")
                traceback.print_exc()

        if results:
            RV.save_rv_results(results, args.dataset, output_dir)

    # ===============================
    # RV Tuning
    # ===============================
    if args.rv_tuning:
        print("\n" + "="*60)
        print("RUNNING RV TUNING (omega x alpha grid)")
        print("="*60)

        # Grid: omega x alpha_multiplier
        # alpha_multiplier is passed to compute_rv_metric as alpha.
        # Inside compute_robust_volume: alpha_internal = 1.0 / (alpha_multiplier * n)
        # So alpha1_1n   -> alpha_multiplier=1   -> alpha_internal = 1/n
        #    alpha1_10n  -> alpha_multiplier=10  -> alpha_internal = 1/(10n)
        #    alpha1_100n -> alpha_multiplier=100 -> alpha_internal = 1/(100n)
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
                # config name matches your grid: e.g. omega0p1_alpha1_10n
                config_name = f"omega{omega_str}_alpha1_{alpha_label}"
                config_dir = tuning_out / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[RV TUNE] Config: {config_name}  "
                      f"(omega={omega}, alpha_multiplier={alpha_multiplier} -> alpha_internal=1/({alpha_multiplier}*n))")

                for rep in range(1, n_rv_repeats + 1):
                    rep_dir = config_dir / f"rep_{rep}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    set_seed(args.seed + rep * 1000)

                    rep_csv = rep_dir / f"rv_results_{args.dataset.lower()}_{config_name}_rep{rep}.csv"
                    rep_header = ['seed', 'bootstrap_size', 'log_volume', 'log_robust_volume',
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
                                feature_extractor_path=args.feature_extractor_path,
                                max_samples=10000,
                                omega=omega,
                                alpha=alpha_multiplier   # passed as the multiplier; RV.py computes 1/(alpha*n)
                            )
                            timing = res.get('timing', {})
                            ts = time.time()
                            row = {
                                'seed': seed,
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
                            traceback.print_exc()

                    print(f"  [rep {rep}] done -> {rep_csv}")

        print(f"\n[INFO] RV tuning complete. Results under {tuning_out}")
        print(f"[INFO] Aggregated summary: {summary_csv}")

    if args.compute_davinz:
        print("\n" + "="*60)
        print("COMPUTING DaVinz METRICS")
        print("="*60)

        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        os.makedirs(args.save_dir, exist_ok=True)
        default_n_batch = 100
        _model_template = create_model()

        init_path = Path(args.save_dir) / f"{args.dataset.lower()}_tabularlmp_init_seed{args.seed}.pt"
        model = None
        if init_path.exists():
            try:
                model = torch.load(str(init_path)).to(device)
                print(f"[INFO] Loaded initial model from {init_path}")
            except Exception as e:
                print(f"[WARN] Failed to load initial model ({init_path}): {e}")
                model = _model_template.to(device)
                try:
                    torch.save(model, str(init_path))
                except Exception:
                    pass
        else:
            model = _model_template.to(device)
            try:
                torch.save(model, str(init_path))
                print(f"[INFO] Created and saved initial model at {init_path}")
            except Exception as e:
                print(f"[WARN] Failed to save initial model at {init_path}: {e}")

        n_batch = default_n_batch

        # If requested, run DaVinz batch-size tuning grid and exit
        if getattr(args, 'davinz_batch_tuning', False):
            print("\n[INFO] Running DaVinz batch-size tuning")
            batch_values = [int(x.strip()) for x in str(getattr(args, 'davinz_batch_values', '10,20,50,100,250,500,1000')).split(',') if x.strip()]
            repeats = int(getattr(args, 'davinz_batch_repeats', 5))

            tuning_out = output_dir / 'Davinz_batchtuning'
            tuning_out.mkdir(parents=True, exist_ok=True)

            seed_list = parse_seed_range(getattr(args, 'bootstrap_seeds', '0-0'))

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

                try:
                    set_seed(rep_model_seed)
                    model_rep = create_model().to(device)
                except Exception as e:
                    print(f"[ERROR] Failed to initialize repeat model (rep={rep}): {e}")
                    traceback.print_exc()
                    model_rep = None

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

                    try:
                        with open(nb_dir / 'model_info.json', 'w') as mf:
                            json.dump({'rep': rep, 'model_seed': rep_model_seed}, mf, indent=2)
                    except Exception:
                        pass

                    results_rows = []
                    skipped_config = False

                    for seed in seed_list:
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
                                diagonal_I_mag=1e-6,
                                n_batch=nb)

                            davinz_result['seed'] = seed
                            davinz_result['bootstrap_size'] = bsize
                            davinz_result['rep'] = rep
                            davinz_result['n_batch'] = nb
                            davinz_result['status'] = 'ok'
                            davinz_result['error'] = ''

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

            print(f"[INFO] DaVinz batch-size tuning completed.")
            return

        results = []
        for seed, size, _, bootstrap_loader in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=128
        ):
            print(f"[INFO] DaVinz - processing bootstrap seed {seed} (size={size})")
            try:
                # Debug: inspect first batch of bootstrap_loader
                try:
                    d0 = next(iter(bootstrap_loader))
                    if isinstance(d0, (list, tuple)) and len(d0) >= 2:
                        xd0, yd0 = d0[0], d0[1]
                        print(f"[DEBUG-DAV-{seed}] first batch x.shape={tuple(xd0.shape)}, y.shape={tuple(yd0.shape)}, y_unique={torch.unique(yd0).tolist()}")
                    else:
                        xd0 = d0
                        print(f"[DEBUG-DAV-{seed}] first batch x.shape={tuple(xd0.shape)}")
                except Exception as e:
                    print(f"[DEBUG-DAV-{seed}] failed to inspect batch: {e}")
                res = DAVINZ.compute_davinz(
                    bootstrap_loader,
                    val_loader,
                    dataset=args.dataset,
                    device=device,
                    max_samples=10000,
                    model=None,
                    n_batch=args.n_batch,
                    n_permute=args.n_permute,
                    feature_extractor_path=args.feature_extractor_path
                )
                res['seed'] = seed
                res['timestamp'] = time.time()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] DaVinz computation failed for seed {seed}: {e}")
                traceback.print_exc()

        if results:
            DAVINZ.save_davinz_results(results, args.dataset, output_dir)

    # ===============================
    # NTK TUNING
    # ===============================
    if args.tune_ntk:
        print("\n" + "="*60)
        print("RUNNING NTK TUNING")
        print("="*60)

        tuning_out = output_dir / 'davinz_tuning'
        tuning_out.mkdir(parents=True, exist_ok=True)

        try:
            requested_gpu = str(args.gpu).split(',')[0]
            print(f"[NTK TUNE] Requested GPU(s) via --gpu: {args.gpu}")
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(int(requested_gpu))
                except Exception:
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
        with open(batch_csv, 'w', newline='') as bf:
            writer = _csv.DictWriter(bf, fieldnames=[
                'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
            ])
            writer.writeheader()

        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        for n_batch in batch_values:
            print(f"\n[NTK TUNE] Testing n_batch={n_batch}")
            results_rows = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_labels, batch_size=128
            ):
                print(f"  Seed {seed} (size {size})...")
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
                ntk_model = create_model().to(device)
                ntk_model_source = f'constructed_tabularmlp_{args.dataset}'

                try:
                    param_count = sum(p.numel() for p in ntk_model.parameters())
                except Exception:
                    param_count = None
                print(f"[DEBUG-NTK] Using model for NTK: source={ntk_model_source}, class={ntk_model.__class__.__name__}, params={param_count}")

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

            mmds = [r['mmd'] for r in results_rows]
            ntks = [r['ntk'] for r in results_rows if r['ntk'] not in (0, None)]
            mean_mmd = float(np.mean(mmds)) if mmds else 0.0
            mean_ntk = float(np.mean(ntks)) if ntks else 0.0
            kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0

            with open(batch_csv, 'a', newline='') as bf:
                writer = _csv.DictWriter(bf, fieldnames=[
                    'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                    'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
                ])
                for r in results_rows:
                    writer.writerow(r)

            print(f"[NTK TUNE] Completed n_batch={n_batch}: mean_mmd={mean_mmd:.6f}, mean_ntk={mean_ntk:.6f}, kappa={kappa:.6f}")

        # Permute tuning: fix n_batch=100
        permute_values = [1, 2, 5, 10, 20, 50, 100, 500]
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
                ntk_model = create_model().to(device)
                ntk_model_source = f'constructed_tabularmlp_{args.dataset}'

                try:
                    param_count = sum(p.numel() for p in ntk_model.parameters())
                except Exception:
                    param_count = None
                print(f"[DEBUG-NTK] Using model for NTK (permute): source={ntk_model_source}, class={ntk_model.__class__.__name__}, params={param_count}")

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
    # Bootstrap training
    # ===============================
    if args.train_bootstrap:
        # Hyperparameters for tabular datasets
        batch_size = 128
        lr = 0.001
        momentum = 0.0  # Not used with Adam
        weight_decay = 1e-5
        num_epochs = 100

        bootstrap_results = []
        
        for bootstrap_seed, size, bootstrap_indices, _ in iter_bootstraps(
            args, output_dir, train_inputs, train_labels, batch_size=batch_size
        ):
            print(f"\n[INFO] Processing bootstrap seed {bootstrap_seed} (size: {size})")
            
            # Create fresh model for this bootstrap
            model = create_model()
            
            # Train model on this bootstrap
            model, checkpoint_path, metrics = train_bootstrap_model(
                args, model, bootstrap_indices,
                TensorDataset(train_inputs, train_labels), val_data, test_data,
                lr, momentum, weight_decay, num_epochs, batch_size, device,
                output_dir, bootstrap_seed
            )
            
            # Add checkpoint path to metrics
            metrics['checkpoint_path'] = checkpoint_path
            bootstrap_results.append(metrics)

        # Write all results to a single CSV
        results_dir = output_dir / 'train_bootstraps'
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{args.dataset.lower()}_bootstrap_results.csv"
        
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
                    'val_accuracy': m.get('val_accuracy', 0.0),
                    'val_loss': m.get('val_loss', 0.0),
                    'test_accuracy': m.get('test_accuracy', 0.0),
                    'test_loss': m.get('test_loss', 0.0),
                    'epochs': m.get('epochs'),
                    'lr': m.get('lr'),
                    'model_arch': m.get('model_arch'),
                    'checkpoint_path': m.get('checkpoint_path', '')
                })
        
        print(f"\n[INFO] All bootstrap results saved to {csv_path}")
        
        # Print summary statistics
        test_accs = [m.get('test_accuracy', 0) for m in bootstrap_results]
        if test_accs:
            print(f"\n[INFO] Summary - Test Accuracy: mean={np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        print("[INFO] Bootstrap training completed.")
        return

    print("\n[INFO] No operations selected. Use --generate_bootstraps or --train_bootstrap")
    if args.robustness:
        print("\n" + "="*60)
        print("RUNNING ROBUSTNESS TESTS (LABEL/FEATURE NOISE)")
        print("="*60)

        # Parse noise levels
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]

        # Corruption functions (from robustness tester)
        def corrupt_labels(labels, noise_frac, seed):
            rng = np.random.RandomState(seed)
            labels = labels.clone()
            n = labels.size(0)
            k = int(round(noise_frac * n))
            if k == 0:
                return labels
            indices = rng.choice(n, size=k, replace=False)
            num_classes = len(torch.unique(labels))
            for i in indices:
                old = int(labels[i].item())
                choices = list(range(num_classes))
                choices.remove(old)
                labels[i] = random.choice(choices)
            return labels

        def corrupt_features(inputs, noise_frac, seed):
            rng = np.random.RandomState(seed)
            X = inputs.clone()
            N = X.size(0)
            flat = X.view(N, -1)
            D = flat.size(1)
            k = int(round(noise_frac * D))
            if k == 0:
                return X
            idxs = rng.choice(D, size=k, replace=False)
            feat_std = flat.std(dim=0)
            noise = torch.zeros_like(flat)
            for j in idxs:
                stdj = float(feat_std[j].item())
                if stdj <= 0:
                    continue
                noise[:, j] = torch.from_numpy(rng.normal(loc=0.0, scale=0.5 * stdj, size=(N,))).to(flat.device)
            flat = flat + noise
            return flat.view_as(X)

        # Validation loader
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        # Feature extractor path (if any)
        feat_path = args.feature_extractor_path

        # Iterate over robustness seeds and build bootstrap samples on-the-fly
        seeds = parse_seed_range(args.robustness_seeds)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            # choose pool from train_data if it's a Subset (preserves stratified selection), otherwise use train_inputs
            if isinstance(train_data, Subset) and hasattr(train_data, 'indices'):
                pool = np.array(train_data.indices)
            else:
                pool = np.arange(len(train_inputs))

            replace = len(pool) < args.base_size
            sel = rng.choice(pool, size=args.base_size, replace=replace)
            indices = sel.astype(int)
            size = len(indices)

            print(f"\n[INFO] Robustness for bootstrap seed {seed} (size={size})")
            rob_seed_dir = output_dir / 'robustness' / f'seed_{seed}'
            rob_seed_dir.mkdir(parents=True, exist_ok=True)

            # Extract bootstrap data
            boot_inputs = train_inputs[indices]
            boot_labels = train_labels[indices]

            for scenario in ['label', 'feature']:
                scenario_dir = rob_seed_dir / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)

                def write_results(nl, ot_res, rv_res, dav_res):
                    ts = time.time()
                    # OT
                    ot_csv = scenario_dir / 'ot_results.csv'
                    if not ot_csv.exists():
                        with open(ot_csv, 'w') as f:
                            f.write('noise_level,size,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp\n')
                    ot_row = [nl, size,
                            ot_res.get('distance', 'nan'),
                            ot_res.get('timing', {}).get('feature_extraction', 'nan'),
                            ot_res.get('timing', {}).get('ot_computation', 'nan'),
                            ot_res.get('timing', {}).get('total', 'nan'),
                            ot_res.get('mem', {}).get('feature_extraction', 'nan'),
                            ot_res.get('mem', {}).get('ot', 'nan'),
                            ot_res.get('mem', {}).get('total', 'nan'),
                            ts]
                    with open(ot_csv, 'a') as f:
                        f.write(','.join(map(str, ot_row)) + '\n')
                    # RV
                    rv_csv = scenario_dir / 'rv_results.csv'
                    if not rv_csv.exists():
                        with open(rv_csv, 'w') as f:
                            f.write('noise_level,size,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp\n')
                    rv_row = [nl, size,
                            rv_res.get('log_volume', 'nan'),
                            rv_res.get('log_robust_volume', 'nan'),
                            rv_res.get('timing', {}).get('feature_extraction', 'nan'),
                            rv_res.get('timing', {}).get('rv_computation', 'nan'),
                            rv_res.get('timing', {}).get('total', 'nan'),
                            rv_res.get('mem', {}).get('feature_extraction', 'nan'),
                            rv_res.get('mem', {}).get('rv', 'nan'),
                            rv_res.get('mem', {}).get('total', 'nan'),
                            ts]
                    with open(rv_csv, 'a') as f:
                        f.write(','.join(map(str, rv_row)) + '\n')
                    # DaVinz
                    dav_csv = scenario_dir / 'davinz_results.csv'
                    if not dav_csv.exists():
                        with open(dav_csv, 'w') as f:
                            f.write('noise_level,size,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp\n')
                    dav_row = [nl, size,
                            dav_res.get('mmd', dav_res.get('mmd_raw', 'nan')),
                            dav_res.get('mmd_raw', 'nan'),
                            dav_res.get('ntk', 'nan'),
                            dav_res.get('davinz_score', 'nan'),
                            dav_res.get('timing', {}).get('mmd_time', 'nan'),
                            dav_res.get('timing', {}).get('ntk_time', 'nan'),
                            dav_res.get('timing', {}).get('total', 'nan'),
                            dav_res.get('mem', {}).get('mmd', 'nan'),
                            dav_res.get('mem', {}).get('ntk', 'nan'),
                            dav_res.get('mem', {}).get('total', 'nan'),
                            ts]
                    with open(dav_csv, 'a') as f:
                        f.write(','.join(map(str, dav_row)) + '\n')
                    # JSONL
                    ot_json = scenario_dir / 'ot_results.jsonl'
                    with open(ot_json, 'a') as f:
                        f.write(json.dumps({'noise_level': nl, 'size': size, 'scenario': scenario, 'result': ot_res, 'timestamp': ts}) + '\n')
                    rv_json = scenario_dir / 'rv_results.jsonl'
                    with open(rv_json, 'a') as f:
                        f.write(json.dumps({'noise_level': nl, 'size': size, 'scenario': scenario, 'result': rv_res, 'timestamp': ts}) + '\n')
                    dav_json = scenario_dir / 'davinz_results.jsonl'
                    with open(dav_json, 'a') as f:
                        f.write(json.dumps({'noise_level': nl, 'size': size, 'scenario': scenario, 'result': dav_res, 'timestamp': ts}) + '\n')

                for nl in noise_levels:
                    print(f"  [Robustness] {scenario} noise level {nl}")
                    corr_seed = hash((seed, scenario, nl)) % 2**32
                    if scenario == 'label':
                        corrupted_labels = corrupt_labels(boot_labels, nl, corr_seed)
                        corrupted_inputs = boot_inputs
                    else:  # feature
                        corrupted_inputs = corrupt_features(boot_inputs, nl, corr_seed)
                        corrupted_labels = boot_labels

                    corrupted_dataset = TensorDataset(corrupted_inputs, corrupted_labels)
                    if len(corrupted_dataset) > args.max_samples:
                        rng_sub = np.random.RandomState(corr_seed)
                        sub_idx = rng_sub.choice(len(corrupted_dataset), size=args.max_samples, replace=False)
                        corrupted_dataset = Subset(corrupted_dataset, sub_idx)
                    corrupted_loader = DataLoader(corrupted_dataset, batch_size=128, shuffle=False)

                    ot_res = OT.compute_ot_distance(corrupted_loader, val_loader,
                                                    dataset=args.dataset,
                                                    feature_extractor_path=feat_path,
                                                    device=device,
                                                    lambda_x=args.lambda_x,
                                                    lambda_y=args.lambda_y,
                                                    entreg=args.entreg)
                    rv_res = RV.compute_rv_metric(corrupted_loader,
                                                dataset=args.dataset,
                                                device=device,
                                                feature_extractor_path=feat_path,
                                                max_samples=args.max_samples)
                    dav_res = DAVINZ.compute_davinz(corrupted_loader, val_loader,
                                                    dataset=args.dataset,
                                                    device=device,
                                                    max_samples=args.max_samples,
                                                    model=None,
                                                    n_batch=args.n_batch,
                                                    n_permute=args.n_permute,
                                                    feature_extractor_path=feat_path)
                    write_results(nl, ot_res, rv_res, dav_res)

            # -------------------------
            # Size and Replication scenarios (no noise)
            # -------------------------
            # Build parties for size and replication
            def build_size_parties(seed_local, party_k=10, per_party_size=args.base_size):
                rng_local = np.random.RandomState(seed_local)
                if isinstance(train_data, Subset) and hasattr(train_data, 'indices'):
                    pool_local = list(train_data.indices)
                else:
                    pool_local = list(range(len(train_inputs)))
                total_needed = party_k * per_party_size
                replace_flag = len(pool_local) < total_needed
                chosen = rng_local.choice(pool_local, size=total_needed, replace=replace_flag)
                parties_local = []
                for i in range(1, party_k + 1):
                    upto = i * per_party_size
                    idxs = chosen[:upto]
                    inputs_local = train_inputs[idxs]
                    labels_local = train_labels[idxs]
                    parties_local.append((inputs_local, labels_local))
                return parties_local

            def build_replication_parties(seed_local, party_k=10, base_size=args.base_size):
                rng_local = np.random.RandomState(seed_local)
                if isinstance(train_data, Subset) and hasattr(train_data, 'indices'):
                    pool_local = np.array(train_data.indices)
                else:
                    pool_local = np.arange(len(train_inputs))
                replace_flag = len(pool_local) < base_size
                base_idxs = rng_local.choice(pool_local, size=base_size, replace=replace_flag)
                base_inputs = train_inputs[base_idxs]
                base_labels = train_labels[base_idxs]
                parties_local = []
                for i in range(1, party_k + 1):
                    if i == 1:
                        inputs_local = base_inputs
                        labels_local = base_labels
                    else:
                        # repeat rows i times
                        inputs_local = base_inputs.repeat(i, 1)
                        labels_local = base_labels.repeat(i)
                    parties_local.append((inputs_local, labels_local))
                return parties_local

            # Size scenario
            # Use args.base_size for label/feature corruption only.
            # For the "size" scenario use fixed per-party size = 1000
            size_party_per_size = 1000
            size_parties = build_size_parties(seed, party_k=10, per_party_size=size_party_per_size)
            size_dir = rob_seed_dir / 'size'
            size_dir.mkdir(parents=True, exist_ok=True)
            # ensure CSVs and JSONL
            if not (size_dir / 'ot_results.csv').exists():
                with open(size_dir / 'ot_results.csv', 'w') as f:
                    f.write('size,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp\n')
            if not (size_dir / 'rv_results.csv').exists():
                with open(size_dir / 'rv_results.csv', 'w') as f:
                    f.write('size,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp\n')
            if not (size_dir / 'davinz_results.csv').exists():
                with open(size_dir / 'davinz_results.csv', 'w') as f:
                    f.write('size,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp\n')
            (size_dir / 'ot_results.jsonl').touch(exist_ok=True)
            (size_dir / 'rv_results.jsonl').touch(exist_ok=True)
            (size_dir / 'davinz_results.jsonl').touch(exist_ok=True)
            # summary
            if not (size_dir / 'size_summary.csv').exists():
                with open(size_dir / 'size_summary.csv', 'w') as f:
                    f.write('party_id,size,ot_distance,rv_log_robust_volume,davinz_score,timestamp\n')

            for p_idx, (p_inputs, p_labels) in enumerate(size_parties, start=1):
                loader = DataLoader(TensorDataset(p_inputs, p_labels), batch_size=128, shuffle=False)
                ot_r = OT.compute_ot_distance(loader, val_loader, dataset=args.dataset, feature_extractor_path=feat_path, device=device, lambda_x=args.lambda_x, lambda_y=args.lambda_y, entreg=args.entreg)
                rv_r = RV.compute_rv_metric(loader, dataset=args.dataset, device=device, feature_extractor_path=feat_path, max_samples=args.max_samples)
                dav_r = DAVINZ.compute_davinz(loader, val_loader, dataset=args.dataset, device=device, max_samples=args.max_samples, model=None, n_batch=args.n_batch, n_permute=args.n_permute, feature_extractor_path=feat_path)
                ts = time.time()
                with open(size_dir / 'ot_results.csv', 'a') as f:
                    f.write(','.join(map(str, [int(p_inputs.size(0)), ot_r.get('distance', 'nan'), ot_r.get('timing', {}).get('feature_extraction', 'nan'), ot_r.get('timing', {}).get('ot_computation', 'nan'), ot_r.get('timing', {}).get('total', 'nan'), ot_r.get('mem', {}).get('feature_extraction', 'nan'), ot_r.get('mem', {}).get('ot', 'nan'), ot_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                with open(size_dir / 'rv_results.csv', 'a') as f:
                    f.write(','.join(map(str, [int(p_inputs.size(0)), rv_r.get('log_volume', 'nan'), rv_r.get('log_robust_volume', 'nan'), rv_r.get('timing', {}).get('feature_extraction', 'nan'), rv_r.get('timing', {}).get('rv_computation', 'nan'), rv_r.get('timing', {}).get('total', 'nan'), rv_r.get('mem', {}).get('feature_extraction', 'nan'), rv_r.get('mem', {}).get('rv', 'nan'), rv_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                with open(size_dir / 'davinz_results.csv', 'a') as f:
                    f.write(','.join(map(str, [int(p_inputs.size(0)), dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')), dav_r.get('mmd_raw', 'nan'), dav_r.get('ntk', 'nan'), dav_r.get('davinz_score', 'nan'), dav_r.get('timing', {}).get('mmd_time', 'nan'), dav_r.get('timing', {}).get('ntk_time', 'nan'), dav_r.get('timing', {}).get('total', 'nan'), dav_r.get('mem', {}).get('mmd', 'nan'), dav_r.get('mem', {}).get('ntk', 'nan'), dav_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                # JSONL
                with open(size_dir / 'ot_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'size', 'result': ot_r, 'timestamp': ts}) + '\n')
                with open(size_dir / 'rv_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'size', 'result': rv_r, 'timestamp': ts}) + '\n')
                with open(size_dir / 'davinz_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'size', 'result': dav_r, 'timestamp': ts}) + '\n')
                with open(size_dir / 'size_summary.csv', 'a') as f:
                    f.write(','.join(map(str, [p_idx, int(p_inputs.size(0)), ot_r.get('distance', 'nan'), rv_r.get('log_robust_volume', 'nan'), dav_r.get('davinz_score', 'nan'), ts])) + '\n')

            # Replication scenario
            # For replication scenario use fixed base size = 1000 (independent of --base_size)
            repl_base_size = 1000
            repl_parties = build_replication_parties(seed, party_k=10, base_size=repl_base_size)
            repl_dir = rob_seed_dir / 'replication'
            repl_dir.mkdir(parents=True, exist_ok=True)
            if not (repl_dir / 'ot_results.csv').exists():
                with open(repl_dir / 'ot_results.csv', 'w') as f:
                    f.write('replication_factor,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp\n')
            if not (repl_dir / 'rv_results.csv').exists():
                with open(repl_dir / 'rv_results.csv', 'w') as f:
                    f.write('replication_factor,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp\n')
            if not (repl_dir / 'davinz_results.csv').exists():
                with open(repl_dir / 'davinz_results.csv', 'w') as f:
                    f.write('replication_factor,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp\n')
            (repl_dir / 'ot_results.jsonl').touch(exist_ok=True)
            (repl_dir / 'rv_results.jsonl').touch(exist_ok=True)
            (repl_dir / 'davinz_results.jsonl').touch(exist_ok=True)
            if not (repl_dir / 'replication_summary.csv').exists():
                with open(repl_dir / 'replication_summary.csv', 'w') as f:
                    f.write('party_id,replication_factor,num_samples,ot_distance,rv_log_robust_volume,davinz_score,timestamp\n')

            for p_idx, (p_inputs, p_labels) in enumerate(repl_parties, start=1):
                loader = DataLoader(TensorDataset(p_inputs, p_labels), batch_size=128, shuffle=False)
                ot_r = OT.compute_ot_distance(loader, val_loader, dataset=args.dataset, feature_extractor_path=feat_path, device=device, lambda_x=args.lambda_x, lambda_y=args.lambda_y, entreg=args.entreg)
                rv_r = RV.compute_rv_metric(loader, dataset=args.dataset, device=device, feature_extractor_path=feat_path, max_samples=args.max_samples)
                dav_r = DAVINZ.compute_davinz(loader, val_loader, dataset=args.dataset, device=device, max_samples=args.max_samples, model=None, n_batch=args.n_batch, n_permute=args.n_permute, feature_extractor_path=feat_path)
                ts = time.time()
                replication_factor = p_idx
                with open(repl_dir / 'ot_results.csv', 'a') as f:
                    f.write(','.join(map(str, [replication_factor, ot_r.get('distance', 'nan'), ot_r.get('timing', {}).get('feature_extraction', 'nan'), ot_r.get('timing', {}).get('ot_computation', 'nan'), ot_r.get('timing', {}).get('total', 'nan'), ot_r.get('mem', {}).get('feature_extraction', 'nan'), ot_r.get('mem', {}).get('ot', 'nan'), ot_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                with open(repl_dir / 'rv_results.csv', 'a') as f:
                    f.write(','.join(map(str, [replication_factor, rv_r.get('log_volume', 'nan'), rv_r.get('log_robust_volume', 'nan'), rv_r.get('timing', {}).get('feature_extraction', 'nan'), rv_r.get('timing', {}).get('rv_computation', 'nan'), rv_r.get('timing', {}).get('total', 'nan'), rv_r.get('mem', {}).get('feature_extraction', 'nan'), rv_r.get('mem', {}).get('rv', 'nan'), rv_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                with open(repl_dir / 'davinz_results.csv', 'a') as f:
                    f.write(','.join(map(str, [replication_factor, dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')), dav_r.get('mmd_raw', 'nan'), dav_r.get('ntk', 'nan'), dav_r.get('davinz_score', 'nan'), dav_r.get('timing', {}).get('mmd_time', 'nan'), dav_r.get('timing', {}).get('ntk_time', 'nan'), dav_r.get('timing', {}).get('total', 'nan'), dav_r.get('mem', {}).get('mmd', 'nan'), dav_r.get('mem', {}).get('ntk', 'nan'), dav_r.get('mem', {}).get('total', 'nan'), ts])) + '\n')
                with open(repl_dir / 'ot_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'replication', 'result': ot_r, 'timestamp': ts}) + '\n')
                with open(repl_dir / 'rv_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'replication', 'result': rv_r, 'timestamp': ts}) + '\n')
                with open(repl_dir / 'davinz_results.jsonl', 'a') as f:
                    f.write(json.dumps({'party_id': p_idx, 'scenario': 'replication', 'result': dav_r, 'timestamp': ts}) + '\n')
                with open(repl_dir / 'replication_summary.csv', 'a') as f:
                    f.write(','.join(map(str, [p_idx, replication_factor, int(p_inputs.size(0)), ot_r.get('distance', 'nan'), rv_r.get('log_robust_volume', 'nan'), dav_r.get('davinz_score', 'nan'), ts])) + '\n')
if __name__ == '__main__':
    main()
