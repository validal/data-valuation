import sys
sys.path.insert(0, '..')

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

# Import model and baseline modules
from model.mlp_reg import MLP
from baselines import OT, RV, DAVINZ
from baselines.ntk import compute_ntk_score_batched_permute, compute_ntk_score_batched
from baselines.mmd import rbf_mmd2
import resource


# ===============================
# Regression metrics
# ===============================
def r2_score_np(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def eval_reg(model, loader, loss_fn, device):
    """Return (mse, mae, r2) on the given DataLoader."""
    model.eval()
    all_pred, all_tgt = [], []
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            total_loss += loss_fn(pred, yb).item() * len(yb)
            total_n   += len(yb)
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(yb.cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_tgt  = np.concatenate(all_tgt)
    mse = total_loss / total_n
    mae = float(np.mean(np.abs(all_pred - all_tgt)))
    r2  = r2_score_np(all_pred, all_tgt)
    return mse, mae, r2


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).squeeze(-1)   # (B,1) -> (B,)  must match yb shape
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_n   += len(yb)
    return total_loss / total_n


# ===============================
# CASP Dataset Loading
# ===============================
def load_casp(casp_path='CASP.csv'):
    """
    Load the CASP dataset (UCI #265).
    CSV layout: first column = RMSD (target), columns 2-10 = features (F1-F9).
    45730 samples, 9 features.
    Downloads from UCI if the file does not exist locally.
    """
    if not os.path.exists(casp_path):
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
               '/00265/CASP.csv')
        print(f"[INFO] Downloading CASP dataset from {url} ...")
        import urllib.request
        urllib.request.urlretrieve(url, casp_path)
        print(f"[INFO] Saved to {casp_path}")

    import csv as _csv
    rows = []
    with open(casp_path, 'r') as f:
        reader = _csv.reader(f)
        next(reader)   # skip header row
        for row in reader:
            rows.append([float(v) for v in row])
    data = np.array(rows, dtype=np.float32)
    y = data[:, 0]   # RMSD
    X = data[:, 1:]  # F1 – F9
    print(f"[INFO] CASP loaded: X={X.shape}, y={y.shape}, "
          f"RMSD range=[{y.min():.2f}, {y.max():.2f}]")
    return X, y


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


def generate_bootstraps(args, train_data, train_targets, output_dir,
                        bootstrap_seeds, bootstrap_size):
    """Generate bootstrap samples; log target statistics (regression)."""
    bootstrap_dir = output_dir / 'bootstraps'
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    metadata_summary = {}
    for seed in bootstrap_seeds:
        rng = np.random.RandomState(seed)
        min_size   = 1000
        max_size   = bootstrap_size
        sample_size = rng.randint(min_size, max_size + 1)
        train_size  = len(train_targets)
        sample_size = min(sample_size, train_size)
        bootstrap_indices = rng.choice(train_size, size=sample_size, replace=True)

        if isinstance(train_data, Subset):
            actual_indices = [train_data.indices[i] for i in bootstrap_indices]
        else:
            actual_indices = list(bootstrap_indices)

        all_targets = (train_targets.cpu().numpy()
                       if torch.is_tensor(train_targets)
                       else np.array(train_targets))
        boot_tgt = all_targets[actual_indices]
        target_stats = {
            'mean': float(boot_tgt.mean()),
            'std':  float(boot_tgt.std()),
            'min':  float(boot_tgt.min()),
            'max':  float(boot_tgt.max()),
        }

        bootstrap_path = bootstrap_dir / f"bootstrap_seed{seed}_size{sample_size}.pt"
        torch.save({'indices': actual_indices, 'seed': seed, 'size': sample_size},
                   bootstrap_path)

        metadata = {
            'seed': seed, 'size': sample_size,
            'dataset': args.dataset,
            'target_statistics': target_stats,
            'sampling_strategy': 'with_replacement_variable_size',
            'size_range': [min_size, max_size],
            'timestamp': time.time()
        }
        with open(bootstrap_dir / f"bootstrap_seed{seed}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        metadata_summary[seed] = metadata
        print(f"[INFO] Bootstrap {seed}: size={sample_size}, "
              f"RMSD mean={target_stats['mean']:.3f} "
              f"(range: {min_size}-{max_size})")

    print(f"[INFO] Generated {len(bootstrap_seeds)} bootstraps in {bootstrap_dir}")
    return metadata_summary


def iter_bootstraps(args, output_dir, train_inputs, train_targets, batch_size=128):
    """Yield (seed, size, indices, loader) for each stored bootstrap sample."""
    bootstrap_seeds = parse_seed_range(args.bootstrap_seeds)
    bootstrap_dir   = output_dir / 'bootstraps'
    if not bootstrap_dir.exists():
        raise FileNotFoundError(f"Bootstrap directory not found: {bootstrap_dir}")

    for seed in bootstrap_seeds:
        bootstrap_files = list(bootstrap_dir.glob(f"bootstrap_seed{seed}_size*.pt"))
        if not bootstrap_files:
            print(f"[WARNING] Bootstrap {seed} not found, skipping.")
            continue
        bootstrap_data = torch.load(bootstrap_files[0])
        indices = bootstrap_data['indices']
        size    = bootstrap_data.get('size', len(indices))
        ds      = Subset(TensorDataset(train_inputs, train_targets), indices)
        loader  = DataLoader(ds, batch_size=batch_size, shuffle=False)
        yield seed, size, indices, loader


def train_bootstrap_model(args, model, bootstrap_indices, full_train_data,
                          val_data, lr, weight_decay,
                          num_epochs, batch_size, device, output_dir,
                          bootstrap_seed):
    """Train regression model on a bootstrap sample, evaluate on val."""
    print(f"\n{'='*60}")
    print(f"[INFO] TRAINING BOOTSTRAP MODEL  (seed={bootstrap_seed}, "
          f"size={len(bootstrap_indices)})")
    print(f"{'='*60}\n")

    model     = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    loaders = {
        'train': DataLoader(Subset(full_train_data, bootstrap_indices),
                            batch_size=batch_size, shuffle=True, drop_last=True),
        'val':   DataLoader(val_data, batch_size=batch_size, shuffle=False),
    }

    train_bs_dir = output_dir / 'train_bootstraps'
    train_bs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (train_bs_dir /
                f"{args.dataset.lower()}_bootstrap_seed{bootstrap_seed}_training_log.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_mse',
                                'val_mse', 'val_mae', 'val_r2',
                                'lr', 'elapsed_time'])

    print(f"[INFO] Training for {num_epochs} epochs (lr={lr})…")
    early_stop_patience = 20
    best_val_mse = float('inf')
    best_epoch   = 0
    no_improve   = 0
    best_state   = None
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, loaders['train'], loss_fn, optimizer, device)
        val_mse, val_mae, val_r2 = eval_reg(model, loaders['val'], loss_fn, device)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_epoch   = epoch
            no_improve   = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        scheduler.step(val_mse)

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, tr_loss,
                                    val_mse, val_mae, val_r2,
                                    current_lr, elapsed])

        marker = '  ← best' if improved else ''
        print(f"  epoch {epoch:03d}  train_mse={tr_loss:.4f}  "
              f"val_mse={val_mse:.4f}  val_r2={val_r2:.4f}  "
              f"lr={current_lr:.2e}  no_improve={no_improve}{marker}")

        if no_improve >= early_stop_patience:
            print(f"[INFO] Early stop at epoch {epoch} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse, val_mae, val_r2 = eval_reg(model, loaders['val'], loss_fn, device)
    total_time = time.time() - start
    print(f"[INFO] Finished in {total_time:.2f}s  best epoch={best_epoch}  "
          f"val_mse={best_val_mse:.4f}  val_r2={val_r2:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = (f"{args.dataset.lower()}_{model.__class__.__name__.lower()}"
                 f"_bootstrap_seed{bootstrap_seed}.pt")
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    metrics = {
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mse': best_val_mse, 'val_mae': val_mae, 'val_r2': val_r2,
        'best_epoch': best_epoch, 'lr': lr, 'seed': args.seed,
        'bootstrap_seed': bootstrap_seed, 'dataset': args.dataset,
        'model_arch': model.__class__.__name__,
        'model_type': 'bootstrap_model',
        'trained_on': f'bootstrap_{bootstrap_seed}',
        'bootstrap_size': len(bootstrap_indices),
        'val_size': len(val_data),
    }
    torch.save(metrics, ckpt_path)
    print(f"[INFO] Checkpoint saved to {ckpt_path}")
    try:
        import shutil
        shutil.copyfile(ckpt_path, str(train_bs_dir / ckpt_name))
    except Exception:
        pass
    return model, ckpt_path, metrics


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap Correlation Experiment — Regression (CASP)')

    parser.add_argument('--gpu',        type=str,   default='0')
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--dataset',    type=str,   default='CASP',
                        choices=['CASP'],
                        help='Regression dataset')
    parser.add_argument('--casp_path',  type=str,   default='CASP.csv',
                        help='Path to CASP.csv (auto-downloaded if missing)')
    parser.add_argument('--save_dir',   type=str,   default='checkpoints')
    parser.add_argument('--output_dir', type=str,   default='outputs')
    parser.add_argument('--val_frac',   type=float, default=0.1)

    # Training params (shared by --train_base and --train_bootstrap)
    parser.add_argument('--num_epochs',   type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size',   type=int,   default=128)

    # Operation flags
    parser.add_argument('--train_base',            action='store_true',
                        help='Train one model on full train set, eval on val')
    parser.add_argument('--generate_bootstraps',  action='store_true')
    parser.add_argument('--train_bootstrap',       action='store_true')
    parser.add_argument('--compute_ot',            action='store_true')
    parser.add_argument('--compute_volume',        action='store_true')
    parser.add_argument('--compute_davinz',        action='store_true')
    parser.add_argument('--rv_tuning',             action='store_true')
    parser.add_argument('--rv_repeats',            type=int, default=1)
    parser.add_argument('--davinz_repeats',        type=int, default=1)
    parser.add_argument('--tune_ntk',              action='store_true')
    parser.add_argument('--davinz_batch_tuning',   action='store_true')
    parser.add_argument('--davinz_batch_values',   type=str,
                        default='10,20,50,100,250,500,1000')
    parser.add_argument('--davinz_batch_repeats',  type=int, default=5)

    # Bootstrap parameters
    parser.add_argument('--bootstrap_seeds', type=str, default='0-5')
    parser.add_argument('--bootstrap_size',  type=int, default=10000)

    # OT / baseline params
    parser.add_argument('--feature_extractor_path', type=str, default=None)
    parser.add_argument('--lambda_x',   type=float, default=1.0)
    parser.add_argument('--lambda_y',   type=float, default=1.0)
    parser.add_argument('--entreg',     type=float, default=1e-1)
    parser.add_argument('--ot_repeats', type=int,   default=1)
    parser.add_argument('--n_batch',        type=int,   default=1)
    parser.add_argument('--n_permute',      type=int,   default=1)
    parser.add_argument('--diagonal_I_mag', type=float, default=1e-6,
                        help='Regularisation added to NTK diagonal (H += diag_I*I)')

    args = parser.parse_args()

    operations = [args.train_base, args.generate_bootstraps, args.train_bootstrap,
                  args.compute_ot, args.compute_volume, args.compute_davinz,
                  args.tune_ntk, args.rv_tuning, args.davinz_batch_tuning]
    if not any(operations):
        parser.print_help()
        return

    # ── Environment setup ──────────────────────────────────────────
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    gpu_list = ([int(x) for x in str(args.gpu).split(',') if x.strip()]
                if args.gpu else [])
    device = torch.device('cpu')
    if torch.cuda.is_available() and gpu_list:
        try:
            torch.cuda.set_device(int(gpu_list[0]))
            device = torch.device(f'cuda:{int(gpu_list[0])}')
        except Exception as e:
            print(f"[WARN] GPU set failed: {e}")
            device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_out = output_dir / args.dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / 'bootstraps').mkdir(parents=True, exist_ok=True)
    (dataset_out / 'train_bootstraps').mkdir(parents=True, exist_ok=True)
    output_dir = dataset_out

    # ── Data loading ───────────────────────────────────────────────
    print(f"[INFO] Loading {args.dataset} dataset…")
    X_raw, y_raw = load_casp(args.casp_path)

    # Standardise features and target
    X_mean, X_std = X_raw.mean(0), X_raw.std(0) + 1e-8
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std()) + 1e-8
    X_norm = (X_raw - X_mean) / X_std
    y_norm = (y_raw - y_mean) / y_std

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_norm, test_size=args.val_frac, random_state=args.seed)

    train_inputs  = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.float32)
    val_inputs    = torch.tensor(X_val,   dtype=torch.float32)
    val_targets   = torch.tensor(y_val,   dtype=torch.float32)

    train_data = TensorDataset(train_inputs, train_targets)
    val_data   = TensorDataset(val_inputs,   val_targets)

    print(f"[INFO] Split: train={len(train_data)}, val={len(val_data)}")
    print(f"[INFO] Feature dim: {train_inputs.shape[1]}")

    input_dim = train_inputs.shape[1]

    def create_model():
        return MLP(in_dim=input_dim, out_dim=1)

    # ── Bootstrap generation ───────────────────────────────────────
    if args.generate_bootstraps:
        bootstrap_seeds = parse_seed_range(args.bootstrap_seeds)
        print(f"\n[INFO] Generating {len(bootstrap_seeds)} bootstraps…")
        generate_bootstraps(args, train_data, train_targets, output_dir,
                            bootstrap_seeds, args.bootstrap_size)
        return

    # ── OT computation (POT — features + continuous label) ──────────
    if args.compute_ot:
        import ot as pot
        import csv as _csv
        print("\n" + "="*60)
        print("COMPUTING OT DISTANCES  (POT sinkhorn, features + lambda_y*y)")
        print("="*60)
        print(f"[OT] lambda_y = {args.lambda_y}")

        ot_base = output_dir / f"OT_entreg{str(args.entreg).replace('.','p')}"
        ot_base.mkdir(parents=True, exist_ok=True)

        n_ot_repeats = max(1, int(args.ot_repeats))

        # Prepare val matrix [X | lambda_y * y] once (cap at 2000)
        max_samples = 2000
        val_X_np = val_inputs.numpy()
        val_y_np = val_targets.numpy().reshape(-1, 1)
        if len(val_X_np) > max_samples:
            rng0 = np.random.RandomState(args.seed)
            val_idx = rng0.choice(len(val_X_np), max_samples, replace=False)
            val_X_np = val_X_np[val_idx]
            val_y_np = val_y_np[val_idx]
        val_np = np.concatenate([val_X_np, args.lambda_y * val_y_np], axis=1)
        b = np.ones(len(val_np)) / len(val_np)

        for rep in range(1, n_ot_repeats + 1):
            print(f"\n[OT] === Repeat {rep}/{n_ot_repeats} ===")
            rep_dir = ot_base / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            set_seed(args.seed + rep * 1000)

            rep_csv = rep_dir / f"ot_results_{args.dataset.lower()}_rep{rep}.csv"
            headers = ['seed', 'rep', 'bootstrap_size', 'ot_distance',
                       'ot_time_s', 'dataset', 'timestamp']
            with open(rep_csv, 'w', newline='') as f:
                _csv.DictWriter(f, fieldnames=headers).writeheader()

            results = []
            for seed, size, indices, _ in iter_bootstraps(
                args, output_dir, train_inputs, train_targets, batch_size=128
            ):
                print(f"[INFO] OT rep={rep} seed={seed} (size={size})")
                try:
                    boot_X_np = train_inputs[indices].numpy()
                    boot_y_np = train_targets[indices].numpy().reshape(-1, 1)
                    if len(boot_X_np) > max_samples:
                        rng = np.random.RandomState(seed + rep * 1000)
                        boot_idx = rng.choice(len(boot_X_np), max_samples, replace=False)
                        boot_X_np = boot_X_np[boot_idx]
                        boot_y_np = boot_y_np[boot_idx]
                    boot_np = np.concatenate(
                        [boot_X_np, args.lambda_y * boot_y_np], axis=1
                    )

                    a   = np.ones(len(boot_np)) / len(boot_np)
                    M   = pot.dist(boot_np, val_np)          # squared euclidean
                    M  /= M.max() + 1e-8                     # normalise to [0,1]

                    t0  = time.time()
                    W   = float(pot.sinkhorn2(a, b, M, reg=args.entreg,
                                              numItermax=500, warn=False))
                    ot_time = time.time() - t0

                    row = {
                        'seed': seed, 'rep': rep,
                        'bootstrap_size': size,
                        'ot_distance': W,
                        'ot_time_s': ot_time,
                        'dataset': args.dataset,
                        'timestamp': time.time()
                    }
                    results.append(row)
                    with open(rep_csv, 'a', newline='') as f:
                        _csv.DictWriter(f, fieldnames=headers).writerow(row)
                    print(f"         W={W:.6f}  ({ot_time:.1f}s)")

                except Exception as e:
                    print(f"[ERROR] OT seed={seed} rep={rep}: {e}")
                    traceback.print_exc()

            print(f"[INFO] OT rep={rep}: {len(results)} seeds saved to {rep_csv}")

        print(f"[INFO] OT done: {n_ot_repeats} repeat(s) under {ot_base}")

    # ── RV / Volume computation ────────────────────────────────────
    if args.compute_volume:
        import csv as _csv
        print("\n" + "="*60)
        print("COMPUTING RV / VOLUME METRICS")
        print("="*60)

        rv_base = output_dir / 'RV'
        rv_base.mkdir(parents=True, exist_ok=True)
        n_rv_repeats = max(1, int(args.rv_repeats))

        rv_headers = ['seed', 'rep', 'bootstrap_size',
                      'log_volume', 'log_robust_volume',
                      'rv_time_s', 'feature_extraction_time_s',
                      'total_time_s', 'timestamp']

        for rep in range(1, n_rv_repeats + 1):
            print(f"\n[RV] === Repeat {rep}/{n_rv_repeats} ===")
            rep_dir = rv_base / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            set_seed(args.seed + rep * 1000)

            rep_csv = rep_dir / f"rv_results_{args.dataset.lower()}_rep{rep}.csv"
            with open(rep_csv, 'w', newline='') as f:
                _csv.DictWriter(f, fieldnames=rv_headers).writeheader()

            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_targets, batch_size=128
            ):
                print(f"[INFO] RV rep={rep} seed={seed} (size={size})")
                try:
                    res = RV.compute_rv_metric(
                        bootstrap_loader,
                        dataset=args.dataset,
                        device=device,
                        feature_extractor_path=args.feature_extractor_path,
                        max_samples=10000)
                    timing = res.get('timing', {})
                    row = {
                        'seed':                     seed,
                        'rep':                      rep,
                        'bootstrap_size':           size,
                        'log_volume':               res.get('log_volume', ''),
                        'log_robust_volume':        res.get('log_robust_volume', ''),
                        'rv_time_s':                timing.get('rv_computation', ''),
                        'feature_extraction_time_s': timing.get('feature_extraction', ''),
                        'total_time_s':             timing.get('total', ''),
                        'timestamp':                time.time(),
                    }
                    with open(rep_csv, 'a', newline='') as f:
                        _csv.DictWriter(f, fieldnames=rv_headers).writerow(row)
                    print(f"         log_rv={res.get('log_robust_volume', 'N/A')}")
                except Exception as e:
                    print(f"[ERROR] RV rep={rep} seed={seed}: {e}")
                    traceback.print_exc()

            print(f"[INFO] RV rep={rep} saved to {rep_csv}")

    # ── RV Tuning ─────────────────────────────────────────────────
    if args.rv_tuning:
        print("\n" + "="*60)
        print("RUNNING RV TUNING (omega x alpha grid)")
        print("="*60)

        omega_values  = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
        alpha_configs = [('1n', 1), ('10n', 10), ('100n', 100)]
        n_rv_repeats  = max(1, int(args.rv_repeats))

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
                config_dir  = tuning_out / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[RV TUNE] {config_name}  "
                      f"(omega={omega}, alpha_mult={alpha_multiplier})")

                for rep in range(1, n_rv_repeats + 1):
                    rep_dir = config_dir / f"rep_{rep}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    set_seed(args.seed + rep * 1000)

                    rep_csv = (rep_dir /
                               f"rv_results_{args.dataset.lower()}"
                               f"_{config_name}_rep{rep}.csv")
                    rep_header = ['seed', 'bootstrap_size',
                                  'log_volume', 'log_robust_volume',
                                  'rv_time_s', 'feature_extraction_time_s',
                                  'total_time_s', 'timestamp']
                    with open(rep_csv, 'w', newline='') as f:
                        _csv.DictWriter(f, fieldnames=rep_header).writeheader()

                    print(f"  [rep {rep}/{n_rv_repeats}]")
                    for seed, size, _, bootstrap_loader in iter_bootstraps(
                        args, output_dir, train_inputs, train_targets, batch_size=128
                    ):
                        try:
                            res = RV.compute_rv_metric(
                                bootstrap_loader,
                                dataset=args.dataset, device=device,
                                feature_extractor_path=args.feature_extractor_path,
                                max_samples=10000,
                                omega=omega, alpha=alpha_multiplier)
                            timing = res.get('timing', {})
                            ts = time.time()
                            row = {
                                'seed': seed, 'bootstrap_size': size,
                                'log_volume':        res.get('log_volume', ''),
                                'log_robust_volume': res.get('log_robust_volume', ''),
                                'rv_time_s':         timing.get('rv_computation', ''),
                                'feature_extraction_time_s': timing.get('feature_extraction', ''),
                                'total_time_s':      timing.get('total', ''),
                                'timestamp':         ts
                            }
                            with open(rep_csv, 'a', newline='') as f:
                                _csv.DictWriter(f, fieldnames=rep_header).writerow(row)
                            with open(summary_csv, 'a', newline='') as f:
                                _csv.DictWriter(f, fieldnames=summary_header).writerow({
                                    'config': config_name, 'omega': omega,
                                    'alpha_multiplier': alpha_multiplier,
                                    'alpha_label': alpha_label,
                                    'rep': rep, 'seed': seed,
                                    'bootstrap_size': size,
                                    'log_volume':        res.get('log_volume', ''),
                                    'log_robust_volume': res.get('log_robust_volume', ''),
                                    'rv_time_s':    timing.get('rv_computation', ''),
                                    'total_time_s': timing.get('total', ''),
                                    'timestamp': ts
                                })
                        except Exception as e:
                            print(f"[ERROR] RV tune {config_name} rep={rep} "
                                  f"seed={seed}: {e}")
                            traceback.print_exc()

                    print(f"  [rep {rep}] done -> {rep_csv}")

        print(f"\n[INFO] RV tuning complete. Under {tuning_out}")

    # ── DaVinz computation ─────────────────────────────────────────
    if args.compute_davinz:
        print("\n" + "="*60)
        print("COMPUTING DaVinz METRICS")
        print("="*60)

        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        os.makedirs(args.save_dir, exist_ok=True)

        init_path = (Path(args.save_dir) /
                     f"{args.dataset.lower()}_regressionmlp_init_seed{args.seed}.pt")
        if init_path.exists():
            try:
                model = torch.load(str(init_path)).to(device)
                print(f"[INFO] Loaded initial model from {init_path}")
            except Exception as e:
                print(f"[WARN] Load failed ({e}); creating fresh model")
                model = create_model().to(device)
        else:
            model = create_model().to(device)
            try:
                torch.save(model, str(init_path))
                print(f"[INFO] Saved initial model to {init_path}")
            except Exception as e:
                print(f"[WARN] Save failed: {e}")

        if getattr(args, 'davinz_batch_tuning', False):
            print("\n[INFO] Running DaVinz batch-size tuning")
            batch_values = [int(x.strip()) for x in
                            str(getattr(args, 'davinz_batch_values',
                                        '10,20,50,100,250,500,1000')).split(',')
                            if x.strip()]
            repeats    = int(getattr(args, 'davinz_batch_repeats', 5))
            tuning_out = output_dir / 'Davinz_batchtuning'
            tuning_out.mkdir(parents=True, exist_ok=True)
            seed_list = parse_seed_range(getattr(args, 'bootstrap_seeds', '0-0'))
            header = ['bootstrap_size', 'dataset', 'timing', 'rep', 'mem',
                      'mmd_raw_time', 'n_batch', 'status', 'n_permute',
                      'davinz_score', 'seed', 'error', 'mmd_raw_mem',
                      'bootstrap_labels_hash', 'maxsamples_used', 'val_size',
                      'mmd', 'ntk', 'bootstrap_labels_sample', 'mmd_raw',
                      'bootstrap_labels_unique']

            for rep_idx in range(repeats):
                rep = rep_idx + 1
                rep_model_seed = int(args.seed) + rep_idx * 1000
                set_seed(rep_model_seed)
                model_rep = create_model().to(device)

                for nb in batch_values:
                    nb_dir = tuning_out / f'n_batch_{nb}' / f'rep_{rep}'
                    nb_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = nb_dir / 'davinz_results.csv'
                    if not out_csv.exists() or out_csv.stat().st_size == 0:
                        with open(out_csv, 'w', newline='') as _f:
                            csv.writer(_f).writerow(header)

                    for seed in seed_list:
                        bfiles = list((output_dir / 'bootstraps').glob(
                            f"bootstrap_seed{seed}_size*.pt"))
                        if not bfiles:
                            miss = {k: '' for k in header}
                            miss.update({'seed': seed, 'status': 'missing_bootstrap'})
                            with open(out_csv, 'a', newline='') as _f:
                                csv.writer(_f).writerow(
                                    [miss.get(h, '') for h in header])
                            continue
                        bp      = torch.load(str(bfiles[0]))
                        indices = bp.get('indices', [])
                        bsize   = int(bp.get('size', len(indices)))
                        boot_ld = DataLoader(
                            Subset(TensorDataset(train_inputs, train_targets), indices),
                            batch_size=128, shuffle=False)
                        try:
                            davinz_result = DAVINZ.compute_davinz(
                                boot_ld, val_loader, dataset=args.dataset,
                                device=device, model=model_rep,
                                diagonal_I_mag=1e-6, n_batch=nb)
                            davinz_result.update({
                                'seed': seed, 'bootstrap_size': bsize,
                                'rep': rep, 'n_batch': nb,
                                'status': 'ok', 'error': ''})
                            row = {k: '' for k in header}
                            timing = davinz_result.get('timing', {})
                            row.update({
                                'bootstrap_size': bsize, 'dataset': args.dataset,
                                'timing': json.dumps(timing), 'rep': rep,
                                'mem': json.dumps(davinz_result.get('mem', {})),
                                'n_batch': nb, 'status': 'ok',
                                'davinz_score': davinz_result.get('davinz_score'),
                                'seed': seed, 'error': '',
                                'mmd': davinz_result.get('mmd'),
                                'ntk': davinz_result.get('ntk'),
                            })
                            with open(out_csv, 'a', newline='') as _f:
                                csv.writer(_f).writerow(
                                    [row.get(h, '') for h in header])
                        except Exception as e:
                            errrow = {k: '' for k in header}
                            errrow.update({'seed': seed, 'rep': rep,
                                           'status': 'failed', 'error': str(e)})
                            with open(out_csv, 'a', newline='') as _f:
                                csv.writer(_f).writerow(
                                    [errrow.get(h, '') for h in header])
                            print(f"[ERROR] DaVinz n_batch={nb} seed={seed}: {e}")
                            traceback.print_exc()

            print("[INFO] DaVinz batch-size tuning completed.")
            return

        import csv as _csv
        diagonal_I_mag = args.diagonal_I_mag
        davinz_base = (output_dir /
                       f"davinz_n_batch_1_n_permute_1_diagI_{diagonal_I_mag:.0e}".replace('e-0', 'e-').replace('e+0','e'))
        davinz_base.mkdir(parents=True, exist_ok=True)
        n_davinz_repeats = max(1, int(args.davinz_repeats))

        dv_headers = ['seed', 'rep', 'bootstrap_size',
                      'mmd', 'ntk', 'davinz_score', 'kappa',
                      'dataset', 'timestamp']

        for rep in range(1, n_davinz_repeats + 1):
            print(f"\n[DaVinZ] === Repeat {rep}/{n_davinz_repeats} ===")
            rep_dir = davinz_base / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            rep_seed = args.seed + rep * 1000
            set_seed(rep_seed)
            model_rep = create_model().to(device)

            rep_csv = rep_dir / f"davinz_results_{args.dataset.lower()}_rep{rep}.csv"
            with open(rep_csv, 'w', newline='') as f:
                _csv.DictWriter(f, fieldnames=dv_headers).writeheader()

            rep_results = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_targets, batch_size=128
            ):
                print(f"[INFO] DaVinZ rep={rep} seed={seed} (size={size})")
                try:
                    res = DAVINZ.compute_davinz(
                        bootstrap_loader, val_loader,
                        dataset=args.dataset, device=device,
                        max_samples=10000, model=model_rep,
                        n_batch=1, n_permute=1, diagonal_I_mag=diagonal_I_mag,
                        mode='reg',
                        feature_extractor_path=args.feature_extractor_path)
                    rep_results.append({'seed': seed, 'size': size,
                                        'mmd': res.get('mmd'),
                                        'ntk': res.get('ntk')})
                except Exception as e:
                    print(f"[ERROR] DaVinZ rep={rep} seed={seed}: {e}")
                    traceback.print_exc()

            # compute kappa from this rep's results, then write CSV
            mmds = np.array([r['mmd'] for r in rep_results
                             if r['mmd'] is not None], dtype=float)
            ntks = np.array([r['ntk'] for r in rep_results
                             if r['ntk'] is not None], dtype=float)
            mean_mmd = float(np.mean(mmds)) if mmds.size > 0 else 0.0
            mean_ntk = float(np.mean(ntks[ntks != 0])) if ntks.size > 0 else 0.0
            kappa = mean_mmd / mean_ntk if mean_ntk != 0.0 else 0.0
            print(f"[DaVinZ rep={rep}] kappa={kappa:.6e}  "
                  f"mean_mmd={mean_mmd:.6e}  mean_ntk={mean_ntk:.6e}")

            with open(rep_csv, 'a', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=dv_headers)
                for r in rep_results:
                    mmd = float(r['mmd']) if r['mmd'] is not None else ''
                    ntk = float(r['ntk']) if r['ntk'] is not None else ''
                    davinz_score = (-(kappa * ntk + mmd)
                                    if isinstance(mmd, float)
                                    and isinstance(ntk, float) else '')
                    writer.writerow({
                        'seed':           r['seed'],
                        'rep':            rep,
                        'bootstrap_size': r['size'],
                        'mmd':            mmd,
                        'ntk':            ntk,
                        'davinz_score':   davinz_score,
                        'kappa':          kappa,
                        'dataset':        args.dataset,
                        'timestamp':      time.time(),
                    })
            print(f"[INFO] DaVinZ rep={rep} saved to {rep_csv}")

    # ── NTK Tuning ────────────────────────────────────────────────
    if args.tune_ntk:
        print("\n" + "="*60)
        print("RUNNING NTK TUNING")
        print("="*60)

        tuning_out = output_dir / 'davinz_tuning'
        tuning_out.mkdir(parents=True, exist_ok=True)
        val_loader    = DataLoader(val_data, batch_size=128, shuffle=False)
        batch_values  = [20, 50, 100, 250, 500, 1000]
        permute_values = [1, 2, 5, 10, 20, 50, 100, 500]
        fixed_batch   = 100

        import csv as _csv

        # n_batch tuning
        batch_csv = tuning_out / f"batch_tuning_{args.dataset.lower()}.csv"
        with open(batch_csv, 'w', newline='') as bf:
            _csv.DictWriter(bf, fieldnames=[
                'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
            ]).writeheader()

        for n_batch in batch_values:
            print(f"\n[NTK TUNE] n_batch={n_batch}")
            results_rows = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_targets, batch_size=128
            ):
                bx, by = DAVINZ._collect_inputs_labels(
                    bootstrap_loader, device, max_samples=10000)
                vx, _  = DAVINZ._collect_inputs_labels(
                    val_loader, device, max_samples=10000)

                mmd_start   = time.time()
                sigma       = DAVINZ._estimate_sigma(bx[:min(1000, bx.size(0))],
                                                     vx[:min(1000, vx.size(0))])
                mmd_squared = rbf_mmd2(bx.reshape(bx.size(0), -1),
                                       vx.reshape(vx.size(0), -1), sigma=sigma)
                mmd         = float(torch.sqrt(mmd_squared).item())
                mmd_time    = time.time() - mmd_start

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                ntk_start = time.time()
                ntk_model = create_model().to(device)
                ntk_val, _ = compute_ntk_score_batched(
                    ntk_model, bx, by, mode='reg',
                    n_batch=n_batch, use_hack=True, diagonal_I_mag=1e-6)
                ntk_time  = time.time() - ntk_start
                ntk_mem   = (torch.cuda.max_memory_allocated()
                             if torch.cuda.is_available() else
                             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             - mem_before)

                results_rows.append({
                    'n_batch': n_batch, 'seed': seed, 'bootstrap_size': size,
                    'mmd': mmd, 'mmd_time': mmd_time,
                    'ntk': float(ntk_val), 'ntk_time': ntk_time,
                    'ntk_mem_bytes': int(ntk_mem),
                    'val_size': int(vx.size(0)), 'dataset': args.dataset
                })

            with open(batch_csv, 'a', newline='') as bf:
                w = _csv.DictWriter(bf, fieldnames=[
                    'n_batch', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                    'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'])
                for r in results_rows:
                    w.writerow(r)

        # n_permute tuning
        permute_csv = tuning_out / f"permute_tuning_{args.dataset.lower()}.csv"
        with open(permute_csv, 'w', newline='') as pf:
            _csv.DictWriter(pf, fieldnames=[
                'n_permute', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'
            ]).writeheader()

        for n_permute in permute_values:
            print(f"\n[NTK TUNE] n_permute={n_permute} (n_batch={fixed_batch})")
            results_rows = []
            for seed, size, _, bootstrap_loader in iter_bootstraps(
                args, output_dir, train_inputs, train_targets, batch_size=128
            ):
                bx, by = DAVINZ._collect_inputs_labels(
                    bootstrap_loader, device, max_samples=10000)
                vx, _  = DAVINZ._collect_inputs_labels(
                    val_loader, device, max_samples=10000)

                mmd_start   = time.time()
                sigma       = DAVINZ._estimate_sigma(bx[:min(1000, bx.size(0))],
                                                     vx[:min(1000, vx.size(0))])
                mmd_squared = rbf_mmd2(bx.reshape(bx.size(0), -1),
                                       vx.reshape(vx.size(0), -1), sigma=sigma)
                mmd         = float(torch.sqrt(mmd_squared).item())
                mmd_time    = time.time() - mmd_start

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                ntk_start = time.time()
                ntk_model = create_model().to(device)
                ntk_val, _ = compute_ntk_score_batched_permute(
                    ntk_model, bx, by, mode='reg',
                    n_batch=fixed_batch, n_permute=n_permute,
                    use_hack=True, diagonal_I_mag=1e-6)
                ntk_time  = time.time() - ntk_start
                ntk_mem   = (torch.cuda.max_memory_allocated()
                             if torch.cuda.is_available() else
                             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             - mem_before)

                results_rows.append({
                    'n_permute': n_permute, 'seed': seed, 'bootstrap_size': size,
                    'mmd': mmd, 'mmd_time': mmd_time,
                    'ntk': float(ntk_val), 'ntk_time': ntk_time,
                    'ntk_mem_bytes': int(ntk_mem),
                    'val_size': int(vx.size(0)), 'dataset': args.dataset
                })

            with open(permute_csv, 'a', newline='') as pf:
                w = _csv.DictWriter(pf, fieldnames=[
                    'n_permute', 'seed', 'bootstrap_size', 'mmd', 'mmd_time',
                    'ntk', 'ntk_time', 'ntk_mem_bytes', 'val_size', 'dataset'])
                for r in results_rows:
                    w.writerow(r)

        print(f"\n[NTK TUNING] Results saved in {tuning_out}")
        return

    # ── Base model training (full train set) ───────────────────────
    if args.train_base:
        print("\n" + "="*60)
        print("TRAINING BASE MODEL ON FULL TRAIN SET")
        print("="*60)

        model     = create_model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
        loss_fn   = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_data,   batch_size=args.batch_size,
                                  shuffle=False)

        os.makedirs(args.save_dir, exist_ok=True)
        log_path = output_dir / f"{args.dataset.lower()}_base_training_log.csv"
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_mse',
                                    'val_mse', 'val_mae', 'val_r2',
                                    'lr', 'elapsed_s'])

        print(f"[INFO] train={len(train_data)}  val={len(val_data)}  "
              f"epochs={args.num_epochs}  lr={args.lr}  patience=10")
        print(f"[INFO] Early stop: patience=20 epochs without val_mse improvement")

        best_val_mse   = float('inf')
        best_epoch     = 0
        no_improve     = 0
        early_stop_patience = 20
        best_state     = None
        start = time.time()

        for epoch in range(1, args.num_epochs + 1):
            tr_mse = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_mse, val_mae, val_r2 = eval_reg(model, val_loader, loss_fn, device)
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start

            # Track best
            improved = val_mse < best_val_mse
            if improved:
                best_val_mse = val_mse
                best_epoch   = epoch
                no_improve   = 0
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            scheduler.step(val_mse)

            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, tr_mse,
                                        val_mse, val_mae, val_r2,
                                        current_lr, elapsed])

            marker = '  ← best' if improved else ''
            print(f"  epoch {epoch:03d}/{args.num_epochs}  "
                  f"train_mse={tr_mse:.4f}  "
                  f"val_mse={val_mse:.4f}  val_r2={val_r2:.4f}  "
                  f"lr={current_lr:.2e}  no_improve={no_improve}{marker}")

            if no_improve >= early_stop_patience:
                print(f"\n[INFO] Early stop: val_mse did not improve for "
                      f"{early_stop_patience} epochs.")
                break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final eval with best weights
        val_mse, val_mae, val_r2 = eval_reg(model, val_loader, loss_fn, device)
        print(f"\n[INFO] Converged at epoch {best_epoch}  "
              f"val_mse={best_val_mse:.4f}  val_mae={val_mae:.4f}  val_r2={val_r2:.4f}")

        ckpt_path = os.path.join(args.save_dir,
                                 f"{args.dataset.lower()}_base_model.pt")
        torch.save({'model_state_dict': model.state_dict(),
                    'val_mse': best_val_mse, 'val_mae': val_mae, 'val_r2': val_r2,
                    'best_epoch': best_epoch, 'lr': args.lr,
                    'dataset': args.dataset}, ckpt_path)
        print(f"[INFO] Best checkpoint saved: {ckpt_path}")
        print(f"[INFO] Log: {log_path}")
        return

    # ── Bootstrap training ─────────────────────────────────────────
    if args.train_bootstrap:
        batch_size   = args.batch_size
        lr           = args.lr
        weight_decay = args.weight_decay
        num_epochs   = args.num_epochs

        bootstrap_results = []
        for bootstrap_seed, size, bootstrap_indices, _ in iter_bootstraps(
            args, output_dir, train_inputs, train_targets, batch_size=batch_size
        ):
            print(f"\n[INFO] Processing bootstrap seed {bootstrap_seed} (size={size})")
            model = create_model()
            model, ckpt_path, metrics = train_bootstrap_model(
                args, model, bootstrap_indices,
                TensorDataset(train_inputs, train_targets),
                val_data,
                lr, weight_decay, num_epochs, batch_size,
                device, output_dir, bootstrap_seed)
            metrics['checkpoint_path'] = ckpt_path
            bootstrap_results.append(metrics)

        results_dir = output_dir / 'train_bootstraps'
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{args.dataset.lower()}_bootstrap_results.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['bootstrap_seed', 'bootstrap_size',
                          'val_mse', 'val_mae', 'val_r2',
                          'epochs', 'lr', 'model_arch', 'checkpoint_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in bootstrap_results:
                writer.writerow({
                    'bootstrap_seed': m.get('bootstrap_seed'),
                    'bootstrap_size': m.get('bootstrap_size'),
                    'val_mse': m.get('val_mse', ''),
                    'val_mae': m.get('val_mae', ''),
                    'val_r2':  m.get('val_r2',  ''),
                    'epochs':  m.get('epochs'),
                    'lr':      m.get('lr'),
                    'model_arch': m.get('model_arch'),
                    'checkpoint_path': m.get('checkpoint_path', '')
                })

        print(f"\n[INFO] All bootstrap results saved to {csv_path}")
        val_r2s = [m.get('val_r2', 0.0) for m in bootstrap_results]
        if val_r2s:
            print(f"[INFO] Summary - Val R2: "
                  f"mean={np.mean(val_r2s):.4f} +/- {np.std(val_r2s):.4f}")
        print("[INFO] Bootstrap training completed.")
        return


if __name__ == "__main__":
    main()
