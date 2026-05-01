"""
OT.py - Optimal Transport distance computation module
Computes OT distance between bootstrap samples and validation set
"""
import os
import time
import csv
import json
from pathlib import Path
import resource
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset

# OT distance imports
import otdd
from otdd.pytorch.distance_fast import DatasetDistance, FeatureCost

# Import model definitions
from model.resnet import BasicBlock, ResNet
from model.cnn import CNN


def compute_ot_distance(bootstrap_loader, val_loader, dataset, 
                       feature_extractor_path=None, device='cuda',
                       lambda_x=1.0, lambda_y=1.0, entreg=1e-1):
    """
    Compute OT distance between bootstrap sample and fixed validation set.
    
    Args:
        bootstrap_loader: DataLoader for bootstrap sample
        val_loader: DataLoader for validation set
        dataset: 'CIFAR_10' or 'MNIST'
        feature_extractor_path: Path to feature extractor (for CIFAR-10)
        device: 'cuda' or 'cpu'
        lambda_x, lambda_y: Regularization parameters
        entreg: Entropic regularization parameter
    
    Returns:
        dict: Dictionary containing distance and metadata with detailed timing
    """
    total_start = time.time()

    # memory tracking: per-phase and total
    mem = {'feature_extraction': None, 'ot': None, 'total': None}
    # handle device whether string or torch.device
    devstr = str(device) if device is not None else ''
    use_cuda = (devstr and 'cuda' in devstr.lower() and torch.cuda.is_available())
    if use_cuda:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    else:
        try:
            cpu_rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            cpu_rss_start = None
    
    # ===== FEATURE EXTRACTOR SETUP & EXTRACTION TIME =====
    feature_start = time.time()
    
    # Fallback to a known MNIST feature extractor if none provided
    if feature_extractor_path is None and dataset == 'MNIST':
        candidate = Path('checkpoints') / 'mnist_cnn_feature_extractor_seed42.pt'
        if candidate.exists():
            feature_extractor_path = str(candidate)

    # Load feature extractor if needed (CIFAR-10 or MNIST)
    embedder = None
    feature_extraction_time = 0.0
    
    if dataset in ('CIFAR_10', 'MNIST') and feature_extractor_path:
        if not os.path.exists(feature_extractor_path):
            raise FileNotFoundError(f"Feature extractor not found: {feature_extractor_path}")
        # Build model architecture depending on dataset
        if dataset == 'CIFAR_10':
            embedder = ResNet(BasicBlock, [2,2,2,2], in_channels=3, num_classes=10).to(device)
            checkpoint = torch.load(feature_extractor_path, map_location=device)
            embedder.load_state_dict(checkpoint['model_state_dict'])
            # remove final classification head — ResNet here names it `linear`
            if hasattr(embedder, 'linear'):
                embedder.linear = torch.nn.Identity()
            elif hasattr(embedder, 'fc'):
                embedder.fc = torch.nn.Identity()
        else:
            embedder = CNN(in_channels=1, num_classes=10).to(device)
            checkpoint = torch.load(feature_extractor_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            else:
                state = checkpoint
            embedder.load_state_dict(state)
            if hasattr(embedder, 'out'):
                embedder.out = torch.nn.Identity()
        embedder.eval()
        for param in embedder.parameters():
            param.requires_grad = False
        print(f"[DEBUG-OT] Loaded embedder from {feature_extractor_path}, type={embedder.__class__.__name__}")
        
        # Time and measure memory for feature extraction on a sample batch
        # This is approximate - actual extraction happens inside OT computation
        sample_batch = next(iter(bootstrap_loader))[0].to(device)
        if use_cuda:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            with torch.no_grad():
                extract_start = time.perf_counter()
                feat = embedder(sample_batch)
                # Debug: report embedding tensor shape
                try:
                    print(f"[DEBUG-OT] embedder output shape={tuple(feat.shape) if hasattr(feat, 'shape') else 'N/A'}")
                except Exception:
                    pass
                try:
                    torch.cuda.synchronize()
                    feat_mem = int(torch.cuda.max_memory_allocated())
                except Exception:
                    feat_mem = None
                feature_extraction_time = time.perf_counter() - extract_start
            print(f"[DEBUG-OT] sample_batch.shape={tuple(sample_batch.shape)} sample_extract_time={feature_extraction_time:.6f}s feat_mem={feat_mem}")
            mem['feature_extraction'] = feat_mem
        else:
            try:
                cpu_feat_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except Exception:
                cpu_feat_start = None
            with torch.no_grad():
                extract_start = time.perf_counter()
                feat = embedder(sample_batch)
                # Debug: report embedding tensor shape
                try:
                    print(f"[DEBUG-OT] embedder output shape={tuple(feat.shape) if hasattr(feat, 'shape') else 'N/A'}")
                except Exception:
                    pass
                feature_extraction_time = time.perf_counter() - extract_start
            try:
                cpu_feat_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem['feature_extraction'] = int((cpu_feat_end - (cpu_feat_start or 0)) * 1024)
            except Exception:
                mem['feature_extraction'] = None

        # Multiply by number of batches for total estimate (rough)
        feature_extraction_time *= len(bootstrap_loader)
        # Compute a sample embedding for validation set to expose embedding shapes
        embed_shape_bootstrap_sample = None
        embed_shape_val_sample = None
        try:
            # collect a single batch from val_loader and extract features (no grad)
            val_sample = next(iter(val_loader))[0].to(device)
            with torch.no_grad():
                val_feat = embedder(val_sample)
            # feat from bootstrap extraction above may exist
            bs_shape = tuple(feat.shape) if 'feat' in locals() and hasattr(feat, 'shape') else None
            val_shape = tuple(val_feat.shape) if hasattr(val_feat, 'shape') else None
            embed_shape_bootstrap_sample = list(bs_shape) if bs_shape is not None else None
            embed_shape_val_sample = list(val_shape) if val_shape is not None else None
            print(f"[DEBUG-OT] embedder sample shapes - bootstrap: {bs_shape}, val: {val_shape}")
        except Exception:
            # best-effort: ignore failures to avoid breaking OT flow
            embed_shape_bootstrap_sample = None
            embed_shape_val_sample = None
    
    feature_setup_time = time.time() - feature_start
    
    # ===== FEATURE COST CREATION =====
    cost_start = time.time()
    
    # Create feature cost for CIFAR-10
    feature_cost = None
    if dataset == 'CIFAR_10' and embedder is not None:
        feature_cost = FeatureCost(src_embedding=embedder,
                                  src_dim=(3,32,32),
                                  tgt_embedding=embedder,
                                  tgt_dim=(3,32,32),
                                  p=2,
                                  device=device)
    elif dataset == 'MNIST' and embedder is not None:
        feature_cost = FeatureCost(src_embedding=embedder,
                                  src_dim=(1,28,28),
                                  tgt_embedding=embedder,
                                  tgt_dim=(1,28,28),
                                  p=2,
                                  device=device)
    if embedder is not None:
        print(f"[DEBUG-OT] FeatureCost will be created and passed to DatasetDistance (embedder present, feature_cost={'set' if feature_cost is not None else 'None'})")
    
    cost_creation_time = time.time() - cost_start
    
    # ===== OT DISTANCE COMPUTATION =====
    ot_start = time.time()
    
    # Compute OT distance
    dist = DatasetDistance(bootstrap_loader, val_loader,
                          inner_ot_method='exact',
                          debiased_loss=True,
                          feature_cost=feature_cost,
                          λ_x=lambda_x,
                          λ_y=lambda_y,
                          sqrt_method='spectral',
                          sqrt_niters=10,
                          precision='single',
                          p=2,
                          entreg=entreg,
                          device=device)
    
    # Calculate distance
    #maxsamples = min(len(bootstrap_loader.dataset), 10000)
    maxsamples = len(bootstrap_loader.dataset)
    print(f"[DEBUG-OT] Initialized DatasetDistance with maxsamples={maxsamples}")

    # Time the actual OT computation
    ot_compute_start = time.perf_counter()

    try:
        print(f"[DEBUG-OT] Calling dist.distance(maxsamples={maxsamples}) feature_cost_present={feature_cost is not None}")
        distance_value = dist.distance(maxsamples=maxsamples)
        coupling = None
    except:
        # Fall back to dual_sol
        coupling, distance_value = dist.dual_sol(maxsamples=maxsamples, return_coupling=True)

    ot_compute_time = time.perf_counter() - ot_compute_start

    # capture OT-phase memory
    if use_cuda:
        try:
            torch.cuda.synchronize()
            mem['ot'] = int(torch.cuda.max_memory_allocated())
        except Exception:
            mem['ot'] = None
    else:
        try:
            cpu_ot_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem['ot'] = int((cpu_ot_end - (cpu_rss_start or 0)) * 1024) if cpu_rss_start is not None else None
        except Exception:
            mem['ot'] = None
    
    # Ensure distance_value is a numeric scalar (handle tensor, numpy, tuple, list)
    def _to_scalar(x):
        import numpy as _np
        # torch tensor
        if isinstance(x, torch.Tensor):
            try:
                if x.numel() == 1:
                    return float(x.item())
                return float(x.mean().item())
            except Exception:
                pass
        # numpy array
        if isinstance(x, (_np.ndarray,)):
            try:
                if _np.size(x) == 1:
                    return float(_np.asarray(x).item())
                return float(_np.mean(x))
            except Exception:
                pass
        # list/tuple
        if isinstance(x, (list, tuple)):
            # try first scalar-like element
            for e in x:
                try:
                    s = _to_scalar(e)
                    if isinstance(s, (int, float)):
                        return float(s)
                except Exception:
                    continue
            # fallback: mean of numeric entries
            try:
                arr = _np.array(x, dtype=float)
                return float(_np.mean(arr))
            except Exception:
                pass
        # scalar numbers
        if isinstance(x, (int, float)):
            return float(x)
        # last resort: try float cast
        try:
            return float(x)
        except Exception:
            return x

    distance_value = _to_scalar(distance_value)
    # ensure numeric scalar; otherwise set NaN to avoid formatting errors
    if not isinstance(distance_value, (int, float, np.floating, np.integer)):
        try:
            distance_value = float(distance_value)
        except Exception:
            distance_value = float('nan')

    ot_total_time = time.time() - ot_start
    total_time = time.time() - total_start

    # capture total peak memory (CUDA peak since function start) or final RSS delta
    if use_cuda:
        try:
            torch.cuda.synchronize()
            total_peak = int(torch.cuda.max_memory_allocated())
            phase_max = max([v for v in (mem.get('feature_extraction'), mem.get('ot')) if isinstance(v, int)], default=0)
            mem['total'] = int(max(total_peak, phase_max))
        except Exception:
            mem['total'] = mem.get('ot') or mem.get('feature_extraction') or None
    else:
        try:
            cpu_rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            total_bytes = int((cpu_rss_end - (cpu_rss_start or 0)) * 1024) if cpu_rss_start is not None else None
            phase_max = max([v for v in (mem.get('feature_extraction'), mem.get('ot')) if isinstance(v, int)], default=0)
            if total_bytes is not None:
                mem['total'] = int(max(total_bytes, phase_max))
            else:
                mem['total'] = phase_max or None
        except Exception:
            mem['total'] = mem.get('ot') or mem.get('feature_extraction') or None
    
    # Return results with detailed timing
    result = {
        'distance': distance_value,
        'timing': {
            'feature_extraction_estimate': feature_extraction_time if dataset == 'CIFAR_10' else 0.0,
            'feature_extraction': feature_extraction_time,
            'feature_setup': feature_setup_time,
            'cost_creation': cost_creation_time,
            'ot_computation': ot_compute_time,
            'ot_total': ot_total_time,
            'total': total_time
        },
        'mem': mem,
        'maxsamples_used': maxsamples,
        'bootstrap_size': len(bootstrap_loader.dataset),
        'val_size': len(val_loader.dataset),
        'dataset': dataset,
        'feature_extractor_used': embedder is not None,
        'embed_shape_bootstrap_sample': embed_shape_bootstrap_sample if 'embed_shape_bootstrap_sample' in locals() else None,
        'embed_shape_val_sample': embed_shape_val_sample if 'embed_shape_val_sample' in locals() else None,
        'lambda_x': lambda_x,
        'lambda_y': lambda_y,
        'entreg': entreg
    }
    
    # Add coupling stats if available
    if coupling is not None:
        if hasattr(coupling, 'shape'):
            result['coupling_shape'] = list(coupling.shape)
        if hasattr(coupling, 'numel'):
            coupling_array = coupling.cpu().numpy() if torch.is_tensor(coupling) else coupling
            result['coupling_sparsity'] = float(np.sum(coupling_array == 0) / coupling_array.size)
    
    return result


def save_ot_results(results, dataset, output_dir):
    """
    Save OT results to CSV with dataset-specific filename.
    
    Args:
        results: List of result dictionaries (each must have 'seed' key)
        dataset: 'CIFAR_10' or 'MNIST'
        output_dir: Output directory path
    """
    if not results:
        print("[WARNING] No results to save")
        return
    
    # Main results CSV - flatten timing for easier analysis
    csv_path = output_dir / f'ot_distances_{dataset.lower()}.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['seed', 'bootstrap_size', 'distance', 
                     'ot_computation_time', 'feature_extraction_time', 'total_time',
                     'memory_total',
                     'maxsamples_used', 'val_size']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {
                'seed': r.get('seed', 'N/A'),
                'bootstrap_size': r.get('bootstrap_size', 'N/A'),
                'distance': r.get('distance', 'N/A'),
                'ot_computation_time': r.get('timing', {}).get('ot_computation', 'N/A'),
                'feature_extraction_time': r.get('timing', {}).get('feature_extraction_estimate', 'N/A'),
                'total_time': r.get('timing', {}).get('total', 'N/A'),
                'memory_total': r.get('mem', {}).get('total', 'N/A'),
                'maxsamples_used': r.get('maxsamples_used', 'N/A'),
                'val_size': r.get('val_size', 'N/A')
            }
            writer.writerow(row)
    
    print(f"\n[INFO] OT distances saved to {csv_path}")
    
    # Save detailed metadata as JSON (with full timing breakdown)
    detailed_path = output_dir / f'ot_distances_{dataset.lower()}_detailed.json'
    
    # Convert non-serializable values
    serializable_results = []
    for r in results:
        r_copy = r.copy()
        for k, v in r_copy.items():
            if isinstance(v, (np.integer, np.floating, np.ndarray)):
                if hasattr(v, 'item'):
                    r_copy[k] = v.item()
                else:
                    r_copy[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, torch.Tensor):
                r_copy[k] = v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, dict) and k == 'timing':
                # Handle timing dict
                r_copy[k] = {tk: float(tv) if isinstance(tv, (int, float)) else tv 
                            for tk, tv in v.items()}
        serializable_results.append(r_copy)
    
    with open(detailed_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"[INFO] Detailed results saved to {detailed_path}")
    
    # Save summary statistics
    distances = [r['distance'] for r in results]
    ot_times = [r['timing']['ot_computation'] for r in results]
    feature_times = [r['timing']['feature_extraction_estimate'] for r in results if r['timing']['feature_extraction_estimate'] > 0]
    total_times = [r['timing']['total'] for r in results]
    
    summary = {
        'dataset': dataset,
        'num_bootstraps': len(results),
        'distance_stats': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances))
        },
        'timing_stats': {
            'ot_computation': {
                'mean': float(np.mean(ot_times)),
                'total': float(np.sum(ot_times))
            },
            'total': {
                'mean': float(np.mean(total_times)),
                'total': float(np.sum(total_times))
            }
        }
    }
    
    # Add feature extraction stats if applicable
    if feature_times:
        summary['timing_stats']['feature_extraction'] = {
            'mean': float(np.mean(feature_times)),
            'total': float(np.sum(feature_times))
        }
    
    summary_path = output_dir / f'ot_distances_{dataset.lower()}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[INFO] Summary statistics saved to {summary_path}")
    
    
def print_timing_summary(results):
    """Print a summary of timing breakdowns"""
    if not results:
        return
    
    print("\n" + "="*60)
    print("TIMING BREAKDOWN SUMMARY")
    print("="*60)
    
    # Calculate averages
    avg_ot = np.mean([r['timing']['ot_computation'] for r in results])
    avg_total = np.mean([r['timing']['total'] for r in results])
    
    print(f"Average OT computation time: {avg_ot:.4f}s")
    print(f"Average total time per bootstrap: {avg_total:.4f}s")
    
    if results[0]['feature_extractor_used']:
        avg_feature = np.mean([r['timing']['feature_extraction_estimate'] for r in results])
        avg_setup = np.mean([r['timing']['feature_setup'] + r['timing']['cost_creation'] for r in results])
        
        print(f"\nFeature extraction (estimated): {avg_feature:.4f}s")
        print(f"Feature setup + cost creation: {avg_setup:.4f}s")
        print(f"OT computation: {avg_ot:.4f}s")
        print(f"  - Percentage of total: {avg_ot/avg_total*100:.1f}%")
    
    print(f"\nTotal time for all bootstraps: {np.sum([r['timing']['total'] for r in results]):.4f}s")
    print("="*60)