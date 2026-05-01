"""
RV.py - Volume and Robust Volume baseline metrics
Extracted main computation parts from robust volume baseline.
"""
import os
import time
import csv
import json
from collections import defaultdict, Counter
from math import ceil, floor
import numpy as np
import torch
import resource
from sklearn.preprocessing import StandardScaler
from model.resnet import BasicBlock, ResNet
from model.cnn import CNN
from pathlib import Path


def _collect_features(loader, device, max_samples):
	features = []
	total = 0
	for batch in loader:
		x = batch[0] if isinstance(batch, (list, tuple)) else batch
		x = x.to(device)
		x = x.view(x.size(0), -1)
		features.append(x)
		total += x.size(0)
		if total >= max_samples:
			break
	if not features:
		return torch.empty(0, device=device)
	feats = torch.cat(features, dim=0)
	return feats[:max_samples]


def _extract_embeddings(loader, embedder, device, max_samples):
	features = []
	total = 0
	extract_time = 0.0
	for batch in loader:
		x = batch[0] if isinstance(batch, (list, tuple)) else batch
		x = x.to(device)
		with torch.no_grad():
			start = time.perf_counter()
			feat = embedder(x)
			extract_time += time.perf_counter() - start
		features.append(feat)
		total += feat.size(0)
		if total >= max_samples:
			break
	if not features:
		return torch.empty(0, device=device), 0.0
	feats = torch.cat(features, dim=0)
	return feats[:max_samples], extract_time


def scale_normal(dataset):
	"""Scale features to standard normal distribution using torch (keeps data on GPU).
	This avoids CPU round-trips and sklearn dependence.
	"""
	# dataset expected shape (N, D) or (N, ...). We normalize over N (samples)
	dev = dataset.device
	X = dataset
	# Flatten features per-sample
	N = X.shape[0]
	flat = X.reshape(N, -1)
	mean = flat.mean(dim=0, keepdim=True)
	# use population std (unbiased=False) to avoid NaNs for small N
	std = flat.std(dim=0, unbiased=False, keepdim=True)
	# avoid zero std
	std[std == 0] = 1.0
	trans = (flat - mean) / std
	trans = trans.reshape(N, *X.shape[1:])
	return trans.to(dev)


# def compute_volume(dataset):
# 	"""Return the log-determinant of Gram = X^T X (i.e., log|X^T X|).
# 	Uses torch.linalg.slogdet when available for stability, falls back to SVD.
# 	"""
# 	Gram = dataset.t() @ dataset
# 	# print(Gram.device)
# 	# # Add regularization to make it positive definite
# 	# eps = 1e-6
# 	# I = torch.eye(Gram.shape[0], device=Gram.device)
# 	# # Use relative regularization based on trace
# 	# reg_param = eps * torch.trace(Gram) / Gram.shape[0]
# 	# Gram_reg = Gram + reg_param * I

# 	# try slogdet for stability
# 	try:
# 		sign, logabsdet = torch.linalg.slogdet(Gram)
# 		# if any sign <= 0, determinant non-positive -> run debug and return NaN
# 		if isinstance(sign, torch.Tensor):
# 			if (sign <= 0).any():
# 				compute_volume_debug(dataset)
# 				return torch.tensor(float('nan'))
# 			return logabsdet
# 		else:
# 			if sign <= 0:
# 				compute_volume_debug(dataset)
# 				return torch.tensor(float('nan'))
# 			return torch.tensor(logabsdet)
# 	except Exception:
# 		# fallback/debug: inspect dataset to find NaN/instability source
# 		try:
# 			return compute_volume_debug(dataset)
# 		except Exception:
# 			# as a final fallback try SVD logdet and if that fails return NaN
# 			try:
# 				sv = torch.linalg.svdvals(dataset)
# 				sv_pos = sv[sv > 0]
# 				if sv_pos.numel() == 0:
# 					return torch.tensor(float('nan'))
# 				logdet = 2.0 * torch.log(sv_pos).sum()
# 				return logdet
# 			except Exception:
# 				return torch.tensor(float('nan'))


def compute_volume_(dataset):
    """
    Return the log-determinant of Gram = X^T X (i.e., log|X^T X|).
    Full debug mode enabled.
    """

    print("\n================ DEBUG VOLUME ================")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset device: {dataset.device}")
    print(f"dataset.mean(): {dataset.mean().item():.6e}")
    print(f"dataset.std():  {dataset.std().item():.6e}")
    print(f"dataset.min():  {dataset.min().item():.6e}")
    print(f"dataset.max():  {dataset.max().item():.6e}")
    print(f"dataset.norm(): {torch.norm(dataset).item():.6e}")

    # Compute Gram
    Gram = dataset.t() @ dataset
    print(f"Gram shape: {Gram.shape}")
    print(f"Gram trace: {torch.trace(Gram).item():.6e}")

    # Rank estimate
    rank = torch.linalg.matrix_rank(Gram)
    print(f"Gram rank: {rank} / {Gram.shape[0]}")

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(Gram)
    print(f"Eigen min: {eigvals.min().item():.6e}")
    print(f"Eigen max: {eigvals.max().item():.6e}")

    # Condition number
    if eigvals.min() > 0:
        cond_number = eigvals.max() / eigvals.min()
        print(f"Condition number: {cond_number.item():.6e}")
    else:
        print("Condition number: INF (singular matrix)")

    # Add regularization
    eps = 1e-6
    I = torch.eye(Gram.shape[0], device=Gram.device)

    reg_param = eps * torch.trace(Gram) / Gram.shape[0]
    print(f"Regularization parameter: {reg_param.item():.6e}")

    Gram_reg = Gram + reg_param * I

    # Determinant via slogdet
    sign, logabsdet = torch.linalg.slogdet(Gram_reg)

    print(f"[DEBUG-VOL] slogdet sign: {sign.item()}")
    print(f"[DEBUG-VOL] logabsdet: {logabsdet.item():.6e}")

    print("==============================================\n")

    return logabsdet

def compute_volume(dataset):
	U, S, V = torch.linalg.svd(dataset, full_matrices=False)
	logdet = 2 * torch.sum(torch.log(S + 1e-12))
	return logdet


def compute_volume_debug(dataset):
	"""Debug version to identify NaN source in volume computation."""
	try:
		print(f"[DEBUG-VOL] Dataset shape: {tuple(dataset.shape)}")
		print(f"[DEBUG-VOL] Dataset dtype: {dataset.dtype}")
		print(f"[DEBUG-VOL] Dataset min/max: {float(dataset.min()):.6f}/{float(dataset.max()):.6f}")
	except Exception as e:
		print(f"[DEBUG-VOL] Failed basic checks: {e}")

	# Check singular values and rank
	try:
		sv = torch.linalg.svdvals(dataset)
		print(f"[DEBUG-VOL] Singular values (first 10): {sv[:10].tolist()}")
		if sv.numel() > 1 and sv[-1] != 0:
			cond = float(sv[0] / (sv[-1] + 1e-30))
			print(f"[DEBUG-VOL] Approx condition number: {cond:.2e}")
		rank = (sv > 1e-10).sum().item()
		print(f"[DEBUG-VOL] Rank (sv>1e-10): {rank} / {min(dataset.shape)}")
		if rank < min(dataset.shape):
			print(f"[WARNING] Matrix is rank-deficient! Rank={rank}")
	except Exception as e:
		print(f"[DEBUG-VOL] SVD failed: {e}")

	# Gram diagnostics
	try:
		Gram = dataset.t() @ dataset
		print(f"[DEBUG-VOL] Gram shape: {tuple(Gram.shape)}")
		try:
			condg = torch.linalg.cond(Gram)
			print(f"[DEBUG-VOL] Gram cond: {float(condg):.2e}")
		except Exception:
			pass
		try:
			eigs = torch.linalg.eigvalsh(Gram)
			print(f"[DEBUG-VOL] Gram min eigenvalue: {float(eigs.min()):.6e}")
		except Exception:
			pass
		# slogdet attempt
		try:
			sign, logabsdet = torch.linalg.slogdet(Gram)
			print(f"[DEBUG-VOL] slogdet sign: {sign} logabsdet: {logabsdet}")
			if isinstance(sign, torch.Tensor):
				if (sign <= 0).any():
					print(f"[ERROR] Non-positive determinant sign: {sign}")
					return torch.tensor(float('nan'))
				return logabsdet
			else:
				if sign <= 0:
					print(f"[ERROR] Non-positive determinant sign: {sign}")
					return torch.tensor(float('nan'))
				return logabsdet
		except Exception as e:
			print(f"[ERROR] slogdet failed: {e}")
			return torch.tensor(float('nan'))
	except Exception as e:
		print(f"[DEBUG-VOL] Gram diagnostics failed: {e}")
		return torch.tensor(float('nan'))


def compute_X_tilde_and_counts(X, omega):
	"""
	Compresses the original feature matrix X to X_tilde with the specified omega.
	Uses torch ops for index computation to remain on the GPU; small tuple
	conversions for dict keys are performed per-hypercube.
	Returns: X_tilde and cube counts.
	"""
	cubes = Counter()
	Omega = defaultdict(list)

	# min per-dimension
	min_ds = torch.min(X, dim=0).values

	# compute integer cube indices per sample (on same device as X)
	# offset and scaled indices
	offset = X - min_ds
	# use torch.floor on GPU, then convert to integer type
	idxs = torch.floor(offset / float(omega)).to(torch.int64)

	# iterate rows to build hypercube buckets (tuple conversion implies small CPU transfers)
	for i in range(idxs.size(0)):
		key = tuple(int(x) for x in idxs[i].tolist())
		cubes[key] += 1
		Omega[key].append(X[i])

	# build X_tilde by averaging tensors (keeps tensors on device)
	X_tilde = torch.stack([
		torch.stack(list(value), dim=0).mean(dim=0) for _, value in Omega.items()
	]) if len(Omega) > 0 else torch.empty((0, X.size(1)), device=X.device)

	return X_tilde, cubes



def compute_robust_volume(X_tilde, hypercubes, alpha=None):
	"""
	Compute robust volume for given X_tilde and hypercube counts.

	If `alpha` is None, use the legacy default alpha = 1.0 / (10 * n).
	Otherwise use the provided alpha value.
	"""
	n = len(X_tilde)
	if alpha is None:
		# legacy default behaviour
		alpha = 1.0 / (10 * n) if n > 0 else 1.0
	else:
		alpha = 1.0/( alpha * n)

	# compute log-volume of X_tilde
	log_volume = compute_volume(X_tilde)

	# compute log of rho_omega_prod (sum of logs)
	log_rho_sum = 0.0
	for _, freq_count in hypercubes.items():
		rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)
		# rho_omega should be positive; guard
		if rho_omega <= 0:
			# non-positive, mark NaN
			return torch.tensor(float('nan'))
		log_rho_sum += float(np.log(rho_omega))

	# robust log-volume = log_volume + log_rho_sum
	try:
		if isinstance(log_volume, torch.Tensor):
			return log_volume + float(log_rho_sum)
		else:
			return torch.tensor(float(log_volume) + float(log_rho_sum))
	except Exception:
		return torch.tensor(float('nan'))


def compute_rv_metric(
	bootstrap_loader,
	dataset,
	device='cpu',
	feature_extractor_path=None,
	max_samples=10000,
	omega=0.1,
	alpha=None
):
	"""
	Compute volume and robust volume metrics for a bootstrap dataset.
	If dataset is CIFAR_10 and feature_extractor_path is provided, embeddings
	are extracted before computing volume metrics.
	"""
	total_start = time.time()
	# memory tracking: per-phase and total
	mem = {'feature_extraction': None, 'rv': None, 'total': None}
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

	embedder = None
	feature_extraction_time = 0.0

	# Fallback to a known MNIST feature extractor if none provided
	if feature_extractor_path is None and dataset == 'MNIST':
		candidate = Path('checkpoints') / 'mnist_cnn_feature_extractor_seed42.pt'
		if candidate.exists():
			feature_extractor_path = str(candidate)

	if dataset in ('CIFAR_10', 'MNIST') and feature_extractor_path:
		if not os.path.exists(feature_extractor_path):
			raise FileNotFoundError(f"Feature extractor not found: {feature_extractor_path}")
		# Choose architecture based on dataset
		if dataset == 'CIFAR_10':
			embedder = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10).to(device)
			checkpoint = torch.load(feature_extractor_path, map_location=device)
			embedder.load_state_dict(checkpoint['model_state_dict'])
			# ResNet implementation uses `linear` as final layer — replace it
			if hasattr(embedder, 'linear'):
				embedder.linear = torch.nn.Identity()
			elif hasattr(embedder, 'fc'):
				embedder.fc = torch.nn.Identity()
		else:
			# MNIST: simple CNN feature extractor
			embedder = CNN(in_channels=1, num_classes=10).to(device)
			checkpoint = torch.load(feature_extractor_path, map_location=device)
			# Some checkpoints store full metrics dict like main.py saves
			if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
				state = checkpoint['model_state_dict']
			else:
				state = checkpoint
			embedder.load_state_dict(state)
			# replace final classification layer with identity to get features
			if hasattr(embedder, 'out'):
				embedder.out = torch.nn.Identity()
		embedder.eval()
		for param in embedder.parameters():
			param.requires_grad = False
		print(f"[DEBUG-RV] Loaded embedder from {feature_extractor_path}, type={embedder.__class__.__name__}")

	if embedder is not None:
		# measure embedding extraction memory/time
		if use_cuda:
			try:
				torch.cuda.reset_peak_memory_stats()
			except Exception:
				pass
			try:
				torch.cuda.synchronize()
			except Exception:
				pass
			feats, feature_extraction_time = _extract_embeddings(
				bootstrap_loader, embedder, device, max_samples
			)
			print(f"[DEBUG-RV] Extracted embeddings: feats.shape={tuple(feats.shape) if hasattr(feats, 'shape') else 'N/A'} extract_time={feature_extraction_time:.6f}s")
			try:
				torch.cuda.synchronize()
				mem['feature_extraction'] = int(torch.cuda.max_memory_allocated())
			except Exception:
				mem['feature_extraction'] = None
		else:
			try:
				cpu_feat_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			except Exception:
				cpu_feat_start = None
			feats, feature_extraction_time = _extract_embeddings(
				bootstrap_loader, embedder, device, max_samples
			)
			try:
				cpu_feat_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
				mem['feature_extraction'] = int((cpu_feat_end - (cpu_feat_start or 0)) * 1024)
			except Exception:
				mem['feature_extraction'] = None
	else:
		feats = _collect_features(bootstrap_loader, device, max_samples)
		print(f"[DEBUG-RV] Using raw features (no embedder). feats.shape={tuple(feats.shape) if hasattr(feats, 'shape') else 'N/A'}")

	if feats.numel() == 0:
		return {
			'volume': 0.0,
			'robust_volume': 0.0,
			'log_volume': float('nan'),
			'log_robust_volume': float('nan'),
			'timing': {
				'feature_extraction': 0.0,
				'rv_computation': 0.0,
				'total': 0.0
			},
			'maxsamples_used': 0,
			'dataset': dataset
		}

	rv_start = time.time()

	scaled_features = scale_normal(feats)
	non_empty_mask = scaled_features.abs().sum(dim=0).bool()
	scaled_features = scaled_features[:, non_empty_mask]

	# compute log-volume and log-robust-volume (compute_volume/compute_robust_volume
	# return logarithmic values for stability)
	log_volume = compute_volume(scaled_features)
	X_tilde, cubes = compute_X_tilde_and_counts(scaled_features, omega=omega)
	# Pass alpha through to compute_robust_volume (None -> legacy default)
	log_robust_volume = compute_robust_volume(X_tilde, cubes, alpha=alpha)

	# Debug: shapes and counts
	print(f"[DEBUG-RV] scaled_features.shape={tuple(scaled_features.shape)} non_empty_dims={int(non_empty_mask.sum())}")
	print(f"[DEBUG-RV] X_tilde.shape={tuple(X_tilde.shape) if hasattr(X_tilde, 'shape') else 'N/A'} hypercube_count={len(cubes)}")
	try:
		top_cubes = sorted(cubes.items(), key=lambda kv: kv[1], reverse=True)[:5]
		#print(f"[DEBUG-RV] top hypercube freqs: {[ (k,v) for k,v in top_cubes ]}")
	except Exception:
		pass
	# Convert tensors to floats for logging
	try:
		lv = float(log_volume.item()) if hasattr(log_volume, 'item') else float(log_volume)
	except Exception:
		lv = float('nan')
	try:
		lrv = float(log_robust_volume.item()) if hasattr(log_robust_volume, 'item') else float(log_robust_volume)
	except Exception:
		lrv = float('nan')
	print(f"[DEBUG-RV] log_volume={lv} log_robust_volume={lrv}")

	rv_time = time.time() - rv_start
	total_time = time.time() - total_start

	# capture memory after computation
	if use_cuda:
		try:
			torch.cuda.synchronize()
			mem['rv'] = int(torch.cuda.max_memory_allocated())
			phase_max = max([v for v in (mem.get('feature_extraction'), mem.get('rv')) if isinstance(v, int)], default=0)
			mem['total'] = int(max(mem['rv'], phase_max))
		except Exception:
			mem['rv'] = mem.get('feature_extraction') or None
			mem['total'] = mem['rv']
	else:
		try:
			cpu_rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			rv_bytes = int(cpu_rss_end - (cpu_rss_start or 0)) if cpu_rss_start is not None else None
			mem['rv'] = rv_bytes
			phase_max = max([v for v in (mem.get('feature_extraction'), mem.get('rv')) if isinstance(v, int)], default=0)
			mem['total'] = int(phase_max) if phase_max else None
		except Exception:
			mem['rv'] = mem.get('feature_extraction') or None
			mem['total'] = mem['rv']

	return {
		# keep raw volume fields as NaN to avoid expensive/unstable exponentiation;
		# prefer `log_volume` and `log_robust_volume` for downstream analysis
		'volume': float('nan'),
		'robust_volume': float('nan'),
		'log_volume': lv,
		'log_robust_volume': lrv,
		'timing': {
			'feature_extraction': float(feature_extraction_time),
			'rv_computation': float(rv_time),
			'total': float(total_time)
		},
		'mem': mem,
		'maxsamples_used': int(scaled_features.size(0)),
		'dataset': dataset,
		'feature_extractor_used': embedder is not None,
		'hypercube_count': len(cubes) if 'cubes' in locals() else 0
	}


def save_rv_results(results, dataset, output_dir):
	"""Save RV results to CSV and JSON summaries."""
	if not results:
		print("[WARNING] No RV results to save")
		return

	csv_path = output_dir / f'rv_metrics_{dataset.lower()}.csv'
	with open(csv_path, 'w', newline='') as f:
		fieldnames = [
				'seed', 'bootstrap_size', 'volume', 'log_volume', 'robust_volume', 'log_robust_volume',
				'feature_extraction_time', 'rv_computation_time', 'total_time',
				'memory_total',
				'maxsamples_used'
			]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in results:
			row = {
				'seed': r.get('seed', 'N/A'),
				'bootstrap_size': r.get('bootstrap_size', 'N/A'),
				'volume': r.get('volume', 'N/A'),
				'log_volume': r.get('log_volume', 'N/A'),
				'robust_volume': r.get('robust_volume', 'N/A'),
				'log_robust_volume': r.get('log_robust_volume', 'N/A'),
				'feature_extraction_time': r.get('timing', {}).get('feature_extraction', 'N/A'),
				'rv_computation_time': r.get('timing', {}).get('rv_computation', 'N/A'),
				'total_time': r.get('timing', {}).get('total', 'N/A'),
				'memory_total': r.get('mem', {}).get('rv', 'N/A'),
				'maxsamples_used': r.get('maxsamples_used', 'N/A')
			}
			writer.writerow(row)

	print(f"\n[INFO] RV metrics saved to {csv_path}")

	detailed_path = output_dir / f'rv_metrics_{dataset.lower()}_detailed.json'
	serializable_results = []
	for r in results:
		r_copy = r.copy()
		for k, v in r_copy.items():
			if isinstance(v, (np.integer, np.floating, np.ndarray)):
				r_copy[k] = v.item() if hasattr(v, 'item') else float(v)
			elif isinstance(v, torch.Tensor):
				r_copy[k] = v.item() if v.numel() == 1 else v.tolist()
		serializable_results.append(r_copy)

	with open(detailed_path, 'w') as f:
		json.dump(serializable_results, f, indent=2)

	# Use log-volume fields for summary statistics (more stable)
	volumes = [float(r.get('log_volume')) for r in results if isinstance(r.get('log_volume'), (int, float)) and np.isfinite(r.get('log_volume'))]
	robust_volumes = [float(r.get('log_robust_volume')) for r in results if isinstance(r.get('log_robust_volume'), (int, float)) and np.isfinite(r.get('log_robust_volume'))]
	rv_times = [
		r.get('timing', {}).get('rv_computation')
		for r in results
		if isinstance(r.get('timing', {}).get('rv_computation'), (int, float))
	]
	total_times = [
		r.get('timing', {}).get('total')
		for r in results
		if isinstance(r.get('timing', {}).get('total'), (int, float))
	]
	feature_times = [
		r.get('timing', {}).get('feature_extraction')
		for r in results
		if isinstance(r.get('timing', {}).get('feature_extraction'), (int, float))
	]

	summary = {
		'dataset': dataset,
		'num_bootstraps': len(results),
		'volume': {
			'mean': float(np.mean(volumes)) if volumes else 0.0,
			'std': float(np.std(volumes)) if volumes else 0.0
		},
		'robust_volume': {
			'mean': float(np.mean(robust_volumes)) if robust_volumes else 0.0,
			'std': float(np.std(robust_volumes)) if robust_volumes else 0.0
		},
		'timing': {
			'rv_computation': {
				'mean': float(np.mean(rv_times)) if rv_times else 0.0,
				'total': float(np.sum(rv_times)) if rv_times else 0.0
			},
			'total': {
				'mean': float(np.mean(total_times)) if total_times else 0.0,
				'total': float(np.sum(total_times)) if total_times else 0.0
			},
			'feature_extraction': {
				'mean': float(np.mean(feature_times)) if feature_times else 0.0,
				'total': float(np.sum(feature_times)) if feature_times else 0.0
			}
		}
	}

	summary_path = output_dir / f'rv_metrics_{dataset.lower()}_summary.json'
	with open(summary_path, 'w') as f:
		json.dump(summary, f, indent=2)

	print(f"[INFO] RV summary saved to {summary_path}")

	# Also write an aggregated CSV under outputs/RV with full details
	rv_out = output_dir / 'RV'
	rv_out.mkdir(parents=True, exist_ok=True)
	rv_csv = rv_out / f"rv_results_{dataset.lower()}.csv"
	import csv as _csv
	with open(rv_csv, 'w', newline='') as f:
		# include timing and mem as separate fields and also JSON blobs for full details
		fieldnames = [
			'seed', 'bootstrap_size', 'dataset', 'volume', 'log_volume', 'robust_volume', 'log_robust_volume',
			'feature_extraction_time_s', 'rv_computation_time_s', 'total_time_s',
			'memory_total', 'timing_json', 'mem_json', 'maxsamples_used',
			'feature_extractor_used', 'timestamp'
		]
		writer = _csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in results:
			timing = r.get('timing', {}) if isinstance(r, dict) else {}
			mem = r.get('mem', {}) if isinstance(r, dict) else {}
			row = {
				'seed': r.get('seed', 'N/A'),
				'bootstrap_size': r.get('bootstrap_size', 'N/A'),
				'dataset': r.get('dataset', dataset),
				'volume': r.get('volume', 'N/A'),
				'log_volume': r.get('log_volume', 'N/A'),
				'robust_volume': r.get('robust_volume', 'N/A'),
				'log_robust_volume': r.get('log_robust_volume', 'N/A'),
				'feature_extraction_time_s': timing.get('feature_extraction', 'N/A'),
				'rv_computation_time_s': timing.get('rv_computation', 'N/A'),
				'total_time_s': timing.get('total', 'N/A'),
				'memory_total': mem.get('rv', np.nan),
				'timing_json': json.dumps(timing),
				'mem_json': json.dumps(mem),
				'maxsamples_used': r.get('maxsamples_used', 'N/A'),
				'feature_extractor_used': r.get('feature_extractor_used', False),
				'timestamp': r.get('timestamp', time.time())
			}
			writer.writerow(row)

	print(f"[INFO] RV aggregated CSV written to {rv_csv}")
