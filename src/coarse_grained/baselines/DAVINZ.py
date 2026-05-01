"""
DAVINZ.py - DaVinz baseline metrics
Computes MMD and NTK on bootstrap data vs validation data.
"""
from pathlib import Path
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import resource
import hashlib
import json

from baselines.mmd import rbf_mmd2
from baselines.ntk import compute_ntk_score_batched, compute_ntk_score_batched_permute


def _collect_inputs_labels(loader, device, max_samples):
	inputs = []
	labels = []
	total = 0
	for batch in loader:
		x = batch[0] if isinstance(batch, (list, tuple)) else batch
		y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
		x = x.to(device)
		inputs.append(x)
		if y is not None:
			labels.append(y.to(device))
		total += x.size(0)
		if total >= max_samples:
			break
	if not inputs:
		return torch.empty(0, device=device), torch.empty(0, device=device)
	x = torch.cat(inputs, dim=0)[:max_samples]
	if labels:
		y = torch.cat(labels, dim=0)[:max_samples]
	else:
		y = torch.empty(0, device=device)
	return x, y


def _estimate_sigma(X, Y):
	X = X.reshape(X.size(0), -1).float()
	Y = Y.reshape(Y.size(0), -1).float()
	with torch.no_grad():
		dists = torch.cdist(X, Y, p=2)
		median = torch.median(dists)
	return float(median.item()) if median.numel() > 0 else 1.0



def compute_davinz(
	bootstrap_loader,
	val_loader,
	dataset,
	device='cpu',
	max_samples=50000,
	model=None,
	n_batch=1,
	n_permute=1,
	diagonal_I_mag=1e-4,
	mode='cls',
	feature_extractor_path=None
):
	"""
	Compute DaVinz metrics with explicit MMD and NTK values and timings.
	"""
	total_start = time.time()

	# Collect raw inputs (used for NTK and fallback MMD)
	raw_bx, raw_by = _collect_inputs_labels(bootstrap_loader, device, max_samples)
	raw_vx, _ = _collect_inputs_labels(val_loader, device, max_samples)

	# Debug: sample and hash of bootstrap labels to verify corrupted labels passed
	bootstrap_labels_sample = None
	bootstrap_labels_unique = 0
	bootstrap_labels_hash = None
	if raw_by.numel() > 0:
		try:
			by_cpu = raw_by.detach().cpu()
			n_sample = min(10, by_cpu.size(0))
			bootstrap_labels_sample = by_cpu[:n_sample].numpy().tolist()
			bootstrap_labels_unique = int(len(torch.unique(by_cpu)))
			bootstrap_labels_hash = hashlib.sha1(by_cpu.numpy().tobytes()).hexdigest()
			print(f"[DEBUG] compute_davinz bootstrap labels sample={bootstrap_labels_sample} unique={bootstrap_labels_unique} hash={bootstrap_labels_hash}")
		except Exception as e:
			print(f"[DEBUG] compute_davinz label debug failed: {e}")

	# MMD uses raw inputs (as in the original DaVinz implementation)
	bx_for_mmd = raw_bx
	vx_for_mmd = raw_vx

	# Keep original variables for NTK (unchanged behavior)
	bx = raw_bx
	by = raw_by

	# Check raw inputs exist (NTK uses raw inputs); use raw_vx here
	if raw_bx.numel() == 0 or raw_vx.numel() == 0:
		return {
			'davinz_score': 0.0,
			'mmd': 0.0,
			'ntk': 0.0,
			'mmd_raw': 0.0,
			'mmd_raw_time': 0.0,
			'mmd_raw_mem': 0,
			'timing': {
				'mmd_time': 0.0,
				'ntk_time': 0.0,
				'total': 0.0
			},
			'mem': {'mmd': 0.0, 'ntk': 0.0, 'total': 0.0},
			'n_batch': n_batch,
			'n_permute': n_permute,
			'maxsamples_used': 0,
			'val_size': 0,
			'dataset': dataset,
			'bootstrap_labels_sample': None,
			'bootstrap_labels_unique': 0,
			'bootstrap_labels_hash': None
		}

	# MMD (median heuristic sigma, as in the main process)
	mem = {'mmd': 0.0, 'ntk': 0.0, 'total': 0.0}
	device_str = str(device).lower() if device is not None else 'cpu'
	use_cuda = 'cuda' in device_str
	if use_cuda:
		try:
			torch.cuda.reset_peak_memory_stats()
		except Exception:
			pass

	# First: compute raw-MMD on raw inputs (always compute for later analysis)
	mmd_raw = 0.0
	mmd_raw_time = 0.0
	mmd_raw_mem = 0
	try:
		if use_cuda:
			try:
				torch.cuda.synchronize()
			except Exception:
				pass
		mmd_raw_start = time.time()
		Xr = raw_bx if raw_bx.numel() > 0 else raw_bx
		Yr = raw_vx if raw_vx.numel() > 0 else raw_vx
		# Debug: report basic stats for raw inputs used in raw-MMD
		try:
			print(f"[DEBUG-DATA] raw_bx shape={list(Xr.shape)} min={float(Xr.min()):.6f} max={float(Xr.max()):.6f} mean={float(Xr.mean()):.6f} std={float(Xr.std()):.6f}")
		except Exception:
			pass
		try:
			print(f"[DEBUG-DATA] raw_vx shape={list(Yr.shape)} min={float(Yr.min()):.6f} max={float(Yr.max()):.6f} mean={float(Yr.mean()):.6f} std={float(Yr.std()):.6f}")
		except Exception:
			pass
		sigma_raw = _estimate_sigma(Xr[: min(1000, Xr.size(0))], Yr[: min(1000, Yr.size(0))])
		print(f"[DEBUG] compute_davinz sigma_raw={sigma_raw}")
		# capture feature dimensionality used for raw-MMD (flattened features)
		try:
			Xr_flat = Xr.reshape(Xr.size(0), -1)
			Yr_flat = Yr.reshape(Yr.size(0), -1)
			mmd_raw_feature_dim = int(Xr_flat.size(1)) if Xr_flat.numel() > 0 else 0
		except Exception:
			mmd_raw_feature_dim = 0
		print(f"[DEBUG] compute_davinz mmd_raw_feature_dim={mmd_raw_feature_dim} Xr_n={Xr.shape[0]} Yr_n={Yr.shape[0]}")
		# Debug: show shapes/dtypes passed to rbf_mmd2
		try:
			print(f"[DEBUG-MMD] raw inputs shapes: Xr_flat={list(Xr_flat.shape)} Yr_flat={list(Yr_flat.shape)} dtype={getattr(Xr_flat, 'dtype', 'n/a')} device={getattr(Xr_flat, 'device', 'n/a')}")
		except Exception:
			pass
		mmd_raw_sq = rbf_mmd2(Xr_flat, Yr_flat, sigma=sigma_raw)
		# Debug: show raw MMD squared value and type
		try:
			if isinstance(mmd_raw_sq, torch.Tensor):
				print(f"[DEBUG-MMD] raw_mmd_squared value={float(mmd_raw_sq.item())} dtype={mmd_raw_sq.dtype} device={mmd_raw_sq.device}")
			else:
				print(f"[DEBUG-MMD] raw_mmd_squared numpy value={mmd_raw_sq}")
		except Exception as e:
			print(f"[DEBUG-MMD] failed to print raw_mmd_squared: {e}")
		if use_cuda:
			try:
				torch.cuda.synchronize()
				mmd_raw_mem = int(torch.cuda.max_memory_allocated())
			except Exception:
				mmd_raw_mem = 0
		else:
			cpu_mem_end_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			mmd_raw_mem = int(max(0, cpu_mem_end_raw - cpu_mem_start_mmd) * 1024) if 'cpu_mem_start_mmd' in locals() else 0
		mmd_raw = torch.sqrt(mmd_raw_sq).item()
		mmd_raw_time = time.time() - mmd_raw_start
		print(f"[DEBUG] compute_davinz raw-MMD={mmd_raw:.6f} time_s={mmd_raw_time:.4f} mem_bytes={mmd_raw_mem}")
	except Exception:
		mmd_raw = 0.0
		mmd_raw_time = 0.0
		mmd_raw_mem = 0

	# Then: compute MMD on embeddings if available (or raw inputs if not)
	if not use_cuda:
		cpu_mem_start_mmd = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	try:
		if use_cuda:
			try:
				torch.cuda.synchronize()
			except Exception:
				pass
		mmd_start = time.time()
		# compute sigma on flattened inputs (use embeddings if available)
		X_for_sigma = bx_for_mmd if bx_for_mmd.numel() > 0 else bx_for_mmd
		Y_for_sigma = vx_for_mmd if vx_for_mmd.numel() > 0 else vx_for_mmd
		# Debug: report stats for inputs used in MMD (embeddings or raw)
		try:
			print(f"[DEBUG-DATA] mmd_input X shape={list(X_for_sigma.shape)} min={float(X_for_sigma.min()):.6f} max={float(X_for_sigma.max()):.6f} mean={float(X_for_sigma.mean()):.6f} std={float(X_for_sigma.std()):.6f}")
		except Exception:
			pass
		try:
			print(f"[DEBUG-DATA] mmd_input Y shape={list(Y_for_sigma.shape)} min={float(Y_for_sigma.min()):.6f} max={float(Y_for_sigma.max()):.6f} mean={float(Y_for_sigma.mean()):.6f} std={float(Y_for_sigma.std()):.6f}")
		except Exception:
			pass
		# capture feature dimensionality used for MMD (flattened features or embeddings)
		try:
			X_flat = X_for_sigma.reshape(X_for_sigma.size(0), -1)
			Y_flat = Y_for_sigma.reshape(Y_for_sigma.size(0), -1)
			mmd_feature_dim = int(X_flat.size(1)) if X_flat.numel() > 0 else 0
		except Exception:
			mmd_feature_dim = 0
		print(f"[DEBUG] compute_davinz mmd_feature_dim={mmd_feature_dim} X_n={X_for_sigma.shape[0]} Y_n={Y_for_sigma.shape[0]}")
		try:
			print(f"[DEBUG-MMD] mmd inputs shapes: X_flat={list(X_flat.shape)} Y_flat={list(Y_flat.shape)} dtype={getattr(X_flat, 'dtype', 'n/a')} device={getattr(X_flat, 'device', 'n/a')}")
		except Exception:
			pass
		sigma = _estimate_sigma(X_for_sigma[: min(1000, X_for_sigma.size(0))], Y_for_sigma[: min(1000, Y_for_sigma.size(0))])
		print(f"[DEBUG] compute_davinz sigma={sigma}")
		mmd_squared = rbf_mmd2(X_flat, Y_flat, sigma=sigma)
		# Debug: show MMD squared value and type
		try:
			if isinstance(mmd_squared, torch.Tensor):
				print(f"[DEBUG-MMD] mmd_squared value={float(mmd_squared.item())} dtype={mmd_squared.dtype} device={mmd_squared.device}")
			else:
				print(f"[DEBUG-MMD] mmd_squared numpy value={mmd_squared}")
		except Exception as e:
			print(f"[DEBUG-MMD] failed to print mmd_squared: {e}")
		if use_cuda:
			try:
				torch.cuda.synchronize()
				mmd_mem = int(torch.cuda.max_memory_allocated())
			except Exception:
				mmd_mem = 0
		else:
			cpu_mem_end_mmd = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			mmd_mem = int(max(0, cpu_mem_end_mmd - cpu_mem_start_mmd) * 1024)
		mmd = torch.sqrt(mmd_squared).item()
		mmd_time = time.time() - mmd_start
		mem['mmd'] = mmd_mem
	except Exception:
		mmd = 0.0
		mmd_time = 0.0
		mem['mmd'] = 0

	# NTK
	ntk_start = time.time()
	cpu_mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	if use_cuda:
		try:
			torch.cuda.reset_peak_memory_stats()
		except Exception:
			pass
	if by.numel() == 0:
		ntk = 0.0
	else:
		# If no model provided, construct a simple linear classifier over flattened inputs
		if model is None:
			num_classes = int(by.max().item()) + 1
			input_dim = int(np.prod(bx.shape[1:]))
			model = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, num_classes)).to(device)
		print(f"[DEBUG] compute_davinz NTK using model: {model.__class__.__name__} with n_batch={n_batch} and n_permute={n_permute} with diagonal_I_mag={diagonal_I_mag}")
		ntk, _ = compute_ntk_score_batched(
			model,
			bx,
			by.long() if mode == 'cls' else by,
			mode=mode,
			n_batch=n_batch,
			use_hack=True,
			diagonal_I_mag=diagonal_I_mag,
			get_eigen=False		)
	ntk_time = time.time() - ntk_start
	if use_cuda:
		try:
			mem_ntk = int(torch.cuda.max_memory_allocated())
		except Exception:
			mem_ntk = 0
		mem['ntk'] = mem_ntk
		mem['total'] = int(max(mem.get('mmd', 0), mem_ntk))
	else:
		cpu_mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		mem_ntk = int(max(0, cpu_mem_end - cpu_mem_start) * 1024)
		mem['ntk'] = mem_ntk
		mem['total'] = int(max(mem.get('mmd', 0), mem_ntk))

	# Do not compute per-run kappa/davinz here. Use group-level kappa
	# computed in save_davinz_results to derive per-bootstrap davinz scores.
	# Return None for per-run davinz_score so downstream aggregation uses group kappa.
	davinz_score = None

	# Safe numeric conversions for return
	try:
		mmd_val = float(mmd)
		if not np.isfinite(mmd_val):
			mmd_val = 0.0
	except Exception:
		mmd_val = 0.0

	try:
		ntk_val = float(ntk)
		if not np.isfinite(ntk_val):
			ntk_val = 0.0
	except Exception:
		ntk_val = 0.0

	return {
		'mmd': float(mmd_val),
		'ntk': float(ntk_val),
		'mmd_raw': float(mmd_raw),
		'mmd_raw_feature_dim': int(mmd_raw_feature_dim) if 'mmd_raw_feature_dim' in locals() else 0,
		'mmd_raw_time': float(mmd_raw_time),
		'mmd_raw_mem': int(mmd_raw_mem),
		'mmd_feature_dim': int(mmd_feature_dim) if 'mmd_feature_dim' in locals() else 0,
		'davinz_score': davinz_score,
		'mem': mem,
		'n_batch': n_batch,
		'n_permute': n_permute,
		'timing': {
			'mmd_time': float(mmd_time) if 'mmd_time' in locals() else 0.0,
			'ntk_time': float(ntk_time) if 'ntk_time' in locals() else 0.0,
			'total': float(time.time() - total_start)
		},
		'maxsamples_used': int(min(bx_for_mmd.size(0), vx_for_mmd.size(0))) if 'bx_for_mmd' in locals() and 'vx_for_mmd' in locals() else 0,
		'val_size': int(vx_for_mmd.size(0)) if 'vx_for_mmd' in locals() else 0,
		'dataset': dataset,
		'bootstrap_labels_sample': bootstrap_labels_sample,
		'bootstrap_labels_unique': int(bootstrap_labels_unique),
		'bootstrap_labels_hash': bootstrap_labels_hash
	}


def save_davinz_results(results, dataset, output_dir,n_batch=1, n_permute=1, diagonal_I_mag=1e-2,
                    mode='reg'):
	"""Save davinz results to CSV only (no summaries)."""
	if not results:
		print("[WARNING] No davinz results to save")
		return

	# Group results by seed and scenario (if scenario missing, use 'all')
	groups = {}
	for r in results:
		seed = r.get('seed', 'all')
		scenario = r.get('scenario', 'all')
		key = (seed, scenario)
		groups.setdefault(key, []).append(r)

	# compute kappa per group
	kappas = {}
	for key, rows in groups.items():
		mmds = np.array([row.get('mmd', np.nan) for row in rows], dtype=float)
		ntks = np.array([row.get('ntk', np.nan) for row in rows], dtype=float)
		valid_mmds = mmds[np.isfinite(mmds)] if mmds.size > 0 else np.array([])
		mean_mmd = float(np.mean(valid_mmds)) if valid_mmds.size > 0 else 0.0
		valid_ntks = ntks[np.isfinite(ntks) & (ntks != 0)] if ntks.size > 0 else np.array([])
		mean_ntk = float(np.mean(valid_ntks)) if valid_ntks.size > 0 else 0.0
		kappa = (mean_mmd / mean_ntk) if mean_ntk not in (0.0, None) else 0.0
		kappas[key] = {'kappa': float(kappa), 'mean_mmd': float(mean_mmd), 'mean_ntk': float(mean_ntk), 'count': len(rows)}

	# Prepare output folder under output_dir/davinz

	output_dir = Path(output_dir) / 'davinz_n_batch_{}_n_permute_{}_diagI_{}'.format(n_batch, n_permute, diagonal_I_mag)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Build aggregated CSV rows and timing/memory JSON
	csv_path = output_dir / f"davinz_results_{dataset.lower()}.csv"
	timing_path = output_dir / f"davinz_timing_{dataset.lower()}.json"

	fieldnames = [
		'seed', 'scenario', 'bootstrap_size', 'mmd', 'ntk', 'davinz_score', 'kappa',
		'maxsamples_used', 'val_size', 'dataset', 'bootstrap_labels_hash', 'bootstrap_labels_unique', 'timestamp'
	]

	rows = []
	timing_list = []
	for r in results:
		seed = r.get('seed', 'all')
		scenario = r.get('scenario', 'all')
		meta = kappas.get((seed, scenario), {'kappa': 0.0, 'mean_mmd': 0.0, 'mean_ntk': 0.0})
		mmd = float(r.get('mmd', 0.0) or 0.0)
		ntk = float(r.get('ntk', 0.0) or 0.0)
		kappa_val = float(meta.get('kappa', 0.0) or 0.0)
		# per-bootstrap davinz score using group kappa
		davinz_score = - (kappa_val * ntk + mmd)
		rows.append({
			'seed': seed,
			'scenario': scenario,
			'bootstrap_size': r.get('bootstrap_size'),
			'mmd': mmd,
			'ntk': ntk,
			'davinz_score': float(davinz_score),
			'kappa': kappa_val,
			'maxsamples_used': r.get('maxsamples_used'),
			'val_size': r.get('val_size'),
			'dataset': r.get('dataset', dataset),
			'bootstrap_labels_hash': r.get('bootstrap_labels_hash'),
			'bootstrap_labels_unique': int(r.get('bootstrap_labels_unique', 0) or 0),
			'timestamp': r.get('timing', {}).get('total', time.time())
		})
		# collect timing/memory info separately
		timing_entry = {
			'seed': seed,
			'scenario': scenario,
			'bootstrap_size': r.get('bootstrap_size'),
			'timing': r.get('timing', {}),
			'mem': r.get('mem', {}),
			'dataset': r.get('dataset', dataset)
		}
		timing_list.append(timing_entry)

	# write CSV
	with open(csv_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)

	# write timing JSON
	try:
		with open(timing_path, 'w') as jf:
			json.dump(timing_list, jf, indent=2)
	except Exception as e:
		print(f"[WARN] Failed to write timing JSON: {e}")

	print(f"[INFO] Saved davinz CSV ({len(rows)} rows) to {csv_path}")
	print(f"[INFO] Saved davinz timing JSON ({len(timing_list)} entries) to {timing_path}")
	# print per-group summaries
	for key, info in kappas.items():
		s_seed, s_scenario = key
		print(f"[INFO] Group kappa seed={s_seed} scenario={s_scenario}: kappa={info['kappa']:.6f} (mean_mmd={info['mean_mmd']:.6f}, mean_ntk={info['mean_ntk']:.6f}, count={info['count']})")

