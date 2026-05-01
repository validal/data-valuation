import os
import time
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import warnings
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils import load_dataset_cls
from baselines import OT, RV, DAVINZ


NOISE_LEVELS_DEFAULT = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class Config:
    dataset: str
    gpu: Optional[str] = '0'
    seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: str = 'outputs'
    base_size: int = 1000
    noise_levels: List[float] = field(default_factory=lambda: NOISE_LEVELS_DEFAULT)
    max_samples: int = 10000
    val_seed: int = 0


class RobustnessTester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._select_device(cfg.gpu)
        self.output_root = Path(cfg.output_dir) / cfg.dataset / 'robustness'
        self.output_root.mkdir(parents=True, exist_ok=True)

        # will be set in load_data()
        self.train_inputs = None
        self.train_labels = None
        self.val_idx = None
        self.dims = None

    def _select_device(self, gpu_arg: Optional[str]):
        gpu_list = [int(x) for x in str(gpu_arg).split(',') if x.strip() != ''] if gpu_arg else []
        if torch.cuda.is_available() and gpu_list:
            try:
                torch.cuda.set_device(int(gpu_list[0]))
                return torch.device(f'cuda:{int(gpu_list[0])}')
            except Exception:
                return torch.device('cuda')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------
    # Data loading / splits
    # -------------------------
    def load_data(self):
        ti, tl, test_i, test_l, dims, _ = load_dataset_cls(
            self.cfg.dataset,
            trim_dataset=50000 if self.cfg.dataset == 'CIFAR_10' else 60000,
            num_parties=10,
        )
        if not torch.is_tensor(ti):
            ti = torch.tensor(ti, dtype=torch.float32)
        if not torch.is_tensor(tl):
            tl = torch.tensor(tl, dtype=torch.long)

        self.train_inputs = ti.view(-1, *dims)
        self.train_labels = tl
        self.dims = dims

        # fixed stratified validation split using cfg.val_seed
        rng = np.ran_ndom.RandomState(self.cfg.val_seed)
        num_classes = 10
        samples_per_class = 1000
        class_indices = {i: [] for i in range(num_classes)}
        for idx in range(len(self.train_labels)):
            class_indices[int(self.train_labels[idx].item())].append(idx)

        train_idx = []
        val_idx = []
        for class_id in range(num_classes):
            lst = class_indices[class_id]
            rng.shuffle(lst)
            val_idx.extend(lst[:samples_per_class])
            train_idx.extend(lst[samples_per_class:])

        self.train_idx = train_idx
        self.val_idx = val_idx

    # -------------------------
    # Base dataset sampling
    # -------------------------
    def sample_base_dataset(self, seed: int):
        rng = np.random.RandomState(seed)
        if len(self.train_idx) < self.cfg.base_size:
            indices = rng.choice(self.train_idx, size=self.cfg.base_size, replace=True)
        else:
            indices = rng.choice(self.train_idx, size=self.cfg.base_size, replace=False)
        inputs = self.train_inputs[indices]
        labels = self.train_labels[indices]
        return inputs, labels, indices

    # -------------------------
    # Corruption methods
    # -------------------------
    def corrupt_labels(self, labels: torch.Tensor, noise_frac: float, seed: int):
        rng = np.random.RandomState(seed)
        labels = labels.clone()
        n = labels.size(0)
        k = int(round(noise_frac * n))
        if k == 0:
            return labels
        indices = rng.choice(n, size=k, replace=False)
        for i in indices:
            old = int(labels[i].item())
            choices = list(range(10))
            choices.remove(old)
            labels[i] = random.choice(choices)
        return labels

    def corrupt_features(self, inputs: torch.Tensor, noise_frac: float, seed: int):
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
            stdj = float(feat_std[j].item()) if torch.is_tensor(feat_std) else float(feat_std[j])
            if stdj <= 0:
                continue
            noise[:, j] = torch.from_numpy(rng.normal(loc=0.0, scale=0.5 * stdj, size=(N,))).to(flat.device)
        flat = flat + noise
        return flat.view_as(X)

    # -------------------------
    # Helpers for model files
    # -------------------------
    def feature_extractor_path(self, seed: int):
        ds = self.cfg.dataset.lower()
        model_name = 'resnet' if self.cfg.dataset == 'CIFAR_10' else 'cnn'
        return Path('checkpoints') / f"{ds}_{model_name}_feature_extractor_seed0.pt"

    # -------------------------
    # Party construction scenarios
    # -------------------------
    def build_size_parties(self, seed: int, party_k=10, per_party_size=1000):
        """Nested parties: D1 subset of D2 subset ... Dk built from disjoint chunks."""
        rng = np.random.RandomState(seed)
        pool = list(self.train_idx)
        total_needed = party_k * per_party_size
        replace = len(pool) < total_needed
        chosen = rng.choice(pool, size=total_needed, replace=replace)
        parties = []
        for i in range(1, party_k + 1):
            upto = i * per_party_size
            idxs = chosen[:upto]
            inputs = self.train_inputs[idxs]
            labels = self.train_labels[idxs]
            parties.append((inputs, labels))
        return parties

    def build_replication_parties(self, seed: int, party_k=10, base_size=1000):
        """Replication parties: sample base D1 then replicate it i times for Di."""
        rng = np.random.RandomState(seed)
        pool = list(self.train_idx)
        replace = len(pool) < base_size
        base_idxs = rng.choice(pool, size=base_size, replace=replace)
        base_inputs = self.train_inputs[base_idxs]
        base_labels = self.train_labels[base_idxs]
        parties = []
        for i in range(1, party_k + 1):
            # replicate base i times
            if i == 1:
                inputs = base_inputs
                labels = base_labels
            else:
                inputs = base_inputs.repeat(i, *([1] * (base_inputs.dim() - 1)))
                labels = base_labels.repeat(i)
            parties.append((inputs, labels))
        return parties

    def davinz_init_path(self, seed: int):
        ds = self.cfg.dataset.lower()
        return Path('checkpoints') / f"{ds}_model_init_seed{seed}.pt"

    def ensure_davinz_init(self, seed: int):
        p = self.davinz_init_path(seed)
        if p.exists():
            data = torch.load(p, map_location='cpu')
            return data.get('state_dict', data)
        # create untrained model and save it
        if self.cfg.dataset == 'CIFAR_10':
            from model.resnet import BasicBlock, ResNet
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10)
        else:
            from model.cnn import CNN
            model = CNN(in_channels=1, num_classes=10)
        sd = model.state_dict()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': sd}, p)
        return sd

    # -------------------------
    # Baseline computations (use feature extractor path)
    # -------------------------
    def compute_ot(self, loader: DataLoader, val_loader: DataLoader, seed: int):
        feat_p = str(self.feature_extractor_path(seed))
        if not os.path.exists(feat_p):
            warnings.warn(f"Feature extractor {feat_p} not found; OT will use untrained extractor")
            feat_p = None
        try:
            res = OT.compute_ot_distance(loader, val_loader, dataset=self.cfg.dataset, device=self.device, feature_extractor_path=feat_p)
            return res
        except Exception as e:
            return {'distance': float('nan'), 'timing': {'feature_extraction': float('nan'), 'ot_computation': float('nan'), 'total': float('nan')}, 'mem': {}}

    def compute_rv(self, loader: DataLoader, seed: int):
        feat_p = str(self.feature_extractor_path(seed))
        if not os.path.exists(feat_p):
            warnings.warn(f"Feature extractor {feat_p} not found; RV will use untrained extractor")
            feat_p = None
        try:
            res = RV.compute_rv_metric(loader, dataset=self.cfg.dataset, device=self.device, feature_extractor_path=feat_p, max_samples=self.cfg.max_samples)
            return res
        except Exception:
            return {'log_volume': float('nan'), 'log_robust_volume': float('nan'), 'timing': {'feature_extraction': float('nan'), 'rv_computation': float('nan'), 'total': float('nan')}, 'mem': {}}

    def compute_davinz(self, loader: DataLoader, val_loader: DataLoader, seed: int):
        # ensure an untrained init is saved and reuse across noise levels
        init_sd = self.ensure_davinz_init(seed)
        # instantiate model and load state_dict
        if self.cfg.dataset == 'CIFAR_10':
            from model.resnet import BasicBlock, ResNet
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10).to(self.device)
        else:
            from model.cnn import CNN
            model = CNN(in_channels=1, num_classes=10).to(self.device)
        try:
            model.load_state_dict(init_sd)
        except Exception:
            # if state dict was wrapped
            try:
                model.load_state_dict(init_sd.get('state_dict', init_sd))
            except Exception:
                pass

        try:
            res = DAVINZ.compute_davinz(loader, val_loader, dataset=self.cfg.dataset, device=self.device, model=model, diagonal_I_mag=1e-6, n_batch=100)
            return res
        except Exception:
            return {'mmd': float('nan'), 'mmd_raw': float('nan'), 'ntk': float('nan'), 'davinz_score': float('nan'), 'timing': {'mmd_time': float('nan'), 'ntk_time': float('nan'), 'total': float('nan')}, 'mem': {}}

    # -------------------------
    # Results saving
    # -------------------------
    def _ensure_headers(self, path: Path, header: str):
        if not path.exists():
            with open(path, 'w') as f:
                f.write(header + '\n')

    def _write_jsonl(self, path: Path, obj: dict):
        with open(path, 'a') as f:
            f.write(json.dumps(obj) + '\n')

    def run_seed(self, seed: int):
        seed_out = self.output_root / f'seed_{seed}'
        seed_out.mkdir(parents=True, exist_ok=True)
        base_inputs, base_labels, base_indices = self.sample_base_dataset(seed)
        torch.save({'inputs': base_inputs, 'labels': base_labels}, seed_out / 'base_dataset.pt')

        # validation loader (fixed split)
        val_dataset = Subset(TensorDataset(self.train_inputs, self.train_labels), self.val_idx)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        for scenario in ('label', 'feature'):
            d = seed_out / scenario
            d.mkdir(parents=True, exist_ok=True)
            # OT header (include party size)
            self._ensure_headers(d / 'ot_results.csv', 'noise_level,size,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp')
            # RV header (include party size)
            self._ensure_headers(d / 'rv_results.csv', 'noise_level,size,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp')
            # DaVinz header (include party size)
            self._ensure_headers(d / 'davinz_results.csv', 'noise_level,size,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp')
            # JSONL files for full dicts
            (d / 'ot_results.jsonl').touch(exist_ok=True)
            (d / 'rv_results.jsonl').touch(exist_ok=True)
            (d / 'davinz_results.jsonl').touch(exist_ok=True)

        # run label/feature noise sweep: compute metrics per noise level and write results
        for scenario in ('label', 'feature'):
            d = seed_out / scenario
            for nl in self.cfg.noise_levels:
                # prepare corrupted dataset depending on scenario
                if scenario == 'label':
                    labels = self.corrupt_labels(base_labels, nl, seed)
                    inputs = base_inputs
                else:
                    inputs = self.corrupt_features(base_inputs, nl, seed)
                    labels = base_labels

                loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=False)
                ot_r = self.compute_ot(loader, val_loader, seed)
                rv_r = self.compute_rv(loader, seed)
                dav_r = self.compute_davinz(loader, val_loader, seed)
                # write per-noise rows and JSONL entries
                self._write_party_results(d, nl, ot_r, rv_r, dav_r, scenario, size=inputs.size(0))

        # -------------------------
        # Size and Replication scenarios
        # -------------------------
        size_parties = self.build_size_parties(seed)
        repl_parties = self.build_replication_parties(seed)

        # run size parties — aggregate per-party results into seed_out/size (disjoint from label/feature)
        size_dir = seed_out / 'size'
        size_dir.mkdir(parents=True, exist_ok=True)
        # keep per-metric CSV/JSONL for debugging, and also create single summary results.csv
        # per-metric CSVs for size scenario (no noise_level column)
        # add `size` column to record party size
        self._ensure_headers(size_dir / 'ot_results.csv', 'size,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp')
        self._ensure_headers(size_dir / 'rv_results.csv', 'size,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp')
        self._ensure_headers(size_dir / 'davinz_results.csv', 'size,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp')
        (size_dir / 'ot_results.jsonl').touch(exist_ok=True)
        (size_dir / 'rv_results.jsonl').touch(exist_ok=True)
        (size_dir / 'davinz_results.jsonl').touch(exist_ok=True)
        # single summary CSV for size scenario
        self._ensure_headers(size_dir / 'size_summary.csv', 'party_id,size,ot_distance,rv_log_robust_volume,davinz_score,timestamp')

        for p_idx, (p_inputs, p_labels) in enumerate(size_parties, start=1):
            # No noise: compute metrics once per party on raw data
            loader = DataLoader(TensorDataset(p_inputs, p_labels), batch_size=128, shuffle=False)
            ot_r = self.compute_ot(loader, val_loader, seed)
            rv_r = self.compute_rv(loader, seed)
            dav_r = self.compute_davinz(loader, val_loader, seed)
            ts = time.time()
            # write detailed per-metric rows (no noise_level) and JSONL (scenario='size')
            # OT row (prefix with party size)
            ot_csv = size_dir / 'ot_results.csv'
            ot_row = [int(p_inputs.size(0)), ot_r.get('distance', 'nan'), ot_r.get('timing', {}).get('feature_extraction', 'nan'), ot_r.get('timing', {}).get('ot_computation', 'nan'), ot_r.get('timing', {}).get('total', 'nan'), ot_r.get('mem', {}).get('feature_extraction', 'nan'), ot_r.get('mem', {}).get('ot', 'nan'), ot_r.get('mem', {}).get('total', 'nan'), ts]
            with open(ot_csv, 'a') as f:
                f.write(','.join(map(str, ot_row)) + '\n')
            self._write_jsonl(size_dir / 'ot_results.jsonl', {'party_id': p_idx, 'scenario': 'size', 'result': ot_r, 'timestamp': ts})
            # RV row
            rv_csv = size_dir / 'rv_results.csv'
            rv_row = [int(p_inputs.size(0)), rv_r.get('log_volume', 'nan'), rv_r.get('log_robust_volume', 'nan'), rv_r.get('timing', {}).get('feature_extraction', 'nan'), rv_r.get('timing', {}).get('rv_computation', 'nan'), rv_r.get('timing', {}).get('total', 'nan'), rv_r.get('mem', {}).get('feature_extraction', 'nan'), rv_r.get('mem', {}).get('rv', 'nan'), rv_r.get('mem', {}).get('total', 'nan'), ts]
            with open(rv_csv, 'a') as f:
                f.write(','.join(map(str, rv_row)) + '\n')
            self._write_jsonl(size_dir / 'rv_results.jsonl', {'party_id': p_idx, 'scenario': 'size', 'result': rv_r, 'timestamp': ts})
            # DaVinz row
            dav_csv = size_dir / 'davinz_results.csv'
            dav_row = [int(p_inputs.size(0)), dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')), dav_r.get('mmd_raw', 'nan'), dav_r.get('ntk', 'nan'), dav_r.get('davinz_score', 'nan'), dav_r.get('timing', {}).get('mmd_time', 'nan'), dav_r.get('timing', {}).get('ntk_time', 'nan'), dav_r.get('timing', {}).get('total', 'nan'), dav_r.get('mem', {}).get('mmd', 'nan'), dav_r.get('mem', {}).get('ntk', 'nan'), dav_r.get('mem', {}).get('total', 'nan'), ts]
            with open(dav_csv, 'a') as f:
                f.write(','.join(map(str, dav_row)) + '\n')
            self._write_jsonl(size_dir / 'davinz_results.jsonl', {'party_id': p_idx, 'scenario': 'size', 'result': dav_r, 'timestamp': ts})
            # append single-line summary for this party (size_summary.csv)
            results_csv = size_dir / 'size_summary.csv'
            with open(results_csv, 'a') as f:
                f.write(','.join(map(str, [p_idx, int(p_inputs.size(0)), ot_r.get('distance', 'nan'), rv_r.get('log_robust_volume', 'nan'), dav_r.get('davinz_score', 'nan'), ts])) + '\n')

        # run replication parties — aggregate per-party results into seed_out/replication (disjoint from label/feature)
        repl_dir = seed_out / 'replication'
        repl_dir.mkdir(parents=True, exist_ok=True)
        # single summary CSV for replication scenario (no noise_level in per-metric CSVs)
        self._ensure_headers(repl_dir / 'replication_summary.csv', 'party_id,replication_factor,num_samples,ot_distance,rv_log_robust_volume,davinz_score,timestamp')
        # per-metric CSVs include replication_factor column
        self._ensure_headers(repl_dir / 'ot_results.csv', 'replication_factor,ot_distance,feature_extraction_time_s,ot_computation_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp')
        self._ensure_headers(repl_dir / 'rv_results.csv', 'replication_factor,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp')
        self._ensure_headers(repl_dir / 'davinz_results.csv', 'replication_factor,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp')
        (repl_dir / 'ot_results.jsonl').touch(exist_ok=True)
        (repl_dir / 'rv_results.jsonl').touch(exist_ok=True)
        (repl_dir / 'davinz_results.jsonl').touch(exist_ok=True)

        for p_idx, (p_inputs, p_labels) in enumerate(repl_parties, start=1):
            # No noise: compute metrics once per party on raw (possibly replicated) data
            loader = DataLoader(TensorDataset(p_inputs, p_labels), batch_size=128, shuffle=False)
            ot_r = self.compute_ot(loader, val_loader, seed)
            rv_r = self.compute_rv(loader, seed)
            dav_r = self.compute_davinz(loader, val_loader, seed)
            ts = time.time()
            # OT row (prefix with replication factor)
            replication_factor = p_idx
            ot_csv = repl_dir / 'ot_results.csv'
            ot_row = [replication_factor, ot_r.get('distance', 'nan'), ot_r.get('timing', {}).get('feature_extraction', 'nan'), ot_r.get('timing', {}).get('ot_computation', 'nan'), ot_r.get('timing', {}).get('total', 'nan'), ot_r.get('mem', {}).get('feature_extraction', 'nan'), ot_r.get('mem', {}).get('ot', 'nan'), ot_r.get('mem', {}).get('total', 'nan'), ts]
            with open(ot_csv, 'a') as f:
                f.write(','.join(map(str, ot_row)) + '\n')
            self._write_jsonl(repl_dir / 'ot_results.jsonl', {'party_id': p_idx, 'scenario': 'replication', 'result': ot_r, 'timestamp': ts})
            # RV row
            rv_csv = repl_dir / 'rv_results.csv'
            rv_row = [replication_factor, rv_r.get('log_volume', 'nan'), rv_r.get('log_robust_volume', 'nan'), rv_r.get('timing', {}).get('feature_extraction', 'nan'), rv_r.get('timing', {}).get('rv_computation', 'nan'), rv_r.get('timing', {}).get('total', 'nan'), rv_r.get('mem', {}).get('feature_extraction', 'nan'), rv_r.get('mem', {}).get('rv', 'nan'), rv_r.get('mem', {}).get('total', 'nan'), ts]
            with open(rv_csv, 'a') as f:
                f.write(','.join(map(str, rv_row)) + '\n')
            self._write_jsonl(repl_dir / 'rv_results.jsonl', {'party_id': p_idx, 'scenario': 'replication', 'result': rv_r, 'timestamp': ts})
            # DaVinz row
            dav_csv = repl_dir / 'davinz_results.csv'
            dav_row = [replication_factor, dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')), dav_r.get('mmd_raw', 'nan'), dav_r.get('ntk', 'nan'), dav_r.get('davinz_score', 'nan'), dav_r.get('timing', {}).get('mmd_time', 'nan'), dav_r.get('timing', {}).get('ntk_time', 'nan'), dav_r.get('timing', {}).get('total', 'nan'), dav_r.get('mem', {}).get('mmd', 'nan'), dav_r.get('mem', {}).get('ntk', 'nan'), dav_r.get('mem', {}).get('total', 'nan'), ts]
            with open(dav_csv, 'a') as f:
                f.write(','.join(map(str, dav_row)) + '\n')
            self._write_jsonl(repl_dir / 'davinz_results.jsonl', {'party_id': p_idx, 'scenario': 'replication', 'result': dav_r, 'timestamp': ts})
            # append single-line summary for this party (replication_summary.csv)
            results_csv = repl_dir / 'replication_summary.csv'
            with open(results_csv, 'a') as f:
                f.write(','.join(map(str, [p_idx, replication_factor, int(p_inputs.size(0)), ot_r.get('distance', 'nan'), rv_r.get('log_robust_volume', 'nan'), dav_r.get('davinz_score', 'nan'), ts])) + '\n')

    def _write_party_results(self, out_dir: Path, nl: float, ot_r: dict, rv_r: dict, dav_r: dict, scenario: str, party_id: Optional[int]=None, size: Optional[int]=None):
        # helper to write CSV + JSONL for party runs; `out_dir` should be the scenario subdir (label/feature)
        ts = time.time()
        ot_csv = out_dir / 'ot_results.csv'
        ot_row = [nl,
              (int(size) if size is not None else 'nan'),
              ot_r.get('distance', 'nan'),
                  ot_r.get('timing', {}).get('feature_extraction', 'nan'),
                  ot_r.get('timing', {}).get('ot_computation', 'nan'),
                  ot_r.get('timing', {}).get('total', 'nan'),
                  ot_r.get('mem', {}).get('feature_extraction', 'nan'),
                  ot_r.get('mem', {}).get('ot', 'nan'),
                  ot_r.get('mem', {}).get('total', 'nan'),
                  ts]
        with open(ot_csv, 'a') as f:
            f.write(','.join(map(str, ot_row)) + '\n')
        ot_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 'scenario': scenario, 'result': ot_r, 'timestamp': ts}
        if party_id is not None:
            ot_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'ot_results.jsonl', ot_json)

        rv_csv = out_dir / 'rv_results.csv'
        rv_row = [nl,
              (int(size) if size is not None else 'nan'),
              rv_r.get('log_volume', 'nan'),
                  rv_r.get('log_robust_volume', 'nan'),
                  rv_r.get('timing', {}).get('feature_extraction', 'nan'),
                  rv_r.get('timing', {}).get('rv_computation', 'nan'),
                  rv_r.get('timing', {}).get('total', 'nan'),
                  rv_r.get('mem', {}).get('feature_extraction', 'nan'),
                  rv_r.get('mem', {}).get('rv', 'nan'),
                  rv_r.get('mem', {}).get('total', 'nan'),
                  ts]
        with open(rv_csv, 'a') as f:
            f.write(','.join(map(str, rv_row)) + '\n')
        rv_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 'scenario': scenario, 'result': rv_r, 'timestamp': ts}
        if party_id is not None:
            rv_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'rv_results.jsonl', rv_json)

        dav_csv = out_dir / 'davinz_results.csv'
        dav_row = [nl,
               (int(size) if size is not None else 'nan'),
               dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')),
                   dav_r.get('mmd_raw', 'nan'),
                   dav_r.get('ntk', 'nan'),
                   dav_r.get('davinz_score', 'nan'),
                   dav_r.get('timing', {}).get('mmd_time', 'nan'),
                   dav_r.get('timing', {}).get('ntk_time', 'nan'),
                   dav_r.get('timing', {}).get('total', 'nan'),
                   dav_r.get('mem', {}).get('mmd', 'nan'),
                   dav_r.get('mem', {}).get('ntk', 'nan'),
                   dav_r.get('mem', {}).get('total', 'nan'),
                   ts]
        with open(dav_csv, 'a') as f:
            f.write(','.join(map(str, dav_row)) + '\n')
        dav_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 'scenario': scenario, 'result': dav_r, 'timestamp': ts}
        if party_id is not None:
            dav_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'davinz_results.jsonl', dav_json)
        # end _write_party_results

    def aggregate(self, seeds: List[int]):
        agg_dir = self.output_root / 'aggregated'
        agg_dir.mkdir(parents=True, exist_ok=True)
        # keep simple aggregated summary (seed,noise,ot,rv,davinz)
        with open(agg_dir / 'label_noise_summary.csv', 'w') as f:
            f.write('seed,noise_level,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_dir = self.output_root / f'seed_{s}' / 'label'
                if not seed_dir.exists():
                    continue
                dav = seed_dir / 'davinz_results.csv'
                rv = seed_dir / 'rv_results.csv'
                ot = seed_dir / 'ot_results.csv'
                if dav.exists() and rv.exists() and ot.exists():
                    dav_rows = [line.strip().split(',') for line in dav.read_text().strip().split('\n')[1:]]
                    rv_rows = [line.strip().split(',') for line in rv.read_text().strip().split('\n')[1:]]
                    ot_rows = [line.strip().split(',') for line in ot.read_text().strip().split('\n')[1:]]
                    for i, nl in enumerate(dav_rows):
                        noise_level = nl[0]
                        davinz_score = nl[4] if len(nl) > 4 else ''
                        ot_dist = ot_rows[i][1] if i < len(ot_rows) else ''
                        rv_log = rv_rows[i][2] if i < len(rv_rows) else ''
                        f.write(f"{s},{noise_level},{ot_dist},{rv_log},{davinz_score}\n")

        with open(agg_dir / 'feature_noise_summary.csv', 'w') as f:
            f.write('seed,noise_level,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_dir = self.output_root / f'seed_{s}' / 'feature'
                if not seed_dir.exists():
                    continue
                dav = seed_dir / 'davinz_results.csv'
                rv = seed_dir / 'rv_results.csv'
                ot = seed_dir / 'ot_results.csv'
                if dav.exists() and rv.exists() and ot.exists():
                    dav_rows = [line.strip().split(',') for line in dav.read_text().strip().split('\n')[1:]]
                    rv_rows = [line.strip().split(',') for line in rv.read_text().strip().split('\n')[1:]]
                    ot_rows = [line.strip().split(',') for line in ot.read_text().strip().split('\n')[1:]]
                    for i, nl in enumerate(dav_rows):
                        noise_level = nl[0]
                        davinz_score = nl[4] if len(nl) > 4 else ''
                        ot_dist = ot_rows[i][1] if i < len(ot_rows) else ''
                        rv_log = rv_rows[i][2] if i < len(rv_rows) else ''
                        f.write(f"{s},{noise_level},{ot_dist},{rv_log},{davinz_score}\n")


def parse_seed_range(seed_range_str: str) -> List[int]:
    if '-' in seed_range_str:
        start, end = seed_range_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in seed_range_str:
        return [int(s.strip()) for s in seed_range_str.split(',')]
    else:
        return [int(seed_range_str)]


def main():
    parser = argparse.ArgumentParser(description='Noise Robustness Testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR_10', 'MNIST'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seeds', type=str, default='0', help='seed or range e.g. 0-2')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--base_size', type=int, default=1000)
    parser.add_argument('--max_samples', type=int, default=10000)
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    cfg = Config(
        dataset=args.dataset,
        gpu=args.gpu,
        seeds=seeds,
        output_dir=args.output_dir,
        base_size=args.base_size,
        max_samples=args.max_samples,
    )

    tester = RobustnessTester(cfg)
    tester.load_data()
    for s in seeds:
        print(f"[INFO] Running seed={s} on device={tester.device}")
        tester.run_seed(s)
    tester.aggregate(seeds)


if __name__ == '__main__':
    main()
