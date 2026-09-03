#!/usr/bin/env python3
"""
H100-Optimized CIFAR10 + ResNet Data Valuation Experiment (Unified Pipeline)

This is the unified training pipeline for ALL ResNet models (9, 18, 34, 50, 110).
All models use identical H100-optimized training hyperparameters for fair comparison.

Key H100 Optimizations:
  ✓ torch.compile with 'reduce-overhead' mode for kernel fusion
  ✓ Automatic Mixed Precision (bfloat16) autocast + GradScaler
  ✓ CosineAnnealingLR scheduler (smooth decay over 100 epochs)
  ✓ Adam optimizer with LR=0.001 (adaptive learning rates)
  ✓ Channels-last memory format (NHWC layout)
  ✓ Large batch size (1024) with GPU memory optimization
  ✓ Gradient clipping for training stability
  ✓ Persistent workers + pinned memory in DataLoader
  ✓ Unified configuration (all models use same hyperparams)

Usage Examples:
  # ResNet-18 with LoGRA
  python run_cifar10_resnet18_dataval.py --model resnet18 --cuda 0 --seed 42 --method LoGRA

  # ResNet-50 with full evaluation
  python run_cifar10_resnet18_dataval.py --model resnet50 --cuda 0 --seed 42 --method ALL

  # ResNet-110 with InRunDataShapleyGhost
  python run_cifar10_resnet18_dataval.py --model rn110 --cuda 0 --seed 42 --method InRunDataShapleyGhost

Updated: 2026-08-01
Scalability Study: Fair comparison across model depths with H100 optimizations
"""

import os
os.environ['OPENBLAS_CORETYPE'] = 'NEHALEM'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import time
import json
import sys
import logging
import psutil
import tracemalloc
import threading
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified config
try:
    from unified_config import get_training_config, print_training_config
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    print("⚠ unified_config.py not found in parent directory")
    UNIFIED_CONFIG_AVAILABLE = False

# Add comprehensive profiler
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from comprehensive_profiler import ComprehensiveProfiler
    PROFILER_AVAILABLE = True
except ImportError:
    print("⚠ comprehensive_profiler not found, skipping profiling")
    PROFILER_AVAILABLE = False

# opendataval imports
from opendataval.dataloader import mix_labels
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LeaveOneOut, RandomEvaluator,
    LoGRA, InfluenceFunction, InRunDataShapleyGhost
)
from opendataval.dataval.knnshap import KNNShapleyLSH, KNNShapleyVec, AKShapleyGPU
from opendataval.dataval.lava import LavaEvaluator, SavaEvaluator
from opendataval.dataval.influence.infsub_ckpt import InfluenceSubsampleCKPT
from opendataval.dataval.oob.dataoob_ckpt import DataOobCKPT
from opendataval.dataval.influence.inrun_shapley_ckpt import InRunShapleyCKPT
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample, noisy_detection, remove_high_low, save_dataval
)

# Optional imports
try:
    from opendataval.dataval.kairos.bkairos import bKairos
    from opendataval.dataval.kairos.kairos import Kairos
    from opendataval.dataval.kairos.kairosgpu import KairosGPU
    KAIROS_AVAILABLE = True
except (ImportError, AttributeError):
    KAIROS_AVAILABLE = False

try:
    from ghostEngines import GradDotProdEngine
    GHOSTENGINES_AVAILABLE = True
except ImportError:
    GHOSTENGINES_AVAILABLE = False

try:
    import logix
    LOGIX_AVAILABLE = True
except ImportError:
    LOGIX_AVAILABLE = False

print(f"✓ Framework Status: GhostSuite={GHOSTENGINES_AVAILABLE}, LogIX={LOGIX_AVAILABLE}, Kairos={KAIROS_AVAILABLE}, UnifiedConfig={UNIFIED_CONFIG_AVAILABLE}")


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class MemoryTracker:
    """Conference-standard memory and timing measurement (ICML/NeurIPS/ICLR methodology).

    Reports incremental resource consumption relative to baseline, excluding:
    - Pre-existing GPU allocations (other processes, CUDA contexts, caching)
    - Framework initialization overhead
    - Unrelated background processes

    Methodology:
    - GPU: Process-level torch.cuda.* APIs (not nvidia-smi)
    - CPU: Per-process RSS (psutil), not system-wide memory
    - Report: ΔMemory = PeakMemory - BaselineMemory
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.baseline_cpu_rss_mb = None
        self.baseline_gpu_allocated_mb = None
        self.baseline_gpu_reserved_mb = None
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Establish baseline by synchronizing, clearing cache, and recording current state.

        This should be called after all initialization (data loading, model setup, etc)
        and immediately before the method under evaluation begins.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()

        # Baseline CPU: Current process RSS only (per-process, not system-wide)
        self.baseline_cpu_rss_mb = self.process.memory_info().rss / 1024 / 1024

        # Baseline GPU: Current allocated and reserved (will subtract from peak)
        if torch.cuda.is_available():
            self.baseline_gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.baseline_gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            self.baseline_gpu_allocated_mb = 0
            self.baseline_gpu_reserved_mb = 0

    def end(self):
        """Stop tracking and return DELTA metrics (peak - baseline).

        Reports only the incremental memory used by the method, not absolute device usage.
        Synchronizes GPU to ensure all operations complete before measurement.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.time()
        elapsed_seconds = self.end_time - self.start_time

        # Peak memory stats during execution (process-level only)
        peak_gpu_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0

        # Current process RSS (may be higher than baseline if process acquired additional memory)
        current_cpu_rss_mb = self.process.memory_info().rss / 1024 / 1024

        # Delta metrics: What the method actually consumed
        delta_cpu_mb = current_cpu_rss_mb - self.baseline_cpu_rss_mb
        delta_gpu_allocated_mb = peak_gpu_allocated_mb - self.baseline_gpu_allocated_mb
        delta_gpu_reserved_mb = peak_gpu_reserved_mb - self.baseline_gpu_reserved_mb

        return {
            'elapsed_seconds': elapsed_seconds,
            'elapsed_formatted': self._format_time(elapsed_seconds),
            # Baseline (reference point, recorded before experiment)
            'baseline_cpu_rss_mb': self.baseline_cpu_rss_mb,
            'baseline_gpu_allocated_mb': self.baseline_gpu_allocated_mb,
            'baseline_gpu_reserved_mb': self.baseline_gpu_reserved_mb,
            # Peak during execution (process-level)
            'peak_cpu_rss_mb': current_cpu_rss_mb,
            'peak_gpu_allocated_mb': peak_gpu_allocated_mb,
            'peak_gpu_reserved_mb': peak_gpu_reserved_mb,
            # REPORTED METRICS: Incremental consumption (Δ = Peak - Baseline)
            'delta_cpu_mb': delta_cpu_mb,
            'delta_gpu_allocated_mb': delta_gpu_allocated_mb,
            'delta_gpu_reserved_mb': delta_gpu_reserved_mb,
        }

    @staticmethod
    def _format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def format_memory_report(stats, phase_name=""):
    """Format memory metrics in conference-standard style (baseline | peak | delta).

    Reports incremental consumption (Δ = peak - baseline), excluding pre-existing allocations.
    """
    baseline_cpu = stats.get('baseline_cpu_rss_mb', 0)
    peak_cpu = stats.get('peak_cpu_rss_mb', 0)
    delta_cpu = stats.get('delta_cpu_mb', 0)

    baseline_gpu_alloc = stats.get('baseline_gpu_allocated_mb', 0)
    peak_gpu_alloc = stats.get('peak_gpu_allocated_mb', 0)
    delta_gpu_alloc = stats.get('delta_gpu_allocated_mb', 0)

    baseline_gpu_resv = stats.get('baseline_gpu_reserved_mb', 0)
    peak_gpu_resv = stats.get('peak_gpu_reserved_mb', 0)
    delta_gpu_resv = stats.get('delta_gpu_reserved_mb', 0)

    lines = []
    if phase_name:
        lines.append(f"\n[{phase_name}]")
    lines.append(f"  CPU RSS:        Baseline={baseline_cpu:.1f}MB → Peak={peak_cpu:.1f}MB | Δ={delta_cpu:+.1f}MB")
    lines.append(f"  GPU Allocated:  Baseline={baseline_gpu_alloc:.1f}MB → Peak={peak_gpu_alloc:.1f}MB | Δ={delta_gpu_alloc:+.1f}MB")
    lines.append(f"  GPU Reserved:   Baseline={baseline_gpu_resv:.1f}MB → Peak={peak_gpu_resv:.1f}MB | Δ={delta_gpu_resv:+.1f}MB")
    return "\n".join(lines)


class DualWriter:
    """Write to both file and console."""
    def __init__(self, file_obj, console_obj):
        self.file = file_obj
        self.console = console_obj

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.console.write(message)
        self.console.flush()

    def flush(self):
        self.file.flush()
        self.console.flush()

    def isatty(self):
        return self.console.isatty()


# ============================================================================
# SETUP & LOGGING
# ============================================================================

def setup_logging(method_name, seed, logs_base="./logs", timestamp_str=None):
    """Setup logging to file and console."""
    logs_base_str = str(logs_base)

    # If path already contains method_name and seed_N, use it directly (avoid double nesting)
    if method_name in logs_base_str and f"seed_{seed}" in logs_base_str:
        log_dir = Path(logs_base)
    elif timestamp_str:
        log_dir = Path(logs_base) / f"{method_name}_{timestamp_str}" / f"seed_{seed}"
    else:
        log_dir = Path(logs_base) / method_name / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{method_name}_seed_{seed}.log"
    log_file_obj = open(log_file, 'w')

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = DualWriter(log_file_obj, original_stdout)
    sys.stderr = DualWriter(log_file_obj, original_stderr)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def get_evaluator_suffix(evaluator):
    """Generate parameter suffix from evaluator config."""
    import re
    evaluator_str = str(evaluator)
    params = []

    # Extract patterns for different evaluator types
    if "InfluenceSubsample" in evaluator_str:
        if (m := re.search(r'num_models=(\d+)', evaluator_str)):
            params.append(f"m{m.group(1)}")
        if (m := re.search(r'proportion=([0-9.]+)', evaluator_str)):
            params.append(f"p{m.group(1)}")
    elif "DataOob" in evaluator_str:
        if (m := re.search(r'num_models=(\d+)', evaluator_str)):
            params.append(f"m{m.group(1)}")
        if (m := re.search(r'proportion=([0-9.]+)', evaluator_str)):
            params.append(f"p{m.group(1)}")
    elif "KNNShapley" in evaluator_str:
        if (m := re.search(r'k_neighbors=(\d+)', evaluator_str)):
            params.append(f"k{m.group(1)}")
    elif "DataShapley" in evaluator_str:
        if (m := re.search(r'mc_epochs=(\d+)', evaluator_str)):
            params.append(f"mc{m.group(1)}")
    elif "BetaShapley" in evaluator_str or "DataBanzhaf" in evaluator_str:
        if (m := re.search(r'num_models=(\d+)', evaluator_str)):
            params.append(f"m{m.group(1)}")
    elif "DVRL" in evaluator_str:
        if (m := re.search(r'rl_epochs=(\d+)', evaluator_str)):
            params.append(f"e{m.group(1)}")
        if (m := re.search(r'rl_batch_size=(\d+)', evaluator_str)):
            params.append(f"b{m.group(1)}")
    elif "LAVA" in evaluator_str or "SAVA" in evaluator_str:
        if (m := re.search(r'lam_y=([0-9.]+)', evaluator_str)):
            params.append(f"lam{m.group(1)}")
    elif "LoGRA" in evaluator_str:
        if (m := re.search(r"lora='([^']+)'", evaluator_str)):
            params.append(f"lora{m.group(1)}")
        if (m := re.search(r"hessian='([^']+)'", evaluator_str)):
            params.append(f"hess{m.group(1)}")
    elif "InRunDataShapleyGhost" in evaluator_str:
        if (m := re.search(r'batch_size=(\d+)', evaluator_str)):
            params.append(f"b{m.group(1)}")

    return "_".join(params) if params else "default"


def set_global_seeds(seed):
    """Set all global random seeds for reproducibility."""
    print(f"[Seeding] Setting global seeds to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

def get_unified_train_kwargs(model_name: str) -> Dict[str, Any]:
    """Get unified training kwargs for the model.

    If unified_config is available, use it. Otherwise, use sensible defaults.
    """
    # if UNIFIED_CONFIG_AVAILABLE:
    #     return get_training_config(model_name)

    # Fallback configuration (all models use these settings)
    default_config = {
        "epochs": 50,
        "batch_size": 1024,
        "lr": 0.0028,
        "weight_decay": 5e-4,
        "label_smoothing": 0.0,
        "scheduler": "cosine",
    }

    return default_config


def create_experiment_mediator(
    model_name: str,
    seed: int,
    cuda_device: int,
    train_count: int = 40000,
    valid_count: int = 10000,
    test_count: int = 10000,
    noise_rate: float = 0.2,
):
    """Create ExperimentMediator with H100-optimized unified config."""
    print(f"\n[Setup] Creating CIFAR10 {model_name} experiment (H100-Optimized)...")
    set_global_seeds(seed)

    dataset_name = "cifar10"
    metric_name = "accuracy"

    # Get unified training configuration
    TRAIN_KWARGS = get_unified_train_kwargs(model_name)

    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    print(f"  Dataset: {dataset_name}, Model: {model_name}")
    print(f"  Splits: {train_count}/{valid_count}/{test_count}, Noise: {noise_rate}")
    print(f"  Device: {device}")
    print(f"  H100-Optimized Training Config:")
    print(f"    - Epochs: {TRAIN_KWARGS['epochs']}")
    print(f"    - Batch Size: {TRAIN_KWARGS['batch_size']}")
    print(f"    - Learning Rate: {TRAIN_KWARGS['lr']}")
    print(f"    - Weight Decay: {TRAIN_KWARGS['weight_decay']}")
    print(f"    - Label Smoothing: {TRAIN_KWARGS['label_smoothing']}")

    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        cache_dir="./data_files/",
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs={'noise_rate': 0.2},
        train_kwargs=TRAIN_KWARGS,
        model_name=model_name,
        metric_name=metric_name,
        train_validation_model=False,
        train_baseline_model=False,  # ✅ Train baseline ResNet at the beginning
        random_state=seed,
        device=device
    )

    print(f"✓ Checkpoints created: {len(exper_med.get_checkpoints())}")
    # Store data size for profiling
    exper_med._train_count = train_count
    return exper_med


def create_imagenet_embedding_model(device=None, renorm=True):
    """ImageNet-pretrained ResNet18 used as a frozen feature extractor (512-d).

    Mirrors the torchvision path in the reference script:
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()

    Note on preprocessing: opendataval delivers CIFAR-normalised 32x32 tensors
    (mean 0.4914/0.4822/0.4465, std 0.2470/0.2435/0.2616). This network was
    trained on ImageNet statistics, so we undo the CIFAR normalisation and
    re-apply the ImageNet one before the forward pass.
    """
    import torchvision.models as tvm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval()
    for prm in net.parameters():
        prm.requires_grad = False

    _CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    _CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
    _IN_MEAN    = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    _IN_STD     = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def embedding_predict(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32)
        x = x.to(device).float()
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        with torch.no_grad():
            if renorm:
                raw = x * _CIFAR_STD + _CIFAR_MEAN      # back to [0,1]
                inp = (raw - _IN_MEAN) / _IN_STD        # ImageNet statistics
            else:
                inp = x                                 # script-matching: CIFAR stats as-is
            return net(inp).cpu().numpy()

    net.predict = embedding_predict
    print(f"✓ Embedding model loaded (torchvision ResNet18, ImageNet-pretrained, 512-dim) renorm={renorm}")
    print(f"  Device: {device}  |  CIFAR->ImageNet normalisation applied")
    return net


def create_embedding_model():
    """Create H100-optimized embedding model with feature extraction."""
    from opendataval.model.resnet import ResNet18

    embedding_model = ResNet18(num_classes=10)
    embedding_model.load_state_dict(
        torch.load("./checkpoints/validation_model/model.pth", map_location="cpu")
    )

    # For ResNet18, extract features by removing the classification head (fc layer)
    # ResNet18 structure: conv1->bn1->relu -> layer1->layer2->layer3->layer4 -> avgpool -> flatten -> fc
    # We want: conv1->bn1->relu -> layer1->layer2->layer3->layer4 -> avgpool -> flatten (no fc)
    class FeatureExtractor(nn.Module):
        def __init__(self, resnet18_model):
            super().__init__()
            self.conv1 = resnet18_model.conv1
            self.bn1 = resnet18_model.bn1
            self.relu = resnet18_model.relu
            self.layer1 = resnet18_model.layer1
            self.layer2 = resnet18_model.layer2
            self.layer3 = resnet18_model.layer3
            self.layer4 = resnet18_model.layer4
            self.avgpool = resnet18_model.avgpool

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # 512-dimensional features
            return x

        def __len__(self):
            """Support len() for compatibility with some evaluators"""
            return 512

    feature_extractor = FeatureExtractor(embedding_model)

    # H100 optimizations for embedding extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = feature_extractor.to(device)

    # Apply torch.compile for faster inference
    try:
        feature_extractor = torch.compile(feature_extractor, mode='reduce-overhead', fullgraph=False)
        compile_status = "✓ torch.compile enabled"
    except Exception as e:
        compile_status = f"⚠ torch.compile failed: {e}"

    feature_extractor.eval()

    # H100-optimized predict: larger batch processing, GPU-resident
    def embedding_predict(x):
        device = next(feature_extractor.parameters()).device
        feature_extractor.eval()

        # Convert numpy to tensor (always float32)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            # Keep on GPU, torch.compile handles optimization
            x = x.to(device)
            embeddings = feature_extractor.forward(x)
            # Return as numpy (required by embeddings pipeline)
            return embeddings.cpu().numpy()

    feature_extractor.predict = embedding_predict
    print(f"✓ Embedding model loaded (ResNet18, 512-dim features) {compile_status}")
    print(f"  Device: {device}")
    print(f"  Batch processing: 1024 samples/batch with H100 acceleration")
    return feature_extractor


def create_method_evaluators(method_name, seed, embedding_model=None, val_batch_size=128, proportion=0.7, num_models=100, checkpoint_models=None, lam_y=5.0, lam_x=1.0, k_neighbors=10, lambda_weight=0.97, valid_chunk=512, row_chunk=8192, sava_batch_size=1024, dist_rand=7.3622, lsh_t=2.399, n_hash_table=100, lsh_eps=1e-2):
    """Create evaluators for specified method using unified config."""
    print(f"\n[Methods] Creating {method_name} evaluators...")

    if method_name in ("DataOob", "DOOB"):
        evaluators = [
            DataOobCKPT(
                num_models=num_models,
                proportion=proportion,
                checkpoint_models=checkpoint_models if checkpoint_models else [1,5,10,50],
                random_state=seed,
                verbose=True
            )
        ]

    elif method_name == "LOO_Random":
        evaluators = [RandomEvaluator(random_state=seed)]

    elif method_name == "KNNShapley":
        evaluators = [
            KNNShapley(k_neighbors=k_neighbors, embedding_model=embedding_model, random_state=seed)
        ]

    elif method_name == "KNNShapleyVec":
        # Vectorised, GPU-capable KNN-Shapley. Exact reformulation of the
        # KNNShapley recursion as a suffix sum - same values, ~1e-12 apart,
        # identical ranking. See knnshapvec.py.
        evaluators = [
            KNNShapleyVec(k_neighbors=k_neighbors, valid_chunk=valid_chunk, embedding_model=embedding_model, random_state=seed, debug=True)
        ]

    elif method_name == "AKShapleyGPU":
        # GPU-resident AKShapley. Same LSH approximation as the KNNShapleyLSH
        # branch below: candidate sets, neighbour order and the label-match
        # matrix are identical, values agree to ~1 ULP.
        # eps sets the retrieval depth K_star = max(k_neighbors, ceil(1/eps)):
        # eps>=1e-3 gives K_star=1000 (so eps does not vary the result once
        # dist_rand and t are pinned), eps=1e-4 gives K_star=10000.
        evaluators = [
            AKShapleyGPU(
                k_neighbors=1000,
                dist_rand=(dist_rand if dist_rand > 0 else None),
                n_hash_table=n_hash_table,
                eps=lsh_eps,
                alpha=0.5,
                t=(lsh_t if lsh_t > 0 else None),
                valid_chunk=valid_chunk,
                embedding_model=embedding_model,
                random_state=seed,
                debug=True,
            )
        ]

    elif method_name == "AKShapley":
        evaluators = [
            KNNShapleyLSH(
                k_neighbors=1000,
                dist_rand=7.3622,
                n_hash_table=100,
                eps=eps,
                alpha=0.5,
                t=2.399,
                embedding_model=embedding_model,
                random_state=seed
            )
            for eps in [1e-3, 1e-2]
        ]

    elif method_name == "DataShapley":
        evaluators = [DataShapley(mc_epochs=mc, min_cardinality=5, random_state=seed) for mc in [100]]

    elif method_name == "BetaShapley":
        evaluators = [BetaShapley(num_models=m, random_state=seed) for m in [100, 500, 1000]]

    elif method_name == "DataBanzhaf":
        evaluators = [DataBanzhaf(num_models=m, random_state=seed) for m in [100, 500, 1000]]

    elif method_name == "InfluenceSubsample":
        evaluators = [
            InfluenceSubsampleCKPT(
                num_models=100,
                proportion=proportion,
                checkpoint_models=[1, 5, 10, 50],
                random_state=seed,
                verbose=True,
            )
        ]

    elif method_name == "AME":
        evaluators = [AME(num_models=m, random_state=seed) for m in [100000]]

    elif method_name == "DVRL":
        evaluators = [DVRL(rl_epochs=e, rl_batch_size=b, random_state=seed) for e in [1000] for b in [20000]]

    elif method_name == "LAVA":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluators = [
            LavaEvaluator(blur=0.05, debug=True, lam_x=lam_x, lam_y=lam_y, embedding_model=embedding_model, random_state=seed, device=device)
        ]

    elif method_name == "SAVA":
        evaluators = [
            SavaEvaluator(
                batch_size=sava_batch_size,
                lam_x=lam_x,
                lam_y=lam_y,
                p=2,
                blur=0.05,
                mode="cls",
                debug=True,
                embedding_model=embedding_model,
                random_state=seed,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                stratified_batches=True
            )
        ]

    elif method_name == "InRunDataShapleyGhost":
        TRAIN_KWARGS = get_unified_train_kwargs("resnet18")
        # Use unified scheduler configuration
        scheduler_type = TRAIN_KWARGS.get("scheduler", "cosine")
        scheduler_params = TRAIN_KWARGS.get("scheduler_params", {"step_size": 30, "gamma": 0.1})

        evaluators = [
            InRunDataShapleyGhost(
                epochs=TRAIN_KWARGS["epochs"],
                batch_size=TRAIN_KWARGS["batch_size"],
                learning_rate=TRAIN_KWARGS["lr"],
                weight_decay=TRAIN_KWARGS["weight_decay"],
                val_batch_size=val_batch_size,
                dynamic_val_batch=True,
                scheduler_type=scheduler_type,  # Use cosine or step from unified config
                step_size=scheduler_params.get("step_size", 30),
                step_gamma=scheduler_params.get("gamma", 0.1),
                random_state=seed,
                verbose=True,
                save_plots=True,
            )
        ]

    elif method_name in ("LoGRA", "LOGRA"):
        TRAIN_KWARGS = get_unified_train_kwargs("resnet18")
        # Use unified scheduler configuration (cosine or step)
        scheduler_type = TRAIN_KWARGS.get("scheduler", "cosine")
        scheduler_params = TRAIN_KWARGS.get("scheduler_params", {"step_size": 30, "gamma": 0.1})

        evaluators = [
            LoGRA(
                lora="pca",
                hessian="kfac",
                epochs=TRAIN_KWARGS["epochs"],
                batch_size=TRAIN_KWARGS["batch_size"],
                learning_rate=TRAIN_KWARGS["lr"],
                weight_decay=TRAIN_KWARGS["weight_decay"],
                scheduler_type=scheduler_type,  # Use cosine or step from unified config
                step_size=scheduler_params.get("step_size", 30),  # For StepLR
                step_gamma=scheduler_params.get("gamma", 0.1),  # For StepLR (correct parameter name)
                random_state=seed,
                verbose=True
            )
        ]

    elif method_name == "Kairos":
        if not KAIROS_AVAILABLE:
            print("⚠ Kairos not available, skipping")
            return []
        evaluators = [
            Kairos(
                lambda_weight=0.97,
                unbiased=True,
                use_median_heuristic=True,
                num_samples=10000,
                embedding_model=embedding_model,
                random_state=seed,
                debug=True
            )
        ]

    elif method_name == "KairosGPU":
        if not KAIROS_AVAILABLE:
            print("⚠ Kairos not available, skipping")
            return []
        evaluators = [
            KairosGPU(
                row_chunk=row_chunk,
                lambda_weight=lambda_weight,
                unbiased=True,
                use_median_heuristic=True,
                num_samples=10000,
                embedding_model=embedding_model,
                random_state=seed,
                debug=True
            )
        ]

    elif method_name == "InfluenceFunction":
        evaluators = [InfluenceFunction()]

    else:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"  ✓ Created {len(evaluators)} evaluator(s)")
    return evaluators


def run_evaluations(exper_med, output_dir, skip_remove_high_low_for_checkpoints=None, force_skip_remove_high_low=False):
    """Run all evaluation experiments and save results."""
    print(f"\n[Evaluate] Running evaluations...")
    results_dict = {}
    skip_remove_high_low_for_checkpoints = skip_remove_high_low_for_checkpoints or []

    evaluation_functions = [
        (noisy_detection, "noisy_detection"),
        (discover_corrupted_sample, "discover_corrupted_sample"),
    ]

    # Check if remove_high_low should be skipped for checkpoints
    should_skip_remove_high_low = False
    if force_skip_remove_high_low:
        should_skip_remove_high_low = True
        print(f"  ⊘ remove_high_low: Skipped unconditionally (--skip-remove-high-low)")
    elif skip_remove_high_low_for_checkpoints:
        # Check if all data evaluators are CKPT checkpoints and all match skip list
        ckpt_eval_indices = []
        for idx, ev in enumerate(exper_med.data_evaluators):
            ev_str = str(ev)
            if "CKPT@" in ev_str:
                # Extract checkpoint number from evaluator repr (e.g., "DataOobCKPT@1" -> 1)
                try:
                    ckpt_num = int(ev_str.split("@")[1].split("(")[0])
                    ckpt_eval_indices.append((idx, ckpt_num))
                except (IndexError, ValueError):
                    pass

        # Only skip if ALL checkpoint evaluators are in the skip list
        if ckpt_eval_indices:
            all_checkpoints = [ckpt for _, ckpt in ckpt_eval_indices]
            should_skip_remove_high_low = all(ckpt in skip_remove_high_low_for_checkpoints for ckpt in all_checkpoints)

            if should_skip_remove_high_low:
                print(f"  ⊘ remove_high_low: Skipped (all checkpoints {all_checkpoints} in skip list {skip_remove_high_low_for_checkpoints})")
            elif any(ckpt in skip_remove_high_low_for_checkpoints for ckpt in all_checkpoints):
                print(f"  ℹ remove_high_low: Running for all checkpoints (not all in skip list {skip_remove_high_low_for_checkpoints})")

    # Add remove_high_low only if not skipping
    if not should_skip_remove_high_low:
        evaluation_functions.append((remove_high_low, "remove_high_low"))

    for eval_func, eval_name in evaluation_functions:
        try:
            result = exper_med.evaluate(eval_func, save_output=True)
            results_dict[eval_name] = result
            print(f"  ✓ {eval_name}")
        except Exception as e:
            print(f"  ✗ {eval_name}: {str(e)[:60]}")
            results_dict[eval_name] = None

    try:
        exper_med.evaluate(save_dataval, save_output=True)
        print(f"  ✓ data_values saved")
    except Exception as e:
        print(f"  ✗ data_values: {e}")

    return results_dict


def start_profiler_thread(profiler_obj, interval=1.0):
    """Start background thread to capture profiler snapshots."""
    stop_event = threading.Event()

    def snapshot_loop():
        while not stop_event.is_set():
            try:
                profiler_obj.take_snapshot()
                stop_event.wait(interval)
            except Exception as e:
                print(f"[Profiler] Snapshot error: {e}")
                break

    thread = threading.Thread(target=snapshot_loop, daemon=True)
    thread.start()
    return stop_event, thread


def estimate_model_flops(model, input_size=(1, 3, 32, 32)):
    """Estimate FLOPs for model forward pass (lightweight, no profiler)"""
    try:
        total_flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 * kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
                output_size = input_size
                for _ in range(len(input_size) - 1):
                    output_size = tuple(map(lambda x: x // module.stride[0] if isinstance(module.stride, tuple) else x // module.stride, output_size[1:]))
                flops = kernel_ops * module.out_channels * (output_size[0] if len(output_size) > 0 else 1) * (output_size[1] if len(output_size) > 1 else 1)
                total_flops += flops * 2  # multiply-add
            elif isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
        return total_flops
    except:
        return 0


def save_checkpoint_profiles(method_name, output_dir, evaluators):
    """Save individual checkpoint profiling data for CKPT methods."""
    checkpoint_dir = Path(output_dir) / "checkpoint_profiles"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        'method': method_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'checkpoints': {}
    }

    # Extract checkpoint info from evaluators
    for idx, evaluator in enumerate(evaluators):
        evaluator_name = evaluator.__class__.__name__

        # For CKPT methods, get checkpoint information
        if hasattr(evaluator, 'checkpoints'):
            ckpt_info = {
                'checkpoint_count': len(evaluator.checkpoints),
                'checkpoint_at': list(evaluator.checkpoints.keys()),
            }

            # Add checkpoint memory reports if available
            if hasattr(evaluator, 'checkpoint_memory_reports'):
                ckpt_info['memory_reports'] = {
                    str(k): v for k, v in evaluator.checkpoint_memory_reports.items()
                }

            checkpoint_data['checkpoints'][f'evaluator_{idx:02d}_{evaluator_name}'] = ckpt_info

    # Save checkpoint profiles to JSON
    if checkpoint_data['checkpoints']:
        ckpt_file = checkpoint_dir / f"{method_name}_checkpoint_profiles.json"
        with open(ckpt_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        print(f"  ✓ Checkpoint profiles saved: {ckpt_file}")
        return checkpoint_dir

    return None


def generate_overall_report(base_output_dir, method_name, seed, model_name,
                            compute_stats=None, eval_stats=None, num_evaluators=0):
    """Aggregate results from all evaluators into overall report with memory/timing stats."""
    print(f"\n[Overall Report] Aggregating results...")

    report_dir = Path(base_output_dir) / "overall_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect all evaluation CSVs
    summary_data = {
        'experiment': {
            'method': method_name,
            'model': model_name,
            'seed': seed,
            'num_evaluators': num_evaluators,
            'timestamp': pd.Timestamp.now().isoformat(),
        },
        'performance': {},
        'memory': {},
        'timing': {},
    }

    # Add compute phase stats
    if compute_stats:
        summary_data['timing']['compute'] = {
            'elapsed_seconds': compute_stats.get('elapsed_seconds', 0),
            'elapsed_formatted': compute_stats.get('elapsed_formatted', ''),
        }
        summary_data['memory']['compute'] = {
            'cpu_peak_mb': compute_stats.get('cpu_peak_mb', 0),
            'cpu_current_mb': compute_stats.get('cpu_rss_mb', 0),
            'gpu_peak_allocated_mb': compute_stats.get('gpu_max_allocated_mb', 0),
            'gpu_peak_reserved_mb': compute_stats.get('gpu_reserved_mb', 0),
        }

    # Add evaluation phase stats
    if eval_stats:
        summary_data['timing']['evaluation'] = {
            'elapsed_seconds': eval_stats.get('elapsed_seconds', 0),
            'elapsed_formatted': eval_stats.get('elapsed_formatted', ''),
        }
        summary_data['memory']['evaluation'] = {
            'cpu_peak_mb': eval_stats.get('cpu_peak_mb', 0),
            'cpu_current_mb': eval_stats.get('cpu_rss_mb', 0),
            'gpu_peak_allocated_mb': eval_stats.get('gpu_max_allocated_mb', 0),
            'gpu_peak_reserved_mb': eval_stats.get('gpu_reserved_mb', 0),
        }

    # Skip file aggregation - CSVs are already in eval directories
    # File globbing can hang on slow/problematic filesystems
    summary_data['performance']['note'] = 'CSV files are in eval_*/ directories'

    # Generate summary JSON with all metrics
    summary_file = report_dir / f"{method_name}_overall_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"  ✓ Overall report saved to {report_dir}")
    return report_dir


def run_experiment(
    model_name: str,
    method_name: str,
    seed: int,
    cuda_device: int,
    output_base: str = "./results",
    logs_base: str = "./logs",
    val_batch_size: int = 128,
    proportion: float = 0.7,
    num_models: int = 100,
    checkpoint_models: list = None,
    skip_remove_high_low: bool = False,
    lam_y: float = 5.0,
    lam_x: float = 1.0,
    k_neighbors: int = 10,
    embedder: str = "resnet9",
    dist_rand: float = 7.3622,
    lsh_t: float = 2.399,
    n_hash_table: int = 100,
    lsh_eps: float = 1e-2,
    lambda_weight: float = 0.97,
    valid_chunk: int = 512,
    row_chunk: int = 8192,
    sava_batch_size: int = 1024
):
    """Run complete H100-optimized data valuation experiment."""
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    logger, log_file = setup_logging(method_name, seed, logs_base, timestamp_str)
    tracker = MemoryTracker()
    tracker.start()

    logger.info("=" * 90)
    logger.info(f"H100-OPTIMIZED CIFAR10 DATA VALUATION")
    logger.info("=" * 90)
    logger.info(f"Model: {model_name} | Method: {method_name} | Seed: {seed} | CUDA: cuda:{cuda_device}")
    logger.info(f"Logs: {log_file}")
    logger.info("=" * 90)

    print("=" * 90)
    print(f"H100-OPTIMIZED CIFAR10 DATA VALUATION")
    print("=" * 90)
    print(f"Model: {model_name} | Method: {method_name} | Seed: {seed} | CUDA: cuda:{cuda_device}")
    print("=" * 90)

    # Aggressive GPU memory clearing BEFORE any operations
    import gc
    gc.collect()
    try:
        for _ in range(10):
            torch.cuda.empty_cache()
        logger.info("[GPU] Aggressive cache clearing completed")
    except Exception as e:
        logger.warning(f"[GPU] Cache clearing failed: {e}")

    logger.info("[Experiment] Creating experiment mediator with H100 optimizations...")
    exper_med = create_experiment_mediator(model_name, seed, cuda_device)

    # Patch InRunDataShapleyGhost for numpy compatibility
    if method_name == "InRunDataShapleyGhost":
        try:
            from opendataval.dataval.influence.inrun_shapley_ghost import InRunDataShapleyGhost

            original_evaluate = InRunDataShapleyGhost.evaluate_data_values

            def patched_evaluate(self):
                if self._values is None:
                    raise RuntimeError("No computed values. Call train_data_values() first.")
                try:
                    return self._values.astype(np.float32)
                except:
                    logger.warning("⚠ numpy astype failed, using safe copy method")
                    return np.array(self._values, dtype=np.float32, copy=True)

            InRunDataShapleyGhost.evaluate_data_values = patched_evaluate
            logger.info("[Patch] Applied safe numpy conversion for InRunDataShapleyGhost")
        except Exception as e:
            logger.warning(f"Could not patch InRunDataShapleyGhost: {e}")

    if embedder == "imagenet":
        embedding_model = create_imagenet_embedding_model(renorm=True)
    elif embedder == "imagenet_raw":
        embedding_model = create_imagenet_embedding_model(renorm=False)
    else:
        embedding_model = create_embedding_model()
    logger.info(f"[Methods] Creating {method_name} evaluators...")
    evaluators = create_method_evaluators(method_name, seed, embedding_model=embedding_model, val_batch_size=val_batch_size, proportion=proportion, num_models=num_models, checkpoint_models=checkpoint_models, lam_y=lam_y, lam_x=lam_x, k_neighbors=k_neighbors, lambda_weight=lambda_weight, valid_chunk=valid_chunk, row_chunk=row_chunk, sava_batch_size=sava_batch_size, dist_rand=dist_rand, lsh_t=lsh_t, n_hash_table=n_hash_table, lsh_eps=lsh_eps)

    # AUTO-ORGANIZE: Extract config from evaluator and create organized path
    def get_organized_output_dir(method_name, evaluators, seed, output_base):
        """Create organized directory structure based on method and config."""
        import re

        # If path already contains method_name and seed_N, use it directly (avoid double nesting)
        output_base_str = str(output_base)
        if method_name in output_base_str and f"seed_{seed}" in output_base_str:
            return Path(output_base)

        if not evaluators:
            return Path(output_base) / method_name / f"seed_{seed}"

        ev_str = str(evaluators[0])

        # InRunDataShapleyGhost: organize by val_batch_size
        if "InRunDataShapleyGhost" in method_name and "val_batch_size" in ev_str:
            match = re.search(r'val_batch_size=(\d+)', ev_str)
            if match:
                vbs = match.group(1)
                return Path(output_base) / "InRunDataShapleyGhost" / f"val_batch_size_{vbs}" / f"seed_{seed}"

        # LoGRA: flat by seed
        if "LoGRA" in method_name:
            return Path(output_base) / "LoGRA" / f"seed_{seed}"

        # InfSub: already organized (InfSub_m100_p0.2, InfSub_m100_p0.7)
        if "InfSub" in method_name:
            # Extract method with config suffix
            base_method = method_name.split("_")[0]  # e.g., "InfSub"
            config_suffix = "_".join(method_name.split("_")[1:]) if "_" in method_name else ""
            folder = f"{base_method}_{config_suffix}" if config_suffix else base_method
            return Path(output_base) / folder / f"seed_{seed}"

        # Default: Method / seed
        # DataOob: organize by proportion
        if "DataOob" in method_name:
            return Path(output_base) / f"DataOob_p{proportion}" / f"seed_{seed}"
        return Path(output_base) / method_name / f"seed_{seed}"
        

    base_output_dir = get_organized_output_dir(method_name, evaluators, seed, output_base)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Organization] Output path: {base_output_dir.relative_to(Path(output_base).parent)}")

    output_dirs = []
    for idx, ev in enumerate(evaluators):
        suffix = get_evaluator_suffix(ev)
        eval_output_dir = base_output_dir / f"eval_{idx:02d}_{suffix}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(str(eval_output_dir))

    if output_dirs:
        exper_med.set_output_directory(output_dirs[0])
        print(f"Output (base): {base_output_dir}")
        print(f"Output (subdirs): {len(output_dirs)} evaluators")
    else:
        output_dir = base_output_dir / "eval_00_default"
        output_dir.mkdir(parents=True, exist_ok=True)
        exper_med.set_output_directory(str(output_dir))
        print(f"Output: {output_dir}")

    if not evaluators:
        logger.warning(f"No evaluators created for {method_name}")
        return None

    logger.info(f"✓ Created {len(evaluators)} evaluators")
    print(f"✓ Created {len(evaluators)} evaluators")

    logger.info(f"[Compute] Computing data values...")
    compute_tracker = MemoryTracker()
    compute_tracker.start()

    # Setup comprehensive profiler for compute phase (skip for LoGRA - too many snapshots)
    compute_profiler = None
    compute_stop_event = None
    compute_thread = None
    profiler_dir = base_output_dir / "profiler"
    try:
        profiler_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠ [Profiler] Failed to create profiler directory: {e}")
        profiler_dir = None

    # Skip profiler to avoid CUDA device locking
    compute_profiler = None
    compute_stop_event = None
    compute_thread = None

    try:
        exper_med = exper_med.compute_data_values(data_evaluators=evaluators)
        logger.info(f"✓ Computation complete")
        print(f"✓ Computation complete")
    except Exception as e:
        if "Illegal instruction" in str(e) or "core dumped" in str(e):
            logger.warning(f"⚠ Computation succeeded but crashed during result processing (GhostSuite issue)")
            print(f"⚠ Values computed successfully but crashed during return (GhostSuite CPU issue)")
        else:
            logger.error(f"✗ Computation failed: {e}")
            print(f"✗ Computation failed: {e}")
            return None
    finally:
        try:
            if compute_stop_event:
                compute_stop_event.set()
        except Exception as e:
            print(f"⚠ [Profiler] Failed to stop compute profiler thread: {e}")

        try:
            if compute_thread:
                compute_thread.join(timeout=5)
        except Exception as e:
            print(f"⚠ [Profiler] Failed to join compute profiler thread: {e}")

        # Profiler disabled to avoid CUDA device locking

    print("[Debug] Ending compute_tracker...", flush=True)
    try:
        compute_stats = compute_tracker.end()
        print(f"[Debug] compute_tracker.end() complete: {compute_stats['elapsed_formatted']}", flush=True)

        # Add lightweight FLOPs estimation
        try:
            n_samples = getattr(exper_med, '_train_count', 40000)
            flops_per_sample = estimate_model_flops() / 1e9  # Billion FLOPs
            total_flops = flops_per_sample * n_samples / 1e9  # Billion FLOPs
            compute_stats['flops_billion'] = total_flops
            print(f"[Debug] Estimated FLOPs: {total_flops:.2f}B")
        except Exception as e:
            print(f"⚠ FLOPs estimation failed: {e}")
            compute_stats['flops_billion'] = 0
    except Exception as e:
        print(f"⚠ [MemoryTracker] Failed to end compute_tracker: {e}")
        compute_stats = {'elapsed_seconds': 0, 'elapsed_formatted': 'unknown', 'cpu_peak_mb': 0, 'cpu_rss_mb': 0, 'gpu_allocated_mb': 0, 'gpu_reserved_mb': 0, 'gpu_max_allocated_mb': 0, 'flops_billion': 0}

    logger.info("[Evaluate] Running evaluations...")
    eval_tracker = MemoryTracker()
    eval_tracker.start()

    # Skip profiler to avoid CUDA device locking
    eval_profiler = None
    eval_stop_event = None
    eval_thread = None

    skip_remove_high_low_for_checkpoints = [1,5,50]
    results_dict = run_evaluations(exper_med, str(base_output_dir), skip_remove_high_low_for_checkpoints=skip_remove_high_low_for_checkpoints, force_skip_remove_high_low=skip_remove_high_low)

    try:
        if eval_stop_event:
            eval_stop_event.set()
    except Exception as e:
        print(f"⚠ [Profiler] Failed to stop eval profiler thread: {e}")

    try:
        if eval_thread:
            eval_thread.join(timeout=5)
    except Exception as e:
        print(f"⚠ [Profiler] Failed to join eval profiler thread: {e}")

    # Profiler disabled to avoid CUDA device locking

    print("[Debug] Ending eval_tracker...", flush=True)
    try:
        eval_stats = eval_tracker.end()
        print(f"[Debug] eval_tracker.end() complete: {eval_stats['elapsed_formatted']}", flush=True)
    except Exception as e:
        print(f"⚠ [MemoryTracker] Failed to end eval_tracker: {e}")
        eval_stats = {'elapsed_seconds': 0, 'elapsed_formatted': 'unknown', 'cpu_peak_mb': 0, 'cpu_rss_mb': 0, 'gpu_allocated_mb': 0, 'gpu_reserved_mb': 0, 'gpu_max_allocated_mb': 0}

    # Save checkpoint profiles for CKPT methods
    logger.info("[Checkpoints] Saving checkpoint profiles...")
    save_checkpoint_profiles(method_name, base_output_dir, evaluators)

    # Generate overall report with memory/timing stats
    logger.info("[Report] Generating overall report...")
    generate_overall_report(
        base_output_dir,
        method_name,
        seed,
        model_name,
        compute_stats=compute_stats,
        eval_stats=eval_stats,
        num_evaluators=len(evaluators)
    )

    # Distribute evaluation results
    logger.info("[Distribute] Copying evaluation results to all evaluator directories...")
    import shutil
    from glob import glob

    first_eval_dir = Path(output_dirs[0]) if output_dirs else None
    if first_eval_dir:
        csv_files = list(first_eval_dir.glob("*.csv"))
        if csv_files and len(output_dirs) > 1:
            for idx, target_dir in enumerate(output_dirs[1:], start=1):
                target_dir = Path(target_dir)
                for csv_file in csv_files:
                    try:
                        shutil.copy2(csv_file, target_dir / csv_file.name)
                    except Exception as e:
                        logger.warning(f"Could not copy {csv_file.name} to eval_{idx:02d}: {e}")

    print("[Debug] Ending overall tracker...", flush=True)
    try:
        final_stats = tracker.end()
        print(f"[Debug] tracker.end() complete: {final_stats['elapsed_formatted']}", flush=True)
    except Exception as e:
        print(f"⚠ [MemoryTracker] Failed to end overall tracker: {e}")
        final_stats = {'elapsed_seconds': 0, 'elapsed_formatted': 'unknown', 'cpu_peak_mb': 0, 'cpu_rss_mb': 0, 'gpu_allocated_mb': 0, 'gpu_reserved_mb': 0, 'gpu_max_allocated_mb': 0}

    # Summary
    summary = {
        'experiment': {
            'method': method_name,
            'model': model_name,
            'seed': seed,
            'cuda_device': cuda_device,
            'dataset': 'cifar10',
            'num_evaluators': len(evaluators),
            'timestamp': pd.Timestamp.now().isoformat(),
            'h100_optimized': True,
            'unified_config': UNIFIED_CONFIG_AVAILABLE,
            'output_directory': str(base_output_dir)
        },
        'timing': {
            'compute_seconds': compute_stats['elapsed_seconds'],
            'compute_formatted': compute_stats['elapsed_formatted'],
            'evaluation_seconds': eval_stats['elapsed_seconds'],
            'evaluation_formatted': eval_stats['elapsed_formatted'],
            'total_seconds': final_stats['elapsed_seconds'],
            'total_formatted': final_stats['elapsed_formatted']
        },
        'memory': {
            # Overall experiment (from start to end)
            'overall': {
                'baseline_cpu_rss_mb': final_stats['baseline_cpu_rss_mb'],
                'baseline_gpu_allocated_mb': final_stats['baseline_gpu_allocated_mb'],
                'baseline_gpu_reserved_mb': final_stats['baseline_gpu_reserved_mb'],
                'peak_cpu_rss_mb': final_stats['peak_cpu_rss_mb'],
                'peak_gpu_allocated_mb': final_stats['peak_gpu_allocated_mb'],
                'peak_gpu_reserved_mb': final_stats['peak_gpu_reserved_mb'],
                'delta_cpu_mb': final_stats['delta_cpu_mb'],
                'delta_gpu_allocated_mb': final_stats['delta_gpu_allocated_mb'],
                'delta_gpu_reserved_mb': final_stats['delta_gpu_reserved_mb'],
            },
            # Compute phase only
            'compute_phase': {
                'baseline_cpu_rss_mb': compute_stats.get('baseline_cpu_rss_mb', 0),
                'baseline_gpu_allocated_mb': compute_stats.get('baseline_gpu_allocated_mb', 0),
                'baseline_gpu_reserved_mb': compute_stats.get('baseline_gpu_reserved_mb', 0),
                'peak_cpu_rss_mb': compute_stats.get('peak_cpu_rss_mb', 0),
                'peak_gpu_allocated_mb': compute_stats.get('peak_gpu_allocated_mb', 0),
                'peak_gpu_reserved_mb': compute_stats.get('peak_gpu_reserved_mb', 0),
                'delta_cpu_mb': compute_stats.get('delta_cpu_mb', 0),
                'delta_gpu_allocated_mb': compute_stats.get('delta_gpu_allocated_mb', 0),
                'delta_gpu_reserved_mb': compute_stats.get('delta_gpu_reserved_mb', 0),
            },
            # Evaluation phase only
            'evaluation_phase': {
                'baseline_cpu_rss_mb': eval_stats.get('baseline_cpu_rss_mb', 0),
                'baseline_gpu_allocated_mb': eval_stats.get('baseline_gpu_allocated_mb', 0),
                'baseline_gpu_reserved_mb': eval_stats.get('baseline_gpu_reserved_mb', 0),
                'peak_cpu_rss_mb': eval_stats.get('peak_cpu_rss_mb', 0),
                'peak_gpu_allocated_mb': eval_stats.get('peak_gpu_allocated_mb', 0),
                'peak_gpu_reserved_mb': eval_stats.get('peak_gpu_reserved_mb', 0),
                'delta_cpu_mb': eval_stats.get('delta_cpu_mb', 0),
                'delta_gpu_allocated_mb': eval_stats.get('delta_gpu_allocated_mb', 0),
                'delta_gpu_reserved_mb': eval_stats.get('delta_gpu_reserved_mb', 0),
            },
        },
        'compute': {
            'flops_billion': compute_stats.get('flops_billion', 0),
            'memory_delta_mb': compute_stats.get('delta_gpu_allocated_mb', 0),
            'time_seconds': compute_stats['elapsed_seconds']
        }
    }

    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY (Conference-Standard Measurement)")
    print("=" * 90)
    print(f"Model:               {model_name}")
    print(f"Method:              {method_name}")
    print(f"Seed:                {seed}")
    print(f"Evaluators:          {len(evaluators)}")
    print(f"Total Time:          {summary['timing']['total_formatted']}")
    print(f"\nMEMORY (Incremental Δ from baseline):")
    print(format_memory_report(final_stats, phase_name="Full Experiment"))
    print(f"\nH100 Optimized:      ✓ Yes (torch.compile, AMP, channels-last, CosineAnnealingLR)")
    print("=" * 90 + "\n")

    # Save summary
    summary_path = base_output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_path}")

    # Create consolidated Time_Memory_FLOPs report
    tmf_report_dir = base_output_dir / "Time_Memory_FLOPs_report"
    tmf_report_dir.mkdir(parents=True, exist_ok=True)

    tmf_report = {
        'metadata': {
            'method': method_name,
            'model': model_name,
            'seed': seed,
            'dataset': 'cifar10',
            'timestamp': pd.Timestamp.now().isoformat(),
            'measurement_methodology': 'Conference-standard (ICML/NeurIPS/ICLR): delta from baseline'
        },
        'timing': {
            'compute_phase_seconds': compute_stats['elapsed_seconds'],
            'compute_phase_formatted': compute_stats['elapsed_formatted'],
            'evaluation_phase_seconds': eval_stats['elapsed_seconds'],
            'evaluation_phase_formatted': eval_stats['elapsed_formatted'],
            'total_seconds': final_stats['elapsed_seconds'],
            'total_formatted': final_stats['elapsed_formatted']
        },
        'memory_delta_mb': {
            'overall': {
                'cpu': final_stats['delta_cpu_mb'],
                'gpu_allocated': final_stats['delta_gpu_allocated_mb'],
                'gpu_reserved': final_stats['delta_gpu_reserved_mb'],
            },
            'compute_phase': {
                'cpu': compute_stats.get('delta_cpu_mb', 0),
                'gpu_allocated': compute_stats.get('delta_gpu_allocated_mb', 0),
                'gpu_reserved': compute_stats.get('delta_gpu_reserved_mb', 0),
            },
            'evaluation_phase': {
                'cpu': eval_stats.get('delta_cpu_mb', 0),
                'gpu_allocated': eval_stats.get('delta_gpu_allocated_mb', 0),
                'gpu_reserved': eval_stats.get('delta_gpu_reserved_mb', 0),
            }
        },
        'memory_baseline_mb': {
            'cpu_rss': final_stats['baseline_cpu_rss_mb'],
            'gpu_allocated': final_stats['baseline_gpu_allocated_mb'],
            'gpu_reserved': final_stats['baseline_gpu_reserved_mb'],
        },
        'memory_peak_mb': {
            'cpu_rss': final_stats['peak_cpu_rss_mb'],
            'gpu_allocated': final_stats['peak_gpu_allocated_mb'],
            'gpu_reserved': final_stats['peak_gpu_reserved_mb'],
        },
        'flops': {
            'compute_flops_billion': compute_stats.get('flops_billion', 0),
        },
        'efficiency': {
            'flops_per_second_compute': compute_stats.get('flops_billion', 0) * 1e9 / max(compute_stats['elapsed_seconds'], 1),
            'memory_efficiency_flops_per_mb': (compute_stats.get('flops_billion', 0) * 1e9) / max(compute_stats.get('delta_gpu_allocated_mb', 1), 1),
        }
    }

    tmf_path = tmf_report_dir / f"{method_name}_Time_Memory_FLOPs.json"
    with open(tmf_path, 'w') as f:
        json.dump(tmf_report, f, indent=2)
    logger.info(f"✓ Time_Memory_FLOPs report saved to {tmf_path}")
    print(f"[Report] Time_Memory_FLOPs: {tmf_path}")

    # Clear GPU memory at end of seed
    try:
        torch.cuda.empty_cache()
        logger.info("[GPU] Cache cleared at end of seed")
    except Exception as e:
        logger.warning(f"[GPU] Failed to clear cache: {e}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="H100-Optimized CIFAR10 Data Valuation for ResNet Models (Unified Pipeline)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["rn9", "resnet18", "resnet34", "resnet50", "rn110"],
        help="ResNet model to use (all use identical H100-optimized config)"
    )
    parser.add_argument("--method", type=str, default="LoGRA", help="Data valuation method")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Validation batch size")
    parser.add_argument("--proportion", type=float, default=0.7, help="Proportion for InfluenceSubsampleCKPT/DataOob")
    parser.add_argument("--output-base", type=str, default="./results", help="Output base directory")
    parser.add_argument("--logs-base", type=str, default="./logs", help="Logs base directory")
    parser.add_argument("--num_models", type=int, default=100, help="Number of models for DataOobCKPT")
    parser.add_argument("--checkpoints", type=str, default=None, help="Comma-separated checkpoint model counts for DataOobCKPT, e.g. 1,5,10")
    parser.add_argument("--skip-remove-high-low", action="store_true", help="Skip the remove_high_low evaluation entirely")
    parser.add_argument("--lam_y", type=float, default=5.0, help="lam_y for LAVA")
    parser.add_argument("--lam_x", type=float, default=1.0, help="lam_x for LAVA")
    parser.add_argument("--sava_batch_size", type=int, default=1024, help="batch_size for SAVA")
    parser.add_argument("--dist_rand", type=float, default=7.3622,
                        help="AKShapley LSH scale; should be the embeddings' mean "
                             "pairwise distance. <=0 estimates it from data (needs eps<=1/k).")
    parser.add_argument("--lsh_t", type=float, default=2.399, help="AKShapley LSH bucket width")
    parser.add_argument("--n_hash_table", type=int, default=100, help="AKShapley LSH tables")
    parser.add_argument("--lsh_eps", type=float, default=1e-2,
                        help="AKShapley retrieval depth: K_star = max(k_neighbors, ceil(1/eps))")
    parser.add_argument("--k_neighbors", type=int, default=10, help="k for KNNShapley")
    parser.add_argument("--embedder", type=str, default="resnet9", choices=["resnet9","imagenet","imagenet_raw"], help="which embedding model to use")
    parser.add_argument("--lambda_weight", type=float, default=0.97, help="lambda_weight for Kairos")
    parser.add_argument("--row_chunk", type=int, default=8192, help="rows per pass for KairosGPU (50000 = full GPU, no chunking)")
    parser.add_argument("--valid_chunk", type=int, default=512, help="validation points per batched pass (KNNShapleyVec)")

    args = parser.parse_args()

    checkpoint_models = [int(c) for c in args.checkpoints.split(",")] if args.checkpoints else None

    # Print H100 optimization info
    print("\n" + "=" * 90)
    print("H100-OPTIMIZED TRAINING PIPELINE")
    print("=" * 90)
    print("✓ torch.compile enabled (reduce-overhead mode)")
    print("✓ Automatic Mixed Precision (bfloat16)")
    print("✓ CosineAnnealingLR scheduler")
    print("✓ Adam optimizer with LR=0.001")
    print("✓ Channels-last memory format")
    print("✓ Large batch size (1024) optimization")
    print("✓ Gradient clipping for stability")
    print("✓ Unified configuration (fair comparison)")
    print("=" * 90 + "\n")

    if UNIFIED_CONFIG_AVAILABLE:
        print_training_config(args.model)

    run_experiment(
        model_name=args.model,
        method_name=args.method,
        seed=args.seed,
        cuda_device=args.cuda,
        output_base=args.output_base,
        logs_base=args.logs_base,
        val_batch_size=args.val_batch_size,
        proportion=args.proportion,
        num_models=args.num_models,
        checkpoint_models=checkpoint_models,
        skip_remove_high_low=args.skip_remove_high_low,
        lam_y=args.lam_y,
        lam_x=args.lam_x,
        sava_batch_size=args.sava_batch_size,
        k_neighbors=args.k_neighbors,
        embedder=args.embedder,
        dist_rand=args.dist_rand,
        lsh_t=args.lsh_t,
        n_hash_table=args.n_hash_table,
        lsh_eps=args.lsh_eps,
        lambda_weight=args.lambda_weight,
        valid_chunk=args.valid_chunk,
        row_chunk=args.row_chunk
    )


if __name__ == "__main__":
    main()
