import os
import time
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings
import random
import gc
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import hashlib
import torch.nn.functional as F

# Import your baseline modules (adjust paths as needed)
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
    tiny_imagenet_path: str = '/home/mehdi.touil/ondemand/data/tiny-imagenet-200'
    num_classes: int = 200  # Default, will be updated if subset_classes is set
    feature_dim: int = 2048  # ResNet50 feature dimension
    subset_classes: Optional[int] = None
    subset_seed: int = 0
    embed_batch_size: int = 256
    # Scenarios to run: subset of ['label','feature','size','replication'] or 'all'
    scenarios: List[str] = field(default_factory=lambda: ['label','feature','size','replication'])


class TinyImageNetSubset:
    """
    Helper class to handle Tiny ImageNet subset selection and label remapping
    """
    def __init__(self, root_path: Path, num_classes: int = 100, seed: int = 0):
        self.root_path = Path(root_path)
        self.num_classes = num_classes
        self.seed = seed
        
        # Get all class folders
        train_dir = self.root_path / 'train'
        self.all_class_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        self.total_classes = len(self.all_class_folders)
        
        print(f"[INFO] Total available classes: {self.total_classes}")
        
        # Select subset of classes
        self.selected_folders = self._select_classes()
        
        # Create mapping from original wnid to new class index (0 to num_classes-1)
        self.wnid_to_new_idx = {
            folder.name: idx for idx, folder in enumerate(self.selected_folders)
        }
        
        # Also create reverse mapping for debugging
        self.new_idx_to_wnid = {
            idx: folder.name for idx, folder in enumerate(self.selected_folders)
        }
        
        print(f"[INFO] Selected {len(self.selected_folders)} classes")
        print(f"[INFO] New label range: 0 to {len(self.selected_folders)-1}")
        
        # Print first 5 classes as sample
        sample_classes = list(self.wnid_to_new_idx.items())[:5]
        print(f"[INFO] Sample class mapping: {sample_classes}")
    
    def _select_classes(self) -> List[Path]:
        """Select subset of classes reproducibly"""
        rng = np.random.RandomState(self.seed)
        
        if self.num_classes >= self.total_classes:
            print(f"[INFO] Requested {self.num_classes} classes, but only {self.total_classes} available. Using all.")
            return self.all_class_folders
        
        # Randomly select num_classes without replacement
        indices = rng.choice(self.total_classes, size=self.num_classes, replace=False)
        indices = sorted(indices)  # Keep them sorted for reproducibility
        
        selected = [self.all_class_folders[i] for i in indices]
        return selected
    
    def remap_label(self, wnid: str) -> int:
        """Map original WordNet ID to new class index (0 to num_classes-1)"""
        if wnid not in self.wnid_to_new_idx:
            return -1  # Class not in subset
        return self.wnid_to_new_idx[wnid]
    
    def get_class_name(self, idx: int) -> str:
        """Get original WordNet ID for a new class index"""
        return self.new_idx_to_wnid.get(idx, "unknown")


class RobustnessTester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._select_device(cfg.gpu)
        # New structure: output_dir/dataset_name/scenario/seed_X/
        # Keep output_root pointing at output_dir/dataset_name
        self.output_root = Path(cfg.output_dir) / cfg.dataset
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.train_features = None      # Features for OT/RV
        self.train_labels = None        # Labels (remapped to 0-99)
        self.train_images = None         # Transformed images used for feature extraction (224x224, normalized)
        self.train_images_orig = None    # Original images for DAVINZ (64x64, unnormalized)
        self.train_image_labels = None   # Labels for images
        
        self.val_features = None
        self.val_labels = None
        self.val_images = None           # Transformed val images (224x224, normalized)
        self.val_images_orig = None      # Original val images for DAVINZ (64x64)
        self.val_image_labels = None
        
        self.train_idx = None
        self.val_idx = None
        
        # Class information
        self.num_classes = cfg.num_classes
        self.class_to_idx = None
        self.subset_handler = None
        
        # Feature extraction time tracking
        self.feature_extraction_time = 0.0
        self.feature_extraction_log = []  # records per-split timings and memory

    def _select_device(self, gpu_arg: Optional[str]):
        gpu_list = [int(x) for x in str(gpu_arg).split(',') if x.strip() != ''] if gpu_arg else []
        if torch.cuda.is_available() and gpu_list:
            try:
                torch.cuda.set_device(int(gpu_list[0]))
                return torch.device(f'cuda:{int(gpu_list[0])}')
            except Exception:
                return torch.device('cuda')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _clear_cuda(self):
        """Clear CUDA cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def _clear_cuda_and_check(self, stage: str = ''):
        """Clear CUDA, run GC, and verify GPU memory is low; print debug info."""
        if not torch.cuda.is_available():
            print(f"[DEBUG-CUDA] CUDA not available; skipping clear/check {stage}")
            return

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()
        gc.collect()

        # Report memory usage
        try:
            alloc = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        except Exception:
            # Fallback to default device stats
            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

        print(f"[DEBUG-CUDA] Stage={stage} allocated={alloc} reserved={reserved}")

        # If memory still allocated, try another empty and collect
        if alloc > 0:
            print(f"[WARN-CUDA] Memory still allocated after clear at stage={stage} ({alloc} bytes). Emptying cache again.")
            try:
                torch.cuda.empty_cache()
                gc.collect()
                alloc2 = torch.cuda.memory_allocated(self.device)
                reserved2 = torch.cuda.memory_reserved(self.device)
                print(f"[DEBUG-CUDA] After retry allocated={alloc2} reserved={reserved2}")
            except Exception:
                pass

    def _gpu_warmup(self):
        """Run a tiny op on GPU to avoid one-time init overhead in timings."""
        if torch.cuda.is_available():
            try:
                with torch.no_grad():
                    tmp = torch.ones((1,), device=self.device)
                    tmp.mul_(1.0)
                    torch.cuda.synchronize()
            except Exception:
                pass

    def _time_start(self):
        """Return timing tokens for wall-clock and CUDA events (if available)."""
        info = {'wall_start': time.time(), 'gpu_start': None}
        if torch.cuda.is_available():
            try:
                info['gpu_start'] = torch.cuda.Event(enable_timing=True)
                info['gpu_end'] = torch.cuda.Event(enable_timing=True)
                info['gpu_start'].record()
            except Exception:
                info['gpu_start'] = None
                info['gpu_end'] = None
        return info

    def _time_end(self, info):
        """Given the tokens from `_time_start`, return dict with wall_s and gpu_s (or None).

        gpu_s is seconds measured by CUDA events (may be None on CPU or if events failed).
        wall_s is wall-clock seconds (always available).
        """
        wall_s = time.time() - info.get('wall_start', time.time())
        gpu_s = None
        if torch.cuda.is_available() and info.get('gpu_start') is not None and info.get('gpu_end') is not None:
            try:
                info['gpu_end'].record()
                torch.cuda.synchronize()
                # elapsed_time returns milliseconds
                ms = info['gpu_start'].elapsed_time(info['gpu_end'])
                gpu_s = float(ms) / 1000.0
            except Exception:
                gpu_s = None
        return {'wall_s': float(wall_s), 'gpu_s': gpu_s}

    # -------------------------
    # Feature extraction utilities
    # -------------------------
    def _get_feature_extractor(self):
        """Get pretrained ResNet50 feature extractor"""
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.eval()
        resnet50 = resnet50.to(self.device)
        extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
        extractor.eval()
        return extractor

    def _extract_features_batched(self, images: torch.Tensor, extractor) -> torch.Tensor:
        """Extract features in batches with memory management"""
        if images is None or len(images) == 0:
            return torch.empty(0, self.cfg.feature_dim)
        
        batch_size = min(self.cfg.embed_batch_size, len(images))
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                out = extractor(batch)
                out = out.view(out.size(0), -1).cpu()
                features.append(out)
                
                # Clear cache periodically
                if i % (batch_size * 4) == 0:
                    self._clear_cuda()
        
        return torch.cat(features, dim=0)

    # -------------------------
    # Main data loading function for Tiny ImageNet with subset selection
    # -------------------------
    def load_tiny_imagenet_subset(self):
        """
        Load Tiny ImageNet with class subset selection and proper label remapping
        """
        base_path = Path(self.cfg.tiny_imagenet_path)
        if not base_path.exists():
            raise RuntimeError(f"Tiny ImageNet path not found: {base_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading Tiny ImageNet with subset selection")
        print(f"{'='*60}")
        
        # Initialize subset handler
        target_classes = self.cfg.subset_classes or 100
        self.subset_handler = TinyImageNetSubset(
            base_path, 
            num_classes=target_classes,
            seed=self.cfg.subset_seed
        )
        
        # Update num_classes in config
        self.num_classes = len(self.subset_handler.selected_folders)
        self.cfg.num_classes = self.num_classes
        print(f"[INFO] Updated num_classes to: {self.num_classes}")
        # Update output root to include dataset name suffixed with num_classes
        try:
            dataset_name = f"tiny_imagenet_{self.num_classes}"
            # Point output_root to dataset-level directory (no 'robustness' suffix)
            self.output_root = Path(self.cfg.output_dir) / dataset_name
            self.output_root.mkdir(parents=True, exist_ok=True)
            # Save the selected class list to outputs for reproducibility
            classes_file = self.output_root / f"classes_{self.num_classes}_seed{self.cfg.subset_seed}.txt"
            try:
                with open(classes_file, 'w') as cf:
                    for idx in range(self.num_classes):
                        wnid = self.subset_handler.get_class_name(idx)
                        cf.write(f"{idx}\t{wnid}\n")
                print(f"[INFO] Saved selected class list to {classes_file}")
            except Exception as e:
                print(f"[WARN] Failed to save class list: {e}")
        except Exception:
            # If anything fails here, fall back to existing output_root
            pass
        
        # Setup transforms
        # For feature extraction (OT/RV): 224x224 with ImageNet normalization
        transform_feat = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # For DAVINZ: original Tiny ImageNet resolution 64x64 (no ImageNet normalization)
        transform_orig = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        # Get feature extractor
        feature_extractor = self._get_feature_extractor()
        # Cache extractor for later use when we need to compute embeddings from corrupted images
        self.feature_extractor = feature_extractor
        
        # Process training set
        print(f"\n[INFO] Processing training set...")
        train_images_list = []        # transformed for feature extraction
        train_images_list_orig = []   # original 64x64 for DAVINZ
        train_labels_list = []
        
        train_dir = base_path / 'train'
        for class_folder in self.subset_handler.selected_folders:
            wnid = class_folder.name
            new_label = self.subset_handler.remap_label(wnid)
            
            # Get images for this class
            class_img_dir = class_folder / 'images'
            if not class_img_dir.exists():
                print(f"[WARN] Images directory not found for {wnid}, skipping")
                continue
            
            for img_file in class_img_dir.glob('*.JPEG'):
                # Load and transform image (keep both original-size and feature transform)
                img = Image.open(img_file).convert('RGB')
                img_tensor_feat = transform_feat(img)
                img_tensor_orig = transform_orig(img)

                train_images_list.append(img_tensor_feat)
                train_images_list_orig.append(img_tensor_orig)
                train_labels_list.append(new_label)
        
        # Stack training images and extract features
        if train_images_list:
            self.train_images = torch.stack(train_images_list)
            self.train_images_orig = torch.stack(train_images_list_orig)
            self.train_image_labels = torch.tensor(train_labels_list, dtype=torch.long)
            
            print(f"[INFO] Training images (for features) shape: {self.train_images.shape}")
            print(f"[INFO] Training images (original) shape: {self.train_images_orig.shape}")
            print(f"[INFO] Training labels shape: {self.train_image_labels.shape}")
            print(f"[INFO] Training label range: {self.train_image_labels.min()} to {self.train_image_labels.max()}")
            
            # Extract features for OT/RV (track wall + GPU time and peak CUDA memory)
            print(f"[INFO] Extracting features for training set...")
            try:
                self._gpu_warmup()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            tkn = self._time_start()
            self.train_features = self._extract_features_batched(self.train_images, feature_extractor)
            times = self._time_end(tkn)
            peak_mem = 0
            if torch.cuda.is_available():
                try:
                    peak_mem = int(torch.cuda.max_memory_allocated(self.device))
                except Exception:
                    try:
                        peak_mem = int(torch.cuda.max_memory_allocated())
                    except Exception:
                        peak_mem = 0
            self.feature_extraction_time += float(times.get('wall_s', 0.0))
            print(f"[INFO] Training features shape: {self.train_features.shape}")
            print(f"[INFO] Training embedding wall_time_s={times.get('wall_s'):.4f} gpu_time_s={times.get('gpu_s')} peak_cuda_bytes={peak_mem}")
            self.feature_extraction_log.append({
                'split': 'train',
                'num_samples': int(self.train_features.size(0)) if hasattr(self.train_features, 'size') else 0,
                'wall_time_s': float(times.get('wall_s', 0.0)),
                'gpu_time_s': float(times.get('gpu_s')) if times.get('gpu_s') is not None else None,
                'peak_cuda_bytes': int(peak_mem)
            })
        
        # Process validation set
        print(f"\n[INFO] Processing validation set...")
        val_images_list = []
        val_images_list_orig = []
        val_labels_list = []
        
        val_dir = base_path / 'val'
        val_images_dir = val_dir / 'images'
        val_annotations = val_dir / 'val_annotations.txt'
        
        if not val_images_dir.exists() or not val_annotations.exists():
            raise RuntimeError(f"Validation annotations not found")
        
        # Read validation annotations
        img_to_wnid = {}
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_to_wnid[parts[0]] = parts[1]
        
        # Process validation images
        for img_file in sorted(val_images_dir.glob('*.JPEG')):
            if img_file.name in img_to_wnid:
                wnid = img_to_wnid[img_file.name]
                new_label = self.subset_handler.remap_label(wnid)
                
                # Only include if class is in our subset
                if new_label >= 0:
                    img = Image.open(img_file).convert('RGB')
                    img_tensor_feat = transform_feat(img)
                    img_tensor_orig = transform_orig(img)

                    val_images_list.append(img_tensor_feat)
                    val_images_list_orig.append(img_tensor_orig)
                    val_labels_list.append(new_label)
        
        # Stack validation images and extract features
        if val_images_list:
            self.val_images = torch.stack(val_images_list)
            self.val_images_orig = torch.stack(val_images_list_orig)
            self.val_image_labels = torch.tensor(val_labels_list, dtype=torch.long)
            
            print(f"[INFO] Validation images (for features) shape: {self.val_images.shape}")
            print(f"[INFO] Validation images (original) shape: {self.val_images_orig.shape}")
            print(f"[INFO] Validation labels shape: {self.val_image_labels.shape}")
            print(f"[INFO] Validation label range: {self.val_image_labels.min()} to {self.val_image_labels.max()}")
            
            # Extract features for validation set (track timings and peak memory)
            print(f"[INFO] Extracting features for validation set...")
            try:
                self._gpu_warmup()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            tkn = self._time_start()
            self.val_features = self._extract_features_batched(self.val_images, feature_extractor)
            times = self._time_end(tkn)
            peak_mem = 0
            if torch.cuda.is_available():
                try:
                    peak_mem = int(torch.cuda.max_memory_allocated(self.device))
                except Exception:
                    try:
                        peak_mem = int(torch.cuda.max_memory_allocated())
                    except Exception:
                        peak_mem = 0
            self.feature_extraction_time += float(times.get('wall_s', 0.0))
            print(f"[INFO] Validation features shape: {self.val_features.shape}")
            print(f"[INFO] Validation embedding wall_time_s={times.get('wall_s'):.4f} gpu_time_s={times.get('gpu_s')} peak_cuda_bytes={peak_mem}")
            self.feature_extraction_log.append({
                'split': 'val',
                'num_samples': int(self.val_features.size(0)) if hasattr(self.val_features, 'size') else 0,
                'wall_time_s': float(times.get('wall_s', 0.0)),
                'gpu_time_s': float(times.get('gpu_s')) if times.get('gpu_s') is not None else None,
                'peak_cuda_bytes': int(peak_mem)
            })
        
        # Create index mappings
        self.train_idx = list(range(len(self.train_features)))
        self.val_idx = list(range(len(self.val_features)))
        
        # Verify class distribution
        train_classes = set(self.train_image_labels.numpy())
        val_classes = set(self.val_image_labels.numpy())
        
        print(f"\n{'='*60}")
        print(f"Dataset Summary:")
        print(f"  Training samples: {len(self.train_features)}")
        print(f"  Validation samples: {len(self.val_features)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes in train: {len(train_classes)}/{self.num_classes}")
        print(f"  Classes in val: {len(val_classes)}/{self.num_classes}")
        print(f"  Missing from train: {set(range(self.num_classes)) - train_classes}")
        print(f"  Missing from val: {set(range(self.num_classes)) - val_classes}")
        #print(f"{'='=60}\n")
        
        # Save cache for faster reloading
        self._save_cache()
        # Write feature extraction timing/memory CSV
        try:
            if self.feature_extraction_log:
                feat_csv = self.output_root / f"feature_extraction_times_{self.num_classes}_seed{self.cfg.subset_seed}.csv"
                with open(feat_csv, 'w') as f:
                    f.write('split,num_samples,wall_time_s,gpu_time_s,peak_cuda_bytes\n')
                    for row in self.feature_extraction_log:
                        f.write(','.join([
                            str(row.get('split')),
                            str(row.get('num_samples')),
                            str(row.get('wall_time_s')),
                            str(row.get('gpu_time_s')) if row.get('gpu_time_s') is not None else 'nan',
                            str(row.get('peak_cuda_bytes'))
                        ]) + '\n')
                print(f"[INFO] Wrote feature extraction details to {feat_csv}")
        except Exception as e:
            print(f"[WARN] Failed to write feature extraction CSV: {e}")

    def _save_cache(self):
        """Save processed data to cache"""
        cache_dir = Path('checkpoints')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = cache_dir / f"tiny_imagenet_subset_{self.num_classes}_seed{self.cfg.subset_seed}.pt"
        
        try:
            cache_data = {
                'train_features': self.train_features.cpu(),
                'train_labels': self.train_image_labels.cpu(),
                'train_images': self.train_images.cpu(),
                'train_images_orig': self.train_images_orig.cpu() if self.train_images_orig is not None else None,
                'val_features': self.val_features.cpu(),
                'val_labels': self.val_image_labels.cpu(),
                'val_images': self.val_images.cpu(),
                'val_images_orig': self.val_images_orig.cpu() if self.val_images_orig is not None else None,
                'num_classes': self.num_classes,
                'subset_seed': self.cfg.subset_seed,
                'class_mapping': self.subset_handler.wnid_to_new_idx if self.subset_handler else None
            }
            torch.save(cache_data, cache_path)
            print(f"[INFO] Saved cache to {cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    def load_data(self):
        """Main data loading function"""
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            self.load_tiny_imagenet_subset()
        else:
            # Original code for CIFAR/MNIST
            ti, tl, test_i, test_l, dims, _ = load_dataset_cls(
                self.cfg.dataset,
                trim_dataset=50000 if self.cfg.dataset == 'CIFAR_10' else 60000,
                num_parties=10,
            )
            if not torch.is_tensor(ti):
                ti = torch.tensor(ti, dtype=torch.float32)
            if not torch.is_tensor(tl):
                tl = torch.tensor(tl, dtype=torch.long)
            self.train_features = ti.view(-1, *dims)
            self.train_labels = tl
            self.dims = dims
            
            # Fixed stratified validation split
            rng = np.random.RandomState(self.cfg.val_seed)
            num_classes = 10 if self.cfg.dataset == 'CIFAR_10' else 100
            samples_per_class = 1000 if self.cfg.dataset == 'CIFAR_10' else 500
            
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
    # Base dataset sampling (ensuring all classes are represented)
    # -------------------------
    def sample_base_dataset(self, seed: int):
        """
        Sample base dataset ensuring equal representation from each class
        """
        rng = np.random.RandomState(seed)
        
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            # Build class-wise indices
            class_indices = defaultdict(list)
            for idx, label in enumerate(self.train_image_labels):
                class_indices[int(label)].append(idx)
            
            # Verify all classes are present
            for c in range(self.num_classes):
                if len(class_indices[c]) == 0:
                    raise RuntimeError(f"Class {c} has no samples!")
            
            # Calculate samples per class
            samples_per_class = self.cfg.base_size // self.num_classes
            if samples_per_class < 1:
                raise ValueError(f"base_size ({self.cfg.base_size}) too small for {self.num_classes} classes")
            
            # Adjust base_size if not perfectly divisible
            total_needed = samples_per_class * self.num_classes
            if total_needed != self.cfg.base_size:
                print(f"    Adjusting base_size from {self.cfg.base_size} to {total_needed}")
                self.cfg.base_size = total_needed
            
            print(f"    Sampling {samples_per_class} images per class")
            
            # Sample from each class
            chosen = []
            for c in range(self.num_classes):
                class_indices_c = class_indices[c]
                if len(class_indices_c) < samples_per_class:
                    # Sample with replacement if needed
                    c_chosen = rng.choice(class_indices_c, size=samples_per_class, replace=True).tolist()
                else:
                    c_chosen = rng.choice(class_indices_c, size=samples_per_class, replace=False).tolist()
                chosen.extend(c_chosen)
            
            rng.shuffle(chosen)
            indices = np.array(chosen)
            
            # Get features and labels
            inputs = self.train_features[indices]
            labels = self.train_image_labels[indices]
            
            # Verify class distribution
            unique_classes = set(labels.numpy())
            print(f"    Sampled {len(indices)} images")
            print(f"    Classes present: {len(unique_classes)}/{self.num_classes}")
            
            return inputs, labels, indices
            
        else:
            # Original sampling for CIFAR/MNIST
            if len(self.train_idx) < self.cfg.base_size:
                indices = rng.choice(self.train_idx, size=self.cfg.base_size, replace=True)
            else:
                indices = rng.choice(self.train_idx, size=self.cfg.base_size, replace=False)
            inputs = self.train_features[indices]
            labels = self.train_labels[indices]
            return inputs, labels, indices

    # -------------------------
    # Corruption methods
    # -------------------------
    def corrupt_labels(self, labels: torch.Tensor, noise_frac: float, seed: int):
        """Corrupt labels by flipping to random different class"""
        if noise_frac == 0:
            return labels.clone()
        
        rng = np.random.RandomState(seed)
        labels = labels.clone()
        n = labels.size(0)
        k = int(round(noise_frac * n))
        
        if k == 0:
            return labels
        
        indices = rng.choice(n, size=k, replace=False)
        for i in indices:
            old = int(labels[i].item())
            choices = list(range(self.num_classes))
            choices.remove(old)
            labels[i] = rng.choice(choices)
        
        return labels

    def corrupt_features(self, inputs: torch.Tensor, noise_frac: float, seed: int):
        """Add Gaussian noise to features"""
        if noise_frac == 0:
            return inputs.clone()
        
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

    def corrupt_images_simple(self, images: torch.Tensor, noise_frac: float, seed: int):
        """Add simple per-pixel Gaussian noise to images"""
        if noise_frac == 0 or images is None:
            return images
        
        rng = np.random.RandomState(seed)
        imgs = images.clone()
        N = imgs.size(0)
        if N == 0:
            return imgs
        
        try:
            glob_std = float(imgs.view(N, -1).std().item())
        except Exception:
            glob_std = 1.0
        
        scale = float(noise_frac) * 0.5 * glob_std
        noise_np = rng.normal(loc=0.0, scale=scale, size=tuple(imgs.shape))
        noise = torch.from_numpy(noise_np).to(imgs.device).type_as(imgs)
        return imgs + noise

    def _images_to_embeddings(self, images_orig: torch.Tensor) -> torch.Tensor:
        """Convert original 64x64 images (0-1 tensors) to normalized 224x224 and extract embeddings."""
        if images_orig is None or len(images_orig) == 0:
            return torch.empty(0, self.cfg.feature_dim)

        # Ensure float tensor on CPU for interpolation to avoid GPU fragmentation
        imgs = images_orig.clone().float()

        # Resize to 224x224
        imgs_resized = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=imgs_resized.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=imgs_resized.dtype).view(1, 3, 1, 1)
        imgs_norm = (imgs_resized - mean) / std

        # Extract features (this helper moves batches to device internally)
        feats = self._extract_features_batched(imgs_norm, getattr(self, 'feature_extractor', None) or self._get_feature_extractor())
        return feats

    # -------------------------
    # Baseline computations
    # -------------------------
    def feature_extractor_path(self, seed: int):
        """Path to pretrained feature extractor"""
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            return Path('checkpoints') / f"imagenet_resnet50_feature_extractor.pt"
        model_name = 'resnet' if self.cfg.dataset == 'CIFAR_10' else 'cnn'
        ds = self.cfg.dataset.lower()
        return Path('checkpoints') / f"{ds}_{model_name}_feature_extractor_seed0.pt"

    def davinz_init_path(self, seed: int):
        """Path to untrained model initialization"""
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            return Path('checkpoints') / f"imagenet_resnet50_init_seed{seed}.pt"
        ds = self.cfg.dataset.lower()
        return Path('checkpoints') / f"{ds}_model_init_seed{seed}.pt"

    def ensure_davinz_init(self, seed: int):
        """Create and save untrained model for DAVINZ"""
        p = self.davinz_init_path(seed)
        # Instantiate model with the desired number of classes (respect subset)
        # Use ResNet-18 (BasicBlock) for Tiny ImageNet to reduce memory
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            from model.resnet import BasicBlock, ResNet
            num_classes = int(self.cfg.num_classes)
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes)
        elif self.cfg.dataset == 'CIFAR_10':
            from model.resnet import BasicBlock, ResNet
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10)
        else:
            from model.cnn import CNN
            model = CNN(in_channels=1, num_classes=10)

        model_sd = model.state_dict()

        if p.exists():
            data = torch.load(p, map_location='cpu')
            ckpt = data.get('state_dict', data)

            # Merge checkpoint into model state dict but only copy parameters with matching shapes
            merged = {k: v.clone() for k, v in model_sd.items()}
            for k, v in ckpt.items():
                if k in merged:
                    try:
                        if v.shape == merged[k].shape:
                            merged[k] = v.clone()
                        else:
                            print(f"[WARN] Skipping loading param '{k}' due to shape mismatch {v.shape} != {merged[k].shape}")
                    except Exception:
                        print(f"[WARN] Skipping loading param '{k}' due to inability to compare shapes")
                else:
                    print(f"[WARN] Checkpoint param '{k}' not found in model; skipping")

            # Return a full, compatible state dict (model defaults where checkpoint incompatible)
            return merged

        # If no checkpoint exists, save the fresh initialization
        sd = model.state_dict()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': sd}, p)
        return sd

    def compute_ot(self, loader: DataLoader, val_loader: DataLoader, seed: int):
        """Compute Optimal Transport distance"""
        feat_p = str(self.feature_extractor_path(seed))
        if not os.path.exists(feat_p):
            warnings.warn(f"Feature extractor {feat_p} not found; OT will use untrained extractor")
            feat_p = None
        
        try:
            # Warmup and timed run (wall-clock + CUDA events)
            self._gpu_warmup()
            tkn = self._time_start()
            res = OT.compute_ot_distance(
                loader, val_loader, 
                dataset=self.cfg.dataset, 
                device=self.device, 
                feature_extractor_path=feat_p
            )
            times = self._time_end(tkn)

            if 'timing' not in res:
                res['timing'] = {}

            # Add measured timings and feature extraction time
            res['timing']['gpu_time_s'] = times['gpu_s']
            res['timing']['wall_time_s'] = times['wall_s']
            res['timing']['feature_extraction'] = float(self.feature_extraction_time)

            return res
        except Exception as e:
            print(f"[WARN] OT computation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'distance': float('nan'), 
                'timing': {'feature_extraction': float('nan'), 'ot_computation': float('nan'), 'wall_time_s': float('nan'), 'gpu_time_s': float('nan')}, 
                'mem': {}
            }

    def compute_rv(self, loader: DataLoader, seed: int):
        """Compute Robust Volume"""
        feat_p = str(self.feature_extractor_path(seed))
        if not os.path.exists(feat_p):
            warnings.warn(f"Feature extractor {feat_p} not found; RV will use untrained extractor")
            feat_p = None
        
        try:
            # Warmup and timed run
            self._gpu_warmup()
            tkn = self._time_start()
            res = RV.compute_rv_metric(
                loader, 
                dataset=self.cfg.dataset, 
                device=self.device, 
                feature_extractor_path=feat_p, 
                max_samples=self.cfg.max_samples
            )
            times = self._time_end(tkn)

            if 'timing' not in res:
                res['timing'] = {}

            res['timing']['gpu_time_s'] = times['gpu_s']
            res['timing']['wall_time_s'] = times['wall_s']
            res['timing']['feature_extraction'] = float(self.feature_extraction_time)

            return res
        except Exception as e:
            print(f"[WARN] RV computation failed: {e}")
            return {
                'log_volume': float('nan'), 
                'log_robust_volume': float('nan'), 
                'timing': {'feature_extraction': float('nan'), 'rv_computation': float('nan'), 'wall_time_s': float('nan'), 'gpu_time_s': float('nan')}, 
                'mem': {}
            }

    def compute_davinz(self, loader: DataLoader, val_loader: DataLoader, seed: int):
        """Compute DAVINZ score using original images"""
        # Ensure we're using image loaders, not feature loaders
        if self.cfg.dataset == 'TINY_IMAGENET_200' and hasattr(self, 'train_images_orig'):
            # If the provided loader is already a TensorDataset of images and labels,
            # use it directly so any corrupted labels are preserved.
            use_provided = False
            try:
                if isinstance(loader.dataset, TensorDataset):
                    tensors = getattr(loader.dataset, 'tensors', None)
                    if tensors and len(tensors) >= 2 and tensors[0].dim() == 4:
                        image_loader = loader
                        # Ensure validation image loader uses original images if not provided
                        if isinstance(val_loader.dataset, TensorDataset) and getattr(val_loader.dataset, 'tensors', None) and val_loader.dataset.tensors[0].dim() == 4:
                            val_image_loader = val_loader
                        else:
                            val_image_dataset = TensorDataset(self.val_images_orig, self.val_image_labels)
                            val_image_loader = DataLoader(val_image_dataset, batch_size=128, shuffle=False)
                        print(f"[DEBUG-DAVINZ] Using provided image TensorDataset for DAVINZ: train={len(image_loader.dataset)} val={len(val_image_loader.dataset)}")
                        use_provided = True
            except Exception:
                use_provided = False

            if not use_provided:
                try:
                    # Extract indices from loader if it's a Subset-like object
                    if hasattr(loader.dataset, 'indices'):
                        indices = loader.dataset.indices
                    elif hasattr(self, 'base_indices'):
                        n = len(loader.dataset)
                        indices = self.base_indices[:n]
                    else:
                        print("[WARN] Cannot map to original images, using provided loader")
                        image_loader = loader
                        val_image_loader = val_loader
                        indices = None

                    if indices is not None:
                        # Map to original 64x64 images and corresponding labels (original labels)
                        images = self.train_images_orig[indices]
                        labels = self.train_image_labels[indices]
                        image_dataset = TensorDataset(images, labels)
                        image_loader = DataLoader(image_dataset, batch_size=128, shuffle=False)

                        # Validation image loader
                        val_image_dataset = TensorDataset(self.val_images_orig, self.val_image_labels)
                        val_image_loader = DataLoader(val_image_dataset, batch_size=128, shuffle=False)

                        print(f"[DEBUG-DAVINZ] Using mapped original images for DAVINZ: train={len(image_dataset)} val={len(val_image_dataset)}")
                except Exception as e:
                    print(f"[ERROR] Failed to create image loaders: {e}")
                    image_loader = loader
                    val_image_loader = val_loader
                    print(f"[DEBUG-DAVINZ] Falling back to provided loaders for DAVINZ: train={len(getattr(loader.dataset, 'tensors', [loader.dataset])) if hasattr(loader, 'dataset') else 'unknown'} val={len(getattr(val_loader.dataset, 'tensors', [val_loader.dataset])) if hasattr(val_loader, 'dataset') else 'unknown'}")
        else:
            image_loader = loader
            val_image_loader = val_loader
        
        # Ensure untrained init exists with correct num_classes
        init_sd = self.ensure_davinz_init(seed)
        
        # Create model with correct number of classes
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            from model.resnet import BasicBlock, ResNet
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=self.num_classes).to(self.device)
        elif self.cfg.dataset == 'CIFAR_10':
            from model.resnet import BasicBlock, ResNet
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=10).to(self.device)
        else:
            from model.cnn import CNN
            model = CNN(in_channels=1, num_classes=10).to(self.device)
        
        # Load initial state
        try:
            model.load_state_dict(init_sd)
        except Exception:
            try:
                model.load_state_dict(init_sd.get('state_dict', init_sd))
            except Exception as e:
                print(f"[WARN] Failed to load init state dict: {e}")
        
        # Compute DAVINZ
        try:
            print(f"[INFO] Computing DAVINZ with {len(image_loader.dataset)} train, {len(val_image_loader.dataset)} val images")
            # Warmup and timed run (wall + CUDA events)
            self._gpu_warmup()
            tkn = self._time_start()
            res = DAVINZ.compute_davinz(
                image_loader, val_image_loader,
                dataset=self.cfg.dataset,
                device=self.device,
                model=model,
                diagonal_I_mag=1e-6,
                n_batch=100
            )
            times = self._time_end(tkn)

            # Ensure timing fields exist and include measured times and feature extraction time
            if isinstance(res, dict):
                if 'timing' not in res:
                    res['timing'] = {}
                res['timing']['gpu_time_s'] = times['gpu_s']
                res['timing']['wall_time_s'] = times['wall_s']
                res['timing']['feature_extraction'] = float(self.feature_extraction_time)
            return res
        except Exception as e:
            print(f"[WARN] DAVINZ computation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mmd': float('nan'),
                'mmd_raw': float('nan'),
                'ntk': float('nan'),
                'davinz_score': float('nan'),
                'timing': {'mmd_time': float('nan'), 'ntk_time': float('nan'), 'wall_time_s': float('nan'), 'gpu_time_s': float('nan')},
                'mem': {}
            }

    def _verify_davinz_labels(self, labels_tensor: torch.Tensor, dav_r: dict, context: str = ''):
        """Verify that DAVINZ received the same labels by comparing SHA1 hash and basic stats.

        Prints a warning if there's a mismatch or if DAVINZ did not return debug label fields.
        """
        if labels_tensor is None or (hasattr(labels_tensor, 'numel') and labels_tensor.numel() == 0):
            return

        try:
            labels_cpu = labels_tensor.detach().cpu()
            local_hash = hashlib.sha1(labels_cpu.numpy().tobytes()).hexdigest()
            local_sample = labels_cpu[:10].numpy().tolist()
            local_unique = int(len(torch.unique(labels_cpu)))

            dav_hash = dav_r.get('bootstrap_labels_hash') if isinstance(dav_r, dict) else None
            dav_sample = dav_r.get('bootstrap_labels_sample') if isinstance(dav_r, dict) else None
            dav_unique = int(dav_r.get('bootstrap_labels_unique')) if isinstance(dav_r, dict) and dav_r.get('bootstrap_labels_unique') is not None else None

            if dav_hash is None:
                print(f"[WARN] DAVINZ did not return label debug fields in context='{context}'")
                print(f"  local sample={local_sample} unique={local_unique} hash={local_hash}")
            else:
                if dav_hash != local_hash:
                    print(f"[ERROR] DAVINZ label hash mismatch in context='{context}'")
                    print(f"  local sample={local_sample} unique={local_unique} hash={local_hash}")
                    print(f"  davinz sample={dav_sample} unique={dav_unique} hash={dav_hash}")
                else:
                    print(f"[DEBUG] DAVINZ label hash OK in context='{context}' sample={local_sample} unique={local_unique} hash={local_hash}")
        except Exception as e:
            print(f"[WARN] _verify_davinz_labels failed: {e} (context='{context}')")

    # -------------------------
    # Results saving helpers
    # -------------------------
    def _ensure_headers(self, path: Path, header: str):
        if not path.exists():
            with open(path, 'w') as f:
                f.write(header + '\n')

    def _write_jsonl(self, path: Path, obj: dict):
        with open(path, 'a') as f:
            f.write(json.dumps(obj) + '\n')

    def _write_party_results(self, out_dir: Path, nl: float, ot_r: dict, rv_r: dict, dav_r: dict, 
                            scenario: str, party_id: Optional[int]=None, size: Optional[int]=None):
        """Write results for a single party/noise level"""
        ts = time.time()
        
        # OT results
        ot_csv = out_dir / 'ot_results.csv'
        ot_row = [nl,
                  (int(size) if size is not None else 'nan'),
                  ot_r.get('distance', 'nan'),
                  ot_r.get('timing', {}).get('feature_extraction', 'nan'),
                  ot_r.get('timing', {}).get('ot_computation', 'nan'),
                  ot_r.get('timing', {}).get('wall_time_s', 'nan'),
                  ot_r.get('timing', {}).get('gpu_time_s', 'nan'),
                  ot_r.get('timing', {}).get('total', 'nan'),
                  ot_r.get('mem', {}).get('feature_extraction', 'nan'),
                  ot_r.get('mem', {}).get('ot', 'nan'),
                  ot_r.get('mem', {}).get('total', 'nan'),
                  ts]
        with open(ot_csv, 'a') as f:
            f.write(','.join(map(str, ot_row)) + '\n')
        
        ot_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 
                   'scenario': scenario, 'result': ot_r, 'timestamp': ts}
        if party_id is not None:
            ot_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'ot_results.jsonl', ot_json)

        # RV results
        rv_csv = out_dir / 'rv_results.csv'
        rv_row = [nl,
                  (int(size) if size is not None else 'nan'),
                  rv_r.get('log_volume', 'nan'),
                  rv_r.get('log_robust_volume', 'nan'),
                  rv_r.get('timing', {}).get('feature_extraction', 'nan'),
                  rv_r.get('timing', {}).get('rv_computation', 'nan'),
                  rv_r.get('timing', {}).get('wall_time_s', 'nan'),
                  rv_r.get('timing', {}).get('gpu_time_s', 'nan'),
                  rv_r.get('timing', {}).get('total', 'nan'),
                  rv_r.get('mem', {}).get('feature_extraction', 'nan'),
                  rv_r.get('mem', {}).get('rv', 'nan'),
                  rv_r.get('mem', {}).get('total', 'nan'),
                  ts]
        with open(rv_csv, 'a') as f:
            f.write(','.join(map(str, rv_row)) + '\n')
        
        rv_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 
                   'scenario': scenario, 'result': rv_r, 'timestamp': ts}
        if party_id is not None:
            rv_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'rv_results.jsonl', rv_json)

        # DAVINZ results
        dav_csv = out_dir / 'davinz_results.csv'
        dav_row = [nl,
                   (int(size) if size is not None else 'nan'),
                   dav_r.get('mmd', dav_r.get('mmd_raw', 'nan')),
                   dav_r.get('mmd_raw', 'nan'),
                   dav_r.get('ntk', 'nan'),
                   dav_r.get('davinz_score', 'nan'),
                   dav_r.get('timing', {}).get('mmd_time', 'nan'),
                   dav_r.get('timing', {}).get('ntk_time', 'nan'),
                   dav_r.get('timing', {}).get('wall_time_s', 'nan'),
                   dav_r.get('timing', {}).get('gpu_time_s', 'nan'),
                   dav_r.get('timing', {}).get('total', 'nan'),
                   dav_r.get('mem', {}).get('mmd', 'nan'),
                   dav_r.get('mem', {}).get('ntk', 'nan'),
                   dav_r.get('mem', {}).get('total', 'nan'),
                   ts]
        with open(dav_csv, 'a') as f:
            f.write(','.join(map(str, dav_row)) + '\n')
        
        dav_json = {'noise_level': nl, 'size': (int(size) if size is not None else None), 
                    'scenario': scenario, 'result': dav_r, 'timestamp': ts}
        if party_id is not None:
            dav_json['party_id'] = int(party_id)
        self._write_jsonl(out_dir / 'davinz_results.jsonl', dav_json)

    # -------------------------
    # Main run function for a single seed
    # -------------------------
    def run_seed(self, seed: int):
        print(f"\n{'='*60}")
        print(f"[INFO] Running seed {seed}")
        print(f"{'='*60}")
        
        seed_shared = self.output_root / f'seed_{seed}'
        seed_shared.mkdir(parents=True, exist_ok=True)

        # Sample base dataset
        base_inputs, base_labels, base_indices = self.sample_base_dataset(seed)
        self.base_indices = base_indices
        # Save a shared copy of the base dataset for this seed
        torch.save({
            'inputs': base_inputs, 
            'labels': base_labels,
            'indices': base_indices
        }, seed_shared / 'base_dataset.pt')
        print(f"[INFO] Base dataset size: {len(base_inputs)} samples")
        
        # Create validation loader
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            val_dataset = TensorDataset(self.val_features, self.val_image_labels)
        else:
            val_dataset = TensorDataset(self.train_features[self.val_idx], self.train_labels[self.val_idx])
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        print(f"[INFO] Validation set size: {len(val_dataset)} samples")
        
        # Create per-scenario/seed directories and headers for requested scenarios
        for scenario in self.cfg.scenarios:
            d = self.output_root / scenario / f'seed_{seed}'
            d.mkdir(parents=True, exist_ok=True)

            self._ensure_headers(d / 'ot_results.csv', 
                               'noise_level,size,ot_distance,feature_extraction_time_s,ot_computation_time_s,wall_time_s,gpu_time_s,total_time_s,feature_extraction_mem_bytes,ot_mem_bytes,total_mem_bytes,timestamp')
            self._ensure_headers(d / 'rv_results.csv', 
                               'noise_level,size,log_volume,log_robust_volume,feature_extraction_time_s,rv_computation_time_s,wall_time_s,gpu_time_s,total_time_s,feature_extraction_mem_bytes,rv_mem_bytes,total_mem_bytes,timestamp')
            self._ensure_headers(d / 'davinz_results.csv', 
                               'noise_level,size,mmd,mmd_raw,ntk,davinz_score,mmd_time_s,ntk_time_s,wall_time_s,gpu_time_s,total_time_s,mmd_mem_bytes,ntk_mem_bytes,total_mem_bytes,timestamp')

            (d / 'ot_results.jsonl').touch(exist_ok=True)
            (d / 'rv_results.jsonl').touch(exist_ok=True)
            (d / 'davinz_results.jsonl').touch(exist_ok=True)

        # Run label noise sweep
        if 'label' in self.cfg.scenarios:
            print(f"\n[INFO] Running label noise sweep...")
            d = self.output_root / 'label' / f'seed_{seed}'
            for nl in self.cfg.noise_levels:
                print(f"  Noise level: {nl}")

                # Corrupt labels
                corrupted_labels = self.corrupt_labels(base_labels, nl, seed)
                # DEBUG: verify corrupted labels differ from original (for label-noise scenario)
                try:
                    n_changed = int((corrupted_labels != base_labels).sum().item())
                    frac = n_changed / float(len(base_labels)) if len(base_labels) > 0 else 0.0
                    print(f"[DEBUG-LABEL] corrupted labels: changed={n_changed}/{len(base_labels)} ({frac:.4f}), unique_classes={len(torch.unique(corrupted_labels))}")
                    print(f"[DEBUG-LABEL] sample corrupted labels (first 10): {corrupted_labels[:10].tolist()}")
                except Exception:
                    print("[DEBUG-LABEL] unable to inspect corrupted_labels")

                # OT/RV loader (uses features)
                dataset = TensorDataset(base_inputs, corrupted_labels)
                loader = DataLoader(dataset, batch_size=128, shuffle=False)

                # DAVINZ loader (uses images)
                if self.cfg.dataset == 'TINY_IMAGENET_200' and hasattr(self, 'train_images_orig'):
                    images = self.train_images_orig[base_indices]
                    img_dataset = TensorDataset(images, corrupted_labels)
                    davinz_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

                    val_img_dataset = TensorDataset(self.val_images_orig, self.val_image_labels)
                    val_img_loader = DataLoader(val_img_dataset, batch_size=128, shuffle=False)
                else:
                    davinz_loader = loader
                    val_img_loader = val_loader

                # Compute metrics
                ot_r = self.compute_ot(loader, val_loader, seed)
                self._clear_cuda_and_check('after_ot_label')

                rv_r = self.compute_rv(loader, seed)
                self._clear_cuda_and_check('after_rv_label')

                dav_r = self.compute_davinz(davinz_loader, val_img_loader, seed)
                self._clear_cuda_and_check('after_davinz_label')

                # Verify DAVINZ received the corrupted labels correctly
                try:
                    self._verify_davinz_labels(corrupted_labels, dav_r, context=f'label_noise_nl={nl}')
                except Exception:
                    pass

                # Save results
                self._write_party_results(d, nl, ot_r, rv_r, dav_r, 'label', size=base_inputs.size(0))
        else:
            print(f"[INFO] Skipping label scenario for seed {seed}")

        # Run feature noise sweep
        if 'feature' in self.cfg.scenarios:
            print(f"\n[INFO] Running feature noise sweep...")
            d = self.output_root / 'feature' / f'seed_{seed}'
            for nl in self.cfg.noise_levels:
                print(f"  Noise level: {nl}")

                # Corrupt features
                # If Tiny ImageNet: inject noise into original 64x64 images, then resize+embed for OT/RV
                if self.cfg.dataset == 'TINY_IMAGENET_200' and hasattr(self, 'train_images_orig'):
                    images_orig = self.train_images_orig[base_indices]
                    corrupted_images = self.corrupt_images_simple(images_orig, nl, seed)

                    # DAVINZ uses corrupted original images (64x64)
                    img_dataset = TensorDataset(corrupted_images, base_labels)
                    davinz_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

                    # For OT/RV: resize+normalize corrupted images and extract embeddings
                    print(f"[DEBUG] Converting {len(corrupted_images)} corrupted images to embeddings for OT/RV (noise={nl})")
                    corrupted_embeddings = self._images_to_embeddings(corrupted_images.cpu())
                    dataset = TensorDataset(corrupted_embeddings, base_labels)
                    loader = DataLoader(dataset, batch_size=128, shuffle=False)

                    # Validation: keep original validation images/features
                    val_img_dataset = TensorDataset(self.val_images_orig, self.val_image_labels)
                    val_img_loader = DataLoader(val_img_dataset, batch_size=128, shuffle=False)
                else:
                    corrupted_features = self.corrupt_features(base_inputs, nl, seed)
                    dataset = TensorDataset(corrupted_features, base_labels)
                    loader = DataLoader(dataset, batch_size=128, shuffle=False)
                    davinz_loader = loader
                    val_img_loader = val_loader

                # Compute metrics
                ot_r = self.compute_ot(loader, val_loader, seed)
                self._clear_cuda_and_check('after_ot_feature')

                rv_r = self.compute_rv(loader, seed)
                self._clear_cuda_and_check('after_rv_feature')

                dav_r = self.compute_davinz(davinz_loader, val_img_loader, seed)
                self._clear_cuda_and_check('after_davinz_feature')

                # Verify DAVINZ received the labels for feature-noise scenario (should be base_labels)
                try:
                    self._verify_davinz_labels(base_labels, dav_r, context=f'feature_noise_nl={nl}')
                except Exception:
                    pass

                # Save results for this noise level (feature scenario)
                self._write_party_results(d, nl, ot_r, rv_r, dav_r, 'feature', size=base_inputs.size(0))
        else:
            print(f"[INFO] Skipping feature scenario for seed {seed}")

        # Size scenario (nested parties)
        if 'size' in self.cfg.scenarios:
            print(f"\n[INFO] Running size scenario...")
            size_dir = self.output_root / 'size' / f'seed_{seed}'
            self._run_size_scenario(size_dir, seed, val_loader)
        else:
            print(f"[INFO] Skipping size scenario for seed {seed}")

        # Replication scenario
        if 'replication' in self.cfg.scenarios:
            print(f"\n[INFO] Running replication scenario...")
            repl_dir = self.output_root / 'replication' / f'seed_{seed}'
            self._run_replication_scenario(repl_dir, seed, val_loader)
        else:
            print(f"[INFO] Skipping replication scenario for seed {seed}")

    def _run_size_scenario(self, out_dir: Path, seed: int, val_loader: DataLoader):
        """Run size scenario with nested parties"""
        party_k = 10
        # per-party increment: use cfg.base_size//party_k if available (e.g., 50000 -> 5000), otherwise default to 5000
        per_party_size = self.cfg.base_size // party_k if self.cfg.base_size >= party_k else 5000
        if per_party_size == 0:
            per_party_size = 5000
        
        rng = np.random.RandomState(seed)
        pool = list(self.train_idx)
        total_needed = party_k * per_party_size
        replace = len(pool) < total_needed
        chosen = rng.choice(pool, size=total_needed, replace=replace)
        
        for p_idx in range(1, party_k + 1):
            upto = p_idx * per_party_size
            indices = chosen[:upto]
            
            if self.cfg.dataset == 'TINY_IMAGENET_200':
                inputs = self.train_features[indices]
                labels = self.train_image_labels[indices]
                images = self.train_images_orig[indices]
            else:
                inputs = self.train_features[indices]
                labels = self.train_labels[indices]
                images = None
            
            print(f"  Party {p_idx}: {inputs.size(0)} samples")
            
            # OT/RV loader
            loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=False)
            
            # DAVINZ loader
            if images is not None:
                img_loader = DataLoader(TensorDataset(images, labels), batch_size=128, shuffle=False)
            else:
                img_loader = loader
            
            ot_r = self.compute_ot(loader, val_loader, seed)
            self._clear_cuda_and_check('after_ot_size')
            rv_r = self.compute_rv(loader, seed)
            self._clear_cuda_and_check('after_rv_size')
            dav_r = self.compute_davinz(img_loader, val_loader, seed)
            self._clear_cuda_and_check('after_davinz_size')

            # Verify DAVINZ received the labels for this party (size scenario)
            try:
                self._verify_davinz_labels(labels, dav_r, context=f'size_party_{p_idx}')
            except Exception:
                pass
            
            self._write_party_results(out_dir, 0.0, ot_r, rv_r, dav_r, 'size', 
                                     party_id=p_idx, size=inputs.size(0))

    def _run_replication_scenario(self, out_dir: Path, seed: int, val_loader: DataLoader):
        """Run replication scenario"""
        party_k = 10
        # Base unique block size: default to cfg.base_size//party_k (e.g., 50000->5000)
        base_size = self.cfg.base_size // party_k if self.cfg.base_size >= party_k else 5000
        if base_size == 0:
            base_size = 5000
        
        rng = np.random.RandomState(seed)
        pool = list(self.train_idx)
        replace = len(pool) < base_size
        base_idxs = rng.choice(pool, size=base_size, replace=replace)
        
        if self.cfg.dataset == 'TINY_IMAGENET_200':
            base_inputs = self.train_features[base_idxs]
            base_labels = self.train_image_labels[base_idxs]
            base_images = self.train_images_orig[base_idxs]
        else:
            base_inputs = self.train_features[base_idxs]
            base_labels = self.train_labels[base_idxs]
            base_images = None
        
        for p_idx in range(1, party_k + 1):
            if p_idx == 1:
                inputs = base_inputs
                labels = base_labels
                images = base_images
            else:
                inputs = base_inputs.repeat(p_idx, *([1] * (base_inputs.dim() - 1)))
                labels = base_labels.repeat(p_idx)
                if images is not None:
                    images = base_images.repeat(p_idx, 1, 1, 1)
            
            print(f"  Party {p_idx}: replication factor {p_idx}, {inputs.size(0)} samples")
            
            loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=False)
            
            if images is not None:
                img_loader = DataLoader(TensorDataset(images, labels), batch_size=128, shuffle=False)
            else:
                img_loader = loader
            
            ot_r = self.compute_ot(loader, val_loader, seed)
            self._clear_cuda_and_check('after_ot_repl')
            rv_r = self.compute_rv(loader, seed)
            self._clear_cuda_and_check('after_rv_repl')
            dav_r = self.compute_davinz(img_loader, val_loader, seed)
            self._clear_cuda_and_check('after_davinz_repl')

            # Verify DAVINZ received the labels for replication scenario
            try:
                self._verify_davinz_labels(labels, dav_r, context=f'repl_party_{p_idx}')
            except Exception:
                pass
            
            self._write_party_results(out_dir, float(p_idx), ot_r, rv_r, dav_r, 'replication',
                                     party_id=p_idx, size=inputs.size(0))

    # -------------------------
    # Aggregate results across seeds
    # -------------------------
    def aggregate(self, seeds: List[int]):
        print(f"\n[INFO] Aggregating results for seeds {seeds}")
        
        agg_dir = self.output_root / 'aggregated'
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        # Label noise summary
        with open(agg_dir / 'label_noise_summary.csv', 'w') as f:
            f.write('seed,noise_level,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_dir = self.output_root / 'label' / f'seed_{s}'
                if not seed_dir.exists():
                    continue
                
                dav_file = seed_dir / 'davinz_results.csv'
                rv_file = seed_dir / 'rv_results.csv'
                ot_file = seed_dir / 'ot_results.csv'
                
                if dav_file.exists() and rv_file.exists() and ot_file.exists():
                    dav_lines = dav_file.read_text().strip().split('\n')[1:]
                    rv_lines = rv_file.read_text().strip().split('\n')[1:]
                    ot_lines = ot_file.read_text().strip().split('\n')[1:]
                    
                    for i, nl_line in enumerate(dav_lines):
                        if i >= len(rv_lines) or i >= len(ot_lines):
                            continue
                        
                        parts = nl_line.split(',')
                        if len(parts) < 5:
                            continue
                        
                        noise_level = parts[0]
                        davinz_score = parts[4]
                        rv_log = rv_lines[i].split(',')[2] if len(rv_lines[i].split(',')) > 2 else ''
                        ot_dist = ot_lines[i].split(',')[1] if len(ot_lines[i].split(',')) > 1 else ''
                        
                        f.write(f"{s},{noise_level},{ot_dist},{rv_log},{davinz_score}\n")
        
        # Feature noise summary
        with open(agg_dir / 'feature_noise_summary.csv', 'w') as f:
            f.write('seed,noise_level,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_dir = self.output_root / 'feature' / f'seed_{s}'
                if not seed_dir.exists():
                    continue
                
                dav_file = seed_dir / 'davinz_results.csv'
                rv_file = seed_dir / 'rv_results.csv'
                ot_file = seed_dir / 'ot_results.csv'
                
                if dav_file.exists() and rv_file.exists() and ot_file.exists():
                    dav_lines = dav_file.read_text().strip().split('\n')[1:]
                    rv_lines = rv_file.read_text().strip().split('\n')[1:]
                    ot_lines = ot_file.read_text().strip().split('\n')[1:]
                    
                    for i, nl_line in enumerate(dav_lines):
                        if i >= len(rv_lines) or i >= len(ot_lines):
                            continue
                        
                        parts = nl_line.split(',')
                        if len(parts) < 5:
                            continue
                        
                        noise_level = parts[0]
                        davinz_score = parts[4]
                        rv_log = rv_lines[i].split(',')[2] if len(rv_lines[i].split(',')) > 2 else ''
                        ot_dist = ot_lines[i].split(',')[1] if len(ot_lines[i].split(',')) > 1 else ''
                        
                        f.write(f"{s},{noise_level},{ot_dist},{rv_log},{davinz_score}\n")
        
        # Size scenario summary
        size_summary = agg_dir / 'size_summary.csv'
        with open(size_summary, 'w') as f:
            f.write('seed,party_id,size,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_file = self.output_root / 'size' / f'seed_{s}' / 'size_summary.csv'
                if seed_file.exists():
                    lines = seed_file.read_text().strip().split('\n')[1:]
                    for line in lines:
                        f.write(f"{s},{line}\n")
        
        # Replication scenario summary
        repl_summary = agg_dir / 'replication_summary.csv'
        with open(repl_summary, 'w') as f:
            f.write('seed,party_id,replication_factor,num_samples,ot_distance,rv_log_robust_volume,davinz_score\n')
            for s in seeds:
                seed_file = self.output_root / 'replication' / f'seed_{s}' / 'replication_summary.csv'
                if seed_file.exists():
                    lines = seed_file.read_text().strip().split('\n')[1:]
                    for line in lines:
                        f.write(f"{s},{line}\n")
        
        print(f"[INFO] Aggregated results saved to {agg_dir}")


def parse_seed_range(seed_range_str: str) -> List[int]:
    """Parse seed string like '0', '0-4', or '0,1,2'"""
    if '-' in seed_range_str:
        start, end = seed_range_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in seed_range_str:
        return [int(s.strip()) for s in seed_range_str.split(',')]
    else:
        return [int(seed_range_str)]


def main():
    parser = argparse.ArgumentParser(description='Noise Robustness Testing')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['CIFAR_10', 'MNIST', 'TINY_IMAGENET_200'],
                        help='Dataset to use')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID(s) to use (e.g., "0" or "0,1,2,3")')
    parser.add_argument('--seeds', type=str, default='0',
                        help='Seed or range (e.g., "0", "0-4", "0,1,2")')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--base_size', type=int, default=50000,
                        help='Base dataset size (will be adjusted to multiple of num_classes)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum samples for RV computation')
    parser.add_argument('--tiny_imagenet_path', type=str, 
                        default='/home/mehdi.touil/ondemand/data/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset')
    parser.add_argument('--noise_levels', type=str, default=None,
                        help='Comma-separated noise levels (overrides default)')
    parser.add_argument('--subset_classes', type=int, default=100,
                        help='If set, select this many classes from Tiny ImageNet')
    parser.add_argument('--subset_seed', type=int, default=0,
                        help='Random seed used to select subset classes')
    parser.add_argument('--embed_batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--scenario', type=str, default='all',
                        help='Scenario to run: label,feature,size,replication or all (default all)')
    
    args = parser.parse_args()

    # Parse seeds
    seeds = parse_seed_range(args.seeds)
    
    # Parse noise levels if provided
    noise_levels = NOISE_LEVELS_DEFAULT
    if args.noise_levels:
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
    
    # For Tiny ImageNet with subset, num_classes will be updated after loading
    num_classes = 200 if args.dataset == 'TINY_IMAGENET_200' else (10 if args.dataset == 'CIFAR_10' else 100)
    
    # Adjust base_size to be multiple of num_classes for Tiny ImageNet
    if args.dataset == 'TINY_IMAGENET_200' and args.subset_classes:
        # We'll adjust after loading when we know actual num_classes
        pass
    elif args.dataset == 'TINY_IMAGENET_200':
        if args.base_size % 200 != 0:
            adjusted = (args.base_size // 200) * 200
            print(f"[WARNING] base_size={args.base_size} not multiple of 200. Adjusting to {adjusted}")
            args.base_size = adjusted
    
    # Create config
    cfg = Config(
        dataset=args.dataset,
        gpu=args.gpu,
        seeds=seeds,
        output_dir=args.output_dir,
        base_size=args.base_size,
        max_samples=args.max_samples,
        subset_classes=args.subset_classes,
        subset_seed=args.subset_seed,
        noise_levels=noise_levels,
        tiny_imagenet_path=args.tiny_imagenet_path,
        num_classes=num_classes,
        embed_batch_size=args.embed_batch_size
    )

    # Parse scenario argument into list
    allowed = ['label', 'feature', 'size', 'replication']
    if args.scenario is None or args.scenario.strip() == '' or args.scenario.strip().lower() == 'all':
        cfg.scenarios = allowed
    else:
        parts = [p.strip().lower() for p in args.scenario.split(',') if p.strip()]
        # validate
        bad = [p for p in parts if p not in allowed]
        if bad:
            raise ValueError(f"Invalid scenario(s): {bad}. Allowed: {allowed} or 'all'.")
        cfg.scenarios = parts

    print(f"\n{'='*60}")
    print(f"Robustness Testing Configuration")
    print(f"{'='*60}")
    print(f"Dataset: {cfg.dataset}")
    print(f"GPUs: {cfg.gpu}")
    print(f"Seeds: {seeds}")
    print(f"Base size: {cfg.base_size}")
    print(f"Subset classes: {cfg.subset_classes}")
    print(f"Subset seed: {cfg.subset_seed}")
    print(f"Noise levels: {cfg.noise_levels}")
    print(f"Max samples (RV): {cfg.max_samples}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Scenarios: {cfg.scenarios}")
    #print(f"{'='=60}\n")

    # Create tester and run
    tester = RobustnessTester(cfg)
    
    print("[INFO] Loading data...")
    tester.load_data()
    
    # Adjust base_size if needed after loading
    if cfg.dataset == 'TINY_IMAGENET_200' and cfg.subset_classes:
        # Ensure base_size is multiple of actual num_classes
        if cfg.base_size % tester.num_classes != 0:
            adjusted = (cfg.base_size // tester.num_classes) * tester.num_classes
            print(f"[INFO] Adjusting base_size from {cfg.base_size} to {adjusted} (multiple of {tester.num_classes})")
            tester.cfg.base_size = adjusted
    
    print(f"[INFO] Data loaded: {len(tester.train_features)} training samples")
    print(f"[INFO] Number of classes: {tester.num_classes}")
    print(f"[INFO] Training indices: {len(tester.train_idx)}")
    print(f"[INFO] Validation indices: {len(tester.val_idx)}")
    
    for s in seeds:
        print(f"\n{'='*60}")
        print(f"[INFO] Running seed={s} on device={tester.device}")
        print(f"{'='*60}")
        tester.run_seed(s)
        tester._clear_cuda_and_check('after_seed')
    
    print("\n[INFO] Aggregating results...")
    tester.aggregate(seeds)
    
    print("\n[INFO] Done!")


if __name__ == '__main__':
    main()