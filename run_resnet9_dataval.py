#!/usr/bin/env python3
"""
Fine-Grained Data Valuation for ResNet9 on CIFAR-10
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "GhostSuite"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fine_grained.opendataval.dataval.random import RandomEvaluator

print("=" * 80)
print("Fine-Grained Data Valuation: ResNet9 on CIFAR-10")
print("=" * 80)

# Check environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ Device: {device}")

# Load CIFAR-10
print("\nLoading CIFAR-10...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

print(f"✓ Loaded {len(trainset)} training samples")

# Test data valuation
print("\nTesting data valuation (Random baseline)...")
images, labels = next(iter(train_loader))
images = images[:32]  # Sample 32 images
labels = labels[:32]

random_eval = RandomEvaluator(random_state=42)
values = random_eval.evaluate(images, labels)

print(f"✓ Computed {len(values)} data values")
print(f"  Top 5 values: {sorted(values, reverse=True)[:5]}")

print("\n" + "=" * 80)
print("✓ Success! Data valuation working.")
print("=" * 80)
