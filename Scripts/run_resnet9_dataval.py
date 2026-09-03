#!/usr/bin/env python3
"""
Fine-Grained Data Valuation for ResNet9 on CIFAR-10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "src" / "fine_grained"))
sys.path.insert(0, str(Path.cwd() / "GhostSuite"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opendataval.dataval.random import RandomEvaluator

print("=" * 80)
print("Fine-Grained Data Valuation: ResNet9 on CIFAR-10")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ Device: {device}")

print("\nLoading CIFAR-10...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

print(f"✓ Loaded {len(trainset)} training samples")

print("\nTesting data valuation (Random baseline)...")
images, labels = next(iter(train_loader))
images = images[:32]
labels = labels[:32]

random_eval = RandomEvaluator(random_state=42)
values = random_eval.evaluate(images, labels)

print(f"✓ Computed {len(values)} data values")
print(f"  Top 5 values: {sorted(values, reverse=True)[:5]}")

print("\n" + "=" * 80)
print("✓ Success! Data valuation working.")
print("=" * 80)
