"""Unified H100-Optimized ResNet Models (9, 18, 34, 50, 110)

All models use identical training hyperparameters for fair comparison.
Integrated optimizations:
- torch.compile (reduce-overhead mode)
- Automatic Mixed Precision (bfloat16)
- Channels-last memory format
- CosineAnnealingLR scheduler
- Adam optimizer with LR=0.001
- Gradient clipping for stability
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader, Subset as TorchSubset
from torchvision import transforms, datasets
import numpy as np
from opendataval.model.grad import TorchGradMixin


# ============================================================================
# BASIC BLOCKS & UTILITIES
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block for deeper ResNets (50+)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        return self.relu(out)


class Mul(nn.Module):
    """Multiply layer for ResNet9"""
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    """Flatten layer for ResNet9"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    """Residual wrapper for ResNet9"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


# ============================================================================
# UNIFIED RESNET BASE CLASS
# ============================================================================

class ResNetBase(TorchGradMixin):
    """Base class for all ResNet variants with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def predict(self, x):
        """Make predictions with proper device handling"""
        device = next(self.parameters()).device
        self.eval()
        if not isinstance(x, torch.Tensor):
            if hasattr(x, '__getitem__') and hasattr(x, '__len__'):
                x_list = [x[i] for i in range(len(x))]
                x = torch.stack(x_list)
            else:
                x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            x = x.to(device)
            return torch.softmax(self.forward(x), dim=1)

    def clone(self):
        """Clone the model for a fresh copy"""
        return copy.deepcopy(self)

    @torch.no_grad()
    def compute_accuracy(self, loader, device):
        """Compute accuracy on a given DataLoader"""
        self.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1 and y.shape[1] > 1:
                y = y.argmax(dim=1)
            else:
                y = y.view(-1).long()
            preds = self(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / total

    def compute_accuracy_from_tensors(self, x, y, device, batch_size):
        """Compute accuracy from tensors in batches"""
        self.eval()
        correct, total = 0, 0
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = self(x_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        return correct / total

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, x_test=None,
            y_test=None, batch_size=128, epochs=50, lr=0.001,
            weight_decay=0.0, label_smoothing=0.0, verbose=True, **kwargs):
        """H100-Optimized training with integrated optimizations

        Accepts additional kwargs for compatibility with ExperimentMediator,
        but only uses the specified parameters.
        """
        device = next(self.parameters()).device

        # H100 Optimization 1: torch.compile
        if device.type == 'cuda':
            try:
                self = torch.compile(self, mode='reduce-overhead', fullgraph=False)
                if verbose:
                    print("[H100] ✓ torch.compile enabled")
            except Exception:
                pass

        # H100 Optimization 2: Channels-last format
        if device.type == 'cuda':
            self = self.to(memory_format=torch.channels_last)

        self.train()

        # Convert Subsets to Tensors
        if isinstance(x_train, TorchSubset):
            x_train = torch.stack([x_train[i] for i in range(len(x_train))])
        if isinstance(x_valid, TorchSubset):
            x_valid = torch.stack([x_valid[i] for i in range(len(x_valid))])
        if isinstance(x_test, TorchSubset):
            x_test = torch.stack([x_test[i] for i in range(len(x_test))])

        # Convert y_train to class indices
        if isinstance(y_train, torch.Tensor):
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train_indices = y_train.argmax(dim=1).long()
            else:
                y_train_indices = y_train.reshape(-1).long()
        else:
            if isinstance(y_train, np.ndarray):
                if y_train.ndim > 1 and y_train.shape[1] > 1:
                    y_train_indices = np.argmax(y_train, axis=1)
                else:
                    y_train_indices = y_train.reshape(-1)
            else:
                y_train_indices = y_train
            y_train_indices = torch.from_numpy(np.asarray(y_train_indices)).long()

        # Convert validation labels
        if y_valid is not None:
            if isinstance(y_valid, torch.Tensor):
                if y_valid.ndim > 1 and y_valid.shape[1] > 1:
                    y_valid_indices = y_valid.argmax(dim=1).long()
                else:
                    y_valid_indices = y_valid.reshape(-1).long()
            else:
                if isinstance(y_valid, np.ndarray):
                    if y_valid.ndim > 1 and y_valid.shape[1] > 1:
                        y_valid_indices = np.argmax(y_valid, axis=1)
                    else:
                        y_valid_indices = y_valid.reshape(-1)
                else:
                    y_valid_indices = y_valid
                y_valid_indices = torch.from_numpy(np.asarray(y_valid_indices)).long()
            y_valid_indices = y_valid_indices.to(device)
        else:
            y_valid_indices = None

        # Convert test labels
        if y_test is not None:
            if isinstance(y_test, torch.Tensor):
                if y_test.ndim > 1 and y_test.shape[1] > 1:
                    y_test_indices = y_test.argmax(dim=1).long()
                else:
                    y_test_indices = y_test.reshape(-1).long()
            else:
                if isinstance(y_test, np.ndarray):
                    if y_test.ndim > 1 and y_test.shape[1] > 1:
                        y_test_indices = np.argmax(y_test, axis=1)
                    else:
                        y_test_indices = y_test.reshape(-1)
                else:
                    y_test_indices = y_test
                y_test_indices = torch.from_numpy(np.asarray(y_test_indices)).long()
            y_test_indices = y_test_indices.to(device)
        else:
            y_test_indices = None

        if verbose:
            print(f"[H100-Training] H100-optimized ResNet training")
            print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
            print(f"  Optimizer: Adam")
            print(f"  Scheduler: CosineAnnealingLR (smooth decay over {epochs} epochs)")
            print(f"  Weight Decay: {weight_decay}, Label Smoothing: {label_smoothing}\n")

        train_dataset = TensorDataset(x_train, y_train_indices)
        can_pin_memory = isinstance(x_train, torch.Tensor) and not x_train.is_cuda
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=can_pin_memory, drop_last=True
        )

        # Balanced loss weights
        class_counts = torch.bincount(y_train_indices, minlength=self.num_classes).float()
        class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        class_weights = class_weights.to(device)

        # Adam optimizer with adaptive learning rates
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights
        )

        # Learning Rate Schedule: CosineAnnealingLR (smooth decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # H100 Optimization 5: GradScaler for AMP
        scaler = GradScaler('cuda') if torch.cuda.is_available() else None

        epoch_losses = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                if device.type == 'cuda':
                    x_batch = x_batch.to(memory_format=torch.channels_last)
                y_batch = y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                # H100 Optimization 6: AMP with bfloat16
                if scaler:
                    with autocast('cuda', dtype=torch.bfloat16):
                        out = self(x_batch)
                        loss = criterion(out, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # H100 Optimization 7: Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self(x_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                preds = out.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_acc = correct / total
            epoch_losses.append(avg_loss)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            val_acc = None
            if x_valid is not None and y_valid_indices is not None:
                val_acc = self.compute_accuracy_from_tensors(x_valid, y_valid_indices, device, batch_size)

            test_acc = None
            if x_test is not None and y_test_indices is not None:
                test_acc = self.compute_accuracy_from_tensors(x_test, y_test_indices, device, batch_size)

            # Print progress every 5 epochs or at start/end
            if verbose and ((epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {train_acc:.4f}"
                if val_acc is not None:
                    msg += f" | Val: {val_acc:.4f}"
                if test_acc is not None:
                    msg += f" | Test: {test_acc:.4f}"
                msg += f" | LR: {current_lr:.6f}"
                print(msg)

        return {"epoch_losses": epoch_losses}


# ============================================================================
# RESNET ARCHITECTURES
# ============================================================================

class ResNet18(ResNetBase):
    """ResNet-18 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet34(ResNetBase):
    """ResNet-34 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet50(ResNetBase):
    """ResNet-50 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 3, stride=1, block=Bottleneck)
        self.layer2 = self._make_layer(128, 4, stride=2, block=Bottleneck)
        self.layer3 = self._make_layer(256, 6, stride=2, block=Bottleneck)
        self.layer4 = self._make_layer(512, 3, stride=2, block=Bottleneck)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride, block=Bottleneck):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet101(ResNetBase):
    """ResNet-101 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 3, stride=1, block=Bottleneck)
        self.layer2 = self._make_layer(128, 4, stride=2, block=Bottleneck)
        self.layer3 = self._make_layer(256, 23, stride=2, block=Bottleneck)
        self.layer4 = self._make_layer(512, 3, stride=2, block=Bottleneck)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride, block=Bottleneck):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet152(ResNetBase):
    """ResNet-152 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 3, stride=1, block=Bottleneck)
        self.layer2 = self._make_layer(128, 8, stride=2, block=Bottleneck)
        self.layer3 = self._make_layer(256, 36, stride=2, block=Bottleneck)
        self.layer4 = self._make_layer(512, 3, stride=2, block=Bottleneck)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride, block=Bottleneck):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet110(ResNetBase):
    """ResNet-110 for CIFAR-10 with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(16, 18, stride=1)
        self.layer2 = self._make_layer(32, 18, stride=2)
        self.layer3 = self._make_layer(64, 18, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet9(ResNetBase):
    """ResNet-9 lightweight variant with H100 optimizations"""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.model = self._build_model(num_classes)

    def _build_model(self, num_classes):
        def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                          stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
            Mul(0.2)
        )

    def forward(self, x):
        return self.model(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
