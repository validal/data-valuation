from collections import OrderedDict
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, RandomSampler, Dataset, Sampler
from torch.utils.data.sampler import BatchSampler
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from opendataval.dataloader.util import CatDataset
from opendataval.dataval.api import DataEvaluator, ModelMixin


class StratifiedBatchSampler(Sampler):
    """Stratified batch sampler that maintains class distribution in each batch.

    This ensures that each batch has a representative mix of classes, which is
    especially important for imbalanced datasets.
    """

    def __init__(self, y: torch.Tensor, batch_size: int, num_batches: int,
                 random_state: Optional[RandomState] = None):
        """
        Parameters
        ----------
        y : torch.Tensor
            Labels for stratification (should be on CPU for numpy operations)
        batch_size : int
            Size of each batch
        num_batches : int
            Total number of batches to sample
        random_state : RandomState, optional
            Random state for reproducibility
        """
        # Ensure y is on CPU for numpy operations
        if torch.is_tensor(y):
            self.y = y.cpu().numpy() if y.is_cuda else y.numpy()
        else:
            self.y = np.array(y)

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.random_state = check_random_state(random_state)

        # Get unique classes and their indices
        self.classes = np.unique(self.y)
        self.class_indices = {c: np.where(self.y == c)[0] for c in self.classes}
        self.class_counts = {c: len(indices) for c, indices in self.class_indices.items()}

        # Calculate how many samples from each class per batch
        self.samples_per_class = {}
        for c in self.classes:
            # Proportional allocation based on class distribution
            proportion = self.class_counts[c] / len(self.y)
            self.samples_per_class[c] = max(1, int(batch_size * proportion))

        # Adjust to ensure batch size is correct
        total_samples = sum(self.samples_per_class.values())
        if total_samples < batch_size:
            # Add remaining samples to the largest class
            largest_class = max(self.classes, key=lambda c: self.samples_per_class[c])
            self.samples_per_class[largest_class] += batch_size - total_samples
        elif total_samples > batch_size:
            # Remove extra samples from the smallest class
            smallest_class = min(self.classes, key=lambda c: self.samples_per_class[c])
            diff = total_samples - batch_size
            self.samples_per_class[smallest_class] = max(1, self.samples_per_class[smallest_class] - diff)

        # Verify total
        assert sum(self.samples_per_class.values()) == batch_size, \
            f"Batch size mismatch: {sum(self.samples_per_class.values())} != {batch_size}"

        logger.info(f"StratifiedBatchSampler initialized:")
        for c in self.classes:
            logger.info(f"  Class {c}: {self.samples_per_class[c]} samples per batch (total {self.class_counts[c]})")

    def __iter__(self):
        """Generate batches with stratified sampling."""
        for _ in range(self.num_batches):
            batch_indices = []
            for c in self.classes:
                # Sample from each class with replacement to handle large num_batches
                indices = self.class_indices[c]
                n_samples = self.samples_per_class[c]

                # Always sample with replacement to support arbitrary num_batches
                sampled = self.random_state.choice(indices, size=n_samples, replace=True)
                batch_indices.extend(sampled)

            # Shuffle the batch
            self.random_state.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


class DVRL(DataEvaluator, ModelMixin):
    """Data valuation using reinforcement learning class, implemented with PyTorch.

    References
    ----------
    .. [1] J. Yoon, S. Arik, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimensions for the RL Multilayer Perceptron Value Estimator (VE)
        (details in :py:class:`DataValueEstimatorRL` class), by default 100
    layer_number : int, optional
        Number of hidden layers for the Value Estimator (VE), by default 5
    comb_dim : int, optional
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    rl_epochs : int, optional
        Number of training epochs for the VE, by default 1000
    rl_batch_size : int, optional
        Batch size for training the VE, by default 32
    lr : float, optional
        Learning rate for the VE, by default 0.01
    threshold : float, optional
        Search rate threshold, the VE may get stuck in certain bounds close to
        :math:`[0, 1]`, thus outside of :math:`[1-threshold, threshold]` we encourage
        searching, by default 0.9
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state : RandomState, optional
        Random initial state, by default None
    stratified_batches : bool, optional
        Whether to use stratified batching to maintain class distribution, by default True
    """

    def __init__(
        self,
        hidden_dim: int = 100,
        layer_number: int = 5,
        comb_dim: int = 10,
        rl_epochs: int = 1000,
        rl_batch_size: int = 32,
        lr: float = 0.01,
        threshold: float = 0.9,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        random_state: Optional[RandomState] = None,
        stratified_batches: bool = True,
    ):
        super().__init__(random_state=random_state)

        # Value estimator parameters
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.device = device

        # Training parameters
        self.rl_epochs = rl_epochs
        self.rl_batch_size = rl_batch_size
        self.lr = lr
        self.threshold = threshold
        self.stratified_batches = stratified_batches
        self.epsilon = 1e-8

        # Performance metric
        from opendataval.metrics import accuracy
        self.perf_metric = accuracy

        # Initialize device-specific generators
        self._init_generators()

        logger.info(f"DVRL initialized with: hidden_dim={hidden_dim}, layer_number={layer_number}, "
                   f"rl_epochs={rl_epochs}, rl_batch_size={rl_batch_size}, lr={lr}, "
                   f"stratified_batches={stratified_batches}, device={device}")

    def _init_generators(self):
        """Initialize random generators for GPU."""
        seed = self.random_state.tomaxint() if self.random_state is not None else 42
        self.gen = torch.Generator(self.device).manual_seed(seed)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for DVRL.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        # Move ALL data to GPU immediately for maximum performance
        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.x_valid = x_valid.to(self.device)
        self.y_valid = y_valid.to(self.device)

        self.num_points, [*self.feature_dim] = len(x_train), x_train[0].shape
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape

        # Check class distribution for stratification
        if self.stratified_batches:
            # Move to CPU only for numpy operations
            y_train_np = self.y_train.cpu().numpy() if torch.is_tensor(self.y_train) else np.array(self.y_train)
            # Convert one-hot to class indices if needed
            if y_train_np.ndim > 1:
                y_train_np = y_train_np.argmax(axis=1)
            unique_classes, counts = np.unique(y_train_np, return_counts=True)
            logger.info(f"Class distribution for stratification:")
            for c, count in zip(unique_classes, counts):
                logger.info(f"  Class {c}: {count} samples ({count/len(y_train_np)*100:.1f}%)")

            if len(unique_classes) < 2:
                logger.warning("⚠️ Only one class detected! Stratification disabled.")
                self.stratified_batches = False

        self.value_estimator = DataValueEstimatorRL(
            x_dim=np.prod(self.feature_dim),
            y_dim=np.prod(self.label_dim),
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            random_state=self.random_state,
        ).to(self.device)

        logger.info(f"Value Estimator created with {sum(p.numel() for p in self.value_estimator.parameters()):,} parameters")
        logger.info(f"All data on device: {self.x_train.device}")

        return self

    def _evaluate_baseline_models(self, *args, **kwargs):
        """Load and train baseline models - everything on GPU."""
        logger.info("Evaluating baseline models...")

        # Final model
        self.final_model = self.pred_model.clone()

        # Train baseline model with input data (on GPU)
        logger.info("Training baseline model on full training data...")
        self.ori_model = self.pred_model.clone()
        self.ori_model.fit(self.x_train, self.y_train, *args, **kwargs)

        # Trains validation model (on GPU)
        logger.info("Training validation model on validation data...")
        self.val_model = self.ori_model.clone()
        self.val_model.fit(self.x_valid, self.y_valid, *args, **kwargs)

        # Eval performance - everything on GPU
        y_valid_hat = self.ori_model.predict(self.x_valid)
        self.valid_perf = self.perf_metric(
            y_valid_hat,
            self.y_valid.unsqueeze(1) if self.y_valid.ndim == 1 else self.y_valid
        )

        # Compute diff - stays on GPU
        y_pred = self.val_model.predict(self.x_train)
        self.y_pred_diff = torch.abs(self.y_train - y_pred)

        logger.info(f"Baseline validation performance: {self.valid_perf:.6f}")
        logger.info(f"Y_pred_diff - min: {self.y_pred_diff.min().item():.6f}, "
                   f"max: {self.y_pred_diff.max().item():.6f}, "
                   f"mean: {self.y_pred_diff.mean().item():.6f}")

    def _create_stratified_dataloader(self, data: Dataset, batch_size: int,
                                     num_batches: int) -> DataLoader:
        """Create a DataLoader with stratified batch sampling - NO pin_memory."""
        # Get labels for stratification - move to CPU for numpy operations
        y_labels = self.y_train.cpu() if torch.is_tensor(self.y_train) else self.y_train

        # Convert one-hot encoded labels to class indices
        if y_labels.dim() > 1:
            y_labels = y_labels.argmax(dim=1)

        # Create stratified batch sampler
        batch_sampler = StratifiedBatchSampler(
            y=y_labels,
            batch_size=batch_size,
            num_batches=num_batches,
            random_state=self.random_state
        )

        # CRITICAL: pin_memory=False because data is already on GPU
        dataloader = DataLoader(
            data,
            batch_sampler=batch_sampler,
            pin_memory=False,  # Data is on GPU
            num_workers=0,
        )

        return dataloader

    def _create_random_dataloader(self, data: Dataset, batch_size: int,
                                 num_batches: int) -> DataLoader:
        """Create a DataLoader with random sampling - NO pin_memory."""
        gen = torch.Generator(self.device).manual_seed(self.random_state.tomaxint())
        rs = RandomSampler(data, True, num_batches * batch_size, generator=gen)

        # CRITICAL: pin_memory=False because data is already on GPU
        dataloader = DataLoader(
            data,
            batch_size,
            sampler=rs,
            generator=gen,
            pin_memory=False,  # Data is on GPU
            num_workers=0,
            persistent_workers=False,
        )
        return dataloader

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):
        """Trains model to predict data values - everything on GPU."""
        batch_size = min(self.rl_batch_size, len(self.x_train))
        total_batches = self.rl_epochs
        self._evaluate_baseline_models(*args, **kwargs)

        # Solver
        optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        criterion = DveLoss(threshold=self.threshold)

        # Re-initialize generator
        self._init_generators()

        data = CatDataset(self.x_train, self.y_train, self.y_pred_diff)

        # Create dataloader with or without stratification
        if self.stratified_batches:
            logger.info(f"Using STRATIFIED batching with {batch_size} samples per batch")
            dataloader = self._create_stratified_dataloader(
                data, batch_size, total_batches
            )
        else:
            logger.info(f"Using RANDOM batching with {batch_size} samples per batch")
            dataloader = self._create_random_dataloader(
                data, batch_size, total_batches
            )

        # Track batch statistics
        batch_stats = {
            'class_counts': [],
            'selection_rates': [],
            'rewards': [],
            'losses': [],
            'batch_accuracies': [],
            'batch_selected_counts': [],
            'batch_class_dist_selected': []
        }

        # Detailed per-batch tracking for CSV export
        batch_details = []

        for batch_idx, (x_batch, y_batch, y_hat_batch) in enumerate(tqdm.tqdm(dataloader, desc="DVRL Training")):
            # Data is already on GPU from DataLoader
            # No need to call .to(device) - it's already there!

            optimizer.zero_grad()

            # Debug: print shapes and devices before forward pass
            if batch_idx == 0:
                logger.info(f"[DVRL DEBUG BATCH 0] Shapes and Devices:")
                logger.info(f"  x_batch: shape={x_batch.shape}, device={x_batch.device}, dtype={x_batch.dtype}")
                logger.info(f"  y_batch: shape={y_batch.shape}, device={y_batch.device}, dtype={y_batch.dtype}")
                logger.info(f"  y_hat_batch: shape={y_hat_batch.shape}, device={y_hat_batch.device}, dtype={y_hat_batch.dtype}")

            try:
                # Generates selection probability
                pred_dataval = self.value_estimator(x_batch, y_batch, y_hat_batch)
            except Exception as e:
                logger.error(f"[DVRL ERROR] Batch {batch_idx} forward pass failed!")
                logger.error(f"  x_batch: {x_batch.shape}")
                logger.error(f"  y_batch: {y_batch.shape}")
                logger.error(f"  y_hat_batch: {y_hat_batch.shape}")
                logger.error(f"  Error: {type(e).__name__}: {e}")
                raise

            # Debug: stats of predicted data values
            p_mean = pred_dataval.mean().item()
            p_std = pred_dataval.std().item()

            if batch_idx % 50 == 0 and batch_idx > 0:
                logger.debug(f"Batch {batch_idx}: pred mean={p_mean:.4f}, std={p_std:.4f}")

            # Samples the selection probability - on GPU
            select_prob = torch.bernoulli(pred_dataval, generator=self.gen)

            if select_prob.sum().item() == 0:  # Exception (select probability is 0)
                logger.warning(f"⚠️ Batch {batch_idx}: No data selected! Resetting to 0.5")
                pred_dataval = 0.5 * torch.ones_like(pred_dataval, requires_grad=True)
                select_prob = torch.bernoulli(pred_dataval, generator=self.gen)

            # Track selection rate and detailed stats
            selection_rate = select_prob.mean().item()
            num_selected = select_prob.sum().item()
            batch_stats['selection_rates'].append(selection_rate)
            batch_stats['batch_selected_counts'].append(num_selected)

            # Get class labels for tracking
            if y_batch.dim() > 1:
                y_batch_classes = y_batch.argmax(dim=1)
            else:
                y_batch_classes = y_batch

            # Get selected indices - keep on GPU for indexing
            selected_mask = select_prob.squeeze() > 0
            selected_indices = torch.where(selected_mask)[0]

            if len(selected_indices) > 0:
                # Get selected classes - on GPU
                selected_classes = y_batch_classes[selected_indices]
                unique_selected, counts_selected = torch.unique(selected_classes, return_counts=True)
                class_dist = {int(c.item()): int(cnt.item()) for c, cnt in zip(unique_selected, counts_selected)}
                batch_stats['batch_class_dist_selected'].append(class_dist)

                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx}: selected {num_selected}/{len(select_prob)} samples, "
                                f"class dist={class_dist}, rate={selection_rate:.3f}")

                    # Warn if only one class is selected
                    if len(unique_selected) == 1:
                        logger.warning(f"⚠️ Batch {batch_idx}: Only class {unique_selected[0].item()} selected! "
                                      f"Batch had {torch.sum(y_batch_classes == 0).item()} class 0, "
                                      f"{torch.sum(y_batch_classes == 1).item()} class 1")
            else:
                class_dist = {}
                logger.warning(f"⚠️ Batch {batch_idx}: No samples selected after bernoulli!")

            # Prediction and training - all on GPU
            new_model = self.pred_model.clone()

            # Fit model with sample weights (all on GPU)
            new_model.fit(
                x_batch,
                y_batch,
                *args,
                sample_weight=select_prob.squeeze().detach(),  # Keep on GPU
                **kwargs
            )

            # Debug: Check accuracy on selected subset before validation
            if len(selected_indices) > 0:
                x_selected = x_batch[selected_indices]
                y_selected = y_batch[selected_indices]
                y_pred_selected = new_model.predict(x_selected)

                # Compute accuracy on selected subset
                if y_selected.dim() > 1:
                    y_selected_classes = y_selected.argmax(dim=1)
                else:
                    y_selected_classes = y_selected

                if y_pred_selected.dim() > 1:
                    y_pred_classes = y_pred_selected.argmax(dim=1)
                else:
                    y_pred_classes = y_pred_selected

                acc_selected = (y_pred_classes == y_selected_classes).float().mean().item()
                batch_stats['batch_accuracies'].append(acc_selected)

                if batch_idx % 10 == 0:
                    logger.debug(f"  → Accuracy on {len(selected_indices)} selected: {acc_selected:.4f}")
            else:
                batch_stats['batch_accuracies'].append(0.0)
                logger.warning(f"Batch {batch_idx}: No selected samples for training!")

            # Reward computation - everything on GPU
            y_valid_hat = new_model.predict(self.x_valid)
            dvrl_perf = self.perf_metric(
                y_valid_hat,
                self.y_valid.unsqueeze(1) if self.y_valid.ndim == 1 else self.y_valid
            )
            print(f"Batch {batch_idx}: Validation performance: {dvrl_perf:.6f}, baseline: {self.valid_perf:.6f}")
            reward_curr = dvrl_perf - self.valid_perf

            if batch_idx % 10 == 0:
                logger.debug(f"  → Validation acc: {dvrl_perf:.4f}, baseline: {self.valid_perf:.4f}, reward: {reward_curr:.6f}")

            batch_stats['rewards'].append(reward_curr)

            # Trains the VE
            try:
                loss = criterion(pred_dataval, select_prob, reward_curr)
            except Exception as e:
                logger.error(f"[DVRL ERROR] Loss computation failed at batch {batch_idx}")
                logger.error(f"  pred_dataval device: {pred_dataval.device}")
                logger.error(f"  select_prob device: {select_prob.device}")
                logger.error(f"  Error: {type(e).__name__}: {e}")
                raise

            # Backward pass - don't retain graph to save memory
            loss.backward(retain_graph=False)
            optimizer.step()

            batch_stats['losses'].append(loss.item())

            # Clear GPU cache periodically
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Record detailed per-batch info for CSV export
            batch_details.append({
                'batch_idx': batch_idx,
                'batch_size': len(x_batch),
                'pred_mean': p_mean,
                'pred_std': p_std,
                'class_0_before': (y_batch_classes == 0).sum().item() if len(selected_indices) > 0 else 0,
                'class_1_before': (y_batch_classes == 1).sum().item() if len(selected_indices) > 0 else 0,
                'selection_rate': selection_rate,
                'num_selected': int(num_selected),
                'class_0_selected': class_dist.get(0, 0) if len(selected_indices) > 0 else 0,
                'class_1_selected': class_dist.get(1, 0) if len(selected_indices) > 0 else 0,
                'sgd_accuracy': batch_stats['batch_accuracies'][-1],
                'validation_accuracy': dvrl_perf,
                've_loss': loss.item(),
                'reward': reward_curr
            })

        # Log batch statistics
        if batch_stats['selection_rates']:
            logger.info(f"Training statistics:")
            logger.info(f"  Avg selection rate: {np.mean(batch_stats['selection_rates']):.4f}")
            logger.info(f"  Avg reward: {np.mean(batch_stats['rewards']):.6f}")
            logger.info(f"  Avg loss: {np.mean(batch_stats['losses']):.6f}")

        # Detailed batch accuracy analysis
        if batch_stats['batch_accuracies']:
            accs = np.array(batch_stats['batch_accuracies'])
            logger.info(f"\nBatch Accuracy Analysis:")
            logger.info(f"  Min accuracy: {accs.min():.4f}")
            logger.info(f"  Max accuracy: {accs.max():.4f}")
            logger.info(f"  Mean accuracy on selected: {accs.mean():.4f}")
            logger.info(f"  Std accuracy: {accs.std():.4f}")
            logger.info(f"  Batches with 0.5 acc (random): {(np.abs(accs - 0.5) < 0.01).sum()}/{len(accs)}")
            logger.info(f"  Batches with <0.55 acc: {(accs < 0.55).sum()}/{len(accs)}")

            # Find problematic batches
            low_acc_batches = np.where(accs < 0.55)[0]
            if len(low_acc_batches) > 0:
                logger.warning(f"⚠️ Found {len(low_acc_batches)} batches with low accuracy (<0.55):")
                for batch_id in low_acc_batches[:5]:  # Show first 5
                    logger.warning(f"    Batch {batch_id}: acc={accs[batch_id]:.4f}, "
                                  f"selected={int(batch_stats['batch_selected_counts'][batch_id])}, "
                                  f"class_dist={batch_stats['batch_class_dist_selected'][batch_id] if batch_id < len(batch_stats['batch_class_dist_selected']) else 'N/A'}")

        # Selection statistics
        if batch_stats['batch_selected_counts']:
            selected_counts = np.array(batch_stats['batch_selected_counts'])
            logger.info(f"\nSelection Statistics:")
            logger.info(f"  Avg samples selected per batch: {selected_counts.mean():.1f}/{batch_size}")
            logger.info(f"  Min selected: {selected_counts.min():.0f}")
            logger.info(f"  Max selected: {selected_counts.max():.0f}")
            logger.info(f"  Batches with <5 selected: {(selected_counts < 5).sum()}/{len(selected_counts)}")
            logger.info(f"  Batches with >90% selected: {(selected_counts > batch_size*0.9).sum()}/{len(selected_counts)}")

        # Calculate final weights - all on GPU
        weights = torch.zeros(0, 1, device=self.device)
        for x_batch, y_batch, y_hat_batch in DataLoader(
            data, batch_size=self.rl_batch_size, shuffle=False,
            pin_memory=False  # Data is on GPU
        ):
            # Data already on GPU
            data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
            weights = torch.cat([weights, data_values])

        # Log final weights statistics
        wcpu = weights.detach().cpu()
        logger.info(f"Final weights stats:")
        logger.info(f"  Min: {wcpu.min().item():.6f}")
        logger.info(f"  Max: {wcpu.max().item():.6f}")
        logger.info(f"  Mean: {wcpu.mean().item():.6f}")
        logger.info(f"  Std: {wcpu.std().item():.6f}")
        logger.info(f"  Near 0 (<0.1): {(wcpu < 0.1).sum().item()}/{wcpu.numel()}")
        logger.info(f"  Near 1 (>0.9): {(wcpu > 0.9).sum().item()}/{wcpu.numel()}")

        self.final_model = self.pred_model.clone()
        self.final_model.fit(
            self.x_train,
            self.y_train,
            *args,
            sample_weight=weights.squeeze().detach(),  # Keep on GPU
            **kwargs,
        )

        # Final model performance against baseline
        y_valid_hat_final = self.final_model.predict(self.x_valid)
        final_valid_perf = self.perf_metric(
            y_valid_hat_final,
            self.y_valid.unsqueeze(1) if self.y_valid.ndim == 1 else self.y_valid
        )
        logger.info(f"Final model performance: {final_valid_perf:.6f}")
        logger.info(f"Improvement over baseline: {final_valid_perf - self.valid_perf:.6f}")

        # Export batch details to CSV
        if batch_details:
            try:
                import pandas as pd
                df_batch_details = pd.DataFrame(batch_details)

                # Save to CSV
                csv_path = f"dvrl_batch_details_epochs{self.rl_epochs}_batch{batch_size}.csv"
                df_batch_details.to_csv(csv_path, index=False)
                logger.info(f"\n✅ Batch details exported to: {csv_path}")
                logger.info(f"   Columns: batch_idx, batch_size, pred_mean, pred_std,")
                logger.info(f"            class distributions (before/after selection),")
                logger.info(f"            sgd_accuracy, validation_accuracy, ve_loss, reward")

                # Print sample of CSV
                logger.info(f"\n📊 First 5 batches:")
                logger.info(f"\n{df_batch_details.head(5).to_string()}")
            except ImportError:
                logger.warning("pandas not available, skipping CSV export")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values for DVRL using the Value Estimator MLP.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        y_valid_pred = self.final_model.predict(self.x_train)
        y_hat = torch.abs(self.y_train - y_valid_pred)
        response = torch.zeros(0, 1, device=self.device)

        # Estimates data value
        with torch.no_grad():  # No dropout layers so no need to set to eval
            data = CatDataset(self.x_train, self.y_train, y_hat)
            for x_batch, y_batch, y_hat_batch in DataLoader(
                data, batch_size=self.rl_batch_size, shuffle=False,
                pin_memory=False  # Data is on GPU
            ):
                # Data already on GPU
                data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
                response = torch.cat([response, data_values])

        # Final data values stats
        resp_cpu = response.detach().cpu()
        logger.info(f"Final data values stats:")
        logger.info(f"  Min: {resp_cpu.min().item():.6f}")
        logger.info(f"  Max: {resp_cpu.max().item():.6f}")
        logger.info(f"  Mean: {resp_cpu.mean().item():.6f}")
        logger.info(f"  Std: {resp_cpu.std().item():.6f}")

        return response.squeeze().cpu().numpy()


class DataValueEstimatorRL(nn.Module):
    """Value Estimator model.

    Here, we assume a simple multi-layer perceptron architecture for the data
    value evaluator model. For data types like tabular, multi-layer perceptron
    is already efficient at extracting the relevant information.
    For high-dimensional data types like images or text,
    it is important to introduce inductive biases to the architecture to
    extract information efficiently. In such cases, there are two options:
    (i) Input the encoded representations (e.g. the last layer activations of
    ResNet for images, or the last layer activations of BERT for  text) and use
    the multi-layer perceptron on top of it. The encoded representations can
    simply come from a pre-trained predictor model using the entire dataset.
    (ii) Modify the data value evaluator model definition below to have the
    appropriate inductive bias (e.g. using convolutions layers for images,
    or attention layers text).

    References
    ----------
    .. [1] J. Yoon, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    x_dim : int
        Data covariates dimension, can be flatten dimension size
    y_dim : int
        Data labels dimension, can be flatten dimension size
    hidden_dim : int
        Hidden dimensions for the Value Estimator
    layer_number : int
        Number of hidden layers for the Value Estimator
    comb_dim : int
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        random_state: Optional[RandomState] = None,
    ):
        super().__init__()

        if random_state is not None:  # Can't pass generators to nn.Module layers
            torch.manual_seed(check_random_state(random_state).tomaxint())

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(x_dim + y_dim, hidden_dim)
        mlp_layers["input_acti"] = nn.ReLU()

        i = 0  # Initialize i in case loop doesn't execute
        for i in range(int(layer_number - 3)):
            mlp_layers[f"{i + 1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i + 1}_acti"] = nn.ReLU()

        mlp_layers[f"{i + 1}_out_lin"] = nn.Linear(hidden_dim, comb_dim)
        mlp_layers[f"{i + 1}_out_acti"] = nn.ReLU()

        self.mlp = nn.Sequential(mlp_layers)

        yhat_combine = OrderedDict()

        # Combines with y_hat
        yhat_combine["reduce_lin"] = nn.Linear(comb_dim + y_dim, comb_dim)
        yhat_combine["reduce_acti"] = nn.ReLU()

        yhat_combine["out_lin"] = nn.Linear(comb_dim, 1)
        yhat_combine["out_acti"] = nn.Sigmoid()  # Sigmoid for binary selection
        self.yhat_comb = nn.Sequential(yhat_combine)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of inputs through value estimator for data values of input.

        Forward pass through Value Estimator. Returns selection probabilities.
        Concats the difference between labels and predicted labels to compute
        selection probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Data covariates
        y : torch.Tensor
            Data labels
        y_hat : torch.Tensor
            Data label predictions (from prediction model)

        Returns
        -------
        torch.Tensor
            Selection probabilities per covariate data point
        """
        # Flattens input dimension in case it is more than 2D (e.g., images)
        if x.dim() > 2:  # [batch, channels, height, width] → [batch, features]
            x = x.flatten(start_dim=1)
        if y.dim() > 2:  # [batch, ...] → [batch, 1] or [batch, classes]
            y = y.flatten(start_dim=1)
        if y_hat.dim() > 2:  # [batch, ...] → [batch, 1] or [batch, classes]
            y_hat = y_hat.flatten(start_dim=1)
        if y.dim() == 1:
            y = y.unsqueeze(1)  # (batch,) → (batch, 1)

        out = torch.concat((x, y), dim=1)
        out = self.mlp(out)
        out = torch.cat((out, y_hat), dim=1)
        out = self.yhat_comb(out)
        return out


class DveLoss(nn.Module):
    """Compute Loss for Value Estimator.

    Custom loss function for the value estimator RL Model. Uses BCE Loss and
    checks average is within threshold to encourage exploration

    Parameters
    ----------
    threshold : float, optional
        Search rate threshold, the VE may get stuck in certain bounds close to
        :math:`[0, 1]`, thus outside of :math:`[1-threshold, threshold]` we encourage
        searching, by default 0.9
    exploration_weight : float, optional
        Large constant to encourage exploration in the Value Estimator, by default 1e3
    """

    def __init__(self, threshold: float = 0.9, exploration_weight: float = 1e3):
        super().__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
        self,
        pred_dataval: torch.Tensor,
        selector_input: torch.Tensor,
        reward_input: float,
    ) -> torch.Tensor:
        """Compute the loss for the Value Estimator.

        Uses REINFORCE Algorithm to compute a loss for the Value Estimator.
        `pred_dataval` is the data values. `selector_input` is a bernoulli random
        variable with `p=pred_dataval`. Computes a BCE between `pred_dataval` and
        `selector_input` and multiplies by the reward signal. Adds an additional loss
        if the Value Estimator is getting stuck outside the threshold.

        References
        ----------
        .. [1] R. J. Williams,
            Simple statistical gradient-following algorithms for connectionist
            reinforcement learning,
            Machine Learning, vol. 8, no. 3-4, pp. 229-256, May 1992,
            doi: https://doi.org/10.1007/bf00992696.


        Parameters
        ----------
        pred_dataval : torch.Tensor
            Predicted values from value estimator
        selector_input : torch.Tensor
            `1` for selected `0` for not selected, bernoulli random variable
        reward_input : float
            Reward/performance signal of prediction model trained on `selector_input`.
            If positive, indicates better than naive model of full sample.

        Returns
        -------
        torch.Tensor
            Computed loss tensor for Value Estimator
        """
        # Ensure both tensors are on the same device
        if pred_dataval.device != selector_input.device:
            selector_input = selector_input.to(pred_dataval.device)

        loss = F.binary_cross_entropy(pred_dataval, selector_input, reduction="sum")

        reward_loss = reward_input * loss
        search_loss = (  # Additional loss when VE is stuck outside threshold range
            F.relu(torch.mean(pred_dataval) - self.threshold)
            + F.relu((1 - self.threshold) - torch.mean(pred_dataval))
        )

        return reward_loss + (self.exploration_weight * search_loss)