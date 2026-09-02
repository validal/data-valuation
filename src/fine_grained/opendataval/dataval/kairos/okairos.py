"""OKairos: Online Kairos using streaming batch updates.

Divides training data into an initial batch + streaming batches, then applies
Kairos.online_update() incrementally. Simulates the online protocol from the paper.

This is different from:
- Kairos: Full kernel matrix computation (fastest for one-shot, O(n²) memory)
- bKairos: Batched computation without online_update (fast offline, O(batch×n) memory)
- OKairos: Online_update streaming (simulates incremental learning, tracks per-batch time)
"""

import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from opendataval.dataval.api import DataEvaluator
from opendataval.dataval.kairos.kairos import Kairos


def _to_f32_2d(x):
    """Convert to numpy float32, flatten if image-shaped."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x


class OKairos(DataEvaluator):
    """Online Kairos: Initialize on first batch, then apply online_update incrementally.

    This simulates the online/streaming setting where training data arrives in batches.
    Useful for benchmarking the streaming protocol and measuring wall-time cost per batch.

    Parameters
    ----------
    lambda_weight : float, optional
        Weight on feature metric, by default 0.97
    sigma_feature : float, optional
        RBF bandwidth. None → auto-compute from full training set, by default None
    use_median_heuristic : bool, optional
        Estimate σ from median distance, by default True
    num_samples : int, optional
        Pairs sampled for median heuristic, by default 10000
    init_batch_size : int, optional
        Size of initial batch for Kairos.input_data(), by default 1000
    batch_size : int, optional
        Size of streaming batches for online_update(), by default 1000
    random_state : Optional, optional
        Random seed, by default None
    verbose : bool, optional
        Print progress, by default False
    unbiased : bool, optional
        Use unbiased feature computation, by default False
    """

    def __init__(
        self,
        lambda_weight: float = 0.97,
        sigma_feature: Optional[float] = None,
        use_median_heuristic: bool = True,
        num_samples: int = 10000,
        init_batch_size: int = 1000,
        batch_size: int = 1000,
        random_state: Optional = None,
        verbose: bool = False,
        unbiased: bool = False,
    ):
        super().__init__(random_state=random_state)
        self.lambda_weight = float(lambda_weight)
        self.sigma_feature = sigma_feature
        self.use_median_heuristic = use_median_heuristic
        self.num_samples = int(num_samples)
        self.init_batch_size = int(init_batch_size)
        self.batch_size = int(batch_size)
        self.verbose = verbose
        self.unbiased = unbiased

        seed = random_state if isinstance(random_state, int) else 42
        self._seed = seed

        self.kairos: Optional[Kairos] = None
        self.batch_runtimes: list[float] = []  # Accumulated seconds per batch
        self.x_train_full = None
        self.y_train_full = None
        self.x_valid = None
        self.y_valid = None

    def _log(self, msg):
        if self.verbose:
            print(f"[OKairos] {msg}", flush=True)

    # ===================================================================== API

    def input_data(self, x_train, y_train, x_valid, y_valid) -> "OKairos":
        """Store training and validation data.

        Parameters
        ----------
        x_train : array-like or torch.Tensor
            Training features, shape (n, d) or image-shaped
        y_train : array-like or torch.Tensor
            Training labels, shape (n,) or (n, c)
        x_valid : array-like or torch.Tensor
            Validation features
        y_valid : array-like or torch.Tensor
            Validation labels

        Returns
        -------
        OKairos
            Self for method chaining
        """
        self.x_train_full = _to_f32_2d(x_train)
        self.y_train_full = _to_f32_2d(y_train)
        self.x_valid = _to_f32_2d(x_valid)
        self.y_valid = _to_f32_2d(y_valid)

        n_train = len(self.x_train_full)
        n_valid = len(self.x_valid)
        d = self.x_train_full.shape[1]

        self._log(f"Data loaded: {n_train:,} train × {d}D, {n_valid:,} validation")
        self._log(f"Init batch: {self.init_batch_size:,}, "
                  f"streaming batch: {self.batch_size:,}")

        # Create Kairos instance for online updates
        self.kairos = Kairos(
            lambda_weight=self.lambda_weight,
            sigma_feature=self.sigma_feature,
            use_median_heuristic=self.use_median_heuristic,
            num_samples=self.num_samples,
            random_state=self._seed,
            debug=False,
            unbiased=self.unbiased,
        )

        # Pre-compute bandwidth on FULL training set if needed
        if self.kairos.sigma_feature is None:
            self.kairos.X_train = torch.from_numpy(self.x_train_full)
            self.kairos.sigma_feature = self.kairos._compute_median_heuristic()
            self.kairos.X_train = None
            self._log(f"Pre-computed sigma_feature = {self.kairos.sigma_feature:.6f}")

        return self

    def train_data_values(self, *args, **kwargs) -> "OKairos":
        """Process training data: init batch + online updates.

        Workflow:
        1. Init batch (size init_batch_size): Kairos.input_data() + train_data_values()
        2. Remaining batches: Kairos.online_update() for each batch
        3. Track wall time after each batch

        Returns
        -------
        OKairos
            Self for method chaining
        """
        x, y = self.x_train_full, self.y_train_full
        n_total = len(x)
        init_size = self.init_batch_size
        batch_size = self.batch_size

        # Compute number of batches
        n_batches = (n_total + batch_size - 1) // batch_size
        n_init_batches = (init_size + batch_size - 1) // batch_size

        self.batch_runtimes = []
        t0 = time.perf_counter()

        self._log(f"Starting online training: {n_batches} batches")

        # ===================================================================
        # PHASE 1: Initialize with first batch
        # ===================================================================
        init_end = min(init_size, n_total)
        x_init = x[:init_end]
        y_init = y[:init_end]

        self._log(f"Phase 1 (init): {init_end:,} samples")

        self.kairos.input_data(
            x_init, y_init,
            self.x_valid, self.y_valid,
        )
        self.kairos.train_data_values(debug=False)

        self.batch_runtimes.append(time.perf_counter() - t0)
        self._log(f"  ✓ Batch 1/{n_batches}: {init_end:,} samples, "
                  f"t={self.batch_runtimes[-1]:.3f}s")

        # ===================================================================
        # PHASE 2: Stream remaining batches with online_update
        # ===================================================================
        if init_end < n_total:
            self._log(f"Phase 2 (online): {n_total - init_end:,} samples in "
                      f"{n_batches - 1} batches")

            # Use tqdm for progress tracking
            batch_iter = tqdm(
                range(1, n_batches),
                desc="Online batches",
                disable=not self.verbose,
                leave=True
            )

            for batch_idx in batch_iter:
                start = init_end + (batch_idx - 1) * batch_size
                end = min(start + batch_size, n_total)

                x_batch = x[start:end]
                y_batch = y[start:end]

                # Apply online_update
                self.kairos.online_update(x_batch, y_batch)

                self.batch_runtimes.append(time.perf_counter() - t0)

                if self.verbose:
                    self._log(f"  ✓ Batch {batch_idx + 1}/{n_batches}: "
                              f"{end - start:,} samples (total: {end:,}), "
                              f"t={self.batch_runtimes[-1]:.3f}s")
                else:
                    batch_iter.set_description(
                        f"Online batches ({end:,}/{n_total:,})"
                    )

        self._log(f"✓ Training complete: {len(self.batch_runtimes)} batches, "
                  f"total time {self.batch_runtimes[-1]:.3f}s")

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Evaluate and return data values.

        Returns
        -------
        np.ndarray
            Data values for each training sample, shape (n_train,)
        """
        return self.kairos.evaluate_data_values()

    # ==================================================================== Utils

    def get_runtimes(self) -> dict:
        """Get runtime statistics.

        Returns
        -------
        dict
            Dictionary with:
            - 'total': total elapsed time
            - 'per_batch': list of per-batch times (differences)
            - 'accumulated': list of accumulated times
            - 'n_batches': number of batches
        """
        if not self.batch_runtimes:
            return {}

        accumulated = self.batch_runtimes
        per_batch = [accumulated[0]] + [
            accumulated[i] - accumulated[i-1]
            for i in range(1, len(accumulated))
        ]

        return {
            'total': accumulated[-1],
            'per_batch': per_batch,
            'accumulated': accumulated,
            'n_batches': len(accumulated),
        }

    def plot_runtimes(self, save_path: Optional[str] = None):
        """Plot streaming runtime.

        Parameters
        ----------
        save_path : Optional[str], optional
            Save plot to this path, by default None

        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[OKairos] matplotlib not available - skipping plot")
            return None

        if not self.batch_runtimes:
            print("[OKairos] No runtimes to plot")
            return None

        runtimes = self.get_runtimes()
        accumulated = runtimes['accumulated']
        per_batch = runtimes['per_batch']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Accumulated time vs batch index
        ax1.plot(accumulated, 'o-', linewidth=2, markersize=4)
        ax1.set_xlabel("Batch index")
        ax1.set_ylabel("Accumulated time (s)")
        ax1.set_title("OKairos: Streaming Runtime")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Per-batch time
        ax2.bar(range(len(per_batch)), per_batch)
        ax2.set_xlabel("Batch index")
        ax2.set_ylabel("Per-batch time (s)")
        ax2.set_title("Per-Batch Processing Time")
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            self._log(f"Saved runtime plot to {save_path}")

        return fig
