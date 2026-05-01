from typing import Optional
import time
import traceback

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.otg.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model

torch.set_num_threads(1)


def macos_fix():
    """Geomloss package has a bug on MacOS remedied as follows.

    `Link to similar bug: https://github.com/NVlabs/stylegan3/issues/75`_.
    """
    import os
    import sys

    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class LavaEvaluator(DataEvaluator, ModelLessMixin):
    """Data valuation using LAVA implementation.

    Parameters
    ----------
    device : torch.device, optional
        Tensor device for acceleration.
    embedding_model : Optional[Model], optional
        A model for computing embeddings, if needed.
    random_state : Optional[RandomState], optional
        Random state for reproducibility.
    lam_x : float, optional
        Regularization weight for features (default=1.0).
    lam_y : float, optional
        Regularization weight for labels (default=1.0).
    p : int, optional
        Power of the cost function (default=2).
    entreg : float, optional
        Entropy regularization (default=1e-1).
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        mode: str = "cls",
        p: int = 2,
        entreg: float = 1e-1,
        loss: str = "sinkhorn",
        feature_cost: Optional[FeatureCost] = None,
        debug: bool = False,
        blur: Optional[float] = None,
        # GeomLoss passthrough knobs
        scaling: float = 0.8,
        backend: Optional[str] = None,
        truncate: Optional[float] = None,
        diameter: Optional[float] = None,
        outer_debias: bool = True,

    ):
        macos_fix()
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.embedding_model = embedding_model
        self.device = device

        # Mode: 'cls' for classification (discrete labels), 'reg' for regression (continuous targets)
        self.mode = mode

        # OTDD-related parameters for convergence studies
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.entreg = entreg
        self.loss = loss
        self.feature_cost = feature_cost
        # Keep a handle to the OTDD distance object for later access (e.g., transport plan)
        self._dist: Optional[DatasetDistance] = None
        self.debug = bool(debug)
        self.blur = blur
        # GeomLoss control
        self.gl_scaling = float(scaling)
        self.gl_backend = backend
        self.gl_truncate = truncate
        self.gl_diameter = diameter
        self.outer_debias = bool(outer_debias)

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values using class-wise Wasserstein distance."""
        feature_cost = None

        if self.embedding_model is not None:
            resize = 32
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=(3, resize, resize),
                tgt_embedding=self.embedding_model,
                tgt_dim=(3, resize, resize),
                p=2,
                device=self.device.type,
            )

        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        # Note: we no longer use lam_y==0 concatenation hack. Use `mode='reg'` to
        # indicate regression and let DatasetDistance handle label distances.

        if self.debug:
            def _stat(t, name):
                try:
                    t_cpu = t.detach().cpu()
                    print(f"[lava] {name}: shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, "
                          f"device={t.device}, min={t_cpu.min().item():.4g}, max={t_cpu.max().item():.4g}")
                except Exception:
                    print(f"[lava] {name}: shape={getattr(t,'shape',None)}, device={getattr(t,'device',None)}")
            print(f"[lava] train_data_values: lam_x={self.lam_x}, lam_y={self.lam_y}, p={self.p}, "
                  f"entreg={self.entreg}, loss={self.loss}, feature_cost={'custom' if self.feature_cost else ('emb' if feature_cost else 'euclidean')}")
            _stat(x_train, 'x_train (emb)')
            _stat(self.y_train, 'y_train')
            _stat(x_valid, 'x_valid (emb)')
            _stat(self.y_valid, 'y_valid')

        # Prefer explicitly provided feature_cost; otherwise use embedding-based one; else euclidean
        _feat_cost = self.feature_cost if self.feature_cost else (feature_cost if feature_cost else "euclidean")

        # Build and solve the OT distance, timing and verbose debug for troubleshooting
        if self.debug:
            print(f"[lava] building DatasetDistance with feature_cost={_feat_cost}, device={self.device}, blur={self.blur}, backend={self.gl_backend}")
        t0 = time.perf_counter()

        # Convert one-hot labels to class indices — otdd expects 1D integer labels
        import torch as _torch
        def _to_labels(y):
            if isinstance(y, _torch.Tensor) and y.dim() == 2 and y.shape[1] > 1:
                return y.argmax(dim=1)
            return y
        y_train_labels = _to_labels(self.y_train)
        y_valid_labels = _to_labels(self.y_valid)

        try:
            print(self.y_train[0])
            dist = DatasetDistance(
                x_train=x_train,
                y_train=y_train_labels,
                x_valid=x_valid,
                y_valid=y_valid_labels,
                feature_cost=_feat_cost,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                p=self.p,
                entreg=self.entreg,
                device=self.device,
                inner_ot_loss=self.loss,
                debug=self.debug,
                blur=self.blur,
                scaling=self.gl_scaling,
                gl_backend=self.gl_backend,
                gl_truncate=self.gl_truncate,
                gl_diameter=self.gl_diameter,
                mode=self.mode,
                outer_debias=self.outer_debias,
            )
        except Exception as ex:
            t1 = time.perf_counter()
            print(f"[lava][error] DatasetDistance construction failed after {t1 - t0:.4f}s: {ex}")
            traceback.print_exc()
            raise
        t1 = time.perf_counter()
        if self.debug:
            print(f"[lava] DatasetDistance constructed in {t1 - t0:.4f}s")

        # Store for downstream usage (e.g., saving transport plan)
        self._dist = dist

        # Call dual_sol() and time the OT solve
        t2 = time.perf_counter()
        try:
            if self.debug:
                print("[lava] starting otdd.dual_sol() (this may take time depending on dataset size and geomloss backend)")
            dual = dist.dual_sol()
            self.dual_sol = dual
        except Exception as ex:
            t3 = time.perf_counter()
            print(f"[lava][error] dual_sol() failed after {t3 - t2:.4f}s: {ex}")
            traceback.print_exc()
            raise
        t3 = time.perf_counter()
        if self.debug:
            try:
                f, g = self.dual_sol
                print(f"[lava] dual potentials: f.shape={tuple(f.shape)}, g.shape={tuple(g.shape)}")
            except Exception:
                print("[lava] dual_sol obtained")
            print(f"[lava] dual_sol() completed in {t3 - t2:.4f}s (construct {t1 - t0:.4f}s, total {t3 - t0:.4f}s)")
        return self

    def save_transport_plan(
        self,
        save_path: str,
        epsilon: Optional[float] = None,
        max_iters: int = 500,
        tol: float = 1e-9,
    ) -> torch.Tensor:
        """Compute and save the OT transport plan using the same augmented cost.

        Parameters
        ----------
        save_path : str
            Path where to save the coupling. Uses .pt (torch.save) or .npy (numpy) based on suffix.
        epsilon : float, optional
            Entropic regularization; if None, uses the default tied to the OTDD blur/p.
        max_iters : int, default 500
            Maximum Sinkhorn iterations.
        tol : float, default 1e-9
            Convergence tolerance for scaling vectors.

        Returns
        -------
        torch.Tensor
            The (N, M) transport plan on CPU.
        """
        # Ensure we have a distance object prepared
        if self._dist is None:
            # Lazily prepare embeddings and distance if train_data_values wasn't called
            x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
            # Do not augment features for regression here; DatasetDistance supports
            # regression mode via `mode='reg'` and combines lam_x/lam_y accordingly.
            feat_cost = self.feature_cost if self.feature_cost else "euclidean"
            self._dist = DatasetDistance(
                x_train=x_train,
                y_train=self.y_train,
                x_valid=x_valid,
                y_valid=self.y_valid,
                feature_cost=feat_cost,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                p=self.p,
                entreg=self.entreg,
                device=self.device,
                inner_ot_loss=self.loss,
                blur=self.blur,
                scaling=self.gl_scaling,
                gl_backend=self.gl_backend,
                gl_truncate=self.gl_truncate,
                gl_diameter=self.gl_diameter,
                mode=self.mode,
                outer_debias=self.outer_debias,
            )
        if self.debug:
            print(f"[lava] save_transport_plan: path={save_path}, epsilon={epsilon}, max_iters={max_iters}, tol={tol}")
        return self._dist.transport_plan(
            save_path=save_path, epsilon=epsilon, max_iters=max_iters, tol=tol
        )

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Gets the calibrated gradient of the dual solution, which can be interpreted as
        the data values.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        f1k = self.dual_sol[0].squeeze()
        num_points = len(f1k) - 1
        train_gradient = f1k * (1 + 1 / (num_points)) - f1k.sum() / num_points

        # We multiply -1 to align LAVA with other data valuation algorithms
        # Low values should indicate detrimental data points
        train_gradient =  -1 * train_gradient
        return train_gradient.numpy(force=True)
