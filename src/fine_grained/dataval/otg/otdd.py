"""Main module for computing exact wasserstein distance between two data sets.

`OTDD Repository <https://github.com/microsoft/otdd>`_.

References
----------
    .. [1] D. Alvarez-Melis and N. Fusi,
        Geometric Dataset Distances via Optimal Transport,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2002.02923.
    .. [2] D. Alvarez-Melis and N. Fusi,
        Dataset Dynamics via Gradient Flows in Probability Space,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2010.12760.
    .. [3] `OTDD repo <https://github.com/microsoft/otdd>`_.
        The following implementation was taken from this repository. It is intended as a
        strict subset of the options provided in the repository, only computing the
        class-wise Wasserstein as needed by the LAVA Paper by H.A. Just et al.

Legacy notation:
    X1, X2: feature tensors of the two datasets
    Y1, Y2: label tensors of the two datasets
    N1, N2 (or N,M): number of samples in datasets
    D1, D2: (feature) dimension of the datasets
    C1, C2: number of classes in the datasets
"""
## Local Imports
import itertools
from functools import partial
from typing import Callable, Literal, Optional, Union

import geomloss
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from opendataval.dataloader import CatDataset

import torch
from importlib.util import find_spec

def _keops_available() -> bool:
    """Return True if PyKeOps is installed and importable.

    GeomLoss' "online" backend relies on KeOps (generic_logsumexp). If it's not
    available, we must avoid selecting the online backend to prevent runtime errors.
    """
    try:
        return find_spec("pykeops") is not None
    except Exception:
        return False

def generic_logsumexp(x, keepdim=False):
    """Numerically stable logsumexp over last dimension."""
    max_val, _ = torch.max(x, dim=-1, keepdim=True)
    return max_val + torch.log(torch.sum(torch.exp(x - max_val), dim=-1, keepdim=keepdim))

cost_routines = {
    1: geomloss.utils.distances,
    2: lambda x, y: geomloss.utils.squared_distances(x, y) / 2,
}

class DatasetDistance:
    """The main class for the Optimal Transport Dataset Distance.

    An object of this class is instantiated with two datasets (the source and target),
    which are stored in it, and various arguments determining how the exact Wasserstein
    distance is to be computed.

    Parameters
    ----------
    x_train : torch.Tensor
        Covariates of the first distribution
    y_train : torch.Tensor
        Labels of the first distribution
    x_valid : torch.Tensor
        Covariates of the second/validation distribution
    y_valid : torch.Tensor
        Labels of the second/validation distribution
    feature_cost : Literal["euclidean"] | Callable, optional
        If not 'euclidean', must be a callable that implements a cost function
        between feature vectors, by default "euclidean"
    p : int, optional
        The coefficient in the OT cost (i.e., the p in p-Wasserstein), by default 2
    entreg : float, optional
        The strength of entropy regularization for sinkhorn, by default 0.1
    lam_x : float, optional
        Weight parameter for feature component of distance, by default 1.0
    lam_y : float, optional
        Weight parameter for label component of distance.=, by default 1.0
    inner_ot_loss : str, optional
        Loss type to exact OT problem, by default "sinkhorn"
    inner_ot_debiased : bool, optional
        Whether to use the debiased version of sinkhorn in the inner OT problem,
        by default False
    inner_ot_p : int, optional
        The coefficient in the inner OT cost., by default 2
    inner_ot_entreg : float, optional
        The strength of entropy regularization for sinkhorn in the inner OT problem,
        by default 0.1
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")
    blur : float, optional
        Entropic scale passed to GeomLoss (epsilon ≈ blur**p). Larger is faster/smoother.
    debug : bool, optional
        Enable verbose prints.
    scaling : float, optional
        GeomLoss epsilon-scaling factor between successive scales (0<scaling<1).
    mode : Literal["cls", "reg"], optional
        Whether to use classification mode (discrete labels) or regression mode
        (continuous targets), by default "cls"
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Union[Literal["euclidean"], Callable] = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        device: torch.device = torch.device("cpu"),
        blur: float = 0.05,
        debug: bool = True,
        scaling: float = 0.9,
        mode: Literal["cls", "reg"] = "cls",
        gl_backend: Optional[str] = None,
        gl_truncate: Optional[float] = None,
        gl_diameter: Optional[float] = None,
        outer_debias: bool = False,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.feature_cost = feature_cost
        self.p = p
        self.entreg = entreg
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.inner_ot_loss = inner_ot_loss
        self.inner_ot_debiased = inner_ot_debiased
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.device = device
        self.blur = blur
        self.debug = debug
        self.gl_scaling = scaling
        self.mode = mode
        self.gl_backend = gl_backend
        self.gl_truncate = gl_truncate
        self.gl_diameter = gl_diameter
        self.outer_debias = outer_debias
        
        # Internal state
        self.label_distances = None
        self.transport_matrix = None
        self._skip_labels = (float(lam_y) == 0.0)

    def _get_label_distances(self) -> Optional[torch.Tensor]:
        """Precompute label-to-label distances.

        Returns tensor of size nclasses_1 x nclasses_2.
        DISTANCE BETWEEN LABEL IN D1 AND LABEL IN D2!

        Returns
        -------
        torch.Tensor
            tensor of size (C1, C2) with pairwise label-to-label distances across
            the two datasets.
        """
        ## If label term disabled, skip computation
        if self._skip_labels:
            if self.debug:
                print("[otdd] lam_y=0: skipping computation of label_distances W")
            return None

        ## In regression mode, labels are continuous: skip precomputed W and use on-the-fly |y1 - y2|^2
        if getattr(self, "mode", "cls") == "reg":
            if self.debug:
                print("[otdd] mode='reg': skipping precomputed W; label term handled in batch_augmented_cost")
            return None

        ## Check if already computed
        if self.label_distances is not None:
            return self.label_distances

        # exact way of computing
        # We just define a function ahead, before loading real data
        # pwdist_exact From Geomloss defined function
        pwdist = partial(
            pwdist_exact,
            symmetric=False,
            p=self.inner_ot_p,
            loss=self.inner_ot_loss,
            debias=self.inner_ot_debiased,
            entreg=self.inner_ot_entreg,
            cost_function=self.feature_cost,
            device=self.device,
            blur=self.blur,
            scaling=self.gl_scaling,
        )

        ## Then we also need within-collection label distances
        DYY1 = pwdist(self.x_train, self.y_train)
        DYY2 = pwdist(self.x_valid, self.y_valid)
        DYY12 = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)

        D = torch.cat([torch.cat([DYY1, DYY12], 1), torch.cat([DYY12.t(), DYY2], 1)])

        ## Collect and save
        self.label_distances = D

        if self.debug:
            Dc = D.detach().cpu()
            print(f"[otdd] label_distances: shape={tuple(Dc.shape)}, min={Dc.min().item():.4g}, max={Dc.max().item():.4g}")

        return self.label_distances

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        """Compute dataset distance.

        Note: Currently requires fully loading dataset into memory, this can probably
        be avoided, e.g., via subsampling.

        Returns
        -------
        tuple[float, torch.Tensor]
            dist (float): the optimal transport dataset distance value.
            pi (tensor, optional): the optimal transport coupling.
        """
        W = self._get_label_distances()
        if W is not None:
            wasserstein = W.to(self.device)
        else:
            wasserstein = None

        if self.debug and wasserstein is not None:
            print(f"[otdd] dual_sol: W.shape={tuple(wasserstein.shape)}, device={wasserstein.device}")

        ## This one leverages precomputed pairwise label distances
        cost_geomloss = partial(
            batch_augmented_cost,
            W=wasserstein,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
            debug=self.debug,
            mode=self.mode,
        )

        N = len(self.x_train)

        # Decide backend, allowing explicit override
        if self.gl_backend in ("tensorized", "multiscale", "online"):
            backend = self.gl_backend
        else:
            #backend = "multiscale" if N > 10000 else "tensorized"
            backend = "tensorized"


        # In regression mode with a label term, ensure tensorized to apply custom joint cost
        if (self.mode == "reg") and (float(self.lam_y) != 0.0):
            if backend != "tensorized" and self.debug:
                print(f"[otdd][info] forcing backend='tensorized' for regression mode to include label term.")
            backend = "tensorized"

        # Fallbacks if KeOps is not available: both 'online' and 'multiscale' require KeOps
        if backend in ("online", "multiscale"):
            if not _keops_available():
                fb = "tensorized"
                if self.debug:
                    print(f"[otdd][warn] backend='{backend}' requested but PyKeOps is not available; falling back to '{fb}'.")
                backend = fb
            else:
                # Some online/backends expect helper functions (e.g., generic_logsumexp) to
                # be available in the calling namespace. If for some reason that's not
                # available, fall back to tensorized to avoid runtime NameError.
                if "generic_logsumexp" not in globals():
                    fb = "tensorized"
                    if self.debug:
                        print(f"[otdd][warn] generic_logsumexp not found in globals; falling back to '{fb}'.")
                    backend = fb

        # Build cost function according to backend
        if backend == "tensorized":
            def cost(x, y):
                # x: (B,N,D_aug) or (N,D_aug) where last dim is label index
                if x.dim() == 2:
                    X = x.unsqueeze(0)
                    Y = y.unsqueeze(0)
                else:
                    X = x
                    Y = y
                # Return (B,N,M)
                return cost_geomloss(X, Y)
        else:
            # Multiscale/online backends require a symbolic cost
            cost = "(SqDist(X,Y) / IntCst(2))"

        if self.debug:
            print(f"[otdd] dual_sol: backend={backend}, blur={self.blur}, inner_loss={self.inner_ot_loss}, inner_p={self.inner_ot_p}, scaling={self.gl_scaling}, truncate={self.gl_truncate}, diameter={self.gl_diameter}, outer_debias={self.outer_debias}")

        loss = geomloss.SamplesLoss(
            loss=self.inner_ot_loss,
            p=self.inner_ot_p,
            cost=cost,
            debias=self.outer_debias,
            blur=self.blur,
            backend=backend,
            scaling=self.gl_scaling,
            **({"truncate": self.gl_truncate} if self.gl_truncate is not None else {}),
            **({"diameter": self.gl_diameter} if self.gl_diameter is not None else {}),
        )

        # Ensure feature tensors are 2D (N, D) before concatenating label column
        X1 = self.x_train
        X2 = self.x_valid
        Y1 = self.y_train
        Y2 = self.y_valid

        if X1.dim() > 2:
            # flatten non-sample dims
            X1 = X1.view(len(X1), -1)
            X2 = X2.view(len(X2), -1)
            if self.debug:
                print(f"[otdd] flattened x_train/x_valid to shapes {X1.shape}/{X2.shape}")

        # Normalize label shapes into 2D arrays for concatenation
        def _label_to_2d(y: torch.Tensor) -> torch.Tensor:
            y = y.float()
            if y.dim() == 1:
                return y.unsqueeze(1)
            # collapse all dimensions after the first into a single feature axis
            if y.dim() >= 2:
                return y.reshape(len(y), -1)
            return y.unsqueeze(1)

        y1_col = _label_to_2d(Y1)
        y2_col = _label_to_2d(Y2)

        if self.debug:
            print(f"[otdd] shapes before concat: X1={X1.shape}, y1={y1_col.shape}, X2={X2.shape}, y2={y2_col.shape}")

        # Concatenate along last dimension (features + label columns)
        Z1 = torch.cat((X1, y1_col), dim=-1)
        Z2 = torch.cat((X2, y2_col), dim=-1)

        with torch.no_grad():
            loss.debias = False
            loss.potentials = True
            torch.cuda.empty_cache()
            F_i, G_j = loss(Z1.to(self.device), Z2.to(self.device))
            pi = [F_i, G_j]

            if self.debug:
                print(f"[otdd] potentials: F_i.shape={tuple(F_i.shape)}, G_j.shape={tuple(G_j.shape)}")

        del Z1, Z2
        torch.cuda.empty_cache()

        return pi

    def _augmented_cost_matrix(self) -> torch.Tensor:
        """Compute the full augmented ground cost matrix C of shape (N, M).

        This uses the same components as dual_sol: precomputed label distances W,
        feature_cost, and the weights lam_x/lam_y.

        Always computed in tensorized manner to obtain an explicit cost matrix.
        """
        W = self._get_label_distances()
        if W is not None:
            W = W.to(self.device)

        # Build augmented features (append label index as last dim)
        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)

        cost_geomloss = partial(
            batch_augmented_cost,
            W=W,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
            debug=self.debug,
            mode=self.mode,
        )

        # Compute (1, N, M) then squeeze
        C = cost_geomloss(Z1.unsqueeze(0).to(self.device), Z2.unsqueeze(0).to(self.device))
        C = C.squeeze(0).detach()

        if self.debug:
            Cc = C.detach().cpu()
            print(f"[otdd] augmented cost C: shape={tuple(Cc.shape)}, min={Cc.min().item():.4g}, max={Cc.max().item():.4g}, mean={Cc.mean().item():.4g}")

        return C

    def transport_plan(
        self,
        save_path: Optional[Union[str, Path]] = None,
        epsilon: Optional[float] = None,
        max_iters: int = 500,
        tol: float = 1e-9,
    ) -> torch.Tensor:
        """Compute and optionally save the entropic OT transport plan P (N x M).

        Uses a simple stabilized Sinkhorn scaling on the explicit augmented cost matrix.
        We assume uniform marginals a=1/N and b=1/M.

        Parameters
        ----------
        save_path : str | Path, optional
            If provided, saves the plan to this path. '.pt' saves with torch.save,
            '.npy' saves with numpy. Directories will be created if needed.
        epsilon : float, optional
            Entropic regularization strength. If None, uses (blur ** p), matching
            the SamplesLoss configuration.
        max_iters : int, default 500
            Maximum number of Sinkhorn iterations.
        tol : float, default 1e-9
            Convergence tolerance on the scaling vectors.

        Returns
        -------
        torch.Tensor
            The transport matrix P of shape (N, M) on CPU.
        """
        C = self._augmented_cost_matrix()  # (N, M) on device
        N, M = C.shape

        # Select epsilon consistent with geomloss 'blur' (heuristic)
        if epsilon is None:
            eps = float(self.blur) ** int(self.inner_ot_p)
        else:
            eps = float(epsilon)

        if self.debug:
            Ccpu = C.detach().cpu()
            span = (Ccpu.max() - Ccpu.min()).item()
            print(f"[otdd] transport_plan: N={N}, M={M}, eps={eps:.4g}, Cmin={Ccpu.min().item():.4g}, Cmax={Ccpu.max().item():.4g}, span={span:.4g}")
            if span / max(eps, 1e-12) > 80:
                print("[otdd][warn] Potential underflow: span/eps is large; consider increasing epsilon or scaling features/lam_y.")

        # Uniform marginals
        a = torch.full((N,), 1.0 / N, device=C.device)
        b = torch.full((M,), 1.0 / M, device=C.device)

        # Gibbs kernel
        K = torch.exp(-C / max(eps, 1e-12))  # (N, M)

        if self.debug:
            Kc = K.detach().cpu()
            print(f"[otdd] K stats: min={Kc.min().item():.4g}, max={Kc.max().item():.4g}")

        # Stabilized Sinkhorn scaling
        u = torch.ones(N, device=C.device)
        v = torch.ones(M, device=C.device)

        for it in range(int(max_iters)):
            u_prev = u
            Kv = K @ v + 1e-38
            u = a / Kv
            Ktu = K.t() @ u + 1e-38
            v = b / Ktu

            if torch.mean(torch.abs(u - u_prev)).item() < tol:
                break

            if self.debug and it in (0, 1, 2, 4, 9, 19, 49, 99):
                r1 = torch.sum(u[:, None] * K * v[None, :], dim=1)
                r2 = torch.sum(u[:, None] * K * v[None, :], dim=0)
                print(f"[otdd] it={it}: row_sum_mean={r1.mean().item():.4g}, col_sum_mean={r2.mean().item():.4g}")

        P = (u[:, None] * K) * v[None, :]  # (N, M)
        P_cpu = P.detach().cpu()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = save_path.suffix.lower()
            if suffix == ".npy":
                np.save(str(save_path), P_cpu.numpy())
            else:  # default to torch
                torch.save(P_cpu, str(save_path))
            if self.debug:
                print(f"[otdd] plan saved to: {save_path}")

        # Store for later access
        self.transport_matrix = P_cpu
        return P_cpu

    def distance_value(self) -> float:
        """Compute the scalar OT distance between current train/valid datasets.

        Uses the same augmented ground cost and backend choice as dual_sol, but
        returns only the scalar SamplesLoss value (no potentials).

        Returns
        -------
        float
            The Sinkhorn (or chosen loss) distance value.
        """
        W = self._get_label_distances()
        if W is not None:
            W = W.to(self.device)

        cost_geomloss = partial(
            batch_augmented_cost,
            W=W,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
            debug=self.debug,
            mode=self.mode,
        )

        N = len(self.x_train)

        # Determine requested backend with override support
        if self.gl_backend in ("tensorized", "multiscale", "online"):
            backend = self.gl_backend
        else:
            #backend = "multiscale" if N > 10000 else "tensorized"
            backend = "tensorized"

        # In regression mode with a label term, ensure tensorized to apply custom joint cost
        if (self.mode == "reg") and (float(self.lam_y) != 0.0):
            if backend != "tensorized" and self.debug:
                print(f"[otdd][info] forcing backend='tensorized' for regression mode to include label term.")
            backend = "tensorized"

        # Fallback if online without KeOps
        if backend == "online" and not _keops_available():
            fb = "multiscale" if N > 10000 else "tensorized"
            if self.debug:
                print(f"[otdd][warn] backend='online' requested but PyKeOps is not available; falling back to '{fb}'.")
            backend = fb

        # Define cost depending on backend
        if backend in ("multiscale", "online"):
            cost = "(SqDist(X,Y) / IntCst(2))"
        else:  # tensorized
            def cost(x, y):
                if x.dim() == 2:
                    X = x.unsqueeze(0)
                    Y = y.unsqueeze(0)
                else:
                    X = x
                    Y = y
                return cost_geomloss(X, Y)

        loss = geomloss.SamplesLoss(
            loss=self.inner_ot_loss,
            p=self.inner_ot_p,
            cost=cost,
            debias=self.outer_debias,
            blur=self.blur,
            backend=backend,
            scaling=self.gl_scaling,
            **({"truncate": self.gl_truncate} if self.gl_truncate is not None else {}),
            **({"diameter": self.gl_diameter} if self.gl_diameter is not None else {}),
        )

        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)

        with torch.no_grad():
            loss.debias = True
            loss.potentials = False
            val = loss(Z1.to(self.device), Z2.to(self.device))
            dist_val = float(val.item() if torch.is_tensor(val) else val)

        if self.debug:
            print(f"[otdd] distance_value: backend={backend}, blur={self.blur}, value={dist_val:.6g}")

        del Z1, Z2
        torch.cuda.empty_cache()

        return dist_val


class FeatureCost:
    """Class implementing a cost (or distance) between feature vectors.

    Arguments:
        p (int): the coefficient in the OT cost (i.e., the p in p-Wasserstein).
        src_embedding (callable, optional): if provided, source data will be
            embedded using this function prior to distance computation.
        tgt_embedding (callable, optional): if provided, target data will be
            embedded using this function prior to distance computation.
    """

    def __init__(
        self,
        src_embedding=None,
        tgt_embedding=None,
        src_dim=None,
        tgt_dim=None,
        p=2,
        device="cpu",
    ):
        assert (src_embedding is None) or (src_dim is not None)
        assert (tgt_embedding is None) or (tgt_dim is not None)

        self.src_emb = src_embedding
        self.tgt_emb = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device

    def _get_batch_shape(self, b):
        if b.ndim == 3:
            return b.shape
        elif b.ndim == 2:
            return (1, *b.shape)
        elif b.ndim == 1:
            return (1, 1, b.shape[0])

    def _batchify_computation(self, X, side="x", slices=20):
        embed = self.src_emb if side == "x" else self.tgt_emb
        out = torch.cat(embed(b).to(self.device) for b in torch.chunk(X, slices, dim=0))
        return out.to(X.device)

    def __call__(self, X1, X2):
        if self.src_emb is not None:
            B1, N1, _ = self._get_batch_shape(X1)
            self.src_emb = self.src_emb.to(self.device)
            X1 = self.src_emb(X1.view(-1, *self.src_dim)).reshape(B1, N1, -1)

        if self.tgt_emb is not None:
            B2, N2, _ = self._get_batch_shape(X2)
            self.tgt_emb = self.tgt_emb.to(self.device)
            X2 = self.tgt_emb(X2.view(-1, *self.tgt_dim)).reshape(B2, N2, -1)

        return cost_routines[self.p](X1, X2)


def extract_dataset(
    x_input: torch.Tensor,
    y_input: torch.Tensor,
    batch_size: int = 256,
    reindex_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads full dataset into memory and reindexes the labels.

    Parameters
    ----------
    x_input : Dataset | torch.Tensor
        Covariate Dataset/tensor to be loaded
    y_input : Dataset | torch.Tensor
        Label Dataset/tensor to be loaded
    batch_size : int, optional
        Batch size of data to be loaded at a time, by default 256
    reindex_start : int, optional
        How much to offset the labels by, useful when comparing different data sets
        so that the data have different labels, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        **x_tensor** Covariates stacked along first dimension
        **y_tensor** Labels, no longer in one-hot-encoding and offset by reindex_start
    """
    loader = DataLoader(CatDataset(x_input, y_input), batch_size=batch_size)
    x_list = []
    y_list = []

    for x, y in tqdm.tqdm(loader, leave=False):
        # Flatten covariates to [B, D]
        x_list.append(x.squeeze().view(x.shape[0], -1))

        # Support two label encodings:
        # - One-hot: y is [B, C] with C>1; use argmax
        # - Integer: y is [B] or [B,1]; use as-is
        if y.dim() == 1:
            y_idx = y.long().view(-1)
        elif y.dim() == 2 and y.shape[1] == 1:
            y_idx = y.view(-1).long()
        else:
            y_idx = y.argmax(dim=1).view(-1).long()

        y_list.append(y_idx)

    x_tensor = torch.cat(x_list)
    y_tensor = torch.cat(y_list)

    return x_tensor, y_tensor + reindex_start


def pwdist_exact(
    X1: torch.Tensor,
    Y1: torch.Tensor,
    X2: Optional[torch.Tensor] = None,
    Y2: Optional[torch.Tensor] = None,
    symmetric: bool = False,
    loss: str = "sinkhorn",
    cost_function: Union[
        Literal["euclidean"], Callable[..., torch.Tensor]
    ] = "euclidean",
    p: int = 2,
    debias: bool = True,
    entreg: float = 1e-1,
    device: torch.device = torch.device("cpu"),
    blur: Optional[float] = None,
    scaling: Optional[float] = None,
):
    """Computation of pairwise Wasserstein distances.

    Efficient computation of pairwise label-to-label Wasserstein distances between
    multiple distributions, without using Gaussian assumption.

    Parameters
    ----------
    X1 : torch.Tensor
        Covariates of first distribution
    Y1 : torch.Tensor
        Labels of first distribution
    X2 : torch.Tensor, optional
        Covariates of second distribution, if None distributions are treated as same,
        by default None
    Y2 : torch.Tensor, optional
        Labels of second distribution, iif None distributions are treated as same,
        by default None
    symmetric : bool, optional
        Whether X1/Y1 and X2/Y2 are to be treated as the same dataset, by default False
    loss : str, optional
        The loss function to compute. Sinkhorn divergence, which interpolates between
        (blur=0) and kernel (blur= :math:`+\infty` ) distances., by default "sinkhorn"
    cost_function : : Literal["euclidean"] | Callable[..., torch.Tensor], optional
        Cost function that should be used instead of
        :math:`\\tfrac{1}{p}\|x-y\|^p`, by default "euclidean"
    p : int, optional
        power of the cost (i.e. order of p-Wasserstein distance), by default 2
    debias : bool, optional
        If true, uses debiased sinkhorn divergence., by default True
    entreg : float, optional
        The strength of entropy regularization for sinkhorn., by default 1e-1
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")

    Returns
    -------
    torch.Tensor
        Computed Wasserstein distance
    """
    if X2 is None:
        # If not specified, assume symmetric
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are asymmetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    if cost_function == "euclidean":
        cost_function = cost_routines[p]

    # Force tensorized backend — online/multiscale require PyKeOps (generic_logsumexp)
    # which may not be available or in scope.
    distance = geomloss.SamplesLoss(
        loss=loss,
        p=p,
        cost=cost_function,
        debias=debias,
        blur=blur if blur else entreg ** (1 / p),
        backend="tensorized",
        **({"scaling": float(scaling)} if scaling is not None else {}),
    )

    D = torch.zeros((n1, n2), device=device, dtype=X1.dtype)

    for i, j in tqdm.tqdm(pairs, leave=False, desc="Computing label-to-label distance"):
        m1 = X1[Y1 == c1[i]].to(device)
        m2 = X2[Y2 == c2[j]].to(device)
        D[i, j] = distance(m1, m2).item()
        if symmetric:
            D[j, i] = D[i, j]

    return D


def batch_augmented_cost(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    feature_cost: Optional[str] = None,
    p: int = 2,
    lam_x: float = 1.0,
    lam_y: float = 1.0,
    debug: bool = True,
    mode: str = "cls",
):
    """Batch ground cost computation on augmented datasets.

    Parameters
    ----------
    Z1 : torch.Tensor
        Tensor of size (B,N,D1), where last position in last dim corresponds to label Y.
    Z2 : torch.Tensor
        Tensor of size (B,M,D2), where last position in last dim corresponds to label Y.
    W : torch.Tensor, optional
        Tensor of size (V1,V2) of precomputed pairwise label distances for all labels
        V1,V2 and returns a batched cost matrix as a (B,N,M) Tensor. W is expected to
        be congruent with p. I.e, if p=2, W[i,j] should be squared Wasserstein
        distance., by default None
    feature_cost : str, optional
        if None or 'euclidean', uses euclidean distances as feature metric, otherwise
        uses this function as metric., by default None
    p : int, optional
        Power of the cost (i.e. order of p-Wasserstein distance), by default 2
    lam_x : float, optional
        Weight parameter for feature component of distance, by default 1.0
    lam_y : float, optional
        Weight parameter for label component of distance, by default 1.0
    debug : bool, optional
        Enable debug printing, by default False
    mode : str, optional
        Either "cls" (classification, discrete labels) or "reg" (regression, continuous
        targets), by default "cls"

    Returns
    -------
    torch.Tensor
        torch Tensor of size (B,N,M)

    Raises
    ------
    ValueError
        If W is not provided in classification mode with lam_y > 0
    """
    _, _, D1 = Z1.shape
    _, M, D2 = Z2.shape

    assert (D1 == D2) or (feature_cost is not None)

    # Extract labels from last dimension
    Y1 = Z1[:, :, -1]  # (B, N)
    Y2 = Z2[:, :, -1]  # (B, M)

    # Compute feature cost C1
    if feature_cost is None or feature_cost == "euclidean":
        # default is euclidean
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])  # Get from GeomLoss
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])  # Feature Embedding

    # If label weight is zero, use feature-only cost
    if float(lam_y) == 0.0:
        if debug:
            try:
                C1c = C1.detach().cpu()
                print(
                    f"[otdd] C1 (features): shape={tuple(C1c.shape)}, min={C1c.min().item():.4g}, "
                    f"max={C1c.max().item():.4g}, mean={C1c.mean().item():.4g}"
                )
                print("[otdd] C2 skipped: lam_y=0; using feature-only cost")
            except Exception:
                print("[otdd] Debug: could not compute C1 statistics")
        D = lam_x * C1
        return D

    # Compute label cost C2
    if mode == "reg":
        # Regression mode: labels are continuous, compute |y1 - y2|^p directly,
        # irrespective of W (W is only for discrete labels).
        Y1_expand = Y1[:, :, None]  # (B, N, 1)
        Y2_expand = Y2[:, None, :]  # (B, 1, M)

        # Compute pairwise distances for labels
        if p == 1:
            C2 = torch.abs(Y1_expand - Y2_expand)
        elif p == 2:
            C2 = (Y1_expand - Y2_expand) ** 2
        else:
            C2 = torch.abs(Y1_expand - Y2_expand) ** p

        if debug:
            try:
                C1c = C1.detach().cpu()
                C2c = C2.detach().cpu()
                print(
                    f"[otdd] C1 (features): shape={tuple(C1c.shape)}, min={C1c.min().item():.4g}, "
                    f"max={C1c.max().item():.4g}, mean={C1c.mean().item():.4g}"
                )
                print(
                    f"[otdd] C2 (reg labels): shape={tuple(C2c.shape)}, min={C2c.min().item():.4g}, "
                    f"max={C2c.max().item():.4g}, mean={C2c.mean().item():.4g}"
                )
            except Exception:
                print("[otdd] Debug: could not compute C1/C2 statistics")

        # Combine feature and label costs
        D = lam_x * C1 + lam_y * (C2 / p)
        return D

    else:
        # Classification mode: require precomputed label-to-label distances W
        if W is None:
            raise ValueError("batch_augmented_cost: W must be provided in classification mode when lam_y>0.")

        Y1_long = Y1.long()
        Y2_long = Y2.long()

        # Label-to-label distances have been precomputed and passed
        # Stores flattened index corresponding to label pairs
        M_labels = W.shape[1] * Y1_long[:, :, None] + Y2_long[:, None, :]
        C2 = W.flatten()[M_labels.flatten(start_dim=1)].reshape(-1, Y1.shape[1], Y2.shape[1])

        assert C1.shape == C2.shape

        ## NOTE: geomloss's cost_routines as defined above already divide by p. We do
        ## so here too for consistency. But as a consequence, need to divide C2 by p too.
        if debug:
            try:
                C1c = C1.detach().cpu()
                C2c = C2.detach().cpu()
                print(
                    f"[otdd] C1 (features): shape={tuple(C1c.shape)}, min={C1c.min().item():.4g}, "
                    f"max={C1c.max().item():.4g}, mean={C1c.mean().item():.4g}"
                )
                print(
                    f"[otdd] C2 (cls labels): shape={tuple(C2c.shape)}, min={C2c.min().item():.4g}, "
                    f"max={C2c.max().item():.4g}, mean={C2c.mean().item():.4g}"
                )
            except Exception:
                print("[otdd] Debug: could not compute C1/C2 statistics")

        # Combine feature and label costs
        D = lam_x * C1 + lam_y * (C2 / p)
        return D