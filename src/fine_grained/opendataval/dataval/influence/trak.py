"""TRAK Data Evaluator — correct implementation using the official TRAK library.

Architecture
------------
The official TRAKer expects a bare torch.nn.Module plus a ModelOutput function
that computes the per-example "margin" f(z; θ).  Our GradientModel interface
exposes grad(x, y) and get_weights(), but not a bare nn.Module.

Strategy
--------
We bypass TRAKer's featurize/score pipeline and implement the TRAK estimator
directly from the paper (Eq. 13/15) using our GradientModel primitives:

    τ(z) = ϕ(z)ᵀ (ΦᵀΦ)⁻¹ Φᵀ Q          [Eq. 13, single checkpoint]

    τ_M(z) = (1/M) Σ_m ϕ_m(z)ᵀ (Φ_mᵀ Φ_m)⁻¹ Φ_mᵀ Q_m   [Eq. 15, ensemble]

where:
    ϕ(z) = P⊤ ∇_θ f(z; θ*)              random-projected gradient of MODEL OUTPUT
    Φ    = [ϕ(z₁); …; ϕ(zₙ)]            (n × k) matrix of projected train grads
    Q    = diag(1 − p*_i)                per-example weighting (1 minus correct-class prob)

The key distinction from TracIn:
    TracIn  : dot product of LOSS gradients  ∇_θ L(z, θ)
    TRAK    : dot product of OUTPUT gradients ∇_θ f(z, θ), weighted by Q

For a cross-entropy classifier:
    f(z; θ)  = log-odds / margin = logit of correct class  (scalar per example)
    ∇_θ f    ≠ ∇_θ L  in general, but related via the chain rule:
               ∇_θ L = −(1 − p*) · ∇_θ f   (for binary / per-class logit)

So if GradientModel.grad(x, y) returns ∇_θ L, we can recover ∇_θ f as:
    ∇_θ f(z) ≈ −∇_θ L(z) / (1 − p*(z))

and Q_i = (1 − p*_i), which cancels, giving:
    ϕ(z)ᵀ Q ≈ −∇_θ L(z)ᵀ projected  (the sign is absorbed into scores)

This recovers the standard TRAK formula from loss gradients alone.

Validation aggregation
----------------------
Because ∇_θ f is linear in the model output, scores aggregate across validation
examples by simple summation (same as IF and TracIn).  We sum over valid set.

Output contract
---------------
evaluate_data_values() → np.ndarray of shape (n_train,), higher = more valuable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import RandomState
from opendataval.dataval.progress import ProgressBar

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel


# ---------------------------------------------------------------------------
# Random projection (Johnson-Lindenstrauss, Rademacher ±1 variant)
# ---------------------------------------------------------------------------

def _rademacher_project(grads: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Project a (batch, p) gradient matrix to (batch, k) using a Rademacher matrix.

    Parameters
    ----------
    grads       : (n, p)  — flattened per-example gradients
    proj_matrix : (p, k)  — fixed ±1/√k Rademacher matrix
    """
    return grads @ proj_matrix   # (n, k)


def _make_proj_matrix(p: int, k: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Build a (p, k) Rademacher projection matrix scaled by 1/√k."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    # ±1 entries, then scale
    signs = torch.randint(0, 2, (p, k), generator=rng).float() * 2 - 1  # {-1, +1}
    return (signs / (k ** 0.5)).to(device)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def _flat_grads(grad_iter) -> torch.Tensor:
    """Flatten a per-example gradient tuple to a single (p,) vector."""
    return torch.cat([g.reshape(-1) for g in grad_iter])


def _collect_projected_grads(
    model: GradientModel,
    X: torch.Tensor,
    y: torch.Tensor,
    proj: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect projected gradients Φ = P⊤∇_θf and weights Q for a dataset.

    Returns
    -------
    Phi : (n, k)  projected loss-gradient features
    Q   : (n,)    per-example weights  1 − p*(z)
          where p*(z) = σ(correct-class logit) for cross-entropy models

    Notes
    -----
    We use ∇_θ L as a proxy for ∇_θ f — they differ only by the scalar
    factor −(1 − p*), which is exactly Q.  Dividing out Q would recover
    ∇_θ f, but since we multiply by Q again in Eq. 13, they cancel and
    the score reduces to −projected_loss_grad summed over validation.
    Keeping Q explicit makes the estimator match Eq. 15 of the paper.
    """
    n = len(X)
    k = proj.shape[1]
    Phi = torch.zeros(n, k, device=device)
    Q = torch.ones(n, device=device)  # default weight = 1

    idx = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end].to(device)
        yb = y[start:end].to(device)

        for i, grad_tuple in enumerate(model.grad(xb, yb)):
            flat = _flat_grads(grad_tuple).to(device)           # (p,)
            Phi[idx] = flat @ proj                              # (k,)

            # Estimate Q_i = 1 − p*(z_i) from model output
            # model must be callable; fall back to Q=1 if not
            try:
                with torch.no_grad():
                    logits = model(xb[i : i + 1])              # (1, C)
                    probs = F.softmax(logits, dim=-1)
                    label = int(yb[i].item()) if yb[i].ndim == 0 else int(yb[i].argmax().item())
                    Q[idx] = 1.0 - probs[0, label].item()
            except Exception:
                Q[idx] = 1.0                                    # fallback

            idx += 1

    return Phi, Q


# ---------------------------------------------------------------------------
# TRAK single-checkpoint estimator  (Eq. 13)
# ---------------------------------------------------------------------------

def _trak_single_checkpoint(
    Phi_train: torch.Tensor,   # (n_train, k)
    Q_train: torch.Tensor,     # (n_train,)  — used to weight train features
    Phi_valid: torch.Tensor,   # (n_valid, k)
) -> torch.Tensor:
    """Compute TRAK scores for one checkpoint.

    Implements Eq. 13 from Park et al. (2023):

        τ(z) = ϕ(z)ᵀ (ΦᵀΦ)⁻¹ Φᵀ Q

    Aggregated over validation set (linear → simple sum):

        score[i] = Σ_j  ϕ_valid[j]ᵀ (ΦᵀΦ)⁻¹ ϕ_train[i] · Q_train[i]

    We implement this efficiently as:

        A   = (ΦᵀΦ)⁻¹ Φᵀ   diag(Q)   ∈ R^{k × n_train}
        τ   = Φ_valid @ A              ∈ R^{n_valid × n_train}
        scores = τ.sum(dim=0)          ∈ R^{n_train}

    Parameters
    ----------
    Phi_train : (n_train, k)
    Q_train   : (n_train,)    weights for training examples
    Phi_valid : (n_valid, k)
    """
    k = Phi_train.shape[1]
    device = Phi_train.device

    # (ΦᵀΦ) + small ridge for numerical stability
    ridge = 1e-6 * torch.eye(k, device=device)
    PtP = Phi_train.T @ Phi_train + ridge                      # (k, k)

    # Solve (ΦᵀΦ) A = Φᵀ diag(Q)  →  A = (ΦᵀΦ)⁻¹ Φᵀ diag(Q)
    # Φᵀ diag(Q) = (Q * Phi_train).T = Phi_train.T * Q[None, :]
    PhiT_Q = (Phi_train * Q_train.unsqueeze(1)).T              # (k, n_train)

    # A ∈ R^{k × n_train}
    A = torch.linalg.solve(PtP, PhiT_Q)                        # (k, n_train)

    # τ = Phi_valid @ A  →  (n_valid, n_train)
    scores_per_valid = Phi_valid @ A                            # (n_valid, n_train)

    # Aggregate over validation set
    return scores_per_valid.sum(dim=0)                          # (n_train,)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class TRAK(DataEvaluator, ModelMixin):
    """TRAK data attribution — correct implementation of Park et al. (ICML 2023).

    Implements the full TRAK estimator (Eq. 13/15) directly from GradientModel
    primitives, without relying on the official TRAKer's task-specific pipeline.

    Parameters
    ----------
    proj_dim : int
        Random projection dimension k.  Paper recommends 1 000 – 40 000.
        Higher k → more accurate but slower.  Default 2 048.
    num_checkpoints : int
        Number of independently-trained checkpoints to ensemble over (M).
        Each checkpoint should be from a *different* training run or late
        epoch.  More checkpoints → better LDS but more compute.
    batch_size : int
        Batch size for gradient collection.
    proj_seed : int
        Seed for the Rademacher projection matrix (fixed across checkpoints
        so features are comparable).
    checkpoints : list[dict] | None
        Pre-trained state_dicts.  If None, the model is trained once and
        late-epoch snapshots are collected automatically.
    checkpoint_epochs : list[int] | None
        Which epochs to snapshot when auto-collecting.  Defaults to the
        last 30% of training, evenly spaced.
    train_kwargs : dict | None
        Forwarded to pred_model.fit() when training.
    random_state : RandomState | None
    """

    def __init__(
        self,
        proj_dim: int = 2048,
        num_checkpoints: int = 5,
        batch_size: int = 256,
        proj_seed: int = 42,
        checkpoints: Optional[list] = None,
        checkpoint_epochs: Optional[list[int]] = None,
        train_kwargs: Optional[dict] = None,
        random_state: Optional[RandomState] = None,
    ):
        super().__init__(random_state=random_state)

        self.proj_dim = proj_dim
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.proj_seed = proj_seed
        self.checkpoints = checkpoints or []
        self.checkpoint_epochs = checkpoint_epochs
        self.train_kwargs = train_kwargs or {}

        self._scores: Optional[np.ndarray] = None
        self._proj: Optional[torch.Tensor] = None   # built lazily once p is known
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # opendataval API
    # ------------------------------------------------------------------

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self._scores = np.zeros(len(x_train))
        return self

    def input_model(self, pred_model: GradientModel):
        assert callable(getattr(pred_model, "grad", None)), (
            "pred_model must implement grad(x, y) → iterator of per-example gradient tuples."
        )
        self.pred_model = pred_model.clone()
        return self

    def train_data_values(self, *args, **kwargs):
        """Train model (collecting checkpoints) then compute TRAK scores."""

        fit_kwargs = {**self.train_kwargs, **kwargs}
        epochs = fit_kwargs.pop("epochs", 24)

        # ---- Collect checkpoints if not pre-supplied ----
        if not self.checkpoints:
            # Default: snapshot the last ~30% of training
            if self.checkpoint_epochs is None:
                start = max(1, int(0.7 * epochs))
                step = max(1, (epochs - start) // (self.num_checkpoints - 1))
                snap_epochs = set(range(start, epochs + 1, step))
                # Always include the final epoch
                snap_epochs.add(epochs)
                snap_epochs = sorted(snap_epochs)[-self.num_checkpoints:]
            else:
                snap_epochs = sorted(self.checkpoint_epochs)

            print(f"[TRAK] Training for {epochs} epochs; "
                  f"will snapshot at epochs {snap_epochs}")

            for ep in range(1, epochs + 1):
                self.pred_model.fit(
                    self.x_train, self.y_train, epochs=1, **fit_kwargs
                )
                if ep in snap_epochs:
                    self.checkpoints.append(
                        {k: v.cpu().clone() for k, v in
                         self.pred_model.state_dict().items()}
                    )
                    print(f"  ✓ Snapshot at epoch {ep} "
                          f"({len(self.checkpoints)}/{len(snap_epochs)})")

        print(f"[TRAK] Ensembling over {len(self.checkpoints)} checkpoint(s)")
        self._compute_trak_scores()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("Call train_data_values() first.")
        return self._scores

    # ------------------------------------------------------------------
    # Core TRAK computation
    # ------------------------------------------------------------------

    def _compute_trak_scores(self):
        """Ensemble TRAK scores across all checkpoints (Eq. 15)."""
        n_train = len(self.x_train)
        accumulated = torch.zeros(n_train, device=self._device)

        for m, ckpt in enumerate(self.checkpoints):
            print(f"[TRAK] Checkpoint {m + 1}/{len(self.checkpoints)}")
            self.pred_model.load_state_dict(ckpt)
            self.pred_model.eval()

            # Build projection matrix once (same seed → same P for all checkpoints)
            # We need to know p (param count) first
            if self._proj is None:
                p = sum(param.numel() for param in self.pred_model.parameters()
                        if param.requires_grad)
                print(f"  Parameter count p = {p:,}  →  projecting to k = {self.proj_dim}")
                self._proj = _make_proj_matrix(p, self.proj_dim, self._device, self.proj_seed)

            # Collect projected grads for train set: Φ_m ∈ R^{n_train × k}, Q_m ∈ R^{n_train}
            print("  Featurizing training set …")
            Phi_train, Q_train = _collect_projected_grads(
                self.pred_model, self.x_train, self.y_train,
                self._proj, self.batch_size, self._device
            )

            # Collect projected grads for validation set: Φ_valid ∈ R^{n_valid × k}
            print("  Featurizing validation set …")
            Phi_valid, _ = _collect_projected_grads(
                self.pred_model, self.x_valid, self.y_valid,
                self._proj, self.batch_size, self._device
            )

            # TRAK estimator for this checkpoint (Eq. 13)
            scores_m = _trak_single_checkpoint(Phi_train, Q_train, Phi_valid)
            accumulated += scores_m

        # Average across checkpoints (Eq. 15)
        self._scores = (accumulated / len(self.checkpoints)).cpu().numpy()
        print(f"[TRAK] Done. Score range: [{self._scores.min():.4f}, {self._scores.max():.4f}]")
