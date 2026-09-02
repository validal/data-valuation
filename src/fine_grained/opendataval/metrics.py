import torch
import torch.nn.functional as F
from opendataval.util import FuncEnum

from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import balanced_accuracy_score as sk_balanced_accuracy

def accuracy(a, b):
    # Handle device mismatch
    if hasattr(a, 'device') and hasattr(b, 'device'):
        if a.device != b.device:
            b = b.to(a.device)

    if a.ndim == 1 and b.ndim == 1:
        return (a == b).float().mean().item()
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()



def neg_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.device != b.device:
        b = b.to(a.device)
    return -torch.square(a - b).sum().sqrt().item()

def neg_mse(a: torch.Tensor, b: torch.Tensor):
    if a.device != b.device:
        b = b.to(a.device)
    return -F.mse_loss(a, b).item()

def inv_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Inverse MSE: 1 / (1 + MSE). Higher is better."""
    if a.device != b.device:
        b = b.to(a.device)
    mse = F.mse_loss(a, b).item()
    return 1.0 / (1.0 + mse)

def neg_cross_entropy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Negative cross-entropy for classification outputs.

    Behavior matches training: multiclass uses cross-entropy on logits; binary uses BCE.

    Inputs
    ------
    a : torch.Tensor
        Model outputs. For multiclass, expected as logits of shape [N, C>2].
        For binary, can be probabilities/logits shaped [N, 1] or class-probabilities shaped [N, 2].
    b : torch.Tensor
        Targets. Can be class indices [N] / [N,1] or one-hot [N, C].

    Returns
    -------
    float
        Negative cross-entropy (so higher is better for maximization).
    """
    # Multiclass case (C > 2) → use CrossEntropy on logits with integer targets
    if a.dim() == 2 and a.shape[1] > 2:
        if b.dim() > 1:
            y = b.argmax(dim=1)
        else:
            y = b.view(-1)
        y = y.to(dtype=torch.long, device=a.device)
        loss = F.cross_entropy(a, y, reduction="mean")
        return -loss.item()

    # Binary cases
    p = a
    # Make sure predictions are 2D for shape handling
    if p.dim() == 1:
        p = p.view(-1, 1)

    # If model outputs two columns (e.g., logistic regression with num_classes==2),
    # we use BCE against one-hot targets to mirror training.
    if p.dim() == 2 and p.shape[1] == 2:
        # Targets → one-hot
        if b.dim() == 1 or (b.dim() == 2 and b.shape[1] == 1):
            y_idx = b.view(-1).to(dtype=torch.long, device=p.device)
            y_onehot = torch.zeros((y_idx.shape[0], 2), device=p.device, dtype=p.dtype)
            y_onehot.scatter_(1, y_idx.unsqueeze(1), 1.0)
        else:
            y_onehot = b.to(dtype=p.dtype, device=p.device)
        # Ensure probabilities
        if (p.min() < 0) or (p.max() > 1):
            p = torch.sigmoid(p)
        loss = F.binary_cross_entropy(p, y_onehot, reduction="mean")
        return -loss.item()

    # Single-probability column [N,1] for positive class; targets 0/1
    # Convert targets to float column
    if b.dim() == 2 and b.shape[1] > 1:
        b_idx = b.argmax(dim=1)
    else:
        b_idx = b.view(-1)
    y_float = b_idx.to(device=p.device, dtype=p.dtype).float().view(-1, 1)
    if (p.min() < 0) or (p.max() > 1):
        p = torch.sigmoid(p)
    loss = F.binary_cross_entropy(p, y_float, reduction="mean")
    return -loss.item()

# Root Mean Squared Error (RMSE)
def neg_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Negative Root Mean Squared Error between a and b."""
    if a.device != b.device:
        b = b.to(a.device)
    return -torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()

# (Optional) Also add a positive RMSE for convenience
def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Root Mean Squared Error between a and b."""
    if a.device != b.device:
        b = b.to(a.device)
    return torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()
    
def recall(a: torch.Tensor, b: torch.Tensor) -> float:
    pred = a.argmax(dim=1).cpu().numpy()
    true = b.argmax(dim=1).cpu().numpy()

    tp = ((pred == 1) & (true == 1)).sum()
    fn = ((pred == 0) & (true == 1)).sum()

    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)

# Root Mean Squared Percentage Error (RMSPE)
def rmspe(a: torch.Tensor, b: torch.Tensor) -> float:
    """Root Mean Squared Percentage Error between a and b."""
    """Root Mean Squared Percentage Error between a and b.
    Args:
        a: true values (torch.Tensor)
        b: predicted values (torch.Tensor)
    Returns:
        float: RMSPE value (e.g., 0.15 = 15% error)
    """
    if a.device != b.device:
        b = b.to(a.device)
    a = a.float()
    b = b.float()
    epsilon = 1e-10
    # Avoid division by zero in denominator
    denom = torch.where(a == 0, torch.full_like(a, epsilon), a)
    pct_error = (a - b) / denom
    rmspe_value = torch.sqrt(torch.mean(pct_error ** 2)).item()
    return rmspe_value

def neg_rmspe(a: torch.Tensor, b: torch.Tensor) -> float:
    """Negative Root Mean Squared Percentage Error between a and b.
    
    Returns negative RMSPE so higher values are better (for maximization).
    RMSPE is a percentage error (e.g., 0.15 = 15% error), so we negate it.
    """
    """Negative Root Mean Squared Percentage Error between a and b.
    Returns negative RMSPE so higher values are better (for maximization).
    """
    return -rmspe(a, b)

# Coefficient of determination (R^2) for regression
def r_squared(a: torch.Tensor, b: torch.Tensor) -> float:
    """Coefficient of determination R^2 for regression.

    R^2 = 1 - SS_res / SS_tot where SS_res = sum((y_true - y_pred)^2)
    and SS_tot = sum((y_true - y_mean)^2).
    Returns 0.0 if the variance of the targets is zero to avoid NaNs.
    """
    if a.device != b.device:
        b = b.to(a.device)
    # Flatten and convert to float
    y_pred = a.float().view(-1)
    y_true = b.float().view(-1)

    # If shapes mismatch, align by truncation to the smallest size
    if y_pred.numel() != y_true.numel():
        n = min(y_pred.numel(), y_true.numel())
        y_pred = y_pred[:n]
        y_true = y_true[:n]

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if ss_tot.item() == 0:
        return 0.0

    return (1.0 - ss_res / ss_tot).item()

# ✅ New metrics
def f1(a: torch.Tensor, b: torch.Tensor) -> float:
    pred = a.argmax(dim=1).cpu().numpy()
    true = b.argmax(dim=1).cpu().numpy()
    return sk_f1_score(true, pred)

def balanced_acc(a: torch.Tensor, b: torch.Tensor) -> float:
    pred = a.argmax(dim=1).cpu().numpy()
    true = b.argmax(dim=1).cpu().numpy()
    return sk_balanced_accuracy(true, pred)

class Metrics(FuncEnum):
    ACCURACY = FuncEnum.wrap(accuracy)
    NEG_L2 = FuncEnum.wrap(neg_l2)
    NEG_MSE = FuncEnum.wrap(neg_mse)
    INV_MSE = FuncEnum.wrap(inv_mse)
    NEG_CROSS_ENTROPY = FuncEnum.wrap(neg_cross_entropy)
    RECALL = FuncEnum.wrap(recall)
    F1 = FuncEnum.wrap(f1)  # ✅ add this
    BALANCED_ACCURACY = FuncEnum.wrap(balanced_acc)  # ✅ add this
    NEG_RMSE = FuncEnum.wrap(neg_rmse)  # ✅ add this
    RMSPE = FuncEnum.wrap(rmspe)
    NEG_RMSPE = FuncEnum.wrap(neg_rmspe)
    R_SQUARED = FuncEnum.wrap(r_squared)

