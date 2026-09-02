import torch
import torch.nn as nn
import torch.nn.functional as F

from opendataval.model.api import TorchClassMixin, TorchPredictMixin
from opendataval.model.grad import TorchGradMixin

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class LogisticRegression(TorchClassMixin, TorchPredictMixin, TorchGradMixin):
    """Initialize LogisticRegression

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the LogisticRegression
    num_classes : int
        Size of the output dimension of the LR, outputs selection probabilities
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Logistic Regression.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of logistic regression
        """
        # Ensure input matches the layer's dtype and device to avoid dtype mismatch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(dtype=self.linear.weight.dtype, device=self.linear.weight.device)

        # Flatten per-sample features if needed (e.g., images)
        # Target in_features
        in_features = self.linear.in_features
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2 and x.shape[1] != in_features:
            # Attempt to reconcile mismatch by flattening
            x = x.reshape(x.size(0), -1)

        logits = self.linear(x)
        # For multiclass classification (num_classes > 2), return raw logits.
        # Cross-entropy in Torch applies log-softmax internally and expects logits.
        # For binary setups represented with 2 outputs, keep sigmoid for BCE.
        if self.num_classes > 2:
            return logits
        else:
            return torch.sigmoid(logits)
    
    # def fit(self, *args, return_logs=False, **kwargs):
    #     # Monkey-patch the loss tracking into TorchClassMixin's fit
    #     self._epoch_losses = []

    #     # Use callback to track losses
    #     def loss_hook(loss_val):
    #         self._epoch_losses.append(loss_val)

    #     # Store original loss function for restoration
    #     original_loss_fn = F.cross_entropy if self.num_classes > 2 else F.binary_cross_entropy

    #     # Wrap the loss function to track loss values
    #     def loss_tracking_fn(output, target, **kw):
    #         loss = original_loss_fn(output, target, **kw)
    #         loss_hook(loss.item())
    #         return loss

    #     # Patch temporarily inside the fit call
    #     import torch.nn.functional as F_global
    #     original = F_global.cross_entropy
    #     try:
    #         if self.num_classes > 2:
    #             F_global.cross_entropy = loss_tracking_fn
    #         else:
    #             F_global.binary_cross_entropy = loss_tracking_fn

    #         result = super().fit(*args, **kwargs)

    #     finally:
    #         # Restore original loss
    #         if self.num_classes > 2:
    #             F_global.cross_entropy = original
    #         else:
    #             F_global.binary_cross_entropy = original

    #     if return_logs:
    #         return {"epoch_losses": self._epoch_losses}
    #     return result

    def fit(self, *args, return_logs=False, **kwargs):
        # Track batch-wise loss
        self._batch_losses = []

        # Store original loss function for restoration
        import torch.nn.functional as F_global
        original_ce = F_global.cross_entropy
        original_bce = F_global.binary_cross_entropy

        # Determine which loss to use
        use_ce = self.num_classes > 2
        original_loss_fn = original_ce if use_ce else original_bce

        # Loss wrapper for tracking
        def loss_tracking_fn(output, target, **kw):
            loss = original_loss_fn(output, target, **kw)
            #self._batch_losses.append(loss.item())
            self._batch_losses.append(loss.mean().item())  # ✅ safe logging

            return loss

        try:
            if use_ce:
                F_global.cross_entropy = loss_tracking_fn
            else:
                F_global.binary_cross_entropy = loss_tracking_fn

            # 🔁 Fit using base class
            result = super().fit(*args, **kwargs)

        finally:
            # Restore original loss function
            if use_ce:
                F_global.cross_entropy = original_ce
            else:
                F_global.binary_cross_entropy = original_bce

        # ✅ Convert batch-level logs to epoch-level
        if return_logs:
            epochs = kwargs.get("epochs", 1)
            batches_per_epoch = len(self._batch_losses) // epochs
            self._epoch_losses = [
                sum(self._batch_losses[i * batches_per_epoch:(i + 1) * batches_per_epoch]) / batches_per_epoch
                for i in range(epochs)
            ]
            return {"epoch_losses": self._epoch_losses}

        return result





