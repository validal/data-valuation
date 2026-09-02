"""InRunDataShapley - Data Shapley values computed during single training run.

Template for incremental Shapley computation during model training.

References
----------
.. [1] Ghorbani et al. "Data Shapley: Equitable Valuation of Data for Machine Learning"
    https://arxiv.org/abs/1904.02868
"""

from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel


class _IndexedDataset(Dataset):
    """Yields (x, y, global_index) so scores map to the right sample under shuffle."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], i


class InRunDataShapley(DataEvaluator, ModelMixin):
    """Data Shapley values computed during single training run.

    Template for computing Shapley values by training once and collecting
    marginal contributions incrementally.

    Parameters
    ----------
    epochs : int, optional
        Number of training epochs, by default 10
    lr : float, optional
        Learning rate, by default 0.01
    batch_size : int, optional
        Training batch size, by default 64
    val_batch_size : int, optional
        Validation batch size (None = use batch_size), by default None
    loss_reduction : str, optional
        Loss reduction method ('mean' or 'sum'), by default "mean"
    random_state : RandomState, optional
        Random state for reproducibility, by default None

    Attributes
    ----------
    data_values : np.ndarray
        Computed Shapley values for training samples
    """

    def __init__(
        self,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        loss_reduction: str = "mean",
        random_state: Optional[RandomState] = None,
    ):
        """Initialize InRunDataShapley evaluator."""
        super().__init__(random_state=random_state)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.loss_reduction = loss_reduction
        self.data_values = None

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store training and validation data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self.data_values = np.zeros(self.num_points)
        return self

    def input_model(self, pred_model: GradientModel):
        """Set the prediction model."""
        assert (
            isinstance(pred_model, GradientModel)
            or callable(getattr(pred_model, "grad"))
        ), "Model with gradient required for InRunDataShapley."

        self.pred_model = pred_model.clone()
        return self

    def _build_val_batch(self, x_valid, y_valid):
        """Pick a class-balanced subset of the validation set as the scoring target.

        Balancing prevents class identity from dominating the gradient-alignment signal.
        Handles one-hot labels by converting to class indices for balancing,
        but keeps val_y in original format for the loss.

        Parameters
        ----------
        x_valid : torch.Tensor
            Validation features
        y_valid : torch.Tensor
            Validation labels (one-hot or class indices)

        Returns
        -------
        tuple
            (val_x, val_y) balanced subset of validation data
        """
        y = y_valid

        # Handle one-hot labels: convert to class indices for balancing
        if y.ndim > 1 and y.shape[1] > 1:
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y.long().view(-1)

        classes = torch.unique(y_idx)
        per_class = max(1, self.val_batch_size // len(classes))

        idx_list = []
        for c in classes:
            c_idx = (y_idx == c).nonzero(as_tuple=True)[0]
            idx_list.append(c_idx[:per_class])
        sel = torch.cat(idx_list)

        val_x = x_valid[sel]
        val_y = y_valid[sel]

        # Store the actual count for later scaling
        self._n_val = len(sel)

        return val_x, val_y

    def train_data_values(self, *args, **kwargs):
        """Compute Shapley values from single training run.

        TODO: Implement incremental Shapley computation during training.

        Parameters
        ----------
        args : tuple
            Additional positional arguments
        kwargs : dict
            Additional keyword arguments
        """
        # Placeholder implementation
        pass
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return computed Shapley values.

        Returns
        -------
        np.ndarray
            Shapley values for each training sample
        """
        if self.data_values is None:
            return np.zeros(self.num_points)
        return self.data_values
