from collections import OrderedDict
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from opendataval.model.api import (
    TorchClassMixin,
    TorchPredictMixin,
    TorchRegressMixin,
)
from opendataval.model.grad import TorchGradMixin
from opendataval.dataloader.util import CatDataset


class ClassifierMLP(TorchClassMixin, TorchPredictMixin, TorchGradMixin):
    """Initializes the Multilayer Perceptron  Classifier.

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the MLP
    num_classes : int
        Size of the output dimension of the MLP, outputs selection probabilities
    layers : int, optional
        Number of layers for the MLP, by default 5
    hidden_dim : int, optional
        Hidden dimension for the MLP, by default 25
    act_fn : Callable, optional
        Activation function for MLP, if none, set to nn.ReLU, by default None
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Optional[Callable] = None,
    ):
        super().__init__()

        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.num_classes = num_classes

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        # Add hidden layers (0 for layers=2, 1 for layers=3, etc.)
        num_hidden = max(0, int(layers - 2))
        for i in range(num_hidden):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        # Output layer (use num_hidden as index to avoid undefined variable when layers=2)
        mlp_layers[f"{num_hidden+1}_out_lin"] = nn.Linear(hidden_dim, num_classes)
        mlp_layers["output"] = nn.Softmax(-1)

        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP Neural Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of MLP
        """
        x = self.mlp(x)
        return x

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
        print_loss: bool = True,
    ):
        """Fits the model with loss tracking and learning verification.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        print_loss : bool, optional
            Whether to print loss during training, by default True

        """

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)
        print("epochs:", epochs, "batch_size:", batch_size, "lr:", lr)
        self.train()

        # Only use pin_memory if data is on CPU (pin_memory fails with GPU tensors)
        use_pin_memory = isinstance(x_train, torch.Tensor) and x_train.device.type == 'cpu'

        # Debug: Track learning metrics
        epoch_losses = []
        batch_losses = []
        grad_norms = []
        param_change_norms = []

        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            num_batches = 0
            epoch_grad_norms = []
            epoch_param_changes = []

            dataloader = DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=use_pin_memory
            )
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{int(epochs)}", leave=True)

            for batch_idx, (x_batch, y_batch, *weights) in progress_bar:
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                # Store params before update for change tracking
                params_before = [p.clone().detach() for p in self.parameters()]

                opt.zero_grad()
                outputs = self.__call__(x_batch)

                if y_batch.dim() > 1:
                    y_t = y_batch.argmax(dim=1)
                else:
                    y_t = y_batch.view(-1)
                y_t = y_t.to(dtype=torch.long, device=self.device)

                if sample_weight is not None:
                    loss = criterion(outputs, y_t, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(outputs, y_t, reduction="mean")

                loss.backward()

                # DEBUG: Compute gradient norm before step
                grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += (p.grad ** 2).sum().item()
                grad_norm = grad_norm ** 0.5
                epoch_grad_norms.append(grad_norm)

                opt.step()

                # DEBUG: Compute parameter change after step
                param_change = 0.0
                for p_before, p_after in zip(params_before, self.parameters()):
                    param_change += ((p_after - p_before) ** 2).sum().item()
                param_change = param_change ** 0.5
                epoch_param_changes.append(param_change)

                epoch_loss += loss.item()
                batch_losses.append(loss.item())
                num_batches += 1

                avg_batch_loss = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_batch_loss:.6f}'})

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)

            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
            grad_norms.append(avg_grad_norm)

            avg_param_change = sum(epoch_param_changes) / len(epoch_param_changes) if epoch_param_changes else 0
            param_change_norms.append(avg_param_change)

            # DEBUG: Print learning progress
            if print_loss or True:  # Always show for debugging
                print(f"  Epoch {epoch+1}/{int(epochs)}: loss={avg_epoch_loss:.6f}, "
                      f"grad_norm={avg_grad_norm:.6f}, param_change={avg_param_change:.6f}")

                # Check for learning issues
                if epoch > 0:
                    loss_improvement = epoch_losses[-2] - epoch_losses[-1]
                    if loss_improvement < 0:
                        print(f"    ⚠️  Loss increased (was {epoch_losses[-2]:.6f})")
                    elif abs(loss_improvement) < 1e-6:
                        print(f"    ⚠️  Loss not improving (change: {loss_improvement:.10f})")

                    if avg_grad_norm < 1e-7:
                        print(f"    ⚠️  Gradient norm very small ({avg_grad_norm:.2e})")

                    if avg_param_change < 1e-8:
                        print(f"    ⚠️  Parameters barely updating ({avg_param_change:.2e})")

                    if avg_grad_norm > 10.0:
                        print(f"    ⚠️  Gradient explosion ({avg_grad_norm:.2e})")

                # Positive indicators
                if avg_grad_norm > 1e-5 and avg_param_change > 1e-6:
                    print(f"    ✓ Learning active")

        return self


class SimpleMLP(TorchClassMixin, TorchPredictMixin, TorchGradMixin):
    """Simple two-layer MLP for MNIST classification.

    A minimal MLP with one hidden layer designed for MNIST (28x28 images).

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension, by default 256
    num_classes : int, optional
        Number of output classes, by default 10
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of simple MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, 1, 28, 28) or (batch_size, 784)

        Returns
        -------
        torch.Tensor
            Output logits (batch_size, num_classes)
        """
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
        print_loss: bool = True,
    ):
        """Fits the model with loss tracking and learning verification.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        print_loss : bool, optional
            Whether to print loss during training, by default True
        """

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)
        print(epochs, batch_size, lr)
        self.train()

        # Only use pin_memory if data is on CPU (pin_memory fails with GPU tensors)
        use_pin_memory = isinstance(x_train, torch.Tensor) and x_train.device.type == 'cpu'

        # Debug: Track learning metrics
        epoch_losses = []
        grad_norms = []
        param_change_norms = []

        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            num_batches = 0
            epoch_grad_norms = []
            epoch_param_changes = []

            dataloader = DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=use_pin_memory
            )
            progress_bar = tqdm(dataloader, total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{int(epochs)}", leave=True)

            for x_batch, y_batch, *weights in progress_bar:
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                # Store params before update for change tracking
                params_before = [p.clone().detach() for p in self.parameters()]

                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if y_batch.dim() > 1:
                    y_t = y_batch.argmax(dim=1)
                else:
                    y_t = y_batch.view(-1)
                y_t = y_t.to(dtype=torch.long, device=self.device)

                if sample_weight is not None:
                    loss = criterion(outputs, y_t, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(outputs, y_t, reduction="mean")

                loss.backward()

                # DEBUG: Compute gradient norm before step
                grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += (p.grad ** 2).sum().item()
                grad_norm = grad_norm ** 0.5
                epoch_grad_norms.append(grad_norm)

                opt.step()

                # DEBUG: Compute parameter change after step
                param_change = 0.0
                for p_before, p_after in zip(params_before, self.parameters()):
                    param_change += ((p_after - p_before) ** 2).sum().item()
                param_change = param_change ** 0.5
                epoch_param_changes.append(param_change)

                epoch_loss += loss.item()
                num_batches += 1

                avg_batch_loss = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_batch_loss:.6f}'})

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)

            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
            grad_norms.append(avg_grad_norm)

            avg_param_change = sum(epoch_param_changes) / len(epoch_param_changes) if epoch_param_changes else 0
            param_change_norms.append(avg_param_change)

            # DEBUG: Print learning progress
            if print_loss or True:  # Always show for debugging
                print(f"  Epoch {epoch+1}/{int(epochs)}: loss={avg_epoch_loss:.6f}, "
                      f"grad_norm={avg_grad_norm:.6f}, param_change={avg_param_change:.6f}")

                # Check for learning issues
                if epoch > 0:
                    loss_improvement = epoch_losses[-2] - epoch_losses[-1]
                    if loss_improvement < 0:
                        print(f"    ⚠️  Loss increased (was {epoch_losses[-2]:.6f})")
                    elif abs(loss_improvement) < 1e-6:
                        print(f"    ⚠️  Loss not improving (change: {loss_improvement:.10f})")

                    if avg_grad_norm < 1e-7:
                        print(f"    ⚠️  Gradient norm very small ({avg_grad_norm:.2e})")

                    if avg_param_change < 1e-8:
                        print(f"    ⚠️  Parameters barely updating ({avg_param_change:.2e})")

                    if avg_grad_norm > 10.0:
                        print(f"    ⚠️  Gradient explosion ({avg_grad_norm:.2e})")

                # Positive indicators
                if avg_grad_norm > 1e-5 and avg_param_change > 1e-6:
                    print(f"    ✓ Learning active")

        return self


class RegressionMLP(TorchRegressMixin, TorchPredictMixin, TorchGradMixin):
    """Initializes the Multilayer Perceptron Regression.

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the MLP
    num_classes : int
        Size of the output dimension of the MLP, >1 means multi dimension output
    layers : int, optional
        Number of layers for the MLP, by default 5
    hidden_dim : int, optional
        Hidden dimension for the MLP, by default 25
    act_fn : Callable, optional
        Activation function for MLP, if none, set to nn.ReLU, by default None
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Optional[Callable] = None,
    ):
        super().__init__()

        act_fn = nn.ReLU() if act_fn is None else act_fn

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        mlp_layers["output"] = nn.Linear(hidden_dim, num_classes)
        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Multilayer Perceptron.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of MLP
        """
        x = self.mlp(x)
        return x

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
        print_loss: bool = True,
    ):
        """Fits the regression model with loss tracking.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        print_loss : bool, optional
            Whether to print loss during training, by default True
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = F.mse_loss
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()

        # Only use pin_memory if data is on CPU (pin_memory fails with GPU tensors)
        use_pin_memory = isinstance(x_train, torch.Tensor) and x_train.device.type == 'cpu'

        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            num_batches = 0

            dataloader = DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=use_pin_memory
            )
            progress_bar = tqdm(dataloader, total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{int(epochs)}", leave=True)

            for x_batch, y_batch, *weights in progress_bar:
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                if sample_weight is not None:
                    loss = criterion(y_hat, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(y_hat, y_batch, reduction="mean")

                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                num_batches += 1

                avg_batch_loss = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_batch_loss:.6f}'})

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if print_loss:
                pass

        return self
