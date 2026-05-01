import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    """
    Simple MLP for tabular data with configurable hidden layers.
    """
    def __init__(self, input_dim, hidden_dims=(256, 128), num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_embeddings(self, x):
        return self.features(x)
