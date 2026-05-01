import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128, 64), num_classes=2):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # expect input shape [N, features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
