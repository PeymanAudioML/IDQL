from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhDeterministic(nn.Module):
    def __init__(
        self,
        base_cls: Type[nn.Module],
        action_dim: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.base_cls = base_cls
        self.action_dim = action_dim

        # Initialize the base network
        self.base = base_cls(*args, **kwargs)

        # Output layer for deterministic actions
        # Get the output dimension from the base network
        if hasattr(self.base, 'layers') and len(self.base.layers) > 0:
            base_output_dim = self.base.layers[-1].out_features
        elif 'hidden_dims' in kwargs:
            base_output_dim = kwargs['hidden_dims'][-1]
        else:
            base_output_dim = 256  # Default fallback

        self.output_layer = nn.Linear(base_output_dim, action_dim)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        x = self.base(inputs, *args, **kwargs)
        means = self.output_layer(x)
        actions = torch.tanh(means)
        return actions