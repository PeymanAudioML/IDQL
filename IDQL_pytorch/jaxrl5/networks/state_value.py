import torch
import torch.nn as nn


class StateValue(nn.Module):
    def __init__(self, base_cls: nn.Module, *args, **kwargs):
        super().__init__()
        self.base = base_cls(*args, **kwargs)

        # Get the output dimension from the base network
        if hasattr(self.base, 'layers') and len(self.base.layers) > 0:
            base_output_dim = self.base.layers[-1].out_features
        elif 'hidden_dims' in kwargs:
            base_output_dim = kwargs['hidden_dims'][-1]
        else:
            base_output_dim = 256  # Default fallback

        # Output layer for value
        self.value_layer = nn.Linear(base_output_dim, 1)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def forward(self, observations: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        outputs = self.base(observations, *args, **kwargs)
        value = self.value_layer(outputs)
        return value.squeeze(-1)