import torch
import torch.nn as nn


class StateActionValue(nn.Module):
    def __init__(self, base_cls: nn.Module, obs_dim: int, action_dim: int, *args, **kwargs):
        super().__init__()
        # Adjust input dimension for concatenated obs and action
        self.base = base_cls(input_dim=obs_dim + action_dim, *args, **kwargs)

        # Get the output dimension from the base network
        if hasattr(self.base, 'layers') and len(self.base.layers) > 0:
            base_output_dim = self.base.layers[-1].out_features
        elif 'hidden_dims' in kwargs:
            base_output_dim = kwargs['hidden_dims'][-1]
        else:
            base_output_dim = 256  # Default fallback

        # Output layer for Q-value
        self.value_layer = nn.Linear(base_output_dim, 1)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        outputs = self.base(inputs, *args, **kwargs)
        value = self.value_layer(outputs)
        return value.squeeze(-1)