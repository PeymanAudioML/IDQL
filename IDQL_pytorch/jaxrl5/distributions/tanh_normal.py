import functools
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal as TorchNormal, MultivariateNormal

from jaxrl5.distributions.tanh_transformed import TanhTransformedDistribution


class Normal(nn.Module):
    def __init__(
        self,
        base_cls: Type[nn.Module],
        action_dim: int,
        log_std_min: Optional[float] = -20,
        log_std_max: Optional[float] = 2,
        state_dependent_std: bool = True,
        squash_tanh: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.base_cls = base_cls
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.squash_tanh = squash_tanh

        # Initialize the base network
        self.base = base_cls(*args, **kwargs)

        # Output layers for mean
        self.mean_layer = nn.Linear(self.base.layers[-1].out_features if hasattr(self.base, 'layers') else kwargs.get('hidden_dims', [256])[-1], action_dim)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

        # Output layer or parameter for log_std
        if self.state_dependent_std:
            self.log_std_layer = nn.Linear(self.base.layers[-1].out_features if hasattr(self.base, 'layers') else kwargs.get('hidden_dims', [256])[-1], action_dim)
            nn.init.xavier_uniform_(self.log_std_layer.weight)
            nn.init.zeros_(self.log_std_layer.bias)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, inputs, *args, **kwargs):
        x = self.base(inputs, *args, **kwargs)

        means = self.mean_layer(x)

        if self.state_dependent_std:
            log_stds = self.log_std_layer(x)
        else:
            log_stds = self.log_std.expand_as(means)

        # Clip log_std values
        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds)

        # Create distribution
        distribution = TorchNormal(means, stds)

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution

    def get_action(self, inputs, deterministic=False, *args, **kwargs):
        """Helper method to get actions from the distribution."""
        distribution = self(inputs, *args, **kwargs)

        if deterministic:
            if hasattr(distribution, 'mode'):
                action = distribution.mode()
            else:
                action = distribution.mean
        else:
            action = distribution.rsample() if hasattr(distribution, 'rsample') else distribution.sample()

        return action


def TanhNormal(base_cls, action_dim, **kwargs):
    """Factory function for creating a TanhNormal distribution."""
    return Normal(base_cls=base_cls, action_dim=action_dim, squash_tanh=True, **kwargs)