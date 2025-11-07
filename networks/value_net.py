"""
Value networks for IQL (Implicit Q-Learning)
Converted from JAX/Flax to PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """Simple MLP network"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: int = 1,
        activation: str = 'relu',
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if layer_norm and i < len(dims) - 2:  # No norm on output layer
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
        
        # Set activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return F.relu
        elif activation == 'mish':
            return F.mish
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'elu':
            return F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            x = self.activation(x)
            
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x


class DoubleCritic(nn.Module):
    """Double Q-networks for IQL"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.q1 = MLP(
            input_dim=observation_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm
        )
        
        self.q2 = MLP(
            input_dim=observation_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        return_both: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through both Q-networks
        
        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim)
            return_both: if True, returns both Q values; if False, returns min
            
        Returns:
            Q values: (2, batch_size, 1) if return_both else (batch_size, 1)
        """
        obs_action = torch.cat([observations, actions], dim=-1)
        
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        
        if return_both:
            return torch.stack([q1, q2], dim=0)
        else:
            return torch.min(q1, q2)
    
    def q1_forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward through Q1 only"""
        return self.q1(torch.cat([observations, actions], dim=-1))
    
    def q2_forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward through Q2 only"""
        return self.q2(torch.cat([observations, actions], dim=-1))
    
    def q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get minimum Q value"""
        return self.forward(observations, actions, return_both=False)


class ValueFunction(nn.Module):
    """State value function for IQL"""
    
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            observations: (batch_size, obs_dim)
            
        Returns:
            values: (batch_size, 1)
        """
        return self.network(observations)


class StateActionValue(nn.Module):
    """Combined state-action value network (for compatibility)"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=observation_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm
        )
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim)
            
        Returns:
            q_values: (batch_size, 1)
        """
        obs_action = torch.cat([observations, actions], dim=-1)
        return self.network(obs_action)


class Ensemble(nn.Module):
    """Ensemble of networks"""
    
    def __init__(
        self,
        network_cls,
        num_networks: int = 2,
        **network_kwargs
    ):
        super().__init__()
        
        self.networks = nn.ModuleList([
            network_cls(**network_kwargs)
            for _ in range(num_networks)
        ])
        self.num_networks = num_networks
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward through all networks
        
        Returns:
            outputs: (num_networks, batch_size, output_dim)
        """
        outputs = []
        for network in self.networks:
            outputs.append(network(*args, **kwargs))
        return torch.stack(outputs, dim=0)
    
    def sample(self, *args, idx: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Forward through a specific network or random one"""
        if idx is None:
            idx = torch.randint(0, self.num_networks, (1,)).item()
        return self.networks[idx](*args, **kwargs)
