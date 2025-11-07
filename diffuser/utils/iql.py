"""
IQL utilities for integration with Diffuser framework
PyTorch implementation to replace JAX version
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import functools

# Import from our PyTorch implementation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.value_net import DoubleCritic
from agents.ddpm_iql import DDPMIQLLearner, DDPMIQLConfig


class IQLWrapper:
    """
    PyTorch wrapper for IQL Q-functions
    Compatible with Diffuser framework
    """
    
    def __init__(
        self,
        env,
        loadpath: str,
        hidden_dims: Tuple[int, ...] = (256, 256),
        device: str = 'cuda'
    ):
        """
        Initialize IQL wrapper
        
        Args:
            env: Gym environment
            loadpath: Path to saved model checkpoint
            hidden_dims: Hidden layer dimensions for Q-networks
            device: Device to load model on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.env = env
        
        # Get dimensions
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Load Q-function
        self.model = self.load_q(loadpath, hidden_dims)
        self.model.eval()
        
        print(f'[ utils/iql ] Loaded Q-function from: {loadpath}')
    
    def load_q(
        self,
        loadpath: str,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ) -> DoubleCritic:
        """
        Load Q-function from checkpoint
        
        Args:
            loadpath: Path to checkpoint file
            hidden_dims: Hidden layer dimensions
            
        Returns:
            Loaded DoubleCritic model
        """
        # Expand path
        loadpath = os.path.expanduser(loadpath)
        
        # Create model
        model = DoubleCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(loadpath):
            checkpoint = torch.load(loadpath, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'critic' in checkpoint:
                # Full DDPM-IQL checkpoint
                model.load_state_dict(checkpoint['critic'])
            elif 'model_state_dict' in checkpoint:
                # Standard PyTorch checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {loadpath}")
        
        return model
    
    @torch.no_grad()
    def forward(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute Q-values (minimum of double Q)
        
        Args:
            observations: (batch_size, obs_dim) or (obs_dim,)
            actions: (batch_size, action_dim) or (action_dim,)
            
        Returns:
            Q-values: (batch_size,) or scalar
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        act_tensor = torch.FloatTensor(actions).to(self.device)
        
        # Add batch dimension if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            act_tensor = act_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute Q-values (minimum of double Q)
        q_values = self.model.q_min(obs_tensor, act_tensor)
        
        # Convert back to numpy
        q_numpy = q_values.squeeze(-1).cpu().numpy()
        
        if squeeze_output:
            q_numpy = q_numpy.squeeze()
        
        return q_numpy
    
    def __call__(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Callable interface for compatibility"""
        return self.forward(observations, actions)


def load_ddpm_iql_agent(
    env,
    loadpath: str,
    config: Optional[DDPMIQLConfig] = None,
    device: str = 'cuda'
) -> DDPMIQLLearner:
    """
    Load a complete DDPM-IQL agent
    
    Args:
        env: Gym environment
        loadpath: Path to checkpoint
        config: Optional config (will be loaded from checkpoint if None)
        device: Device to load on
        
    Returns:
        Loaded DDPMIQLLearner
    """
    loadpath = os.path.expanduser(loadpath)
    
    # Load checkpoint
    checkpoint = torch.load(loadpath, map_location=device)
    
    # Get config from checkpoint or use provided
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Create default config
            config = DDPMIQLConfig(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                device=device
            )
    
    # Create agent
    agent = DDPMIQLLearner(config)
    
    # Load weights
    agent.load(loadpath)
    
    return agent


def create_value_function_for_diffuser(
    env,
    q_loadpath: Optional[str] = None,
    agent_loadpath: Optional[str] = None,
    device: str = 'cuda'
) -> callable:
    """
    Create a value function callable for use with Diffuser's guided sampling
    
    Args:
        env: Gym environment
        q_loadpath: Path to Q-function checkpoint
        agent_loadpath: Path to full agent checkpoint (alternative to q_loadpath)
        device: Device to use
        
    Returns:
        Value function: (observations, actions) -> values
    """
    if q_loadpath is not None:
        # Load just Q-function
        wrapper = IQLWrapper(env, q_loadpath, device=device)
        return wrapper
    
    elif agent_loadpath is not None:
        # Load full agent and extract Q-function
        agent = load_ddpm_iql_agent(env, agent_loadpath, device=device)
        
        @torch.no_grad()
        def value_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
            obs_tensor = torch.FloatTensor(observations).to(device)
            act_tensor = torch.FloatTensor(actions).to(device)
            
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                act_tensor = act_tensor.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            q_values = agent.critic_target.q_min(obs_tensor, act_tensor)
            q_numpy = q_values.squeeze(-1).cpu().numpy()
            
            if squeeze:
                q_numpy = q_numpy.squeeze()
            
            return q_numpy
        
        return value_fn
    
    else:
        raise ValueError("Either q_loadpath or agent_loadpath must be provided")


# Compatibility layer for existing Diffuser code
def load_q(env, loadpath: str, hidden_dims: Tuple[int, ...] = (256, 256), seed: int = 42):
    """
    Direct replacement for JAX version's load_q function
    Maintains same interface for compatibility
    """
    print(f'[ utils/iql ] Loading Q: {loadpath}')
    wrapper = IQLWrapper(env, loadpath, hidden_dims)
    return wrapper


# For backward compatibility with JAX version
JaxWrapper = IQLWrapper  # Alias for compatibility


if __name__ == '__main__':
    # Test loading and using IQL wrapper
    import gym
    
    # Create environment
    env = gym.make('Hopper-v2')
    
    # Test with dummy checkpoint (would need real checkpoint in practice)
    try:
        # Example usage
        wrapper = IQLWrapper(
            env,
            loadpath='./checkpoints/hopper-medium-v2/best_model.pt'
        )
        
        # Test forward pass
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        
        q_value = wrapper(obs, action)
        print(f"Q-value shape: {q_value.shape if hasattr(q_value, 'shape') else 'scalar'}")
        print(f"Q-value: {q_value}")
        
    except FileNotFoundError:
        print("No checkpoint found - this is expected for testing")
        
        # Create a dummy model for testing
        wrapper = IQLWrapper.__new__(IQLWrapper)
        wrapper.device = 'cpu'
        wrapper.observation_dim = env.observation_space.shape[0]
        wrapper.action_dim = env.action_space.shape[0]
        wrapper.model = DoubleCritic(
            wrapper.observation_dim,
            wrapper.action_dim
        )
        wrapper.model.eval()
        
        # Test forward pass with dummy model
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        q_value = wrapper.forward(obs, action)
        print(f"Dummy Q-value: {q_value}")
