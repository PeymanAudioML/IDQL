"""
DDPM-IQL Learner: Implicit Q-Learning with Diffusion Policies
Converted from JAX to PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np
import copy
from dataclasses import dataclass

from ..networks.diffusion import DDPM, DDPMSampler, cosine_beta_schedule, vp_beta_schedule
from ..networks.value_net import DoubleCritic, ValueFunction


@dataclass
class DDPMIQLConfig:
    """Configuration for DDPM-IQL"""
    # Model dimensions
    observation_dim: int
    action_dim: int
    
    # Network architecture
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    value_hidden_dims: Tuple[int, ...] = (256, 256)
    time_dim: int = 64
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    
    # IQL parameters
    discount: float = 0.99
    tau: float = 0.005  # Target network update rate
    actor_tau: float = 0.001  # Actor target network update rate
    expectile: float = 0.7  # For expectile regression
    temperature: float = 3.0  # IQL temperature
    
    # Diffusion parameters
    num_timesteps: int = 100
    beta_schedule: str = 'cosine'  # 'cosine', 'linear', or 'vp'
    ddpm_temperature: float = 1.0
    clip_sample: bool = True
    clip_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Sampling parameters
    num_samples: int = 10  # Number of actions to sample per state
    num_inference_timesteps: Optional[int] = None  # Defaults to num_timesteps
    
    # Training parameters
    batch_size: int = 256
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    
    # Loss weights
    actor_loss_weight: float = 1.0
    bc_loss_weight: float = 0.0  # Behavior cloning regularization
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """Expectile regression loss for IQL"""
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update of target network parameters"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


class DDPMIQLLearner:
    """DDPM-IQL: Combining diffusion models with implicit Q-learning"""
    
    def __init__(self, config: DDPMIQLConfig):
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self._build_networks()
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize diffusion sampler
        self._build_diffusion_sampler()
        
        # Training statistics
        self.training_step = 0
        
    def _build_networks(self):
        """Build all networks"""
        config = self.config
        
        # Actor (DDPM)
        self.actor = DDPM(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            time_dim=config.time_dim,
            hidden_dims=config.actor_hidden_dims
        ).to(self.device)
        
        self.actor_target = DDPM(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            time_dim=config.time_dim,
            hidden_dims=config.actor_hidden_dims
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic (Double Q-networks)
        self.critic = DoubleCritic(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dims=config.critic_hidden_dims
        ).to(self.device)
        
        self.critic_target = DoubleCritic(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dims=config.critic_hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Value function
        self.value = ValueFunction(
            observation_dim=config.observation_dim,
            hidden_dims=config.value_hidden_dims
        ).to(self.device)
    
    def _build_optimizers(self):
        """Build optimizers for all networks"""
        config = self.config
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay
        )
        
        self.value_optimizer = optim.Adam(
            self.value.parameters(),
            lr=config.value_lr,
            weight_decay=config.weight_decay
        )
    
    def _build_diffusion_sampler(self):
        """Build diffusion sampler with precomputed schedules"""
        config = self.config
        
        # Select beta schedule
        if config.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(config.num_timesteps)
        elif config.beta_schedule == 'linear':
            from ..networks.diffusion import linear_beta_schedule
            betas = linear_beta_schedule(config.num_timesteps)
        elif config.beta_schedule == 'vp':
            betas = vp_beta_schedule(config.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
        
        # Create samplers for both actor and target
        self.sampler = DDPMSampler(self.actor, betas, self.device)
        self.target_sampler = DDPMSampler(self.actor_target, betas, self.device)
    
    def update_value(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update value function using expectile regression"""
        observations = batch['observations']
        actions = batch['actions']
        
        with torch.no_grad():
            # Get Q values from target critic
            target_q = self.critic_target.q_min(observations, actions)
        
        # Value function prediction
        values = self.value(observations)
        
        # Expectile loss
        value_loss = expectile_loss(
            target_q - values,
            self.config.expectile
        ).mean()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.grad_clip)
        self.value_optimizer.step()
        
        return {'value_loss': value_loss.item()}
    
    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Q-functions using TD learning"""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        
        with torch.no_grad():
            # Compute target values
            next_values = self.value(next_observations)
            targets = rewards + self.config.discount * (1 - dones) * next_values
        
        # Get Q predictions
        q_values = self.critic(observations, actions, return_both=True)
        
        # Compute TD loss for both critics
        critic_loss = (
            F.mse_loss(q_values[0], targets) +
            F.mse_loss(q_values[1], targets)
        ) / 2
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        
        # Soft update target critic
        soft_update(self.critic_target, self.critic, self.config.tau)
        
        return {'critic_loss': critic_loss.item()}
    
    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update diffusion policy using weighted regression"""
        observations = batch['observations']
        actions = batch['actions']
        
        batch_size = observations.shape[0]
        
        # Sample random timesteps
        t = torch.randint(
            0, self.config.num_timesteps,
            (batch_size,), device=self.device
        )
        
        # Add noise to actions
        noisy_actions, noise = self.sampler.add_noise(actions, t)
        
        # Predict noise
        pred_noise = self.actor(observations, noisy_actions, t, training=True)
        
        # Compute base diffusion loss
        diffusion_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        
        # Compute advantage weights for IQL
        with torch.no_grad():
            # Get Q values for the actions
            q_values = self.critic_target.q_min(observations, actions)
            v_values = self.value(observations)
            advantages = q_values - v_values
            
            # Compute weights using temperature
            weights = torch.exp(advantages / self.config.temperature)
            weights = torch.clamp(weights, max=100.0)  # Clip for stability
            weights = weights / weights.mean()  # Normalize
        
        # Weighted diffusion loss
        actor_loss = (diffusion_loss * weights.squeeze()).mean()
        
        # Optional: Add behavior cloning regularization
        if self.config.bc_loss_weight > 0:
            bc_loss = diffusion_loss.mean()
            actor_loss = actor_loss + self.config.bc_loss_weight * bc_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        (self.config.actor_loss_weight * actor_loss).backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        
        # Soft update target actor
        soft_update(self.actor_target, self.actor, self.config.actor_tau)
        
        return {
            'actor_loss': actor_loss.item(),
            'mean_weight': weights.mean().item(),
            'max_weight': weights.max().item(),
        }
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Full update step for all networks"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Update in order: value -> critic -> actor
        metrics = {}
        metrics.update(self.update_value(batch))
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        
        self.training_step += 1
        metrics['training_step'] = self.training_step
        
        return metrics
    
    @torch.no_grad()
    def sample_actions(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        num_samples: Optional[int] = None,
        deterministic: bool = False,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample actions from the diffusion policy
        
        Args:
            observations: observations to condition on
            num_samples: number of actions to sample per observation
            deterministic: if True, use lower temperature and no noise at t=0
            temperature: sampling temperature (overrides config)
            
        Returns:
            actions: (batch_size * num_samples, action_dim) if num_samples > 1
                    else (batch_size, action_dim)
        """
        self.actor.eval()
        
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations).to(self.device)
        
        if num_samples is None:
            num_samples = 1 if deterministic else self.config.num_samples
        
        if temperature is None:
            temperature = 0.1 if deterministic else self.config.ddpm_temperature
        
        # Sample from diffusion model
        actions = self.sampler.sample(
            observations=observations,
            num_samples=num_samples,
            temperature=temperature,
            num_timesteps=self.config.num_inference_timesteps,
            clip_sample=self.config.clip_sample,
            clip_range=self.config.clip_range
        )
        
        return actions
    
    def get_action(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Get a single action for deployment
        
        Args:
            observation: single observation
            deterministic: if True, use deterministic sampling
            
        Returns:
            action: (action_dim,) numpy array
        """
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        action = self.sample_actions(
            observation,
            num_samples=1,
            deterministic=deterministic
        )
        
        return action.squeeze(0).cpu().numpy()
    
    def save(self, path: str):
        """Save model checkpoints"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'value': self.value.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.value.load_state_dict(checkpoint['value'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        self.training_step = checkpoint['training_step']
