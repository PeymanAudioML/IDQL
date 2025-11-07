"""
DDPM (Denoising Diffusion Probabilistic Models) implementation in PyTorch
Converted from JAX/Flax to pure PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Type, Tuple
import numpy as np


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear beta schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


def vp_beta_schedule(timesteps: int) -> torch.Tensor:
    """Variance preserving beta schedule"""
    t = torch.arange(1, timesteps + 1).float()
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


class FourierFeatures(nn.Module):
    """Fourier features for encoding time steps"""
    
    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        
        if self.learnable:
            self.kernel = nn.Parameter(
                torch.randn(output_size // 2, 1) * 0.2
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        elif x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(-1)
            
        if self.learnable:
            f = 2 * math.pi * x @ self.kernel.T
        else:
            half_dim = self.output_size // 2
            f = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim, device=x.device) * -f)
            f = x * f.unsqueeze(0)
            
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with layer norm and mish activation"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.mish(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = F.mish(x)
        x = self.linear2(x)
        return x + residual


class MLPResNet(nn.Module):
    """MLP with residual connections"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: Optional[int] = None,
        num_blocks: int = 2
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        self.res_blocks = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(num_blocks):
                self.res_blocks.append(ResidualBlock(hidden_dim))
            
            if i < len(hidden_dims) - 1:
                self.res_blocks.append(
                    nn.Linear(hidden_dim, hidden_dims[i + 1])
                )
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.input_layer(x)
        
        for block in self.res_blocks:
            if isinstance(block, nn.Linear):
                x = F.mish(x)
                x = block(x)
            else:
                x = block(x)
        
        x = F.mish(x)
        x = self.output_layer(x)
        return x


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        time_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        num_blocks: int = 2
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Time preprocessing
        self.time_preprocess = FourierFeatures(time_dim, learnable=True)
        
        # Condition encoder (for time embeddings)
        self.cond_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Reverse process encoder
        input_dim = action_dim + observation_dim + hidden_dims[0]
        self.reverse_encoder = MLPResNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            num_blocks=num_blocks
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
        training: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of DDPM
        
        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim) - noisy actions
            time: (batch_size,) or (batch_size, 1) - diffusion timestep
            training: whether in training mode
            
        Returns:
            predicted_noise: (batch_size, action_dim)
        """
        # Preprocess time
        t_emb = self.time_preprocess(time)
        
        # Encode time condition
        cond = self.cond_encoder(t_emb)
        
        # Concatenate inputs
        reverse_input = torch.cat([actions, observations, cond], dim=-1)
        
        # Predict noise
        predicted_noise = self.reverse_encoder(reverse_input, training=training)
        
        return predicted_noise


class DDPMSampler:
    """Sampling utilities for DDPM"""
    
    def __init__(
        self,
        model: DDPM,
        betas: torch.Tensor,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        
        # Precompute diffusion coefficients
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alpha_bars[:-1]
        ])
        
        # Precompute coefficients for sampling
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        
        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1 - self.alpha_bars)
        )
    
    @torch.no_grad()
    def sample(
        self,
        observations: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
        num_timesteps: Optional[int] = None,
        clip_sample: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> torch.Tensor:
        """
        Sample actions from the diffusion model
        
        Args:
            observations: (batch_size, obs_dim) or (obs_dim,)
            num_samples: number of samples per observation
            temperature: sampling temperature
            num_timesteps: number of denoising steps (defaults to training timesteps)
            clip_sample: whether to clip samples
            clip_range: range for clipping
            
        Returns:
            samples: (batch_size * num_samples, action_dim)
        """
        self.model.eval()
        
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        batch_size = observations.shape[0]
        
        # Repeat observations for multiple samples
        if num_samples > 1:
            observations = observations.repeat(num_samples, 1)
        
        # Initialize from noise
        shape = (observations.shape[0], self.model.action_dim)
        x_t = torch.randn(shape, device=self.device) * temperature
        
        if num_timesteps is None:
            num_timesteps = len(self.betas)
        
        # Reverse diffusion process
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.full((x_t.shape[0],), t, device=self.device)
            
            # Predict noise
            pred_noise = self.model(observations, x_t, t_tensor, training=False)
            
            # Remove noise (reverse diffusion)
            x_t = self._p_sample(x_t, pred_noise, t, temperature, clip_sample, clip_range)
        
        return x_t
    
    def _p_sample(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: int,
        temperature: float = 1.0,
        clip_sample: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> torch.Tensor:
        """Single denoising step"""
        
        # Compute x_0 prediction
        x_0_pred = (
            x_t - self.sqrt_one_minus_alpha_bars[t] * pred_noise
        ) / self.sqrt_alpha_bars[t]
        
        if clip_sample:
            x_0_pred = torch.clamp(x_0_pred, clip_range[0], clip_range[1])
        
        # Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_0_pred +
            self.posterior_mean_coef2[t] * x_t
        )
        
        if t > 0:
            # Add noise for t > 0
            noise = torch.randn_like(x_t) * temperature
            posterior_variance = self.posterior_variance[t]
            x_t_minus_1 = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            # No noise at t = 0
            x_t_minus_1 = posterior_mean
        
        return x_t_minus_1
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to data for training
        
        Args:
            x_0: clean data
            t: timesteps
            noise: optional noise (will be generated if None)
            
        Returns:
            x_t: noisy data
            noise: noise added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Expand dimensions for broadcasting
        sqrt_alpha_bars_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bars_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        
        x_t = sqrt_alpha_bars_t * x_0 + sqrt_one_minus_alpha_bars_t * noise
        
        return x_t, noise
