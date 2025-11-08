from functools import partial
from typing import Callable, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(
        beta_start, beta_end, timesteps
    )
    return betas


def vp_beta_schedule(timesteps):
    t = torch.arange(1, timesteps + 1, dtype=torch.float32)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable

        if self.learnable:
            # Learnable Fourier features
            self.kernel = nn.Parameter(
                torch.randn(self.output_size // 2, 1) * 0.2  # Will be resized based on input
            )

    def forward(self, x: torch.Tensor):
        if self.learnable:
            # Ensure kernel has the right dimensions
            if self.kernel.shape[1] != x.shape[-1]:
                self.kernel = nn.Parameter(
                    torch.randn(self.output_size // 2, x.shape[-1]) * 0.2
                )
            f = 2 * np.pi * x @ self.kernel.T
        else:
            # Fixed Fourier features (sinusoidal encoding)
            half_dim = self.output_size // 2
            f = np.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -f)
            f = x * f.unsqueeze(0)

        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class DDPM(nn.Module):
    def __init__(
        self,
        cond_encoder_cls: Type[nn.Module],
        reverse_encoder_cls: Type[nn.Module],
        time_preprocess_cls: Type[nn.Module],
        obs_dim: int,
        action_dim: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize time preprocessing
        if time_preprocess_cls == FourierFeatures:
            self.time_preprocess = time_preprocess_cls(kwargs.get('time_embed_size', 32))
        else:
            self.time_preprocess = time_preprocess_cls(*args, **kwargs)

        # Initialize conditional encoder
        time_embed_size = kwargs.get('time_embed_size', 32)
        cond_hidden_dims = kwargs.get('cond_hidden_dims', [256, 256])
        self.cond_encoder = cond_encoder_cls(
            input_dim=time_embed_size,
            hidden_dims=cond_hidden_dims,
            **(kwargs.get('cond_encoder_kwargs', {}))
        )

        # Initialize reverse encoder
        reverse_hidden_dims = kwargs.get('reverse_hidden_dims', [256, 256])
        # Input is concatenation of action, state, and condition embedding
        reverse_input_dim = action_dim + obs_dim + cond_hidden_dims[-1]
        self.reverse_encoder = reverse_encoder_cls(
            input_dim=reverse_input_dim,
            hidden_dims=reverse_hidden_dims + [action_dim],
            **(kwargs.get('reverse_encoder_kwargs', {}))
        )

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        time: torch.Tensor,
        training: bool = False
    ):
        # Preprocess time
        t_ff = self.time_preprocess(time)

        # Get conditional embedding
        cond = self.cond_encoder(t_ff, training=training)

        # Concatenate inputs for reverse process
        reverse_input = torch.cat([a, s, cond], dim=-1)

        # Apply reverse encoder
        return self.reverse_encoder(reverse_input, training=training)


def ddpm_sampler(
    model,
    T,
    act_dim,
    observations,
    alphas,
    alpha_hats,
    betas,
    sample_temperature=1.0,
    repeat_last_step=0,
    clip_sampler=False,
    device='cuda',
    generator=None
):
    """
    DDPM sampling function for PyTorch

    Args:
        model: The DDPM model
        T: Number of diffusion timesteps
        act_dim: Action dimension
        observations: Batch of observations
        alphas: Alpha schedule
        alpha_hats: Cumulative product of alphas
        betas: Beta schedule
        sample_temperature: Temperature for sampling
        repeat_last_step: Number of times to repeat the last denoising step
        clip_sampler: Whether to clip actions to [-1, 1]
        device: Device to run on
        generator: Random number generator for reproducibility
    """
    batch_size = observations.shape[0]

    # Initialize with random noise
    current_x = torch.randn(
        batch_size, act_dim, device=device, generator=generator
    )

    # Reverse diffusion process
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            # Prepare time tensor
            t = torch.full((batch_size, 1), time, device=device, dtype=torch.float32)

            # Predict noise
            eps_pred = model(observations, current_x, t, training=False)

            # Denoise step
            alpha_1 = 1 / torch.sqrt(alphas[time])
            alpha_2 = (1 - alphas[time]) / torch.sqrt(1 - alpha_hats[time])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            # Add noise (except for t=0)
            if time > 0:
                if generator is not None:
                    z = torch.randn(current_x.shape, generator=generator, device=device)
                else:
                    z = torch.randn_like(current_x)
                z_scaled = sample_temperature * z
                current_x = current_x + torch.sqrt(betas[time]) * z_scaled

            if clip_sampler:
                current_x = torch.clamp(current_x, -1, 1)

        # Repeat last step if specified
        for _ in range(repeat_last_step):
            t = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            eps_pred = model(observations, current_x, t, training=False)

            alpha_1 = 1 / torch.sqrt(alphas[0])
            alpha_2 = (1 - alphas[0]) / torch.sqrt(1 - alpha_hats[0])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            if clip_sampler:
                current_x = torch.clamp(current_x, -1, 1)

    return torch.clamp(current_x, -1, 1)