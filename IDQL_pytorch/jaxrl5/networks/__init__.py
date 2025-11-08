from jaxrl5.networks.ensemble import Ensemble, subsample_ensemble
from jaxrl5.networks.mlp import MLP, get_weight_decay_mask
from jaxrl5.networks.state_action_value import StateActionValue
from jaxrl5.networks.state_value import StateValue
from jaxrl5.networks.diffusion import (
    DDPM, FourierFeatures, cosine_beta_schedule,
    ddpm_sampler, vp_beta_schedule, linear_beta_schedule
)

# Default initialization for PyTorch
import torch.nn as nn
default_init = nn.init.xavier_uniform_

__all__ = [
    'Ensemble', 'subsample_ensemble', 'MLP', 'get_weight_decay_mask',
    'StateActionValue', 'StateValue', 'DDPM', 'FourierFeatures',
    'cosine_beta_schedule', 'ddpm_sampler', 'vp_beta_schedule',
    'linear_beta_schedule', 'default_init'
]