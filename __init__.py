"""
IDQL: Implicit Q-Learning with Diffusion Policies
PyTorch implementation (JAX-independent)
"""

from .agents.ddpm_iql import DDPMIQLLearner, DDPMIQLConfig
from .networks.diffusion import DDPM, DDPMSampler
from .networks.value_net import DoubleCritic, ValueFunction
from .data.dataset import OfflineRLDataset, D4RLDataset

__version__ = '1.0.0'

__all__ = [
    'DDPMIQLLearner',
    'DDPMIQLConfig',
    'DDPM',
    'DDPMSampler',
    'DoubleCritic',
    'ValueFunction',
    'OfflineRLDataset',
    'D4RLDataset'
]
