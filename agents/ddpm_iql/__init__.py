"""
DDPM-IQL: Implicit Q-Learning with Diffusion Policies
PyTorch implementation
"""

from .ddpm_iql_learner import DDPMIQLLearner, DDPMIQLConfig

__all__ = [
    'DDPMIQLLearner',
    'DDPMIQLConfig'
]
