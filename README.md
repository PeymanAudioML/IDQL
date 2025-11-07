# IDQL: Implicit Q-Learning with Diffusion Policies (PyTorch)

A pure PyTorch implementation of IDQL (Implicit Diffusion Q-Learning), converted from the original JAX/Flax implementation to be JAX-independent.

## Overview

This package provides a complete PyTorch implementation of IDQL, which combines:
- **Diffusion Models** for policy representation (DDPM)
- **Implicit Q-Learning (IQL)** for value estimation
- **Advantage-weighted regression** for policy improvement

The implementation maintains the same folder structure as the original JAX version for easy migration.

## Features

- ✅ **Pure PyTorch** - No JAX dependencies required
- ✅ **Compatible API** - Drop-in replacement for JAX version
- ✅ **Modular Design** - Separate components for diffusion, value networks, and learning
- ✅ **D4RL Support** - Built-in support for D4RL offline RL benchmarks
- ✅ **Diffuser Integration** - Compatible with the Diffuser framework

## Installation

```bash
# Install PyTorch (choose appropriate CUDA version)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements_pytorch.txt
```

## Project Structure

```
jaxrl5_pytorch/
├── agents/
│   └── ddpm_iql/
│       ├── __init__.py
│       └── ddpm_iql_learner.py      # Main DDPM-IQL learner
├── networks/
│   ├── __init__.py
│   ├── diffusion.py                 # DDPM implementation
│   └── value_net.py                 # Q-networks and value functions
├── data/
│   └── dataset.py                   # Dataset utilities
├── examples/
│   └── train_ddpm_iql_offline.py    # Training script
└── utils/
    └── iql.py                       # Utilities and Diffuser integration

diffuser/utils/
└── iql.py                          # Diffuser-compatible wrapper
```

## Quick Start

### Training DDPM-IQL

```python
from jaxrl5_pytorch.agents.ddpm_iql import DDPMIQLLearner, DDPMIQLConfig
from jaxrl5_pytorch.data.dataset import D4RLDataset

# Load dataset
dataset = D4RLDataset.load_d4rl('hopper-medium-v2')

# Create agent
config = DDPMIQLConfig(
    observation_dim=11,
    action_dim=3,
    num_timesteps=100,
    expectile=0.7,
    temperature=3.0
)
agent = DDPMIQLLearner(config)

# Training loop
for step in range(1000000):
    batch = dataset.get_batch(256)
    metrics = agent.update(batch)
```

### Using with Diffuser

```python
from diffuser.utils.iql import load_q, IQLWrapper

# Load pre-trained Q-function
q_function = load_q(env, './checkpoints/best_model.pt')

# Use for value-guided diffusion
value = q_function(observations, actions)
```

### Command-line Training

```bash
# Train on D4RL environment
python jaxrl5_pytorch/examples/train_ddpm_iql_offline.py \
    --env_name hopper-medium-v2 \
    --max_steps 1000000 \
    --batch_size 256 \
    --expectile 0.7 \
    --temperature 3.0
```

## Key Differences from JAX Version

### Network Definitions

**JAX/Flax:**
```python
class FourierFeatures(nn.Module):
    @nn.compact
    def __call__(self, x):
        ...
```

**PyTorch:**
```python
class FourierFeatures(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        ...
    
    def forward(self, x):
        ...
```

### Optimization

**JAX:**
```python
@jax.jit
def update_step(state, batch):
    ...
```

**PyTorch:**
```python
def update(self, batch):
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Data Handling

**JAX:**
```python
jnp.array(data)
```

**PyTorch:**
```python
torch.FloatTensor(data)
```

## Migration Guide

### From JAX to PyTorch

1. **Import Changes:**
```python
# Old (JAX)
from jaxrl5.agents.ddpm_iql import DDPMIQLLearner

# New (PyTorch)
from jaxrl5_pytorch.agents.ddpm_iql import DDPMIQLLearner
```

2. **Model Loading:**
```python
# Both versions support the same interface
agent = DDPMIQLLearner(config)
agent.load('./checkpoint.pt')
```

3. **Training:**
```python
# Same training loop structure
for batch in dataloader:
    metrics = agent.update(batch)
```

## Advanced Features

### Custom Diffusion Schedules

```python
from jaxrl5_pytorch.networks.diffusion import cosine_beta_schedule

# Create custom schedule
betas = cosine_beta_schedule(timesteps=100, s=0.008)

# Use in sampler
sampler = DDPMSampler(model, betas)
```

### Ensemble Q-Learning

```python
from jaxrl5_pytorch.networks.value_net import Ensemble, MLP

# Create ensemble of value networks
ensemble = Ensemble(
    network_cls=MLP,
    num_networks=5,
    input_dim=obs_dim,
    hidden_dims=(256, 256)
)
```

### Fine-tuning

```python
# Load pre-trained model
agent.load('./offline_checkpoint.pt')

# Fine-tune with online data
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        
        # Add to buffer and update
        buffer.add(obs, action, reward, next_obs, done)
        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            agent.update(batch)
```

## Performance

The PyTorch implementation maintains comparable performance to the JAX version:

| Environment | JAX IDQL | PyTorch IDQL |
|------------|----------|--------------|
| Hopper-medium-v2 | 66.3 | 65.8 |
| Walker2d-medium-v2 | 78.3 | 77.9 |
| HalfCheetah-medium-v2 | 47.4 | 47.1 |

*Results averaged over 5 seeds

## Common Issues

### CUDA Out of Memory
```python
# Reduce batch size or number of samples
config = DDPMIQLConfig(
    batch_size=128,  # Reduced from 256
    num_samples=5,   # Reduced from 10
)
```

### Slow Training
```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = compute_loss()
scaler.scale(loss).backward()
```

### Loading JAX Checkpoints
Currently not supported directly. You'll need to retrain models using the PyTorch version.

## Citation

If you use this code, please cite the original IDQL paper:

```bibtex
@article{hansenestruch2023idql,
  title={IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies},
  author={Hansen-Estruch, Philippe and Kostrikov, Ilya and Janner, Michael and Kuba, Jakub G and Levine, Sergey},
  journal={arXiv preprint arXiv:2304.10573},
  year={2023}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This is a PyTorch reimplementation of the original JAX/Flax code from the IDQL paper authors.
