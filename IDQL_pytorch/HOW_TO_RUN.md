# How to Run IDQL_pytorch

## ⚠️ Current Status

The IDQL_pytorch package is **partially converted** from JAX to PyTorch. Here's what's available:

### ✅ **Converted & Ready**
- Core infrastructure (types, data structures)
- Neural network modules (MLP, DDPM, State-Value, State-Action-Value)
- Distribution modules (TanhNormal, TanhTransformed)
- Dataset handling
- Environment wrappers

### ❌ **Not Yet Converted - Cannot Run**
- **Agent Learners** (IQLLearner, DDPMIQLLearner, BCLearner) - **REQUIRED FOR TRAINING**
- Training loop scripts (train_diffusion_offline.py, etc.)
- Evaluation functions
- D4RL dataset loaders
- Replay buffers

## Why Can't It Run Yet?

The launcher scripts try to import and use components that haven't been converted:

```python
# launcher/examples/train_ddpm_iql_offline.py
from examples.states.train_diffusion_offline import call_main  # ❌ Not converted

# Original train_diffusion_offline.py uses:
from jaxrl5.agents import DDPMIQLLearner  # ❌ Not converted from JAX
from jaxrl5.data.d4rl_datasets import D4RLDataset  # ❌ Not converted
from jaxrl5.evaluation import evaluate  # ❌ Not converted
```

## Entry Points (Currently Non-Functional)

The intended entry points are:

### 1. **DDPM-IQL Offline Training**
```bash
# ❌ Does not work yet
python launcher/examples/train_ddpm_iql_offline.py --variant 0
```

**What it needs:**
- Environment: D4RL MuJoCo environments (walker2d, halfcheetah, hopper, antmaze)
- Agent: DDPMIQLLearner (not converted)
- Training: Offline RL from pre-collected datasets

### 2. **DDPM-IQL Finetuning**
```bash
# ❌ Does not work yet
python launcher/examples/train_ddpm_iql_finetune.py --variant 0
```

**What it needs:**
- Pre-trained DDPM-IQL agent
- Online finetuning capability

### 3. **IQL Training**
```bash
# ❌ Does not work yet
python launcher/iql/train_iql_oneparam.py --variant 0
```

**What it needs:**
- IQLLearner agent (not converted)

## What You Can Use Right Now

While the full training pipeline isn't ready, you can use the converted modules:

### Example: Using Network Modules

```python
import torch
from jaxrl5.networks import MLP, DDPM, StateValue, FourierFeatures
from jaxrl5.distributions import TanhNormal

# Create an MLP network
mlp = MLP(
    input_dim=17,  # observation dimension
    hidden_dims=[256, 256],
    activate_final=False
)

# Create a value network
value_net = StateValue(
    base_cls=MLP,
    input_dim=17,
    hidden_dims=[256, 256]
)

# Forward pass
obs = torch.randn(32, 17)  # batch of observations
value = value_net(obs)
print(f"Value shape: {value.shape}")  # [32]

# Create DDPM diffusion model
ddpm = DDPM(
    cond_encoder_cls=MLP,
    reverse_encoder_cls=MLP,
    time_preprocess_cls=FourierFeatures,
    obs_dim=17,
    action_dim=6,
    time_embed_size=32,
    cond_hidden_dims=[256, 256],
    reverse_hidden_dims=[256, 256]
)

# Sample from diffusion model
from jaxrl5.networks import cosine_beta_schedule, ddpm_sampler
T = 5
betas = cosine_beta_schedule(T)
alphas = 1 - betas
alpha_hats = torch.cumprod(alphas, dim=0)

actions = ddpm_sampler(
    model=ddpm,
    T=T,
    act_dim=6,
    observations=obs,
    alphas=alphas,
    alpha_hats=alpha_hats,
    betas=betas,
    sample_temperature=1.0,
    clip_sampler=True,
    device='cpu'
)
print(f"Sampled actions: {actions.shape}")  # [32, 6]
```

### Example: Using Dataset

```python
import numpy as np
from jaxrl5.data import Dataset

# Create a dummy dataset
dataset_dict = {
    'observations': np.random.randn(1000, 17),
    'actions': np.random.randn(1000, 6),
    'rewards': np.random.randn(1000),
    'dones': np.random.randint(0, 2, 1000),
    'next_observations': np.random.randn(1000, 17)
}

dataset = Dataset(dataset_dict, seed=42)

# Sample a batch
batch = dataset.sample(batch_size=32)
print(f"Batch keys: {batch.keys()}")

# Sample as PyTorch tensors
torch_batch = dataset.sample_torch(batch_size=32, device='cpu')
print(f"Observations shape: {torch_batch['observations'].shape}")
```

## What's Needed to Make It Fully Functional

To get the training scripts working, you need to convert:

### 1. **Agent Learners** (Priority: HIGH)

**Files to convert:**
- `IDQL/jaxrl5/agents/iql/iql_learner.py` → PyTorch
- `IDQL/jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py` → PyTorch
- `IDQL/jaxrl5/agents/bc/bc_learner.py` → PyTorch

**Key conversions needed:**
- JAX's `@jax.jit` → PyTorch (no decorator needed, or use `@torch.jit.script`)
- JAX's `vmap` → `torch.vmap` or manual batching
- Optax optimizers → PyTorch optimizers (Adam, AdamW, etc.)
- Flax TrainState → Custom PyTorch training state
- JAX random number generation → PyTorch generators

### 2. **Training Scripts** (Priority: HIGH)

**Files to convert:**
- `IDQL/examples/states/train_diffusion_offline.py`
- `IDQL/examples/states/train_offline.py`
- `IDQL/examples/states/train_online.py`

**Key conversions:**
- Replace JAX-specific code with PyTorch
- Update data loading to use PyTorch DataLoader (optional)
- Fix device handling (CPU/CUDA)

### 3. **Data Loaders** (Priority: MEDIUM)

**Files to convert:**
- `IDQL/jaxrl5/data/d4rl_datasets.py`
- `IDQL/jaxrl5/data/replay_buffer.py`
- `IDQL/jaxrl5/data/binary_datasets.py`

### 4. **Evaluation Functions** (Priority: MEDIUM)

**Files to convert:**
- `IDQL/jaxrl5/evaluation.py`

## Recommended Next Steps

### Option 1: Complete the Conversion
1. Convert the agent learners (IQL, DDPM-IQL)
2. Convert the training scripts
3. Test on a simple environment

### Option 2: Create a Minimal Example
Create a standalone PyTorch training script using the converted modules:

```python
# minimal_train.py
import torch
from jaxrl5.networks import DDPM, MLP, StateValue
from jaxrl5.data import Dataset
# ... implement minimal training loop
```

### Option 3: Use Existing PyTorch RL Libraries
Integrate the converted DDPM and network modules with existing PyTorch RL frameworks:
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://docs.cleanrl.dev/)
- [TorchRL](https://pytorch.org/rl/)

## Architecture Overview

```
Training Flow (When Complete):
┌─────────────────────────────────────────┐
│  launcher/examples/                     │
│    - train_ddpm_iql_offline.py         │
│    - train_ddpm_iql_finetune.py        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  examples/states/                       │
│    - train_diffusion_offline.py        │  ❌ Not converted
│      (main training loop)               │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  jaxrl5/agents/                         │
│    - DDPMIQLLearner                     │  ❌ Not converted
│    - IQLLearner                         │  ❌ Not converted
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  jaxrl5/networks/  ✅ CONVERTED         │
│    - DDPM (diffusion model)             │
│    - MLP, StateValue, etc.              │
└─────────────────────────────────────────┘
```

## Support & Resources

- **CONVERSION_NOTES.md** - Detailed conversion information
- **SETUP_GUIDE.md** - Environment setup guide
- **.vscode/README.md** - VSCode configuration help

## Summary

**Current State:** 🟡 Partially Functional
- ✅ Core modules converted and usable
- ❌ Training pipeline not yet functional
- 🔧 Agent learners need conversion

**To Run Full Training:** Convert agent learners and training scripts from JAX to PyTorch (significant work required)

**Quick Start for Development:** Use the converted network modules in custom PyTorch code