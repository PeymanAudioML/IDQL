#!/usr/bin/env python3
"""
Example Usage of Converted IDQL_pytorch Modules

This demonstrates how to use the converted PyTorch modules that are currently available.
Full training pipeline requires agent learners to be converted.
"""

import torch
import numpy as np
from jaxrl5.networks import MLP, DDPM, StateValue, StateActionValue, FourierFeatures
from jaxrl5.networks import cosine_beta_schedule, ddpm_sampler
from jaxrl5.distributions import TanhNormal, TanhDeterministic
from jaxrl5.data import Dataset


def example_mlp():
    """Example: Using MLP network"""
    print("\n" + "="*60)
    print("Example 1: MLP Network")
    print("="*60)

    # Create MLP
    mlp = MLP(
        input_dim=17,
        hidden_dims=[256, 256],
        activate_final=False,
        use_layer_norm=True,
        dropout_rate=0.1
    )

    # Forward pass
    batch_size = 32
    obs = torch.randn(batch_size, 17)

    # Training mode (with dropout)
    mlp.train()
    output_train = mlp(obs, training=True)
    print(f"✓ MLP output shape (training): {output_train.shape}")

    # Eval mode (no dropout)
    mlp.eval()
    output_eval = mlp(obs, training=False)
    print(f"✓ MLP output shape (eval): {output_eval.shape}")


def example_value_networks():
    """Example: Using Value and Q networks"""
    print("\n" + "="*60)
    print("Example 2: Value Networks")
    print("="*60)

    obs_dim = 17
    action_dim = 6
    batch_size = 32

    # Create value network
    value_net = StateValue(
        base_cls=MLP,
        input_dim=obs_dim,
        hidden_dims=[256, 256]
    )

    # Create Q-network
    q_net = StateActionValue(
        base_cls=MLP,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256]
    )

    # Forward passes
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)

    values = value_net(obs)
    q_values = q_net(obs, actions)

    print(f"✓ Value network output shape: {values.shape}")
    print(f"✓ Q-network output shape: {q_values.shape}")


def example_ddpm():
    """Example: Using DDPM diffusion model"""
    print("\n" + "="*60)
    print("Example 3: DDPM Diffusion Model")
    print("="*60)

    obs_dim = 17
    action_dim = 6
    batch_size = 32
    T = 5  # diffusion timesteps

    # Create DDPM model
    ddpm = DDPM(
        cond_encoder_cls=MLP,
        reverse_encoder_cls=MLP,
        time_preprocess_cls=FourierFeatures,
        obs_dim=obs_dim,
        action_dim=action_dim,
        time_embed_size=32,
        cond_hidden_dims=[256, 256],
        reverse_hidden_dims=[256, 256, action_dim]
    )

    # Forward pass (predicting noise)
    obs = torch.randn(batch_size, obs_dim)
    noisy_actions = torch.randn(batch_size, action_dim)
    timesteps = torch.randint(0, T, (batch_size, 1), dtype=torch.float32)

    predicted_noise = ddpm(obs, noisy_actions, timesteps, training=False)
    print(f"✓ DDPM predicted noise shape: {predicted_noise.shape}")

    # Sampling from diffusion model
    print("\n  Sampling actions from diffusion model...")
    betas = cosine_beta_schedule(T)
    alphas = 1 - betas
    alpha_hats = torch.cumprod(alphas, dim=0)

    ddpm.eval()
    with torch.no_grad():
        sampled_actions = ddpm_sampler(
            model=ddpm,
            T=T,
            act_dim=action_dim,
            observations=obs,
            alphas=alphas,
            alpha_hats=alpha_hats,
            betas=betas,
            sample_temperature=1.0,
            clip_sampler=True,
            device='cpu'
        )

    print(f"✓ Sampled actions shape: {sampled_actions.shape}")
    print(f"  Action range: [{sampled_actions.min():.3f}, {sampled_actions.max():.3f}]")


def example_distributions():
    """Example: Using policy distributions"""
    print("\n" + "="*60)
    print("Example 4: Policy Distributions")
    print("="*60)

    obs_dim = 17
    action_dim = 6
    batch_size = 32

    # Create stochastic policy (TanhNormal)
    stochastic_policy = TanhNormal(
        base_cls=MLP,
        action_dim=action_dim,
        input_dim=obs_dim,
        hidden_dims=[256, 256],
        state_dependent_std=True
    )

    # Create deterministic policy
    deterministic_policy = TanhDeterministic(
        base_cls=MLP,
        action_dim=action_dim,
        input_dim=obs_dim,
        hidden_dims=[256, 256]
    )

    # Sample actions
    obs = torch.randn(batch_size, obs_dim)

    # Stochastic policy
    stochastic_policy.eval()
    with torch.no_grad():
        stochastic_actions = stochastic_policy.get_action(obs, deterministic=False)
        deterministic_from_stochastic = stochastic_policy.get_action(obs, deterministic=True)

    # Deterministic policy
    deterministic_policy.eval()
    with torch.no_grad():
        deterministic_actions = deterministic_policy(obs)

    print(f"✓ Stochastic actions shape: {stochastic_actions.shape}")
    print(f"  Range: [{stochastic_actions.min():.3f}, {stochastic_actions.max():.3f}]")
    print(f"✓ Deterministic actions shape: {deterministic_actions.shape}")
    print(f"  Range: [{deterministic_actions.min():.3f}, {deterministic_actions.max():.3f}]")


def example_dataset():
    """Example: Using Dataset class"""
    print("\n" + "="*60)
    print("Example 5: Dataset Handling")
    print("="*60)

    # Create dummy dataset
    n_samples = 1000
    obs_dim = 17
    action_dim = 6

    dataset_dict = {
        'observations': np.random.randn(n_samples, obs_dim).astype(np.float32),
        'actions': np.random.randn(n_samples, action_dim).astype(np.float32),
        'rewards': np.random.randn(n_samples).astype(np.float32),
        'dones': np.random.randint(0, 2, n_samples).astype(np.float32),
        'next_observations': np.random.randn(n_samples, obs_dim).astype(np.float32)
    }

    dataset = Dataset(dataset_dict, seed=42)
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Sample a batch
    batch_size = 32
    batch = dataset.sample(batch_size)
    print(f"✓ Sampled batch with keys: {list(batch.keys())}")
    print(f"  Observations shape: {batch['observations'].shape}")

    # Sample as PyTorch tensors
    torch_batch = dataset.sample_torch(batch_size, device='cpu')
    print(f"✓ Sampled PyTorch batch")
    print(f"  Observations type: {type(torch_batch['observations'])}")
    print(f"  Observations device: {torch_batch['observations'].device}")

    # Split dataset
    train_dataset, val_dataset = dataset.split(ratio=0.8)
    print(f"✓ Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")


def example_training_loop():
    """Example: Minimal training loop structure"""
    print("\n" + "="*60)
    print("Example 6: Minimal Training Loop (Structure Only)")
    print("="*60)

    # Setup
    obs_dim = 17
    action_dim = 6
    batch_size = 32

    # Create networks
    q_net = StateActionValue(
        base_cls=MLP,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256]
    )

    # Create optimizer
    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)

    # Create dummy dataset
    dataset_dict = {
        'observations': np.random.randn(1000, obs_dim).astype(np.float32),
        'actions': np.random.randn(1000, action_dim).astype(np.float32),
        'rewards': np.random.randn(1000).astype(np.float32),
        'next_observations': np.random.randn(1000, obs_dim).astype(np.float32),
        'dones': np.random.randint(0, 2, 1000).astype(np.float32)
    }
    dataset = Dataset(dataset_dict, seed=42)

    print("✓ Training setup complete")
    print(f"  Network parameters: {sum(p.numel() for p in q_net.parameters()):,}")

    # Training loop (just 10 steps as demo)
    q_net.train()
    for step in range(10):
        # Sample batch
        batch = dataset.sample_torch(batch_size, device='cpu')

        # Forward pass
        q_values = q_net(batch['observations'], batch['actions'])

        # Dummy loss (replace with actual RL loss)
        target_q = batch['rewards']  # Simplified
        loss = torch.nn.functional.mse_loss(q_values, target_q)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    print("✓ Training loop completed successfully")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("IDQL_pytorch - Example Usage of Converted Modules")
    print("="*60)
    print("\nThis script demonstrates the converted PyTorch modules.")
    print("Full training requires agent learners to be converted.\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run examples
    example_mlp()
    example_value_networks()
    example_ddpm()
    example_distributions()
    example_dataset()
    example_training_loop()

    print("\n" + "="*60)
    print("All Examples Completed Successfully!")
    print("="*60)
    print("\nNext Steps:")
    print("  • Review HOW_TO_RUN.md for full training pipeline info")
    print("  • Check CONVERSION_NOTES.md for conversion details")
    print("  • Convert agent learners to enable full training")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()