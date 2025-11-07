"""
Offline training script for DDPM-IQL
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import gym
from tqdm import tqdm
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ddpm_iql import DDPMIQLLearner, DDPMIQLConfig
from data.dataset import D4RLDataset, create_dataloader


def evaluate_policy(
    agent: DDPMIQLLearner,
    env: gym.Env,
    num_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, float]:
    """Evaluate policy in environment"""
    returns = []
    lengths = []
    
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            action = agent.get_action(observation, deterministic=deterministic)
            observation, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    return {
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
    }


def train_offline(
    env_name: str,
    seed: int = 42,
    batch_size: int = 256,
    max_steps: int = 1000000,
    eval_freq: int = 5000,
    save_freq: int = 10000,
    normalize_obs: bool = True,
    normalize_actions: bool = False,
    log_wandb: bool = False,
    save_dir: str = './checkpoints',
    **config_kwargs
):
    """
    Train DDPM-IQL on offline dataset
    
    Args:
        env_name: D4RL environment name
        seed: random seed
        batch_size: training batch size
        max_steps: maximum training steps
        eval_freq: evaluation frequency
        save_freq: checkpoint save frequency
        normalize_obs: normalize observations
        normalize_actions: normalize actions
        log_wandb: log to Weights & Biases
        save_dir: directory to save checkpoints
        **config_kwargs: additional config parameters
    """
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Load dataset
    print(f"Loading dataset for {env_name}...")
    dataset = D4RLDataset.load_d4rl(
        env_name,
        normalize_obs=normalize_obs,
        normalize_actions=normalize_actions
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Create agent
    config = DDPMIQLConfig(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        batch_size=batch_size,
        **config_kwargs
    )
    
    agent = DDPMIQLLearner(config)
    
    # Setup logging
    if log_wandb:
        wandb.init(
            project='ddpm-iql',
            name=f'{env_name}_seed{seed}',
            config={
                'env_name': env_name,
                'seed': seed,
                'batch_size': batch_size,
                'max_steps': max_steps,
                **config_kwargs
            }
        )
    
    # Create save directory
    save_path = Path(save_dir) / env_name / f'seed_{seed}'
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    pbar = tqdm(total=max_steps)
    
    data_iter = iter(dataloader)
    best_return = -float('inf')
    
    for step in range(max_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Normalize observations if needed
        if dataset.obs_mean is not None:
            batch['observations'] = (batch['observations'] - torch.FloatTensor(dataset.obs_mean)) / torch.FloatTensor(dataset.obs_std)
            batch['next_observations'] = (batch['next_observations'] - torch.FloatTensor(dataset.obs_mean)) / torch.FloatTensor(dataset.obs_std)
        
        # Update agent
        metrics = agent.update(batch)
        
        # Log metrics
        if log_wandb:
            wandb.log(metrics, step=step)
        
        # Evaluate
        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate_policy(agent, eval_env, num_episodes=10)
            
            print(f"\nStep {step + 1}: Return = {eval_metrics['return_mean']:.2f} ± {eval_metrics['return_std']:.2f}")
            
            if log_wandb:
                wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=step)
            
            # Save best model
            if eval_metrics['return_mean'] > best_return:
                best_return = eval_metrics['return_mean']
                agent.save(save_path / 'best_model.pt')
        
        # Save checkpoint
        if (step + 1) % save_freq == 0:
            agent.save(save_path / f'checkpoint_{step + 1}.pt')
        
        pbar.update(1)
        pbar.set_description(f"Loss: {metrics.get('actor_loss', 0):.4f}")
    
    pbar.close()
    
    # Final evaluation
    final_metrics = evaluate_policy(agent, eval_env, num_episodes=100)
    print(f"\nFinal evaluation: Return = {final_metrics['return_mean']:.2f} ± {final_metrics['return_std']:.2f}")
    
    if log_wandb:
        wandb.log({f'final/{k}': v for k, v in final_metrics.items()})
        wandb.finish()
    
    # Save final model
    agent.save(save_path / 'final_model.pt')
    
    return agent, final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--env_name', type=str, default='hopper-medium-v2',
                       help='D4RL environment name')
    
    # Training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=10000)
    
    # Data preprocessing
    parser.add_argument('--normalize_obs', action='store_true', default=True)
    parser.add_argument('--normalize_actions', action='store_true', default=False)
    
    # Model parameters
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--value_lr', type=float, default=3e-4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--expectile', type=float, default=0.7)
    parser.add_argument('--temperature', type=float, default=3.0)
    
    # Diffusion parameters
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                       choices=['cosine', 'linear', 'vp'])
    parser.add_argument('--ddpm_temperature', type=float, default=1.0)
    
    # Logging
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Train
    train_offline(
        env_name=args.env_name,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        normalize_obs=args.normalize_obs,
        normalize_actions=args.normalize_actions,
        log_wandb=args.log_wandb,
        save_dir=args.save_dir,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        ddpm_temperature=args.ddpm_temperature
    )
