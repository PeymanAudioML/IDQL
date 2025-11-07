"""
Dataset utilities for offline reinforcement learning
PyTorch implementation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import h5py
import pickle


class OfflineRLDataset(Dataset):
    """Dataset for offline reinforcement learning"""
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        normalize_obs: bool = False,
        normalize_actions: bool = False
    ):
        """
        Initialize offline RL dataset
        
        Args:
            observations: (N, obs_dim)
            actions: (N, action_dim)
            rewards: (N,) or (N, 1)
            next_observations: (N, obs_dim)
            dones: (N,) or (N, 1)
            normalize_obs: whether to normalize observations
            normalize_actions: whether to normalize actions
        """
        self.size = len(observations)
        
        # Ensure correct shapes
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        if dones.ndim == 1:
            dones = dones.reshape(-1, 1)
        
        # Store data
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        
        # Compute normalization statistics if needed
        if normalize_obs:
            self.obs_mean = observations.mean(axis=0)
            self.obs_std = observations.std(axis=0) + 1e-6
            self.observations = (observations - self.obs_mean) / self.obs_std
            self.next_observations = (next_observations - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = None
            self.obs_std = None
        
        if normalize_actions:
            self.action_mean = actions.mean(axis=0)
            self.action_std = actions.std(axis=0) + 1e-6
            self.actions = (actions - self.action_mean) / self.action_std
        else:
            self.action_mean = None
            self.action_std = None
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'observations': torch.FloatTensor(self.observations[idx]),
            'actions': torch.FloatTensor(self.actions[idx]),
            'rewards': torch.FloatTensor(self.rewards[idx]),
            'next_observations': torch.FloatTensor(self.next_observations[idx]),
            'dones': torch.FloatTensor(self.dones[idx])
        }
    
    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using dataset statistics"""
        if self.obs_mean is not None:
            return (obs - self.obs_mean) / self.obs_std
        return obs
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using dataset statistics"""
        if self.action_mean is not None:
            return action * self.action_std + self.action_mean
        return action
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations': torch.FloatTensor(self.observations[idx]),
            'actions': torch.FloatTensor(self.actions[idx]),
            'rewards': torch.FloatTensor(self.rewards[idx]),
            'next_observations': torch.FloatTensor(self.next_observations[idx]),
            'dones': torch.FloatTensor(self.dones[idx])
        }


class D4RLDataset(OfflineRLDataset):
    """Dataset for D4RL offline RL benchmark"""
    
    @staticmethod
    def load_d4rl(env_name: str, **kwargs) -> 'D4RLDataset':
        """
        Load D4RL dataset
        
        Args:
            env_name: D4RL environment name
            **kwargs: additional arguments for OfflineRLDataset
        """
        try:
            import d4rl
            import gym
        except ImportError:
            raise ImportError("Please install d4rl: pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl")
        
        env = gym.make(env_name)
        dataset = env.get_dataset()
        
        # Handle different D4RL dataset formats
        if 'observations' in dataset:
            observations = dataset['observations']
            next_observations = dataset['next_observations']
        else:
            observations = dataset['observations']
            next_observations = np.concatenate([
                dataset['observations'][1:],
                dataset['observations'][-1:] 
            ], axis=0)
        
        return D4RLDataset(
            observations=observations,
            actions=dataset['actions'],
            rewards=dataset['rewards'],
            next_observations=next_observations,
            dones=dataset['terminals'] if 'terminals' in dataset else dataset['dones'],
            **kwargs
        )


class TrajectoryDataset(Dataset):
    """Dataset that maintains trajectory structure"""
    
    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        max_length: Optional[int] = None
    ):
        """
        Initialize trajectory dataset
        
        Args:
            trajectories: list of trajectory dictionaries
            max_length: maximum trajectory length (for padding)
        """
        self.trajectories = trajectories
        self.max_length = max_length or max(len(traj['observations']) for traj in trajectories)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        
        # Pad trajectory if needed
        traj_len = len(traj['observations'])
        
        if traj_len < self.max_length:
            # Pad with zeros
            pad_len = self.max_length - traj_len
            
            padded_traj = {}
            for key, value in traj.items():
                if isinstance(value, np.ndarray):
                    pad_shape = ((0, pad_len),) + ((0, 0),) * (value.ndim - 1)
                    padded_traj[key] = np.pad(value, pad_shape, mode='constant')
                else:
                    padded_traj[key] = value
            
            # Create mask
            mask = np.concatenate([
                np.ones(traj_len, dtype=bool),
                np.zeros(pad_len, dtype=bool)
            ])
            padded_traj['mask'] = mask
            
            traj = padded_traj
        else:
            traj['mask'] = np.ones(traj_len, dtype=bool)
        
        # Convert to tensors
        return {
            key: torch.FloatTensor(value) if isinstance(value, np.ndarray) else value
            for key, value in traj.items()
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """Create a DataLoader for training"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def split_dataset(
    dataset: OfflineRLDataset,
    train_ratio: float = 0.9
) -> Tuple[OfflineRLDataset, OfflineRLDataset]:
    """Split dataset into train and validation sets"""
    n_train = int(len(dataset) * train_ratio)
    indices = np.random.permutation(len(dataset))
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = OfflineRLDataset(
        observations=dataset.observations[train_indices],
        actions=dataset.actions[train_indices],
        rewards=dataset.rewards[train_indices],
        next_observations=dataset.next_observations[train_indices],
        dones=dataset.dones[train_indices]
    )
    
    val_dataset = OfflineRLDataset(
        observations=dataset.observations[val_indices],
        actions=dataset.actions[val_indices],
        rewards=dataset.rewards[val_indices],
        next_observations=dataset.next_observations[val_indices],
        dones=dataset.dones[val_indices]
    )
    
    return train_dataset, val_dataset


class ReplayBuffer:
    """Simple replay buffer for online fine-tuning"""
    
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer"""
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions"""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': torch.FloatTensor(self.observations[idx]),
            'actions': torch.FloatTensor(self.actions[idx]),
            'rewards': torch.FloatTensor(self.rewards[idx]),
            'next_observations': torch.FloatTensor(self.next_observations[idx]),
            'dones': torch.FloatTensor(self.dones[idx])
        }
    
    def __len__(self):
        return self.size
