"""
Test script for PyTorch IDQL implementation
Verifies that all components work correctly
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_diffusion_networks():
    """Test diffusion model components"""
    print("Testing Diffusion Networks...")
    
    from networks.diffusion import (
        DDPM, DDPMSampler, FourierFeatures,
        cosine_beta_schedule, linear_beta_schedule
    )
    
    # Test Fourier features
    ff = FourierFeatures(output_size=64)
    time = torch.randn(32, 1)
    time_emb = ff(time)
    assert time_emb.shape == (32, 64), f"Fourier features shape mismatch: {time_emb.shape}"
    
    # Test DDPM
    ddpm = DDPM(
        observation_dim=11,
        action_dim=3,
        time_dim=64,
        hidden_dims=(256, 256)
    )
    
    obs = torch.randn(32, 11)
    actions = torch.randn(32, 3)
    timesteps = torch.randint(0, 100, (32,))
    
    pred_noise = ddpm(obs, actions, timesteps)
    assert pred_noise.shape == (32, 3), f"DDPM output shape mismatch: {pred_noise.shape}"
    
    # Test sampler
    betas = cosine_beta_schedule(100)
    sampler = DDPMSampler(ddpm, betas, device='cpu')
    
    samples = sampler.sample(obs, num_samples=1)
    assert samples.shape == (32, 3), f"Sampler output shape mismatch: {samples.shape}"
    
    print("✓ Diffusion networks test passed")


def test_value_networks():
    """Test value network components"""
    print("Testing Value Networks...")
    
    from networks.value_net import (
        DoubleCritic, ValueFunction, MLP, Ensemble
    )
    
    obs_dim = 11
    act_dim = 3
    batch_size = 32
    
    # Test DoubleCritic
    critic = DoubleCritic(obs_dim, act_dim)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    
    q_values = critic(obs, actions, return_both=True)
    assert q_values.shape == (2, batch_size, 1), f"DoubleCritic shape mismatch: {q_values.shape}"
    
    q_min = critic.q_min(obs, actions)
    assert q_min.shape == (batch_size, 1), f"Q-min shape mismatch: {q_min.shape}"
    
    # Test ValueFunction
    value = ValueFunction(obs_dim)
    v_values = value(obs)
    assert v_values.shape == (batch_size, 1), f"Value function shape mismatch: {v_values.shape}"
    
    # Test Ensemble
    ensemble = Ensemble(MLP, num_networks=3, input_dim=obs_dim, output_dim=1)
    ensemble_out = ensemble(obs)
    assert ensemble_out.shape == (3, batch_size, 1), f"Ensemble shape mismatch: {ensemble_out.shape}"
    
    print("✓ Value networks test passed")


def test_ddpm_iql_learner():
    """Test DDPM-IQL learner"""
    print("Testing DDPM-IQL Learner...")
    
    from agents.ddpm_iql import DDPMIQLLearner, DDPMIQLConfig
    
    # Create config
    config = DDPMIQLConfig(
        observation_dim=11,
        action_dim=3,
        batch_size=32,
        num_timesteps=100,
        device='cpu'  # Use CPU for testing
    )
    
    # Create learner
    learner = DDPMIQLLearner(config)
    
    # Create fake batch
    batch = {
        'observations': torch.randn(32, 11),
        'actions': torch.randn(32, 3),
        'rewards': torch.randn(32, 1),
        'next_observations': torch.randn(32, 11),
        'dones': torch.zeros(32, 1)
    }
    
    # Test update
    metrics = learner.update(batch)
    assert 'value_loss' in metrics
    assert 'critic_loss' in metrics
    assert 'actor_loss' in metrics
    
    # Test action sampling
    obs = torch.randn(1, 11)
    action = learner.sample_actions(obs, num_samples=1)
    assert action.shape == (1, 3), f"Action shape mismatch: {action.shape}"
    
    # Test single action
    obs_single = np.random.randn(11)
    action_single = learner.get_action(obs_single)
    assert action_single.shape == (3,), f"Single action shape mismatch: {action_single.shape}"
    
    print("✓ DDPM-IQL learner test passed")


def test_dataset():
    """Test dataset utilities"""
    print("Testing Dataset Utilities...")
    
    from data.dataset import (
        OfflineRLDataset, ReplayBuffer, create_dataloader
    )
    
    # Create fake data
    n_samples = 1000
    obs_dim = 11
    act_dim = 3
    
    dataset = OfflineRLDataset(
        observations=np.random.randn(n_samples, obs_dim),
        actions=np.random.randn(n_samples, act_dim),
        rewards=np.random.randn(n_samples),
        next_observations=np.random.randn(n_samples, obs_dim),
        dones=np.zeros(n_samples),
        normalize_obs=True
    )
    
    assert len(dataset) == n_samples
    
    # Test batch sampling
    batch = dataset.get_batch(32)
    assert batch['observations'].shape == (32, obs_dim)
    assert batch['actions'].shape == (32, act_dim)
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=32)
    batch = next(iter(dataloader))
    assert batch['observations'].shape == (32, obs_dim)
    
    # Test replay buffer
    buffer = ReplayBuffer(capacity=100, observation_dim=obs_dim, action_dim=act_dim)
    
    for _ in range(50):
        buffer.add(
            observation=np.random.randn(obs_dim),
            action=np.random.randn(act_dim),
            reward=np.random.randn(),
            next_observation=np.random.randn(obs_dim),
            done=False
        )
    
    batch = buffer.sample(16)
    assert batch['observations'].shape == (16, obs_dim)
    
    print("✓ Dataset utilities test passed")


def test_iql_wrapper():
    """Test IQL wrapper for Diffuser integration"""
    print("Testing IQL Wrapper...")
    
    # Change to diffuser directory
    original_dir = os.getcwd()
    diffuser_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(diffuser_dir)
    
    from diffuser.utils.iql import IQLWrapper
    
    # Create dummy environment
    class DummyEnv:
        class Space:
            def __init__(self, shape):
                self.shape = shape
            def sample(self):
                return np.random.randn(*self.shape)
        
        def __init__(self):
            self.observation_space = self.Space((11,))
            self.action_space = self.Space((3,))
    
    env = DummyEnv()
    
    # Create wrapper with dummy model
    wrapper = IQLWrapper.__new__(IQLWrapper)
    wrapper.device = 'cpu'
    wrapper.observation_dim = 11
    wrapper.action_dim = 3
    
    from networks.value_net import DoubleCritic
    wrapper.model = DoubleCritic(11, 3)
    wrapper.model.eval()
    
    # Test forward pass
    obs = np.random.randn(32, 11)
    actions = np.random.randn(32, 3)
    
    q_values = wrapper.forward(obs, actions)
    assert q_values.shape == (32,), f"Wrapper Q-values shape mismatch: {q_values.shape}"
    
    # Test single sample
    obs_single = np.random.randn(11)
    action_single = np.random.randn(3)
    q_single = wrapper(obs_single, action_single)
    assert isinstance(q_single, (float, np.floating)), f"Single Q-value type mismatch: {type(q_single)}"
    
    os.chdir(original_dir)
    print("✓ IQL wrapper test passed")


def test_soft_update():
    """Test soft update functionality"""
    print("Testing Soft Update...")
    
    from agents.ddpm_iql.ddpm_iql_learner import soft_update
    from networks.value_net import MLP
    
    # Create two networks
    source = MLP(10, (64, 64), 1)
    target = MLP(10, (64, 64), 1)
    
    # Initialize target differently
    for param in target.parameters():
        param.data.fill_(0)
    
    # Soft update
    tau = 0.1
    soft_update(target, source, tau)
    
    # Check that parameters are updated
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        expected = tau * source_param.data + (1 - tau) * 0
        assert torch.allclose(target_param.data, expected, atol=1e-6)
    
    print("✓ Soft update test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running PyTorch IDQL Implementation Tests")
    print("=" * 50)
    
    try:
        test_diffusion_networks()
        test_value_networks()
        test_ddpm_iql_learner()
        test_dataset()
        test_iql_wrapper()
        test_soft_update()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
