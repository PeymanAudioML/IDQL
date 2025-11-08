from typing import Optional
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, constraints
from torch.distributions.transforms import TanhTransform, ComposeTransform, AffineTransform

class TanhTransformedDistribution(Distribution):
    """A distribution that applies a tanh transformation to another distribution."""

    def __init__(self, base_distribution: Distribution, validate_args: bool = False):
        self.base_distribution = base_distribution
        self.transforms = [TanhTransform()]
        super().__init__(batch_shape=base_distribution.batch_shape,
                         event_shape=base_distribution.event_shape,
                         validate_args=validate_args)

    def mode(self) -> torch.Tensor:
        """Return the mode of the distribution after tanh transformation."""
        # For normal distribution, mode is the mean
        if hasattr(self.base_distribution, 'mean'):
            base_mode = self.base_distribution.mean
        elif hasattr(self.base_distribution, 'mode'):
            base_mode = self.base_distribution.mode
        else:
            base_mode = self.base_distribution.loc

        # Apply tanh transformation
        return torch.tanh(base_mode)

    def rsample(self, sample_shape=torch.Size()):
        """Reparameterized sampling."""
        x = self.base_distribution.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def sample(self, sample_shape=torch.Size()):
        """Sampling."""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        """Compute log probability of a value."""
        # Apply inverse transform and compute log determinant
        x = value
        log_det_jacobian = 0

        # For tanh, we need to compute arctanh and the log determinant
        eps = 1e-6
        x_clipped = torch.clamp(x, -1 + eps, 1 - eps)
        atanh_x = torch.atanh(x_clipped)

        # Log determinant of tanh transform: log(1 - tanh^2(y))
        log_det_jacobian = torch.log(1 - x_clipped**2 + eps).sum(dim=-1)

        # Get log prob from base distribution
        log_prob = self.base_distribution.log_prob(atanh_x)

        # Adjust for the transformation
        return log_prob - log_det_jacobian

    @property
    def mean(self):
        """Approximate mean via sampling."""
        return torch.tanh(self.base_distribution.mean)

    @property
    def variance(self):
        """Approximate variance via sampling."""
        # This is an approximation
        return self.base_distribution.variance