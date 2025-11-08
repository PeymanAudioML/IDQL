from typing import Type, Optional
import torch
import torch.nn as nn
import numpy as np

class Ensemble(nn.Module):
    def __init__(self, net_cls: Type[nn.Module], num: int = 2, *args, **kwargs):
        super().__init__()
        self.num = num
        self.nets = nn.ModuleList([net_cls(*args, **kwargs) for _ in range(num)])

    def forward(self, *args, **kwargs):
        outputs = []
        for net in self.nets:
            outputs.append(net(*args, **kwargs))
        # Stack outputs along the first dimension (ensemble dimension)
        return torch.stack(outputs, dim=0)

def subsample_ensemble(rng: Optional[torch.Generator], params: dict, num_sample: int, num_qs: int):
    """Subsample ensemble parameters.

    Args:
        rng: Random number generator (torch.Generator or None for numpy)
        params: Model parameters dictionary
        num_sample: Number of ensemble members to sample
        num_qs: Total number of ensemble members

    Returns:
        Subsampled parameters
    """
    if num_sample is not None and num_sample < num_qs:
        all_indx = np.arange(0, num_qs)

        if rng is not None:
            # Use torch for random sampling
            indx = torch.randperm(num_qs, generator=rng)[:num_sample].numpy()
        else:
            # Use numpy for random sampling
            indx = np.random.choice(all_indx, size=num_sample, replace=False)

        # Subsample the ensemble parameters
        if isinstance(params, dict):
            new_params = {}
            for key, value in params.items():
                if isinstance(value, torch.nn.Module):
                    # If it's an ensemble module, subsample its internal nets
                    if hasattr(value, 'nets'):
                        sampled_nets = nn.ModuleList([value.nets[i] for i in indx])
                        new_ensemble = Ensemble.__new__(Ensemble)
                        nn.Module.__init__(new_ensemble)
                        new_ensemble.num = num_sample
                        new_ensemble.nets = sampled_nets
                        new_params[key] = new_ensemble
                    else:
                        new_params[key] = value
                elif isinstance(value, (list, tuple)):
                    new_params[key] = [value[i] for i in indx]
                elif isinstance(value, torch.Tensor) and value.shape[0] == num_qs:
                    new_params[key] = value[indx]
                else:
                    new_params[key] = value
            return new_params
        elif isinstance(params, nn.Module) and hasattr(params, 'nets'):
            # Direct ensemble module subsampling
            sampled_nets = nn.ModuleList([params.nets[i] for i in indx])
            new_ensemble = Ensemble.__new__(Ensemble)
            nn.Module.__init__(new_ensemble)
            new_ensemble.num = num_sample
            new_ensemble.nets = sampled_nets
            return new_ensemble

    return params