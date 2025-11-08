from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight_decay_mask(model):
    """Returns a dict indicating which parameters should have weight decay applied."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if 'bias' in name or 'Input' in name or 'Output' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return {'decay': decay_params, 'no_decay': no_decay_params}

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        activate_final: bool = False,
        use_layer_norm: bool = False,
        scale_final: Optional[float] = None,
        dropout_rate: Optional[float] = None
    ):
        super().__init__()
        self.activations = activations
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        for i in range(len(hidden_dims)):
            layer = nn.Linear(dims[i], dims[i+1])
            # Initialize weights
            if i + 1 == len(hidden_dims) and scale_final is not None:
                nn.init.xavier_uniform_(layer.weight, gain=scale_final)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.use_layer_norm:
            x = self.layer_norm(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i + 1 < len(self.layers) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = self.dropout(x) if training else x
                x = self.activations(x)

        return x