from typing import Any, Dict, Union

import torch
import numpy as np

DataType = Union[np.ndarray, torch.Tensor, Dict[str, "DataType"]]
PRNGKey = Any  # For PyTorch, we'll use torch.Generator or int seeds
Params = Dict[str, Any]  # In PyTorch, parameters are typically dict or OrderedDict