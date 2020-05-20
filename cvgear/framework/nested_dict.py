from collections import OrderedDict
import copy
import numpy
import torch

"""
`state_dict` or `nested_dict`?

    When `torch.nn.Module` loads from state_dict, the keys in state_dict
    contains prefix and parameter/buffer name. The prefix determines the 
    specific module to load the parameter/buffer. Meanwhile the parameter/buffer
    name prevents mismatch among parameters/buffers.

    With nested dict, we don't need to care about the specific module 
    name(prefix) or the complicated structure(e.g. model.stem.conv1).
    We ONLY care the order, the order of the module determines which module to 
    load.

Use cases:
    1. When some of submodule names of `torch.nn.Module` have been changed,
        or some submodules are wrapped into one big abstract module,
        use TorchNestedLoader to save/load between previous and new module

    2. Convert between darknet weights and torch weights
    .cfg ------> DarknetParser                           torch.nn.Module
                        |                                         |
    .weights <-> DarknetNestedLoader <-> nested_dict <-> TorchNestedLoader <-> state_dict <-> .pth file

"""

__all__ = ["nested_dict_counter", "nested_dict_tensor2ndarray", "nested_dict_ndarray2tensor"]

def nested_dict_counter(nested_dict: OrderedDict) -> int:
    """
    Count total number of parameters of `nested_dict`
    """
    counter = 0
    for state_dict_block in nested_dict.values():
        for params in state_dict_block.values():
            if isinstance(params, torch.Tensor):
                counter += params.numel()
            elif isinstance(params, numpy.ndarray):
                counter += params.size
            else:
                raise TypeError(
                "params inside nested_dict should be "
                "torch.Tensor or numpy.ndarry"
                )
    return counter


def nested_dict_tensor2ndarray(nested_dict: OrderedDict) -> OrderedDict:
    """
    Convert OrderedDict[str, OrderedDict[str, torch.Tensor]] 
    to OrderedDict[str, OrderedDict[str, numpy.ndarray]]
    """
    nested_dict = copy.copy(nested_dict)
    for state_dict_block in nested_dict.values():
        for name, params in state_dict_block.items():
            assert isinstance(params, torch.Tensor)
            state_dict_block[name] = params.numpy()
    return nested_dict


def nested_dict_ndarray2tensor(nested_dict: OrderedDict) -> OrderedDict:
    """
    Convert OrderedDict[str, OrderedDict[str, numpy.ndarray]] 
    to OrderedDict[str, OrderedDict[str, torch.Tensor]]
    """
    nested_dict = copy.copy(nested_dict)
    for state_dict_block in nested_dict.values():
        for name, params in state_dict_block.items():
            assert isinstance(params, numpy.ndarray)
            state_dict_block[name] = torch.from_numpy(params)
    return nested_dict