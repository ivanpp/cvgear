# --------------------------------------------
# CVGear | Written by ivan Ding(a.k.a ivanpp)
#
# Nested dictionary loader for torch.nn.Module
# --------------------------------------------
import itertools
from collections import OrderedDict, namedtuple
from typing import List
import torch.nn as nn

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _save_to_nested_dict(module: nn.Module, nested_dict: OrderedDict,
                        prefix: str, keep_vars: bool):
    """
    Saves module state to a dictionary(if the module has parameters/buffers to 
    save), then save the dict to an ordered dict(makes it a nested dict).
    
    Note:
        1. This method will save module itself, but not its descendants.
        2. Nothing happens if `module` has no parameter/buffer to save.
    """
    state_dict_block = OrderedDict()
    for name, param in module._parameters.items():
        if param is not None:
            state_dict_block[name] = param if keep_vars else param.data
    for name, buffer in module._buffers.items():
        if buffer is not None:
            state_dict_block[name] = buffer if keep_vars else buffer.data
    if len(state_dict_block) != 0:
        state_dict_block._metadata = dict(version=module._version)
        nested_dict[prefix] = state_dict_block


def nested_dict(module: nn.Module, destination: OrderedDict = None, 
                prefix: str = '', keep_vars: bool = False):
    """
    Returns a nested ordered containing all parameters and registered buffers 
    of `module`.

    Returns:
        OrderedDict: a nested OrderedDict cotaining several state_dict
            blocks, each containing parameters/buffers of one loadable module
    """
    if destination is None:
        destination = OrderedDict()
    _save_to_nested_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            nested_dict(child, destination, prefix + name + '.', keep_vars=keep_vars)
    return destination


def _load_from_nested_dict(module: nn.Module, nested_dict: OrderedDict, prefix: str,
                          missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
    """
    A warpped function of `torch.nn.Module._load_state_dict()`, to load
    from nested dict(`OrderedDict` that contains one state_dict for each loadable 
    module).
    
    Copies parameters and buffers from `nested_dict` into only `module` (if
    necessary), but not its descendents. This is called on every submodule in
    `load_nested_dict()` function. If `module` has no parameter or buffer,
    nothing will happen.

    Args:
        module (torch.nn.Module): module to be loaded
        nested_dict (OrderedDict): an `OrderedDict` containing several dicts,
            each containing parameters and persistent buffers for one loadable
            module.
        prefix (str): the prefix for parameters and buffers used in this module
        missing_keys (list of str): (used in `torch.nn.Module._load_state_dict()`)
            if `strict=True`, add missing keys to this list
        unexpected_keys (list of str): (used in `torch.nn.Module._load_state_dict()`)
            if `strict=True`, add unexpected keys to this list
        error_msgs (list of str):(used in `torch.nn.Module._load_state_dict()`)
            error messages should be added to this list, and will be reported 
            together in `load_nested_dict()` function
    """
    if len(module._parameters) == 0 and len(module._buffers) == 0:
        return
    try:
        _, state_dict_block = nested_dict.popitem(last=False)
    except KeyError:
        state_dict_block = {}
    state_dict_block = {prefix+key: val for key, val in state_dict_block.items()}
    metadata = getattr(state_dict_block, '_metadata', {})
    module._load_from_state_dict(
        state_dict_block, prefix, metadata, True, missing_keys, unexpected_keys, error_msgs)


class TorchNestedLoader:
    """
    A loader for `torch.nn.Module` to save/load parameters/buffers as nested dict

    `state_dict` or `nested_dict`?

    When `torch.nn.Module` loads from state_dict, the keys in state_dict
    contains prefix and parameter/buffer name. The prefix determines the 
    specific module to load the parameter/buffer. Meanwhile the parameter/buffer
    name prevents mismatch among parameters/buffers.

    With nested dict, we don't need to care about the specific module 
    name(prefix) or the complicated structure(e.g. model.stem.conv1).
    We ONLY care the order, the order of the module determines which module to 
    load.
    """
    def __init__(self, module: nn.Module):
        """
        Args:
            module (nn.Module): module to save/load
        """
        if not isinstance(module, nn.Module):
            raise ValueError(
                "module should be an instance of torch.nn.Module"
        )
        self.module = module

    def nested_dict(
        self, destination: OrderedDict = None, prefix: str = '', keep_vars: bool = False
    ) -> OrderedDict():
        """
        Returns a nested ordered dict containing all parameters and registered
        buffers of the module(`self.module`)

        Returns:
            OrderedDict: a nested OrderedDict cotaining several state_dict
                blocks, each containing whole state of one loadable module(not
                its descendants)
        """
        if destination is None:
            destination = OrderedDict()

        def save(module, prefix=''):
            _save_to_nested_dict(module, destination, prefix, keep_vars)
            for name, child in module._modules.items():
                if child is not None:
                    save(child, prefix + name + '.')

        save(self.module)
        save = None

        return destination

    def load_nested_dict(self, nested_dict: OrderedDict, strict: bool = True):
        """
        Copies parameters and buffers from `nested_dict` into `module` and its
        descendants. If `strict` is `True`, the keys of the inner dict (appended
        with corresponding prefix) must exactly match the keys returned by this
        module's `torch.nn.Module.state_dict` function

        For example:
            state_dict["stem.conv1.weight"] <-> nested_dict["conv"]["weight"]
            If `self.module.stem.conv1` module is mapped to `"conv"` block,
            nested_dict["conv"] will be poped as `state_dict_block`,
            then the key "weight" will be appended with prefix, becomes
            "stem.conv1.weight". Then the `state_dict_block` will be simply
            loaded like state_dict.
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        nested_dict = nested_dict.copy()
        def load(module: nn.Module, prefix: str = ''):
            _load_from_nested_dict(
                module, nested_dict, prefix, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        
        load(self.module)
        load = None

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.module.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @classmethod
    def nested_dict_to_state_dict(cls, nested_dict: OrderedDict) -> OrderedDict:
        """
        Convert nested_dict to state_dict.
        """
        state_dict = OrderedDict()
        state_dict._metadata = OrderedDict()
        for prefix, state_dict_block in nested_dict.items():
            state_dict._metadata[prefix[:-1]] = state_dict_block._metadata
            for name, param in state_dict_block.items():
                state_dict[prefix + name] = param
        return state_dict

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + ("(\n")
        tmpstr += "module="
        tmpstr += self.module.__repr__() + (")")
        return tmpstr
