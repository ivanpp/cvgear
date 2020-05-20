# -----------------------------------------------------------
# CVGear | Written by ivan Ding(a.k.a ivanpp)
# 
# Nested dictionary loader for darknet network(DarknetParser)
# -----------------------------------------------------------
import os
import copy
from collections import OrderedDict
import numpy as np
import torch
from yacs.config import CfgNode

from .parser import DarknetParser
from ..nested_dict import nested_dict_counter, nested_dict_tensor2ndarray

__all__ = ["DarknetNestedLoader", "build_darknet_nested_loader"]


class DarknetNestedLoader:
    """
    A loader that save/load darknet weights as nested_dict
    """

    _SAVE_NESTED_DICT_FUNC = {
        "convolutional": "_save_convolutional_to_nested_dict",
        "connected": "_save_connected_to_nested_dict",
    }

    _LOAD_NESTED_DICT_FUNC = {
        "convolutional": "_load_convolutional_from_nested_dict",
        "connected": "_load_connected_from_nested_dict",
    }

    def __init__(self, parser: DarknetParser = None, weights: np.ndarray = None):
        """
        Args:
            parser (DarknetParser):
            weights (np.ndarry):
        """
        if not isinstance(parser, DarknetParser):
            raise TypeError(
                "parser should be an instance of "
                "cvgear.framework.darknet.DarknetParser"
            )
        self.parser = parser
        self._weights = weights

    def load_darknet_weights(self, weights_filename: str):
        """
        Load darknet weights from binary file to `self._weights`(a flattened 
        `numpy.ndarray`).
        """
        with open(weights_filename, "rb") as f:
            major    = np.fromfile(f, dtype=np.int32, count=1)
            minor    = np.fromfile(f, dtype=np.int32, count=1)
            revision = np.fromfile(f, dtype=np.int32, count=1)
            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                seen = np.fromfile(f, dtype=np.int64, count=1)
            else:
                seen = np.fromfile(f, dtype=np.int32, count=1)
            self._weights  = np.fromfile(f, dtype=np.float32)

    def save_darknet_weights(self, weights_filename: str):
        """
        Save `self._weights` to binary file `weight_filename`.
        """
        if self._weights is None:
            raise ValueError(
                "No weights to save, "
                "load_darknet_weights() to load weights from binary file, or "
                "load_nested_dict() to load weights from nested dict"
        )
        major, minor, revision = 0, 2, 0
        version = np.array([major, minor, revision], dtype=np.int32)
        seen = np.array([0], dtype=np.int64)
        with open(weights_filename, "wb") as f:
            version.tofile(f)
            seen.tofile(f)
            self._weights.tofile(f)
        print("binary saved to {}".format(os.path.abspath(weights_filename)))

    def nested_dict(self, destination: OrderedDict = None) -> OrderedDict:
        """
        Returns a nested ordered dict containing all parameters and buffers of
        a darknet network(`self.parser`).

        Returns:
            OrderedDict: a nested OrderedDict containing sevearl state_dict
                blocks, each containing whole state of one loadable torch module/
                darknet layer
        """
        if self._weights is None:
            raise Exception(
                "Should load binary first:"
                ".load_darknet_weights(weights_filename)"
            )
        if destination is None:
            destination = OrderedDict()

        weights = self._weights.copy()
        for idx, options in enumerate(self.parser.weighted_layers):
            if options.name in DarknetNestedLoader._SAVE_NESTED_DICT_FUNC:
                save_nested_func = getattr(DarknetNestedLoader,
                                           DarknetNestedLoader._SAVE_NESTED_DICT_FUNC[options.name])
                save_nested_func(self, options, destination, idx)
            else:
                raise NotImplementedError(
                    "Method to save '{}' is not implemented, "
                    "please implement it as "
                    "_save_layer_to_nested_dict(self, options, destination, index) -> None,"
                    " and register it in _SAVE_NESTED_DICT_FUNC".format(options.name)
            )
        self._weights = weights

        print("<{} params to save to nested dict, {} saved>".format(
            self._weights.size, nested_dict_counter(destination)))
        return destination

    """
    Note:
        _save_layer_to_nested_dict() method should have the following signature:
        func(self, options: CfgNode, destination: OrderedDict, index: int) -> None
        It should be registered in DarknetNestedLoader._SAVE_NESTED_DICT_FUNC,
        as: "layer_name": "func"

        Args:
            options (CfgNode): configuration of the layer to save
            destination (OrderedDict): nested dict to store params/buffers
            index (int): index of the layer to save, used as suffix of the key
    """
    def _save_connected_to_nested_dict(
        self, options: CfgNode, destination: OrderedDict, index: int
    ):
        """
        Save params/buffers of a darknet connected layer to nested dict.
        """
        outputs = options.out_c
        inputs = options.c
        biases,  self._weights = np.split(self._weights, [outputs])
        weights, self._weights = np.split(self._weights, [outputs*inputs])
        weights = weights.reshape(outputs, inputs)
        if options.batch_normalize:
            scales,            self._weights = np.split(self._weights, [outputs])
            rolling_mean,      self._weights = np.split(self._weights, [outputs])
            rolling_variance,  self._weights = np.split(self._weights, [outputs])
        destination["linear"+str(index)] = OrderedDict()
        destination["linear"+str(index)]["weight"] = torch.from_numpy(weights)
        if options.batch_normalize:
            destination["norm"+str(index)] = OrderedDict()
            destination["norm"+str(index)]["weight"]       = torch.from_numpy(scales)
            destination["norm"+str(index)]["bias"]         = torch.from_numpy(biases)
            destination["norm"+str(index)]["running_mean"] = torch.from_numpy(rolling_mean)
            destination["norm"+str(index)]["running_var"]  = torch.from_numpy(rolling_variance)
        else:
            destination["linear"+str(index)]["bias"] = torch.from_numpy(biases)

    def _save_convolutional_to_nested_dict(
        self, options: CfgNode, destination: OrderedDict, index: int
    ):
        """
        Save params/buffers of a darknet convolutional layer to nested dict.
        """
        n = options.out_c
        num = int(n * options.c/options.groups * options.size * options.size)
        biases, self._weights = np.split(self._weights, [n])
        if options.batch_normalize:
            scales,            self._weights = np.split(self._weights, [n])
            rolling_mean,      self._weights = np.split(self._weights, [n])
            rolling_variance,  self._weights = np.split(self._weights, [n])
        weights, self._weights = np.split(self._weights, [num])
        weights = weights.reshape(n, options.c, options.size, options.size)
        destination["conv"+str(index)] = OrderedDict()
        destination["conv"+str(index)]["weight"] = torch.from_numpy(weights)
        if options.batch_normalize:
            destination["norm"+str(index)] = OrderedDict()
            destination["norm"+str(index)]["weight"]       = torch.from_numpy(scales)
            destination["norm"+str(index)]["bias"]         = torch.from_numpy(biases)
            destination["norm"+str(index)]["running_mean"] = torch.from_numpy(rolling_mean)
            destination["norm"+str(index)]["running_var"]  = torch.from_numpy(rolling_variance)
        else:
            destination["conv"+str(index)]["bias"] = torch.from_numpy(biases)

    def load_nested_dict(self, nested_dict: OrderedDict):
        """
        Copies parameters and buffers from `nested_dict` into `self._weights`(
        which is a flattened numpy.ndarry)
        """
        self._weights = np.array([], dtype=np.float32)
        nested_dict_ndarray =nested_dict_tensor2ndarray(nested_dict)

        for idx, options in enumerate(self.parser.weighted_layers):
            if options.name in DarknetNestedLoader._LOAD_NESTED_DICT_FUNC:
                load_nested_func = getattr(DarknetNestedLoader,
                                           DarknetNestedLoader._LOAD_NESTED_DICT_FUNC[options.name])
                load_nested_func(self, options, nested_dict_ndarray)
            else:
                raise NotImplementedError(
                    "Method to load '{}' is not implemented, "
                    "please implement it as "
                    "_load_layer_from_nested_dict(self, options, nested_dict) -> None,"
                    " and register it in _LOAD_NESTED_DICT_FUNC".format(options.name)
            )
        print("<{} params to load from nested dict, {} loaded>".format(
            nested_dict_counter(nested_dict), self._weights.size))

    """
    Note:
        _load_layer_to_nested_dict() method should have the following signature:
        func(self, options: CfgNode, nested_dict: OrderedDict) -> None
        It should be registered in DarknetNestedLoader._LOAD_NESTED_DICT_FUNC,
        as: "layer_name": "func"

        Args:
            options (CfgNode): configuration of the layer to load
            nested_dict (OrderedDict): nested dict that ready to load from
    """
    def _load_connected_from_nested_dict(
        self, options: CfgNode, nested_dict: OrderedDict
    ):
        """
        Load params/buffers of a darknet connected layer to `self._weights`
        """
        linear = nested_dict.popitem(last=False)[1]
        if options.batch_normalize:
            norm = nested_dict.popitem(last=False)[1]
            self._weights = np.concatenate((self._weights, norm["biases"]))
            self._weights = np.concatenate((self._weights, linear["weight"]))
            self._weights = np.concatenate((self._weights, norm["weight"]))
            self._weights = np.concatenate((self._weights, norm["running_mean"]))
            self._weights = np.concatenate((self._weights, norm["running_var"]))
        else:
            self._weights = np.concatenate((self._weights, linear["bias"]))
            self._weights = np.concatenate((self._weights, linear["weight"].flatten()))

    def _load_convolutional_from_nested_dict(
        self, options: CfgNode, nested_dict: OrderedDict
    ):
        """
        Load params/buffers of a darknet convolutional layer to `self._weights`
        """
        conv = nested_dict.popitem(last=False)[1]
        if options.batch_normalize:
            norm = nested_dict.popitem(last=False)[1]
            self._weights = np.concatenate((self._weights, norm["bias"]))
            self._weights = np.concatenate((self._weights, norm["weight"]))
            self._weights = np.concatenate((self._weights, norm["running_mean"]))
            self._weights = np.concatenate((self._weights, norm["running_var"]))
        else:
            self._weights = np.concatenate((self._weights, conv["bias"]))
        self._weights = np.concatenate((self._weights, conv["weight"].flatten()))

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "(\n"
        if self._weights is None:
            tmpstr += "<.weights not loaded>\n"
        else:
            tmpstr += "#params={}\n".format(self._weights.size)
        tmpstr += "parser="
        tmpstr += self.parser.__repr__()
        tmpstr += ")"
        return tmpstr


def build_darknet_nested_loader(
    cfg_filename: str, weights_filename: str, name: str = None
) -> DarknetNestedLoader:
    """
    Build a nested dict loader for darknet network

    1. Use the cfg file(`cfg_filename`) to build a darknet Parser
    2. Use the weights file(`weights_filename`) to build a darknet loader

    Args:
        cfg_filename (str): path to cfg file of the darknet network
        weights_filename (str): path to weights file of the darknet network
        name (str): name of the network
    
    Returns:
        a darknet nested dict loader
    """
    if name is None:
        name = os.path.basename(cfg_filename).split('.')[0]
    parser = DarknetParser(name)
    parser.load_darknet_cfg(cfg_filename)
    loader = DarknetNestedLoader(parser)
    loader.load_darknet_weights(weights_filename)
    return loader