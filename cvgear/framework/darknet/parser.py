# -------------------------------------------
# CVGear | Written by ivan Ding(a.k.a ivanpp)
# 
# Darknet network parser
# -------------------------------------------
import os
from typing import List, Dict, Tuple
from collections import OrderedDict
from yacs.config import CfgNode

__all__ = ["read_cfg", "DarknetParser", "build_darknet_parser"]


def _read_option(line: str) -> (str, str):
    assert len(line.split('=')) == 2
    key, val = line.split('=')
    key = key.strip()
    val = val.strip()
    return key, val


def read_cfg(filename: str) -> List[CfgNode]:
    """
    Parse darknet network configuration from .cfg file
    python version of
    list *read_cfg(char *filename);
    in pjreddie/darknet/src/parser.c

    Return:
        List of CfgNode(dict-like container)
    
    Note:
        The first CfgNode contains the network configuration, each of others
        contains the configurations to build one darknet layer.
    """
    assert filename.endswith(".cfg"), filename
    configs = []
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    for line in lines:
        if line.startswith('['):
            configs.append(CfgNode())
            configs[-1].name = line[1:-1]
        elif line.startswith('\0') or line.startswith('#') or line.startswith(';'):
            continue
        else:
            key, val = _read_option(line)
            configs[-1][key] = val
    assert configs[0].name == "network" or configs[0].name == "net"
    return configs


def _set_default_with_type(dict: dict, type: type, key: str, default: any):
    """
    Convert `dict[key]` to specific `type`, set to `type(default)` if `dict[key]`
    doesn't exist. Then return `dict[key]`.

    For example:
    
    `val = _set_default_with_type(dict, int, key, 1)`
        is equivalent to:
    `val = dict[key] = int(dict.get(key, 1))`
    """
    val = type(dict.get(key, default))
    dict[key] = val
    return val


class DarknetParser:
    """
    A parser that parses darknet .cfg file, holds necessary information for 
    network structure display, weights loading/converting, network building...

    Attributes:
        name (str): name of the darknet network to be parsed.
        net_options (CfgNode): stores basic configurations of the network.
        layers_options (list[CfgNode]): stores configurations of layers in a 
            list of CfgNode, each CfgNode stores configurations of one darknet
            layer.
        params (CfgNode): size information(`params.b`, `params.c`, `params.h`,
            `params.w`) of the current layer(speified by `params.index`)

    Examples:

    .. code-block:: python
        
        darknet = DarknetParser("darknet53")
        darknet.load_darknet_cfg_from_file("path/to/darknet53.cfg")
        print(darknet)
    """

    _WEIGHTED_LAYERS = ["convolutional", "conv", "connected", "conn"]

    _PARSE_LAYER_FUNC = {
        "convolutional":    "_parse_convolutional",
        "conv":             "_parse_convolutional",
        "connected":        "_parse_connected",
        "conn":             "_parse_connected",
        "shortcut":         "_parse_shortcut",
        "route":            "_parse_route",
        "upsample":         "_parse_upsample",
        "maxpool":          "_parse_maxpool",
        "max":              "_parse_maxpool",
        "avgpool":          "_parse_avgpool",
        "avg":              "_parse_avgpool",
        "softmax":          "_parse_softmax",
        "soft":             "_parse_softmax",
        "yolo":             "_parse_yolo",
        "dropout":          "_parse_dropout",
    }

    def __init__(self, name: str):
        """
        Args:
            name (str): the name of this darknet network
        """
        self._name = name
        self._net_options: CfgNode = None
        self._layers_options: List[CfgNode] = []
        self._params: CfgNode = CfgNode()

    def __len__(self) -> int:
        """
        Returns:
            int: num of layers of the network
        """
        return len(self._layers_options)

    def __getitem__(self, idx: int) -> CfgNode:
        """
        Returns:
            CfgNode: configuration of the layer of given index
        """
        return self._layers_options[idx]

    @property
    def weighted_layers(self) -> List[CfgNode]:
        """
        Returns:
            list[CfgNode]: weighted layers configurations of the network
        """
        return [layer for layer in self if layer.name in DarknetParser._WEIGHTED_LAYERS]

    @property
    def num_weighted_layers(self):
        """
        Returns:
            int: num of weighted layers of the network
        """
        return len(self.weighted_layers)

    def load_darknet_cfg(self, cfg_filename: str, name: str = None, parse: bool = True):
        """
        Load(read and parse) darknet network configuration(.cfg file).
        """
        configs = read_cfg(cfg_filename)
        net_options = configs.pop(0)
        assert net_options.name == "net" or net_options.name == "network"
        self._net_options = net_options
        self._layers_options = configs
        if name is not None:
            self._name = name
        if parse:
            self._parse_network()

    def _set_input_shape(self, options: CfgNode, shape: Tuple[int, int, int] = None):
        """
        Set input shape of the layer(`options`)

        Args:
            options (CfgNode): the layer to be set
            shape (tuple(C, H, W)) or None: given shape(copy from self._params
                if `None` is given)
        """
        if shape is None:
            options.c, options.h, options.w = self._params.c, self._params.h, self._params.w
        else:
            options.c, options.h, options.h = shape

    def _set_output_shape(self, options: CfgNode, shape: Tuple[int, int, int]):
        """
        Set output shape of the layer(`options`) and current size `self._params`

        Args:
            options (CfgNode): the layer to be set
            shape (tuple(C, H, W)): given output shape
        """
        c, h, w = shape
        assert c*h*w
        self._params.c = options.out_c = c
        self._params.h = options.out_h = h
        self._params.w = options.out_w = w

    def _parse_net_options(self):
        """
        Initialize/reset the size information of the network
        1. Set current size as input size of the network
        2. Set current location `self._params.index` to first layer
        """
        self._params.b = _set_default_with_type(self._net_options, int, "batch", 1)
        self._params.c = _set_default_with_type(self._net_options, int, "channels", 0)
        self._params.h = _set_default_with_type(self._net_options, int, "height", 0)
        self._params.w = _set_default_with_type(self._net_options, int, "width", 0)
        assert self._params.b*self._params.c*self._params.h*self._params.w
        self._params.index = 0

    """
    Note:
        _parse_layername() method should have the following signature:
        func(self, options: CfgNode) -> None
        It should be registerd in DarknetParser._PARSE_LAYER_FUNC,
        as "layer_name": "func"

        Args:
            options (CfgNode): configuration of the layer to parse
    """
    def _parse_connected(self, options: CfgNode):
        """
        Parse connected layer
        """
        output = _set_default_with_type(options, int, "output", 1)
        self._set_input_shape(options, (self._params.c*self._params.h*self._params.w, 1, 1))
        self._set_output_shape(options, (output, 1, 1))
        _set_default_with_type(options, int, "batch_normalize", 0)

    def _parse_convolutional(self, options: CfgNode):
        """
        Parse convolutional layer
        """
        n      = _set_default_with_type(options, int, "filters", 1)
        size   = _set_default_with_type(options, int, "size", 1)
        stride = _set_default_with_type(options, int, "stride", 1)
        pad    = _set_default_with_type(options, int, "pad", 0)
        groups = _set_default_with_type(options, int, "groups", 1)
        if pad: padding = _set_default_with_type(options, int, "padding", size/2)
        else: padding = _set_default_with_type(options, int, "padding", 0)
        
        self._set_input_shape(options)
        assert options.c*options.h*options.w
        out_h = int((options.h + 2*padding - size) / stride + 1)
        out_w = int((options.w + 2*padding - size) / stride + 1)
        self._set_output_shape(options, (n, out_h, out_w))
        _set_default_with_type(options, int, "batch_normalize", 0)

    def _parse_shortcut(self, options: CfgNode):
        """
        Parse shortcut layer
            Upper stream index of shortcut connection will be stored in `.index`.
            The shape of shortcut input will be stored in `.c2`, `.h2`, `.w2`.
        """
        index = int(options["from"])
        options.index = self._params.index + index if index < 0 else index
        self._set_input_shape(options)
        self._set_output_shape(options, (self._params.c, self._params.h, self._params.w))
        options.c2 = self._layers_options[options.index].out_c
        options.h2 = self._layers_options[options.index].out_h
        options.w2 = self._layers_options[options.index].out_w

    def _parse_route(self, options: CfgNode):
        """
        Parse route layer
            Upper stream indexes will be stored in list `options.indexes`.
        """
        indexes = options.layers.split(',')
        indexes = [int(index) for index in indexes]
        indexes = [index+self._params.index if index<0 else index for index in indexes]
        options.indexes = indexes.copy()
        first = indexes.pop(0)
        out_c = self._layers_options[first].out_c
        out_h = self._layers_options[first].out_h
        out_w = self._layers_options[first].out_w
        tmpstr = "out_c = {}".format(out_c)
        for index in indexes:
            assert self._layers_options[index].out_h == out_h and self._layers_options[index].out_w == out_w
            c = self._layers_options[index].out_c
            tmpstr += " + {}".format(c)
            out_c += c
        options.info = tmpstr
        self._set_output_shape(options, (out_c, out_h, out_w))

    def _parse_upsample(self, options: CfgNode):
        """
        Parse upsample layer
        """
        stride = _set_default_with_type(options, int, "stride", 2)
        self._set_input_shape(options)
        self._set_output_shape(options, (self._params.c, stride*self._params.h, stride*self._params.w))

    def _parse_maxpool(self, options: CfgNode):
        """
        Parse maxpool layer
        """
        stride = _set_default_with_type(options, int, "stride", 2)
        size = _set_default_with_type(options, int, "size", stride)
        padding = _set_default_with_type(options, int, "padding", size-1)
        self._set_input_shape(options)
        out_h = int((options.h + padding - size) / stride + 1)
        out_w = int((options.w + padding - size) / stride + 1)
        self._set_output_shape(options, (self._params.c, out_h, out_w))

    def _parse_avgpool(self, options: CfgNode):
        """
        Parse avgpool layer
        """
        self._set_input_shape(options)
        self._set_output_shape(options, (options.c, 1, 1))
    
    def _parse_softmax(self, options: CfgNode):
        """
        Parse softmax layer
        """
        self._set_input_shape(options)
        groups = _set_default_with_type(options, int, "groups", 1)
        options.inputs = self._params.c * self._params.h * self._params.w
        assert options.inputs % groups == 0

    def _parse_yolo(self, options: CfgNode):
        """
        Parse yolo layer(yolov3)
            Masks of current yolo layer will be stored in list `options.indexes`.
            Anchors that used in this layer will be stored in `options.local_anchors`
            as list of tuples.
        """
        classes = _set_default_with_type(options, int, "classes", 20)
        total = _set_default_with_type(options, int, "num", 1)
        try:
            indexes = options.mask.split(',')
            options.indexes = [int(index) for index in indexes]
        except AttributeError:
            options.indexes = [0]
        anchors = options.anchors.split(',')
        anchors = [float(anchor) for anchor in anchors]
        anchors = list(zip(anchors[::2], anchors[1::2]))
        assert len(anchors) == total
        options.local_anchors = [anchors[i] for i in options.indexes]
        self._set_input_shape(options)

    def _parse_dropout(self, options: CfgNode):
        """
        Parse dropout layer
        """
        _set_default_with_type(options, float, "probability", .5)
        self._set_input_shape(options)
        self._set_output_shape(options, (self._params.c, self._params.h, self._params.w))

    def _parse_network(self):
        """
        Parse the entire network
        """
        self._parse_net_options()
        for options in self:
            if options.name in DarknetParser._PARSE_LAYER_FUNC:
                parse_func = getattr(DarknetParser,
                                     DarknetParser._PARSE_LAYER_FUNC[options.name])
                parse_func(self, options)
            else:
                print("<Parsing: skipped unsupported layer: [{}] {}>".format(
                    self._params.index, options.name))
            self._params.index += 1

    def __repr__(self) -> str:
        tmpstr =  self.__class__.__name__ + "(\n"
        tmpstr += "name={},\n".format(self._name)
        if self._net_options is None:
            tmpstr += "<.cfg not loaded>\n"
        else:
            tmpstr += "#weighted_layers={},\n".format(self.num_weighted_layers)
            tmpstr += self._network_structure()
        tmpstr += ")"
        return tmpstr

    def _network_structure(self) -> str:
        """
        Returns:
            str: network structure string
        """
        template = "{idx:3} {name:<8}   {filters:^8} {size:<18} {input:<18} {output:<18}\n"
        tmpstr = template.format(
            idx="idx",
            name="layer",
            filters="filters",
            size="size/stride",
            input="input",
            output="output"
        )
        for idx, options in enumerate(self):
            if options.name == "convolutional" or options.name == "conv":
                tmpstr += template.format(
                    idx=idx, 
                    name="conv",
                    filters=options.filters,
                    size="{} x {} / {}".format(options.size, options.size, options.stride),
                    input="{} x {} x {}".format(options.w, options.h, options.c),
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "connected" or options.name == "conn":
                tmpstr += template.format(
                    idx=idx,
                    name="conn",
                    filters="",
                    size="",
                    input="{}".format(options.c),
                    output="{}".format(options.out_c)
                )
            elif options.name == "shortcut":
                tmpstr += template.format(
                    idx=idx,
                    name="shortcut",
                    filters="[{}]".format(options.index),
                    size="{} x {} x {} ->".format(options.w2, options.h2, options.c2),
                    input="{} x {} x {}".format(options.w, options.h, options.c),
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "route":
                tmpstr += template.format(
                    idx=idx,
                    name="route",
                    filters=str(options.indexes),
                    size=options.info,
                    input="",
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "upsample":
                tmpstr += template.format(
                    idx=idx,
                    name="upsample",
                    filters="",
                    size="{}x".format(options.stride),
                    input="{} x {} x {}".format(options.w, options.h, options.c),
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "maxpool" or options.name == "max":
                tmpstr += template.format(
                    idx=idx,
                    name="maxpool",
                    filters="",
                    size="{} x {} / {}".format(options.size, options.size, options.stride),
                    input="{} x {} x {}".format(options.w, options.h, options.c),
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "avgpool" or options.name == "avg":
                tmpstr += template.format(
                    idx=idx,
                    name="avgpool",
                    filters="",
                    size="",
                    input="{} x {} x {}".format(options.w, options.h, options.c),
                    output="{} x {} x {}".format(options.out_w, options.out_h, options.out_c)
                )
            elif options.name == "softmax" or options.name == "soft":
                tmpstr += template.format(
                    idx=idx,
                    name="softmax",
                    filters="",
                    size="groups={}".format(options.groups),
                    input="",
                    output="{}".format(options.inputs)
                )
            elif options.name == "yolo":
                tmpstr += template.format(
                    idx=idx,
                    name="yolo",
                    filters="{}/{}".format(len(options.indexes), options.num),
                    size="{}".format(options.indexes),
                    input="",
                    output="{}".format(options.classes)
                )
            elif options.name == "dropout":
                tmpstr += template.format(
                    idx=idx,
                    name="dropout",
                    filters="p={}".format(options.probability),
                    size="",
                    input="{}".format(options.w * options.h * options.c),
                    output="{}".format(options.out_w * options.out_h * options.out_c)
                )
            else:
                tmpstr += "{:3} unspupported layer: {}\n".format(idx, options.name)
        return tmpstr


def build_darknet_parser(cfg_filename: str, name: str = None) -> DarknetParser:
    """
    Build a darknet parser for a network defined by `cfg_filename`
    """
    if name is None:
        name = os.path.basename(cfg_filename).split('.')[0]
    parser = DarknetParser(name)
    parser.load_darknet_cfg(cfg_filename)
    return parser