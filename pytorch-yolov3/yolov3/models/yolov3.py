import torch
import torch.nn as nn

from collections import defaultdict
from yolov3.models.yolo_layer import YOLOLayer


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a Conv2d / BatchNorm2d / leaky ReLU block.

    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    pad = (ksize - 1) // 2

    sequential = nn.Sequential()
    sequential.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=False,
        ),
    )
    sequential.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    sequential.add_module("leaky", nn.LeakyReLU(0.1))

    return sequential


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of two convolution layers.

    Args:
        ch (int): number of input and output channels.
        n_blocks (int): number of residual blocks.
    """

    def __init__(self, ch, n_blocks):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(n_blocks):
            resblock = nn.ModuleList(
                [add_conv(ch, ch // 2, 1, 1), add_conv(ch // 2, ch, 3, 1)]
            )
            self.module_list.append(resblock)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h

        return x


def create_yolov3_modules(config_model, ignore_threshold):
    """
    Build yolov3 layer modules.
    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        module_list (ModuleList): YOLOv3 module list.
    """
    # layer order is same as yolov3.cfg
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    module_list = nn.ModuleList()

    #
    # Darknet 53
    #

    module_list.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    # 1
    module_list.append(resblock(ch=64, n_blocks=1))
    module_list.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    # 2
    module_list.append(resblock(ch=128, n_blocks=2))
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
    # 3
    module_list.append(resblock(ch=256, n_blocks=8))
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
    # 4
    module_list.append(resblock(ch=512, n_blocks=8))
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
    # 5
    module_list.append(resblock(ch=1024, n_blocks=4))

    #
    # additional layers for YOLOv3
    #

    # A
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    # B
    module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    module_list.append(
        YOLOLayer(
            config_model, layer_no=0, in_ch=1024, ignore_threshold=ignore_threshold
        )
    )
    # C
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    module_list.append(nn.Upsample(scale_factor=2, mode="nearest"))

    # A
    module_list.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    # B
    module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    module_list.append(
        YOLOLayer(
            config_model, layer_no=1, in_ch=512, ignore_threshold=ignore_threshold
        )
    )
    # C
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    module_list.append(nn.Upsample(scale_factor=2, mode="nearest"))

    # A
    module_list.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    module_list.append(
        YOLOLayer(
            config_model, layer_no=2, in_ch=256, ignore_threshold=ignore_threshold
        )
    )

    return module_list


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """

    def __init__(self, config_model, ignore_threshold=0.7):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super().__init__()
        self.module_list = create_yolov3_modules(config_model, ignore_threshold)

    def forward(self, x, labels=None):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            labels (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = labels is not None
        self.loss_dict = defaultdict(float)

        output = []
        layers = []
        for i, module in enumerate(self.module_list):
            if i == 18:
                x = layers[i - 3]
            if i == 20:
                x = torch.cat((layers[i - 1], layers[8]), dim=1)
            if i == 27:
                x = layers[i - 3]
            if i == 29:
                x = torch.cat((layers[i - 1], layers[6]), dim=1)

            if isinstance(module, YOLOLayer):
                if train:
                    x, *losses = module(x, labels)
                    for name, loss in zip(["xy", "wh", "conf", "cls", "l2"], losses):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)

                output.append(x)
            else:
                x = module(x)

            layers.append(x)

        if train:
            return sum(output)
        else:
            return torch.cat(output, dim=1)