from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImBEVNeck(BaseModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResModule(in_channels, in_channels),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels * 2, in_channels * 2),
            ConvModule(
                in_channels=in_channels * 2,
                out_channels=in_channels * 4,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels * 4, in_channels * 4),
            ConvModule(
                in_channels=in_channels * 4,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU', inplace=True)))

    def forward(self, x):
        x = self.model.forward(x)
        return [x]

    def init_weights(self):
        pass


class ResModule(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=None)
        if stride != 1:
            self.downsample = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=None)
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        if self.stride != 1:
            identity = self.downsample(identity)
        x = x + identity
        x = self.activation(x)
        return x
