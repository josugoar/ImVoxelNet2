from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImBEVNeck(BaseModule):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Number of channels in an input tensor.
        out_channels (int): Number of channels in all output tensors.
    """

    def __init__(self, in_channels, out_channels, feat_channels=None):
        super(ImBEVNeck, self).__init__()
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
        if feat_channels is not None:
            self.model.insert(0, ConvModule(
                in_channels=feat_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU', inplace=True)))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in * N_z, N_y, N_x).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        return self.model.forward(x)

    def init_weights(self):
        """Initialize weights of neck."""
        pass


class ResModule(nn.Module):
    """2d residual block for ImBEVNeck.

    Args:
        in_channels (int): Number of channels in input tensor.
        out_channels (int): Number of channels in output tensor.
        stride (int, optional): Stride of the block. Defaults to 1.
    """

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
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        if self.stride != 1:
            identity = self.downsample(identity)
        x = x + identity
        x = self.activation(x)
        return x
