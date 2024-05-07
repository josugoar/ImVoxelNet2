from mmdet.models import ResNet

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImBEVResNet(ResNet):

    def forward(self, x):
        return super().forward(x.permute(0, 1, 4, 2, 3).flatten(1, 2).transpose(-1, -2))
