from mmdet.models.backbones import ResNet

from mmdet3d.registry import MODELS


@MODELS.register_module()
class NoStemResNet(ResNet):

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
