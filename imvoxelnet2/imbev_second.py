from mmdet3d.models import SECOND
from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImBEVSECOND(SECOND):

    def forward(self, x):
        mlvl_x = super().forward(x.permute(0, 1, 4, 2, 3).flatten(1, 2))
        # Anchor3DHead axis order is (y, x).
        return [x.transpose(-1, -2) for x in mlvl_x]
