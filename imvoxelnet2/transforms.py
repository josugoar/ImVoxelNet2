from typing import Tuple, Union

import numpy as np
from torch import Tensor

from mmdet3d.structures import points_cam2img, points_img2cam
from mmdet3d.utils import array_converter


@array_converter(apply_to=('points_3d', 'proj_mat'))
def bbox3d_flip(bboxes: Union[Tensor, np.ndarray],
                cam2img: Union[Tensor, np.ndarray],
                img_shape: Tuple[int],
                direction: str = 'horizontal',
                with_yaw: bool = True) -> Union[Tensor, np.ndarray]:
    centers_2d_with_depth = points_cam2img(
        bboxes[:, :3], cam2img, with_depth=True)
    if direction == 'horizontal':
        centers_2d_with_depth[:,
                              0] = img_shape[1] - centers_2d_with_depth[:, 0]
    elif direction == 'vertical':
        centers_2d_with_depth[:,
                              1] = img_shape[0] - centers_2d_with_depth[:, 1]
    bboxes[:, :3] = points_img2cam(centers_2d_with_depth, cam2img)

    if with_yaw:
        bboxes[:, 6] = -bboxes[:, 6] + np.pi

    return bboxes
