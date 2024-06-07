from copy import deepcopy
from os import path as osp
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate

from mmdet3d.structures import get_box_type

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_mono_3d_detector(model: nn.Module,
                               imgs: ImagesType,
                               data_list: Union[dict, Sequence[dict]],
                               cam_type: str = 'CAM_FRONT'):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        data_list (dict, Sequence[dict]): Annotations.
        cam_type (str): Image of Camera chose to infer.
            For kitti dataset, it should be 'CAM_2',
            and for nuscenes dataset, it should be
            'CAM_FRONT'. Defaults to 'CAM_FRONT'.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        data_list = [data_list]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    assert len(imgs) == len(data_list)

    data = []
    for index, img in enumerate(imgs):
        # get data info containing calib
        data_info = data_list[index]
        img_path = data_info['images'][cam_type]['img_path']
        if osp.basename(img_path) != osp.basename(img):
            raise ValueError(f'the info file of {img_path} is not provided.')

        for proj_mat in ['lidar2img', 'depth2img', 'cam2img']:
            if proj_mat in data_info['images'][cam_type]:
                data_info['images'][cam_type][proj_mat] = \
                    np.array(data_info['images'][cam_type][proj_mat])
        # replace the img_path in data_info with img
        data_info['images'][cam_type]['img_path'] = img
        # avoid data_info['images'] has multiple keys anout camera views.
        mono_img_info = {f'{cam_type}': data_info['images'][cam_type]}
        data_ = dict(
            images=mono_img_info,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)

        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0]
    else:
        return results
