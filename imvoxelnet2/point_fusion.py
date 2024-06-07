from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.layers.fusion_layers import apply_3d_transformation
from mmdet3d.structures.bbox_3d import points_cam2img

EPSILON = 1e-6


def integral_image(features: Tensor) -> Tensor:
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


def point_sample(img_meta: dict,
                 img_features: Tensor,
                 points: Tensor,
                 proj_mat: Tensor,
                 coord_type: str,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True,
                 valid_flag: bool = False,
                 pooling: bool = False,
                 n_voxels: Optional[List] = None) -> Tensor:
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to image coordinate
    if valid_flag:
        proj_pts = points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0).clamp(
                         -1 - EPSILON, 1 + EPSILON)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    if not pooling:
        point_features = F.grid_sample(
            img_features,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # 1xCx1xN feats
    else:
        # Get top-left and bottom-right coordinates of voxel bounding boxes
        norm_corners = grid.view(1, *n_voxels[::-1],
                                 2).permute([0, 1, 3, 2, 4])
        bbox_corners = torch.cat([
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1,
                                                                   1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1,
                                                                1:]),
        ],
                                 dim=-1)
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box
        img_height, img_width = h, w
        area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) *
                img_height * img_width * 0.25 + EPSILON).unsqueeze(1)
        visible = (area > EPSILON)

        # Sample integral image at bounding box locations
        features = img_features
        integral_img = integral_image(features)
        top_left = F.grid_sample(
            integral_img,
            bbox_corners[..., [0, 1]],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)
        btm_right = F.grid_sample(
            integral_img,
            bbox_corners[..., [2, 3]],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)
        top_right = F.grid_sample(
            integral_img,
            bbox_corners[..., [2, 1]],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)
        btm_left = F.grid_sample(
            integral_img,
            bbox_corners[..., [0, 3]],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = (top_left + btm_right - top_right - btm_left) / area
        vox_feats = vox_feats * visible.float()
        vox_feats = vox_feats.view(1, features.shape[1], -1, depth, width)
        point_features = vox_feats.permute([0, 1, 2, 4, 3
                                            ]).reshape(1, features.shape[1], 1,
                                                       -1)

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
                depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()
