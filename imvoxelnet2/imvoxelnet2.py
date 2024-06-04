import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.detectors import ImVoxelNet
from mmdet3d.models.layers.fusion_layers import apply_3d_transformation
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type, points_cam2img

EPSILON = 1e-6


@MODELS.register_module()
class ImVoxelNet2(ImVoxelNet):

    def __init__(
        self,
        backbone,
        neck,
        neck_3d,
        bbox_head,
        prior_generator,
        n_voxels,
        coord_type,
        backbone_3d=None,
        aligned=False,
        mlvl_features=False,
        pooling=False,
        use_ground_plane=False,
        bev=False,
        middle_in_channels=None,
        middle_out_channels=None,
        voxel_pooling=None,
        prev_feats=False,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        if use_ground_plane:
            n_voxels[2] = 1
        super().__init__(backbone,
                         neck,
                         neck_3d,
                         bbox_head,
                         prior_generator,
                         n_voxels,
                         coord_type,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        if backbone_3d is not None:
            self.backbone_3d = MODELS.build(backbone_3d)
        self.aligned = aligned
        self.mlvl_features = mlvl_features
        self.pooling = pooling
        self.use_ground_plane = use_ground_plane
        self.bev = bev
        self.prev_feats = prev_feats
        if bev and middle_in_channels is not None and middle_out_channels is not None:
            if voxel_pooling is None or voxel_pooling == "linear":
                pooling_layer = nn.Linear(middle_in_channels,
                                          middle_out_channels,
                                          bias=False)
            elif voxel_pooling == "max":
                pooling_layer = nn.AdaptiveMaxPool1d(middle_out_channels)
            elif voxel_pooling == "avg":
                pooling_layer = nn.AdaptiveAvgPool1d(middle_out_channels)
            else:
                raise ValueError(f"Invalid voxel pooling type {voxel_pooling}")
            self.voxel_pooling = nn.Sequential(
                pooling_layer, nn.BatchNorm1d(middle_out_channels),
                nn.ReLU(inplace=True))
        else:
            self.voxel_pooling = nn.Identity()

    @property
    def with_backbone_3d(self):
        """Whether the detector has a 3D backbone."""
        return hasattr(self, 'backbone_3d') and self.backbone_3d is not None

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        x = self.backbone(img)
        x = self.neck(x)
        if not self.mlvl_features:
            x = [x[0]]
        x = list(map(list, zip(*x)))
        if self.pooling:
            n_voxels = [n_voxel + 1 for n_voxel in self.n_voxels]
        else:
            n_voxels = self.n_voxels
        points = self.prior_generator.grid_anchors([n_voxels[::-1]],
                                                   device=img.device)[0][:, :3]
        mlvl_volumes, mlvl_valid_preds = [], []
        for features, img_meta in zip(x, batch_img_metas):
            if self.use_ground_plane:
                points[:, 2] = -img_meta['plane'][3]
            volumes, valid_preds = [], []
            for feature in features:
                img_scale_factor = (points.new_tensor(
                    img_meta['scale_factor'][:2]) if 'scale_factor'
                                    in img_meta.keys() else 1)
                img_flip = img_meta['flip'] if 'flip' in img_meta.keys(
                ) else False
                img_crop_offset = (points.new_tensor(
                    img_meta['img_crop_offset']) if 'img_crop_offset'
                                   in img_meta.keys() else 0)
                proj_mat = points.new_tensor(
                    get_proj_mat_by_coord_type(img_meta, self.coord_type))
                volume = point_sample(img_meta,
                                      img_features=feature[None, ...],
                                      points=points,
                                      proj_mat=points.new_tensor(proj_mat),
                                      coord_type=self.coord_type,
                                      img_scale_factor=img_scale_factor,
                                      img_crop_offset=img_crop_offset,
                                      img_flip=img_flip,
                                      img_pad_shape=img.shape[-2:],
                                      img_shape=img_meta['img_shape'][:2],
                                      aligned=self.aligned,
                                      pooling=self.pooling,
                                      n_voxels=n_voxels)
                volumes.append(
                    volume.reshape(self.n_voxels[::-1] + [-1]).permute(
                        3, 2, 1, 0))
                valid_preds.append(
                    ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
            valid_pred = torch.stack(valid_preds).sum(0)
            valid_pred[valid_pred == 0] = 1
            volume = torch.stack(volumes).sum(0) / valid_pred
            mlvl_volumes.append(volume)
            mlvl_valid_preds.append(valid_pred)
        volumes = mlvl_volumes
        valid_preds = mlvl_valid_preds
        x = torch.stack(volumes)
        if self.bev:
            N, _, N_x, N_y, _ = x.size()
            x = x.permute(0, 2, 3, 1, 4)
            x = x.flatten(0, 2)
            x = x.flatten(1, 2)
            x = self.voxel_pooling(x)
            x = x.view(N, N_x, N_y, -1)
            x = x.permute(0, 3, 1, 2)
            # Anchor3DHead axis order is (y, x).
            x = x.transpose(-1, -2)
        if self.with_backbone_3d:
            prev_x = x
            x = self.backbone_3d(x)
            if self.prev_feats:
                x = (prev_x, *x)
        x = self.neck_3d(x)
        return x, torch.stack(valid_preds).float()


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True,
                 valid_flag=False,
                 pooling=False,
                 n_voxels=None):
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)

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
        area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) \
             * img_height * img_width * 0.25 + EPSILON).unsqueeze(1)
        visible = (area > EPSILON)

        # Sample integral image at bounding box locations
        features = img_features
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img,
                                 bbox_corners[..., [0, 1]],
                                 mode=mode,
                                 padding_mode=padding_mode,
                                 align_corners=align_corners)
        btm_right = F.grid_sample(integral_img,
                                  bbox_corners[..., [2, 3]],
                                  mode=mode,
                                  padding_mode=padding_mode,
                                  align_corners=align_corners)
        top_right = F.grid_sample(integral_img,
                                  bbox_corners[..., [2, 1]],
                                  mode=mode,
                                  padding_mode=padding_mode,
                                  align_corners=align_corners)
        btm_left = F.grid_sample(integral_img,
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
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
