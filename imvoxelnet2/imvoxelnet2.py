from typing import List

import torch
from torch import nn

from mmdet3d.models.detectors import ImVoxelNet
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from .point_fusion import point_sample


@MODELS.register_module()
class ImVoxelNet2(ImVoxelNet):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 coord_type: str,
                 backbone_3d: OptConfigType = None,
                 aligned: bool = False,
                 mlvl_features: bool = False,
                 pooling: bool = False,
                 use_ground_plane: bool = False,
                 bev: bool = False,
                 middle_in_channels: int = 0,
                 middle_out_channels: int = 0,
                 voxel_pooling: str = 'linear',
                 prev_feats: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        if use_ground_plane:
            n_voxels[2] = 1
        super().__init__(
            backbone,
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
        if (bev and middle_in_channels != 0 and middle_out_channels != 0):
            if voxel_pooling == "linear":
                pooling_layer = nn.Linear(
                    middle_in_channels, middle_out_channels, bias=False)
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
        return hasattr(self, 'backbone_3d') and self.backbone_3d is not None

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
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
                img_scale_factor = (
                    points.new_tensor(img_meta['scale_factor'][:2])
                    if 'scale_factor' in img_meta.keys() else 1)
                img_flip = img_meta['flip'] if 'flip' in img_meta.keys(
                ) else False
                img_crop_offset = (
                    points.new_tensor(img_meta['img_crop_offset'])
                    if 'img_crop_offset' in img_meta.keys() else 0)
                proj_mat = points.new_tensor(
                    get_proj_mat_by_coord_type(img_meta, self.coord_type))
                volume = point_sample(
                    img_meta,
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
