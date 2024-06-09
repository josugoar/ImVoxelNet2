from typing import List, Optional, Tuple

import numpy as np
import torch
from mmdet.structures.bbox import bbox_flip
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import (BaseInstance3DBoxes, Det3DDataSample,
                                xywhr2xyxyr)
from mmdet3d.structures.bbox_3d.utils import points_cam2img, points_img2cam


@MODELS.register_module()
class MonoDet3DTTAModel(BaseTTAModel):

    def __init__(self, num_classes=1, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.tta_cfg = tta_cfg

    def merge_aug_bboxes(self, aug_bboxes_3d: List[BaseInstance3DBoxes],
                         aug_bboxes_2d: Optional[List[Tensor]],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        recovered_bboxes_3d = []
        recovered_bboxes_3d_for_nms = []
        recovered_bboxes_2d = []

        for idx, (bboxes_3d,
                  img_info) in enumerate(zip(aug_bboxes_3d, img_metas)):
            if aug_bboxes_2d is not None:
                bboxes_2d = aug_bboxes_2d[idx]

            ori_shape = img_info['ori_shape']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            cam2img = img_info['cam2img']

            if flip:
                assert flip_direction == 'horizontal'

                centers_2d_with_depth = points_cam2img(
                    bboxes_3d.center.numpy(force=True),
                    cam2img,
                    with_depth=True)
                centers_2d_with_depth[:, 0] = ori_shape[
                    1] - centers_2d_with_depth[:, 0]

                bboxes_3d.tensor[:, :3] = bboxes_3d.tensor.new_tensor(
                    points_img2cam(centers_2d_with_depth, cam2img))
                if bboxes_3d.with_yaw:
                    bboxes_3d.tensor[:, 6] = -bboxes_3d.tensor[:, 6] + np.pi

                if aug_bboxes_2d is not None:
                    bboxes_2d = bbox_flip(
                        bboxes=bboxes_2d,
                        img_shape=ori_shape,
                        direction=flip_direction)

            bboxes_3d_for_nms = xywhr2xyxyr(bboxes_3d.bev)

            recovered_bboxes_3d.append(bboxes_3d.tensor)
            recovered_bboxes_3d_for_nms.append(bboxes_3d_for_nms)
            if aug_bboxes_2d is not None:
                recovered_bboxes_2d.append(bboxes_2d)

        bboxes_3d = torch.cat(recovered_bboxes_3d, dim=0)
        bboxes_3d_for_nms = torch.cat(recovered_bboxes_3d_for_nms, dim=0)
        scores = torch.cat(aug_scores, dim=0)

        if aug_bboxes_2d is None:
            return bboxes_3d, bboxes_3d_for_nms, scores
        else:
            bboxes_2d = torch.cat(recovered_bboxes_2d, dim=0)
            return bboxes_3d, bboxes_3d_for_nms, bboxes_2d, scores

    def merge_preds(self, data_samples_list: List[List[Det3DDataSample]]):
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(
            self, data_samples: List[Det3DDataSample]) -> Det3DDataSample:
        box_type_3d = data_samples[0].metainfo['box_type_3d']
        pred_bbox2d = len(data_samples[0].pred_instances) > 0

        aug_bboxes_3d = []
        aug_bboxes_2d = None
        if pred_bbox2d:
            aug_bboxes_2d = []
        aug_scores = []
        img_metas = []

        for data_sample in data_samples:
            num_boxes = data_sample.pred_instances_3d.bboxes_3d.shape[0]

            bboxes_3d = data_sample.pred_instances_3d.bboxes_3d
            aug_bboxes_3d.append(bboxes_3d)

            if pred_bbox2d:
                bboxes_2d = data_sample.pred_instances.bboxes
                aug_bboxes_2d.append(bboxes_2d)

            labels = data_sample.pred_instances_3d.labels_3d
            indices = labels.new_tensor(list(range(num_boxes)))
            scores = bboxes_3d.tensor.new_zeros(
                (num_boxes, self.num_classes + 1))
            scores[indices, labels] = data_sample.pred_instances_3d.scores_3d
            aug_scores.append(scores)

            img_metas.append(data_sample.metainfo)

        merged_bboxes_3d, merged_bboxes_3d_for_nms, merged_bboxes_2d, \
            merged_scores = self.merge_aug_bboxes(aug_bboxes_3d, aug_bboxes_2d,
                                                  aug_scores, img_metas)

        if merged_bboxes_3d.numel() == 0:
            return data_samples[0]

        nms_results = box3d_multiclass_nms(
            merged_bboxes_3d,
            merged_bboxes_3d_for_nms,
            merged_scores,
            self.tta_cfg.score_thr,
            self.tta_cfg.max_per_img,
            self.tta_cfg,
            mlvl_bboxes2d=merged_bboxes_2d)

        det_bboxes_3d, det_scores, det_labels = nms_results[:3]
        if pred_bbox2d:
            det_bboxes_2d = nms_results[3]

        results_3d = InstanceData()
        _det_bboxes_3d = det_bboxes_3d.clone()
        results_3d.bboxes_3d = box_type_3d(
            _det_bboxes_3d,
            box_dim=aug_bboxes_3d[0].box_dim,
            with_yaw=aug_bboxes_3d[0].with_yaw)
        results_3d.scores_3d = det_scores
        results_3d.labels_3d = det_labels

        results_2d = InstanceData()
        if pred_bbox2d:
            _det_bboxes_2d = det_bboxes_2d.clone()
            results_2d.bboxes = _det_bboxes_2d
            results_2d.scores = det_scores
            results_2d.labels = det_labels

        det_results = data_samples[0]
        det_results.pred_instances = results_2d
        det_results.pred_instances_3d = results_3d

        return det_results
