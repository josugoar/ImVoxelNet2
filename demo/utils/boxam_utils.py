import bisect
import copy
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmdet.utils import ConfigType
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet3d.apis import convert_SyncBN
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import box3d_to_bbox, get_box_type, points_cam2img

try:
    from pytorch_grad_cam import (AblationCAM, AblationLayer,
                                  ActivationsAndGradients)
    from pytorch_grad_cam import GradCAM as Base_GradCAM
    from pytorch_grad_cam import GradCAMPlusPlus as Base_GradCAMPlusPlus
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    pass


def init_detector(config: Union[str, Path, Config],
                  checkpoint: Optional[str] = None,
                  device: str = 'cuda:0',
                  palette: str = 'none',
                  cfg_options: Optional[dict] = None):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    convert_SyncBN(config.model)
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': config.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                      'Some functions are not supported for now.')

    model.to(device)
    model.eval()
    return model


def reshape_transform(feats: Union[Tensor, List[Tensor]],
                      max_shape: Tuple[int, int] = (20, 20),
                      is_need_grad: bool = False):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.

    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if len(max_shape) == 1:
        max_shape = max_shape * 2

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    else:
        if is_need_grad:
            raise NotImplementedError('The `grad_base` method does not '
                                      'support output multi-activation layers')

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class BoxAMDetectorWrapper(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self,
                 cfg: ConfigType,
                 checkpoint: str,
                 score_thr: float,
                 device: str = 'cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.detector = init_detector(self.cfg, self.checkpoint, device=device)

        pipeline_cfg = copy.deepcopy(self.cfg.test_dataloader.dataset.pipeline)
        pipeline_cfg[0].type = 'mmdet3d.LoadImageFromNDArray'

        new_test_pipeline = []
        for pipeline in pipeline_cfg:
            # FIX: AttributeError: 'InstanceData' object has no attribute 'bboxes_3d'
            if pipeline['type'].endswith('Pack3DDetInputs'):
                pipeline['keys'].extend(['gt_bboxes',
                                         'gt_bboxes_labels',
                                         'gt_bboxes_3d',
                                         'gt_labels_3d',
                                         'centers_2d',
                                         'depths'])
            if not pipeline['type'].endswith('LoadAnnotations3D'):
                new_test_pipeline.append(pipeline)
        self.test_pipeline = Compose(new_test_pipeline)

        self.is_need_loss = False
        self.input_data = None
        self.image = None

    def need_loss(self, is_need_loss: bool):
        """Grad-based methods require loss."""
        self.is_need_loss = is_need_loss

    def set_input_data(self,
                       image: np.ndarray,
                       ann_file: str,
                       cam_type: str,
                       pred_instances: Optional[InstanceData] = None):
        """Set the input data to be used in the next step."""
        self.image = image

        data_list = mmengine.load(ann_file)['data_list']
        data_info = data_list[0]

        box_type_3d, box_mode_3d = get_box_type('camera')

        if self.is_need_loss:
            assert pred_instances is not None
            pred_instances = pred_instances.numpy()
            centers_2d_with_depth = points_cam2img(
                pred_instances.bboxes_3d.gravity_center.numpy(force=True),
                data_info['images'][cam_type]['cam2img'],
                with_depth=True)
            data = dict(
                img=self.image,
                img_id=0,
                **data_info['images'][cam_type],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d,
                gt_bboxes=box3d_to_bbox(
                    pred_instances.bboxes_3d.tensor.numpy(force=True),
                    data_info['images'][cam_type]['cam2img']),
                gt_bboxes_labels=pred_instances.labels_3d,
                gt_bboxes_3d=pred_instances.bboxes_3d,
                gt_labels_3d=pred_instances.labels_3d,
                centers_2d=centers_2d_with_depth[:, :2],
                depths=centers_2d_with_depth[:, 2])
            data = self.test_pipeline(data)
        else:
            data = dict(img=self.image,
                        img_id=0,
                        **data_info['images'][cam_type],
                        box_type_3d=box_type_3d,
                        box_mode_3d=box_mode_3d)
            data = self.test_pipeline(data)
            # FIX: KeyError: 'imgs'
            data['inputs'] = data['inputs']
            data['data_samples'] = data['data_samples']
        # FIX: AssertionError: The input of `ImgDataPreprocessor` should be a NCHW tensor or a list of tensor, but got a tensor with shape: torch.Size([C, H, W])
        data = pseudo_collate([data])
        self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.is_need_loss:
            # Maybe this is a direction that can be optimized
            # self.detector.init_weights()
            if hasattr(self.detector.bbox_head, 'head_module'):
                self.detector.bbox_head.head_module.training = True
            else:
                self.detector.bbox_head.training = True
            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                # Prevent the model algorithm error when calculating loss
                self.detector.bbox_head.featmap_sizes = None

            data_ = {}
            data_['inputs'] = self.input_data['inputs']
            data_['data_samples'] = self.input_data['data_samples']
            data = self.detector.data_preprocessor(data_, training=False)
            loss = self.detector._run_forward(data, mode='loss')

            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                self.detector.bbox_head.featmap_sizes = None

            return [loss]
        else:
            if hasattr(self.detector.bbox_head, 'head_module'):
                self.detector.bbox_head.head_module.training = False
            else:
                self.detector.bbox_head.training = False
            with torch.no_grad():
                results = self.detector.test_step(self.input_data)
                return results


class BoxAMDetectorVisualizer:
    """Box AM visualization class."""

    def __init__(self,
                 method_class,
                 model: nn.Module,
                 target_layers: List,
                 reshape_transform: Optional[Callable] = None,
                 is_need_grad: bool = False,
                 extra_params: Optional[dict] = None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == 'AblationCAM':
            batch_size = extra_params.get('batch_size', 1)
            ratio_channels_to_ablate = extra_params.get(
                'ratio_channels_to_ablate', 1.)
            self.cam = AblationCAM(
                model,
                target_layers,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params['ablation_layer'],
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        else:
            self.cam = method_class(
                model,
                target_layers,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        self.classes = model.detector.dataset_meta['classes']
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def switch_activations_and_grads(self, model) -> None:
        """In the grad-based method, we need to switch
        ``ActivationsAndGradients`` layer, otherwise an error will occur."""
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_am(self,
                image: np.ndarray,
                pred_instance_3d: InstanceData,
                grayscale_am: np.ndarray,
                cam2img: np.ndarray,
                with_norm_in_bboxes: bool = False):
        """Normalize the AM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""

        bboxes_3d = pred_instance_3d.bboxes_3d
        labels_3d = pred_instance_3d.labels_3d

        boxes = box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img)
        labels = labels_3d

        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_am = np.zeros(grayscale_am.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_am * 0
                img[y1:y2, x1:x2] = scale_cam_image(
                    [grayscale_am[y1:y2, x1:x2].copy()])[0]
                images.append(img)

            renormalized_am = np.max(np.float32(images), axis=0)
            renormalized_am = scale_cam_image([renormalized_am])[0]
        else:
            renormalized_am = grayscale_am

        am_image_renormalized = show_cam_on_image(
            image / 255, renormalized_am, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(
            boxes,
            labels,
            am_image_renormalized,
            pred_instance_3d.get('scores_3d'))
        return image_with_bounding_boxes

    def _draw_boxes(self,
                    boxes: List,
                    labels: List,
                    image: np.ndarray,
                    scores: Optional[List] = None):
        """draw boxes on image."""
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            if scores is not None:
                score = scores[i]
                text = str(self.classes[label]) + ': ' + str(
                    round(score * 100, 1))
            else:
                text = self.classes[label]

            cv2.putText(
                image,
                text, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image


class DetAblationLayer(AblationLayer):
    """Det AblationLayer."""

    def __init__(self):
        super().__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super().set_next_batch(input_batch_index, activations,
                                          num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices."""
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super().__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetBoxScoreTarget:
    """Det Score calculation class.

    In the case of the grad-free method, the calculation method is that
    for every original detected bounding box specified in "bboxes_3d",
    assign a score on how the current bounding boxes match it,

        1. In Bbox IoU
        2. In the classification score.

    If there is not a large enough overlap, or the category changed,
    assign a score of 0. The total score is the sum of all the box scores.

    In the case of the grad-based method, the calculation method is
    the sum of losses after excluding a specific key.
    """

    def __init__(self,
                 pred_instance_3d: InstanceData,
                 match_iou_thr: float = 0.5,
                 device: str = 'cuda:0',
                 ignore_loss_params: Optional[List] = None):
        self.focal_bboxes_3d = pred_instance_3d.bboxes_3d
        self.focal_labels_3d = pred_instance_3d.labels_3d
        self.match_iou_thr = match_iou_thr
        self.device = device
        if ignore_loss_params is not None:
            self.ignore_loss_params = ignore_loss_params
        else:
            self.ignore_loss_params = []
        assert isinstance(self.ignore_loss_params, list)

    def __call__(self, results):
        output = torch.tensor([0.], device=self.device)

        if 'loss_cls' in results:
            # grad-based method
            # results is dict
            for loss_key, loss_value in results.items():
                if 'loss' not in loss_key or \
                        loss_key in self.ignore_loss_params:
                    continue
                if isinstance(loss_value, list):
                    output += sum(loss_value)
                else:
                    output += loss_value
            return output
        else:
            # grad-free method
            # results is DetDataSample
            pred_instances_3d = results.pred_instances_3d
            if len(pred_instances_3d) == 0:
                return output

            pred_bboxes_3d = pred_instances_3d.bboxes_3d
            pred_scores_3d = pred_instances_3d.scores_3d
            pred_labels_3d = pred_instances_3d.labels_3d

            # FIX: AssertionError: Indexing on Boxes with (slice(None, None, None), 2) failed to return a matrix!
            pred_bboxes_3d = pred_bboxes_3d.tensor

            for focal_box, focal_label in zip(self.focal_bboxes_3d,
                                              self.focal_labels_3d):
                ious = mmcv.ops.boxes_iou3d(focal_box[None],
                                            pred_bboxes_3d[..., :7])
                index = ious.argmax()
                if ious[0, index] > self.match_iou_thr and pred_labels_3d[
                        index] == focal_label:
                    # TODO: Adaptive adjustment of weights based on algorithms
                    score = ious[0, index] + pred_scores_3d[index]
                    output = output + score
            return output


class SpatialBaseCAM(BaseCAM):
    """CAM that maintains spatial information.

    Gradients are often averaged over the spatial dimension in CAM
    visualization for classification, but this is unreasonable in detection
    tasks. There is no need to average the gradients in the detection task.
    """

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor, target_layer, targets,
                                       activations, grads)
        weighted_activations = weights * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam


class GradCAM(SpatialBaseCAM, Base_GradCAM):
    """Gradients are no longer averaged over the spatial dimension."""

    def get_cam_weights(self, input_tensor, target_layer, target_category,
                        activations, grads):
        return grads


class GradCAMPlusPlus(SpatialBaseCAM, Base_GradCAMPlusPlus):
    """Gradients are no longer averaged over the spatial dimension."""

    def get_cam_weights(self, input_tensor, target_layers, target_category,
                        activations, grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 +
            sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        return weights
