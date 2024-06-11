import os.path as osp
from typing import Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmdet3d.engine import Det3DVisualizationHook
from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample


@HOOKS.register_module()
class BEVDet3DVisualizationHook(Det3DVisualizationHook):

    def __init__(self,
                 *args,
                 draw_bev: bool = False,
                 bev_shape: int = 900,
                 scale: int = 15,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_bev = draw_bev
        self.bev_shape = bev_shape
        self.scale = scale

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[Det3DDataSample]) -> None:
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        data_input = dict()

        # Visualize only the first data
        if self.vis_task in [
                'mono_det', 'multi-view_det', 'multi-modality_det'
        ]:
            assert 'img_path' in outputs[0], 'img_path is not in outputs[0]'
            img_path = outputs[0].img_path
            if isinstance(img_path, list):
                img = []
                for single_img_path in img_path:
                    img_bytes = get(
                        single_img_path, backend_args=self.backend_args)
                    single_img = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    img.append(single_img)
            else:
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_input['img'] = img

        if self.vis_task in ['lidar_det', 'multi-modality_det', 'lidar_seg']:
            assert 'lidar_path' in outputs[
                0], 'lidar_path is not in outputs[0]'
            lidar_path = outputs[0].lidar_path
            num_pts_feats = outputs[0].num_pts_feats
            pts_bytes = get(lidar_path, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, num_pts_feats)
            data_input['points'] = points

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                'val sample',
                data_input,
                data_sample=outputs[0],
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter,
                show_pcd_rgb=self.show_pcd_rgb)

            if self.draw_bev:
                self._visualizer.set_bev_image(bev_shape=self.bev_shape)
                self._visualizer.draw_bev_bboxes(
                    outputs[0].pred_instances_3d.bboxes_3d,
                    scale=self.scale,
                    edge_colors='orange')
                if self.show:
                    self._visualizer.show(
                        win_name='val sample',
                        wait_time=self.wait_time,
                        vis_task=self.vis_task)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            data_input = dict()
            assert 'img_path' in data_sample or 'lidar_path' in data_sample, \
                "'data_sample' must contain 'img_path' or 'lidar_path'"

            out_file = o3d_save_path = None

            if self.vis_task in [
                    'mono_det', 'multi-view_det', 'multi-modality_det'
            ]:
                assert 'img_path' in data_sample, \
                    'img_path is not in data_sample'
                img_path = data_sample.img_path
                if isinstance(img_path, list):
                    img = []
                    for single_img_path in img_path:
                        img_bytes = get(
                            single_img_path, backend_args=self.backend_args)
                        single_img = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img.append(single_img)
                else:
                    img_bytes = get(img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                data_input['img'] = img
                if self.test_out_dir is not None:
                    if isinstance(img_path, list):
                        img_path = img_path[0]
                    out_file = osp.basename(img_path)
                    out_file = osp.join(self.test_out_dir, out_file)

            if self.vis_task in [
                    'lidar_det', 'multi-modality_det', 'lidar_seg'
            ]:
                assert 'lidar_path' in data_sample, \
                    'lidar_path is not in data_sample'
                lidar_path = data_sample.lidar_path
                num_pts_feats = data_sample.num_pts_feats
                pts_bytes = get(lidar_path, backend_args=self.backend_args)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, num_pts_feats)
                data_input['points'] = points
                if self.test_out_dir is not None:
                    o3d_save_path = osp.basename(lidar_path).split(
                        '.')[0] + '.png'
                    o3d_save_path = osp.join(self.test_out_dir, o3d_save_path)

            self._visualizer.add_datasample(
                'test sample',
                data_input,
                data_sample=data_sample,
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                o3d_save_path=o3d_save_path,
                step=self._test_index,
                show_pcd_rgb=self.show_pcd_rgb)

            if self.draw_bev:
                self._visualizer.set_bev_image(bev_shape=self.bev_shape)
                self._visualizer.draw_bev_bboxes(
                    data_sample.pred_instances_3d.bboxes_3d,
                    scale=self.scale,
                    edge_colors='orange')
                if self.show:
                    self._visualizer.show(
                        win_name='test sample',
                        wait_time=self.wait_time,
                        vis_task=self.vis_task)
