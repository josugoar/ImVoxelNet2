_base_ = 'mmdet3d::imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py'

custom_imports = dict(imports=['projects.ImVoxelNet2.imvoxelnet2'])

model = dict(type='ImVoxelNet2')

meta_keys = [
    'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'num_pts_feats', 'pcd_trans', 'sample_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pcd_rotation_angle', 'lidar_path',
    'transformation_3d_flow', 'trans_mat', 'affine_aug', 'sweep_img_metas',
    'ori_cam2img', 'cam2global', 'crop_offset', 'img_crop_offset',
    'resize_img_shape', 'lidar2cam', 'ori_lidar2img', 'num_ref_frames',
    'num_views', 'ego2global', 'axis_align_matrix', 'plane'
]

backend_args = None

train_pipeline = [
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='RandomResize', scale=[(1173, 352), (1387, 416)],
        keep_ratio=True),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=meta_keys)
]

val_dataloader = dict(batch_size=4, num_workers=4)

train_cfg = dict(val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=-1))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(vis_backends=vis_backends)
