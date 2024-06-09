_base_ = 'mmdet3d::imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py'

custom_imports = dict(imports=['projects.ImVoxelNet2.imvoxelnet2'])

model = dict(type='ImVoxelNet2')

backend_args = None

tta_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='mmdet.Resize', scale_factor=scale_factor)
                for scale_factor in [0.95, 1.0, 1.05]
            ],
            [
                dict(
                    type='RandomFlip3D',
                    flip_ratio_bev_horizontal=flip_ratio_bev_horizontal,
                    flip_box3d=False)
                for flip_ratio_bev_horizontal in [0.0, 1.0]
            ],
            [
                dict(
                    type='Pack3DDetInputs',
                    keys=['img'],
                    meta_keys=[
                        'img_path', 'ori_shape', 'img_shape', 'lidar2img',
                        'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                        'flip', 'flip_direction', 'pcd_horizontal_flip',
                        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                        'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                        'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                        'pcd_rotation_angle', 'lidar_path',
                        'transformation_3d_flow', 'trans_mat', 'affine_aug',
                        'sweep_img_metas', 'ori_cam2img', 'cam2global',
                        'crop_offset', 'img_crop_offset', 'resize_img_shape',
                        'lidar2cam', 'ori_lidar2img', 'num_ref_frames',
                        'num_views', 'ego2global', 'axis_align_matrix'
                    ])
            ]
        ])
]

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)

train_cfg = dict(val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=-1))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(
    type='MonoDet3DTTAModel',
    num_classes=1,
    tta_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.05,
        score_thr=0.001,
        min_bbox_size=0,
        max_per_img=20))
