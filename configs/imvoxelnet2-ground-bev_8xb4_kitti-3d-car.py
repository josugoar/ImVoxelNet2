_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(neck_3d=dict(type='ImBEVNeck'), use_ground_plane=True)

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
_base_.train_pipeline[-1].update(meta_keys=meta_keys)
_base_.test_pipeline[-1].update(meta_keys=meta_keys)

train_dataloader = dict(
    batch_size=8,
    dataset=dict(dataset=dict(pipeline=_base_.train_pipeline)))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader
