_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(
    neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[64, 64, 64, 64]),
    backbone_3d=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[128, 256, 512]),
    neck_3d=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[64, 128, 256, 512],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[64, 64, 64, 64]),
    bbox_head=dict(in_channels=256, feat_channels=256),
    n_voxels=[224, 256, 12],
    bev=True,
    middle_in_channels=256 * 12,
    middle_out_channels=64)

find_unused_parameters = False
