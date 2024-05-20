_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(
    backbone_3d=dict(
        type='SECOND',
        in_channels=64 * 12,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck_3d=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(in_channels=512, feat_channels=512),
    bev=True)

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
