_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(
    backbone_3d=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[64, 128]),
    neck_3d=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128]),
    bev=True,
    middle_in_channels=64 * 12,
    middle_out_channels=64)

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
