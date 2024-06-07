_base_ = './imvoxelnet_8xb4_kitti-3d-car.py'

model = dict(
    neck_3d=dict(type='ImBEVNeck'),
    bev=True,
    middle_in_channels=64 * 12,
    middle_out_channels=64)

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
