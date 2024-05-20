_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(neck_3d=dict(type='ImBEVNeck', feat_channels=64 * 12))

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
