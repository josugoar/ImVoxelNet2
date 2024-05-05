_base_ = ['./imvoxelnet2_1xb4_kitti-3d-car.py']

model = dict(
    neck_3d=dict(type='ImBEVNeck', feat_channels=64 * 18),
    n_voxels=[324, 372, 18])

train_dataloader = dict(
    batch_size=4)
val_dataloader = dict(
    batch_size=4)
test_dataloader = val_dataloader
