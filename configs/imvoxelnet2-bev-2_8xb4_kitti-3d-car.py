_base_ = './imvoxelnet_8xb4_kitti-3d-car.py'

model = dict(
    backbone=dict(depth=101),
    neck_3d=dict(type='ImBEVNeck', in_channels=128, out_channels=512),
    bbox_head=dict(in_channels=512, feat_channels=512),
    bev=True,
    middle_in_channels=64 * 16,
    middle_out_channels=128,
    n_voxels=[288, 330, 16])
