_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(
    neck_3d=dict(type='ImBEVNeck', feat_channels=64 * 18),
    n_voxels=[324, 372, 18],
    bev=True)
