_base_ = './imvoxelnet2-bev_8xb4_kitti-3d-car.py'

model = dict(voxel_pooling="avg")
