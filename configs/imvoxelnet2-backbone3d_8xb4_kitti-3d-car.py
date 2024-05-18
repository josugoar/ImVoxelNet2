_base_ = './imvoxelnet2_8xb4_kitti-3d-car.py'

model = dict(
    backbone_3d=dict(
        type='ImBEVResNet',
        depth=50,
        in_channels=64 * 12,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_eval=True,
        style='pytorch'),
    neck_3d=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=1,
        end_level=0),
    n_voxels=[216 * 2, 248 * 2, 12])
