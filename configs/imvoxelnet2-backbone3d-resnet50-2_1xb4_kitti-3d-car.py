_base_ = ['./imvoxelnet2_1xb4_kitti-3d-car.py']

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=64),
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
        in_channels=[64 * 4, 128 * 4, 256 * 4, 512 * 4],
        out_channels=64 * 4,
        num_outs=4),
    bbox_head=dict(
        in_channels=64 * 4,
        feat_channels=64 * 4,
        anchor_generator=dict(
            scales=[1, 1, 1, 1])),
    n_voxels=[216 * 2, 248 * 2, 12])

train_dataloader = dict(
    batch_size=4)
val_dataloader = dict(
    batch_size=4)
test_dataloader = val_dataloader
