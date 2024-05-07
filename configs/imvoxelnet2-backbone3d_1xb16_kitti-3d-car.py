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
        depth=18,
        in_channels=64 * 12,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_eval=True,
        style='pytorch'),
    neck_3d=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            scales=[1, 1, 1, 1])))

train_dataloader = dict(
    batch_size=16)
val_dataloader = dict(
    batch_size=16)
test_dataloader = val_dataloader
