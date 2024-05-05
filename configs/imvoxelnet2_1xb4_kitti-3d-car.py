_base_ = ['../../../configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py']

custom_imports = dict(imports=['projects.ImVoxelNet2.imvoxelnet2'])

model = dict(
    type='ImVoxelNet2',
    backbone=dict(
        depth=18,
        init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=64))

train_dataloader = dict(
    dataset=dict(
        times=1))
val_dataloader = dict(
    batch_size=4,
    num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(val_interval=3)

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=-1))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    vis_backends=vis_backends)
