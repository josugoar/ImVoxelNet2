_base_ = ['./imvoxelnet2_1xb4_kitti-3d-car.py']

model = dict(
    neck_3d=dict(type='ImBEVNeck', feat_channels=64 * 12))

train_dataloader = dict(
    batch_size=8)
val_dataloader = dict(
    batch_size=8)
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=24)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
