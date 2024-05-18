_base_ = 'mmdet3d::imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py'

custom_imports = dict(imports=['projects.ImVoxelNet2.imvoxelnet2'])

model = dict(type='ImVoxelNet2')

train_dataloader = dict(dataset=dict(times=3))
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=-1))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(vis_backends=vis_backends)
