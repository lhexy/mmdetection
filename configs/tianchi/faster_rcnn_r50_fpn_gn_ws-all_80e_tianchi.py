_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    './_base_/tianchi_lumbar.py',
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    type='TianchiRCNN',
    backbone=dict(frozen_stages=-1, conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        type='TianchiRoIHead',
        bbox_head=dict(
            type='TianchiBBoxHead',
            num_classes=11,
            num_tags=7,
            conv_out_channels=256,
            reg_class_agnostic=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)))
lr_config = dict(step=[72, 78])
checkpoint_config = dict(interval=5)
log_config = dict(interval=15)
total_epochs = 80
