dataset_type = 'TianchiImageDataset'
data_root = '/data/tianchi/'
img_norm_cfg = dict(
    mean=[41.1, 41.1, 41.1], std=[45.73, 45.73, 45.73], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TianchiLoadAnnotations'),
    dict(type='TianchiResize', img_scale=(512, 512), keep_ratio=True),
    dict(type='TianchiRandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512)),
    dict(type='TianchiFormatBundle'),
    dict(
        type='Collect',
        # keys=['img', 'gt_bboxes', 'gt_labels', 'gt_points', 'part_inds'])
        keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(512, 512)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lumbar_train150.json',
        img_prefix=data_root + 'images/lumbar_train150',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lumbar_train51.json',
        img_prefix=data_root + 'image/lumbar_train51',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lumbar_testA50.ext.json',
        img_prefix=data_root + 'images/lumbar_testA50',
        pipeline=test_pipeline,
        test_mode=True))
