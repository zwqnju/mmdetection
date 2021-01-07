dataset_type = 'CocoDataset'
data_root = '/media/samba/weiqiang/Datasets/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = (
    "边异常", 
    "角异常", 
    "白色点瑕疵", 
    "浅色块瑕疵", 
    "深色点块瑕疵", 
    "光圈瑕疵"
)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'tile_round1_train_20201231/train_coco.json',
        img_prefix=data_root + 'tile_round1_train_20201231/train_imgs/',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'tile_round1_train_20201231/train_coco.json',
        img_prefix=data_root + 'tile_round1_train_20201231/train_imgs/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'tile_round1_testA_20201231/test_annos.json',
        img_prefix=data_root + 'tile_round1_testA_20201231/testA_imgs/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
