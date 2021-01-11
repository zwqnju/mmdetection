_base_ = [
    '../cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco.py'
]

dataset_type = 'DefectDataset'
classes = (
    "边异常", 
    "角异常", 
    "白色点瑕疵", 
    "浅色块瑕疵", 
    "深色点块瑕疵", 
    "光圈瑕疵"
)
data_root = '/media/samba/weiqiang/Datasets/dataset/'
data = dict(
    train=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root + 'tile_round1_train_20201231/',
        ann_file=data_root + 'tile_round1_train_20201231/train_annos.json',
        img_prefix=data_root + 'tile_round1_train_20201231/train_imgs/',
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root + 'tile_round1_train_20201231/',
        ann_file=data_root + 'tile_round1_train_20201231/train_annos.json',
        img_prefix=data_root + 'tile_round1_train_20201231/train_imgs/',
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root + 'tile_round1_testA_20201231/',
        ann_file=None,
        img_prefix=data_root + 'tile_round1_testA_20201231/testA_imgs/',
    ),
)


model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    )
)

optimizer = dict(
    lr=0.02/4,
)

evaluation = dict(
    metric='mAP',
)
load_from = 'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth'