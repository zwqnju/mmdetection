_base_ = '../faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco.py'

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
    samples_per_gpu=2,
    workers_per_gpu=2,
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
        bbox_head=dict(
            num_classes=6,
        )
    )
)
load_from = 'checkpoints/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth'
optimizer = dict(
    lr=0.02/2,
)

evaluation = dict(
    metric='mAP',
)