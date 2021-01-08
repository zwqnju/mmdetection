_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

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
    samples_per_gpu=4,
    workers_per_gpu=4,
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
            num_classes=6
        )
    )
)

load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
optimizer = dict(
    lr=0.02/4,
)
lr_config = dict(
    warmup=None,
)

evaluation = dict(
    metric='mAP',
)