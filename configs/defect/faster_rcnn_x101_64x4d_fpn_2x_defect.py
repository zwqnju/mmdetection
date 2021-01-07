_base_ = './faster_rcnn_r50_fpn_2x_defect.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'
    ),
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


# data = dict(
#     train=dict(
#         classes=classes,
#     )
# )