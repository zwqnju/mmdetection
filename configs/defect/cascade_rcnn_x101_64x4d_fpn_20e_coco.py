_base_ = [
    '../cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py',
    './defect_base.py'
]

load_from = 'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'