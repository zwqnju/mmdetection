import json

img_json = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/img_info.json'
anno_json = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/rotate_annos.json'
dst_json = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/img_rotate_annos.json'

with open(img_json, 'r') as f:
    img_info = json.load(f)

with open(anno_json, 'r') as f:
    anno_info = json.load(f)

with open(dst_json, 'w') as f:
    json.dump(dict(img=img_info, anno=anno_info), f)