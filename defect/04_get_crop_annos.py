import numpy as np
from tqdm import tqdm
import json

src_anno_file = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/train_annos.json'
dst_anno_file = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/train_annos.json'
crop_folder = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/crop_infos/'

def read_crop_info(img_file):
    crop_file = crop_folder + img_file[:-3] + 'json'
    with open(crop_file, 'r') as f:
        info = json.load(f)
    return info

if __name__ == '__main__':
    with open(src_anno_file, 'r') as f:
        src_annos = json.load(f)

    dst_annos = []
    for src_anno in tqdm(src_annos):
        name = src_anno['name']
        left, top, right, bottom = read_crop_info(name)
        bbox = src_anno['bbox']
        bbox[0] -= left
        bbox[1] -= top
        bbox[2] -= left
        bbox[3] -= top
        dst_annos.append(dict(
            name=name,
            image_height=bottom-top,
            image_width=right-left,
            category=src_anno['category'],
            bbox=bbox,
        ))
    
    with open(dst_anno_file, 'w') as f:
        json.dump(dst_annos, f)