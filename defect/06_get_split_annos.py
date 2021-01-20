import numpy as np
from tqdm import tqdm
import json

src_anno_file = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/train_annos.json'
dst_anno_file = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/split_annos.json'
split_folder = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/split_infos/'

def read_split_info(spilt_file):
    spilt_file = split_folder + split_file
    with open(spilt_file, 'r') as f:
        info = json.load(f)
    return info

def iou(box1, box2):
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    area1 = (r1-l1)*(b1-t1)
    max_l = max(l1, l2)
    min_r = min(r1, r2)
    max_t = max(t1, t2)
    min_b = min(b1, b2)
    if min_r <= max_l or min_b <= max_t:
        return 0
    area_i = (min_r - max_l) * (min_b - max_t)
    return area_i / area1

if __name__ == '__main__':
    with open(src_anno_file, 'r') as f:
        src_annos = json.load(f)

    dst_annos = []
    for src_anno in tqdm(src_annos):
        name = src_anno['name']
        bbox = src_anno['bbox']
        if name[-5] == '3':
            total = 4
        else:
            total = 9
        for number in range(1, total+1):
            split_file = name[:-4] + '-%d.json' % number
            split_box = read_split_info(split_file)
            if iou(bbox, split_box) > 0.1:
                width = split_box[2] - split_box[0]
                height = split_box[3] - split_box[1] 
                new_left = bbox[0] - split_box[0]
                new_top = bbox[1] - split_box[1]
                new_right = bbox[2] - split_box[0]
                new_bottom = bbox[3] - split_box[1]
                new_left = max(0, min(width, new_left))
                new_right = max(0, min(width, new_right))
                new_top = max(0, min(height, new_top))
                new_bottom = max(0, min(height, new_bottom))
                dst_annos.append(dict(
                    name=name[:-4] + '-%d.jpg' % number,
                    image_height=height,
                    image_width=width,
                    category=src_anno['category'],
                    bbox=[new_left, new_top, new_right, new_bottom],
                ))
        
    
    with open(dst_anno_file, 'w') as f:
        json.dump(dst_annos, f)