import numpy as np
from tqdm import tqdm
import json

src_result_file = './0_try/0121_crop_nms0.1.json'
dst_result_file = './0_try/0121_initial_nms0.1.json'
crop_folder = '/home/allride/workspace/2021/Detection/dataset/crop_testA_set/crop_infos/'

def read_crop_info(img_file):
    crop_file = crop_folder + img_file[:-3] + 'json'
    with open(crop_file, 'r') as f:
        info = json.load(f)
    return info

if __name__ == '__main__':
    with open(src_result_file, 'r') as f:
        src_results = json.load(f)

    dst_results = []
    for src_result in tqdm(src_results):
        name = src_result['name']
        left, top, right, bottom = read_crop_info(name)
        bbox = src_result['bbox']
        bbox[0] += left
        bbox[1] += top
        bbox[2] += left
        bbox[3] += top
        dst_results.append(src_result)
    
    with open(dst_result_file, 'w') as f:
        json.dump(dst_results, f)