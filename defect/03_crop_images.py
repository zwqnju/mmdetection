
import multiprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

src_root = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/train_imgs/'
dst_root = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/crop_imgs/'
edge_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/edge_infos/'
crop_folder = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/crop_infos/'

def read_edge_info(img_file):
    edge_file = edge_folder + img_file[:-3] + 'json'
    with open(edge_file, 'r') as f:
        info = json.load(f)
    return info

def save_crop_info(img_file, crop_rect):
    crop_file = crop_folder + img_file[:-3] + 'json'
    with open(crop_file, 'w') as f:
        json.dump(crop_rect, f)

def crop(img_file):
    edge_info = read_edge_info(img_file)
    img = cv2.imread(src_root + img_file, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    left, top, right, bottom = width, height, 0, 0
    for x, y in edge_info['corners'].values():
        left = min(left, x)
        right = max(right, x)
        top = min(top, y)
        bottom = max(bottom, y)
    left = int(np.floor(left)) - 1
    top = int(np.floor(top)) - 1
    right = int(np.ceil(right)) - 1
    bottom = int(np.ceil(bottom)) - 1
    save_crop_info(img_file, [left, top, right, bottom])
    cv2.imwrite(dst_root + img_file, img[top:bottom, left:right])
    

if __name__ == '__main__':
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    src_file_list = os.listdir(src_root)
    pool = multiprocessing.Pool(processes=10)
    pbar = tqdm(total=len(src_file_list))
    for img_file in src_file_list:
        pool.apply_async(
            crop, 
            (img_file, ), 
            callback=lambda n: pbar.update()
        )
    pool.close()
    pool.join()