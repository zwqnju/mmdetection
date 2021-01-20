
import multiprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

OVERLAP = 100

src_root = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/crop_imgs/'
dst_root = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/split_imgs/'
split_folder = '/home/allride/workspace/2021/Detection/dataset/crop_train_set/split_infos/'

def save_crop_img_and_info(img_file, number, img, crop_rect):
    name = img_file[:-4] + '-%d' % number
    img_file = dst_root + name + '.jpg'
    left, top, right, bottom = crop_rect
    cv2.imwrite(img_file, img[top:bottom, left:right])

    json_file = split_folder + name + '.json'
    with open(json_file, 'w') as f:
        json.dump(crop_rect, f)

def crop(img_file):
    img = cv2.imread(src_root + img_file, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    if img_file[-5] == '3':
        split = 2
    else:
        split = 3
    each_width = (width + OVERLAP * split) // split
    each_height = (height + OVERLAP * split) // split
    left_list = [(each_width-OVERLAP) * i for i in range(split)]
    right_list = [width - (each_width-OVERLAP) * i for i in range(split)][::-1]
    top_list = [(each_height-OVERLAP) * i for i in range(split)]
    bottom_list = [height - (each_height-OVERLAP) * i for i in range(split)][::-1]

    number = 0
    for top, bottom in zip(top_list, bottom_list):
        for left, right in zip(left_list, right_list):
            number += 1
            save_crop_img_and_info(img_file, number, img, [left, top, right, bottom])
    

if __name__ == '__main__':
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
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