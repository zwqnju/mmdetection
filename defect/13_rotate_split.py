
import multiprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr

src_root = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/split_imgs/'
dst_root = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/rotate_imgs/'
img_info = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/img_info.json'

def rotate(img_file):
    img = cv2.imread(src_root + img_file, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    src_img = img_file[:-4]+'-src.jpg'
    left_img = img_file[:-4]+'-90l.jpg'
    right_img = img_file[:-4]+'-90r.jpg'
    up_down_img = img_file[:-4]+'-180.jpg'
    cv2.imwrite(dst_root + src_img, img)
    cv2.imwrite(dst_root + left_img, flip90_left(img))
    cv2.imwrite(dst_root + right_img, flip90_right(img))
    cv2.imwrite(dst_root + up_down_img, flip180(img))
    return [
        [src_img, height, width],
        [left_img, width, height],
        [right_img, width, height],
        [up_down_img, height, width],
    ]

if __name__ == '__main__':
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    src_file_list = os.listdir(src_root)

    img_size = []
    def add(img_size, result, pbar):
        img_size += result
        pbar.update()
    
    pool = multiprocessing.Pool(processes=10)
    pbar = tqdm(total=len(src_file_list))
    for img_file in src_file_list:
        pool.apply_async(
            rotate, 
            (img_file, ), 
            callback=lambda info: add(img_size, info, pbar)
        )
    pool.close()
    pool.join()
    with open(img_info, 'w') as f:
        json.dump(img_size, f)