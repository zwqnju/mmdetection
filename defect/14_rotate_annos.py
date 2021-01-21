
import multiprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

def src(arr):
    name = src_anno['name']
    height = src_anno['image_height']
    width = src_anno['image_width']
    bbox = src_anno['bbox']
    return dict(
        name = name[:-4] + '-src.jpg',
        image_height = height,
        image_width = width,
        category = src_anno['category'],
        bbox = bbox
    )

def flip180(arr):
    name = src_anno['name']
    height = src_anno['image_height']
    width = src_anno['image_width']
    bbox = src_anno['bbox']
    return dict(
        name = name[:-4] + '-180.jpg',
        image_height = height,
        image_width = width,
        category = src_anno['category'],
        bbox = [width-bbox[2], height-bbox[3], width-bbox[0], height-bbox[1]],
    )

def flip90_left(anno):
    name = src_anno['name']
    height = src_anno['image_height']
    width = src_anno['image_width']
    bbox = src_anno['bbox']
    return dict(
        name = name[:-4] + '-90l.jpg',
        image_height = width,
        image_width = height,
        category = src_anno['category'],
        bbox = [bbox[1], width-bbox[2], bbox[3], width-bbox[0]],
    )

def flip90_right(arr):
    name = src_anno['name']
    height = src_anno['image_height']
    width = src_anno['image_width']
    bbox = src_anno['bbox']
    return dict(
        name = name[:-4] + '-90r.jpg',
        image_height = width,
        image_width = height,
        category = src_anno['category'],
        bbox = [height-bbox[3], bbox[0], height-bbox[1], bbox[2]],
    )

src_anno = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/split_annos.json'
dst_anno = '/media/samba/weiqiang/Datasets/dataset/crop_train_set/rotate_annos.json'

if __name__ == '__main__':
    
    with open(src_anno, 'r') as f:
        src_annos = json.load(f)
    
    dst_annos = []
    for src_anno in src_annos:
        dst_annos.append(src(src_anno))
        dst_annos.append(flip90_left(src_anno))
        dst_annos.append(flip90_right(src_anno))
        dst_annos.append(flip180(src_anno))
    
    with open(dst_anno, 'w') as f:
        json.dump(dst_annos, f)
