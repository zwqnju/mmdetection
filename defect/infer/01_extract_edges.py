# Canny算子提取边缘
# 得到地板砖边缘

import warnings  
warnings.filterwarnings('error')  
import multiprocessing
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

roi_range = dict(
    CAM1 = dict(
        up=[100, 1000, 2000, 5800],
        down=[5000, 5900, 2000, 5800],
        left=[1000, 5000, 500, 2000],
        right=[1000, 5000, 5800, 7300],
    ),
    CAM2 = dict(
        up=[100, 1100, 2000, 5800],
        down=[5000, 5900, 2000, 5800],
        left=[1000, 5000, 1000, 2500],
        right=[1000, 5000, 6000, 7500],
    ),
    CAM3 = dict(
        up=[200, 1200, 1000, 3200],
        down=[2800, 3450, 1000, 3200],
        left=[1000, 3000, 200, 1200],
        right=[1000, 3000, 3000, 3800],
    ),
)


img_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_testA_20201231/testA_imgs'
output_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_testA_20201231/edge_infos'

def crop(matrix, crop_range):
    top, bottom, left, right = crop_range
    return matrix[top:bottom, left:right].copy()

def get_line(edge_img, cam, loc):
    crop_range = roi_range[cam][loc]
    roi = crop(edge_img, crop_range)
    y_margin, x_margin = crop_range[0], crop_range[2]
    if loc in set(['left', 'right']):
        roi = roi.transpose()
        y_margin, x_margin = x_margin, y_margin
    point_count_per_column = roi.sum(axis=0)
    min_point = max(1, int(point_count_per_column.min()))
    mask = (point_count_per_column <= min_point)
    if mask.sum() < 10:
        min_point += 1
        mask = (point_count_per_column <= min_point)
    assert mask.sum() >= 10, (min_point, (point_count_per_column <= min_point+1).sum())#mask.sum()
    roi[:, ~mask] = False
    y, x = np.where(roi)
    a, b = np.polyfit(x, y, 1)
    b += y_margin - a * x_margin
    return a, b

def get_cross_point(h_line, v_line):
    a1, b1 = h_line
    a2, b2 = v_line
    x = (a2*b1+b2) / (1-a1*a2)
    y = (a1*b2+b1) / (1-a1*a2)
    return x, y

def f2i(num):
    return tuple(np.round(num).astype(np.int).tolist())

def save_img(image, corners, out_file):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for h_loc in ['up', 'down']:
        cv2.line(image, f2i(corners[h_loc+'_left']), f2i(corners[h_loc+'_right']), (0, 0, 255), 25)
    for w_loc in ['left', 'right']:
        cv2.line(image, f2i(corners['up_'+w_loc]), f2i(corners['down_'+w_loc]), (0, 0, 255), 25)
    cv2.imwrite(out_file, image)

def save_edges(lines, corners, out_file):
    with open(out_file, 'w') as f:
        json.dump(dict(lines=lines, corners=corners), f)

def extract_square_edge(src_file, out_file):
    image = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
    edge_img = cv2.Canny(image, 100, 150).astype(np.bool)
    height, width = edge_img.shape
    cam = src_file[-8:-4]
    assert cam in roi_range, cam
    lines = {}
    for loc in ['up', 'down', 'left', 'right']:
        with warnings.catch_warnings():
            try:
                lines[loc] = get_line(edge_img, cam, loc)
            except np.RankWarning:
                print(img_file)
                exit()
            except TypeError as e:
                print(img_file)
                raise e
            except AssertionError as e:
                print(img_file)
                raise e
    corners = {}
    for h_loc in ['up', 'down']:
        for w_loc in ['left', 'right']:
            corners[h_loc+'_'+w_loc] = get_cross_point(lines[h_loc], lines[w_loc])
    # save_img(image, corners, out_file)
    save_edges(lines, corners, out_file[:-3]+'json')

if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        exist_result = []
    else:
        exist_result = os.listdir(output_folder)
    exist_file_set = set(exist_result)
    src_file_list = []
    for img_file in os.listdir(img_folder):
        if img_file[:-3] + 'json' not in exist_file_set:
            src_file_list.append(img_file)
    # print(src_file_list[278])
    # exit()
    pool = multiprocessing.Pool(processes=10)
    pbar = tqdm(total=len(src_file_list))
    for img_file in src_file_list:
        if img_file[:4] in set(['198_']):
            continue
        pool.apply_async(extract_square_edge, (
            os.path.join(img_folder, img_file),
            os.path.join(output_folder, img_file),
        ), callback=lambda n: pbar.update())
        extract_square_edge(
            os.path.join(img_folder, img_file),
            os.path.join(output_folder, img_file),
        )
    pool.close()
    pool.join()