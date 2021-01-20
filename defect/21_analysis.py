
import cv2
import numpy as np
import json
from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm

img_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/train_imgs/'
edge_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/edge_infos/'
root_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/'
anno_file = root_folder + 'train_annos.json'

def get_anno_dict(anno_file):
    with open(anno_file, 'r') as f:
        annos = json.load(f)
    anno_dict = {}
    for anno in annos:
        name = anno['name']
        if name not in anno_dict:
            anno_dict[name] = []
        anno_dict[name].append(anno)
    return anno_dict

anno_dict = get_anno_dict(anno_file)

def get_gt_boxes(img, cat):
    annos = anno_dict[img]
    boxes = []
    for anno in annos:
        if anno['category'] == cat:
            boxes.append(anno['bbox'])
    return boxes

def get_edge_img(img_name):
    img_file = img_folder + img_name
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.Canny(img, 80, 150)
    img = img.astype(np.bool)
    return img

def in_square(x, y, lines):
    if y < lines['up'](x) + 2:
        return False
    if y > lines['down'](x) - 2:
        return False
    if x < lines['left'](y) + 2:
        return False
    if x > lines['right'](y) - 2:
        return False
    return True

def filter_outside(img_name, img):
    edge_file = edge_folder + img_name[:-3] + 'json'

    with open(edge_file, 'r') as f:
        edge_info = json.load(f)
    lines = {}
    for loc, line in edge_info['lines'].items():
        lines[loc] = np.poly1d(line)
    
    y_array, x_array = np.where(img)
    for y, x in zip(y_array, x_array):
        if not in_square(x, y, lines):
            img[y, x] = False

def get_cluster_boxes(img):
    points = np.vstack(np.where(img)).transpose()
    if len(points) == 0:
        return []
    cluster = DBSCAN(eps=5, min_samples=5).fit(points)
    ys = points[:, 0]
    xs = points[:, 1]
    boxes = []
    for i in range(cluster.labels_.max() + 1):
        mask = cluster.labels_ == i
        y_c = ys[mask]
        x_c = xs[mask]
        boxes.append([x_c.min(), y_c.min(), x_c.max(), y_c.max()])
    return boxes

FP = 0
FN = 0
TP = 0
offset = [[], [], [], []]

def check(img_name):
    gt_boxes = get_gt_boxes(img_name, 1)
    if not gt_boxes:
        return
    img = get_edge_img(img_name)
    filter_outside(img_name, img)
    cluster_boxes = get_cluster_boxes(img)
    evaluate(gt_boxes, cluster_boxes)
    print('GT:', len(gt_boxes), 'DT:', len(cluster_boxes), 'TP:', TP, ' FP:', FP, ' FN:', FN)

def piou(pp_box, gt_box):
    l1, t1, r1, b1 = pp_box
    l2, t2, r2, b2 = gt_box
    area1 = (r1-l1)*(b1-t1)
    max_l = max(l1, l2)
    min_r = min(r1, r2)
    max_t = max(t1, t2)
    min_b = min(b1, b2)
    if min_r <= max_l or min_b <= max_t:
        return 0
    area_i = (min_r - max_l) * (min_b - max_t)
    return area_i / area1

def evaluate(gt_boxes, dt_boxes):
    global FP, FN, TP, offset
    if not gt_boxes:
        FP += len(dt_boxes)
        return
    if not dt_boxes:
        FN += len(gt_boxes)
        return
    
    iou_matrix = []
    for gt_box in gt_boxes:
        metric = []
        for dt_box in dt_boxes:
            metric.append(piou(dt_box, gt_box))
        iou_matrix.append(metric)
    iou_matrix = np.array(iou_matrix)
    
    correct_matrix = iou_matrix > 0.5
    dt_num_for_gt = correct_matrix.sum(axis=1)
    FN += (dt_num_for_gt == 0).sum()
    TP += (dt_num_for_gt > 0).sum()
    gt_num_for_dt = correct_matrix.sum(axis=0)
    FP += (gt_num_for_dt == 0).sum()
    gt_indexes, dt_indexes = np.where(correct_matrix)
    for dt_i, gt_i in zip(dt_indexes, gt_indexes):
        dt_box = dt_boxes[dt_i]
        gt_box = gt_boxes[gt_i]
        for i, (dt_coord, gt_coord) in enumerate(zip(dt_box, gt_box)):
            offset[i].append(gt_coord-dt_coord)

if __name__ == '__main__':
    for i, img_name in enumerate(os.listdir(img_folder)):
        check(img_name)
        print(i, img_name)

    with open('box_offset.json', 'w') as f:
        json.dump(offset, f)