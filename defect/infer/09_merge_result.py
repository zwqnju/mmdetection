import json
import torch
import numpy as np
from tqdm import tqdm

def NMS_wrapper(bboxes, scores):
    dets = []
    for box, score in zip(bboxes, scores):
        box.append(score)
        dets.append(box)
    return nms(np.array(dets))

def nms(dets, thresh=0.1):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) #所有box面积
    # print("all box aress: ", areas)
    order = scores.argsort()[::-1] #降序排列得到scores的坐标索引
 
    keep = []
    while order.size > 0:
        i = order[0] #最大得分box的坐标索引
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]]) #最高得分的boax与其他box的公共部分(交集)
 
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1) #求高和宽，并使数值合法化
        inter = w * h #其他所有box的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  #IOU:交并比
 
        inds = np.where(ovr <= thresh)[0] #ovr小表示两个box交集少，可能是另一个物体的框，故需要保留
        order = order[inds + 1]  #iou小于阈值的框
 
    return keep



split_folder = '/home/allride/workspace/2021/Detection/dataset/crop_testA_set/split_infos/'

# 读取原始结果
src_result_file = './0_try/0121_split.json'
with open(src_result_file, 'r') as f:
    split_result = json.load(f)

def get_split_info(img_name):
    with open(split_folder + img_name[:-3]+'json', 'r') as f:
        return json.load(f)

def translation(result_info):
    x_offset, y_offset, _, _ = get_split_info(result_info['name'])
    result_info['name'] = result_info['name'].split('-')[0] + '.jpg'
    left, top, right, bottom = result_info['bbox']
    result_info['bbox'] = [left+x_offset, top+y_offset, right+x_offset, bottom+y_offset]
    return result_info

# 根据split信息，转换：图片名，box坐标
# 聚集为dict：
result_dict = {}
for result in tqdm(split_result):
    result = translation(result)
    name = result['name']
    category = result['category']
    if name not in result_dict:
        result_dict[name] = {}
    if category not in result_dict[name]:
        result_dict[name][category] = dict(
            bboxes=[],
            scores=[],
        )
    result_dict[name][category]['bboxes'].append(result['bbox'])
    result_dict[name][category]['scores'].append(result['score'])

# 针对每张图片每一类做NMS, 记录结果
filtered_count = 0

merge_result = []

for name in tqdm(list(result_dict.keys())):
    for category in result_dict[name]:
        bboxes = result_dict[name][category]['bboxes']
        scores = result_dict[name][category]['scores']
        keep = NMS_wrapper(bboxes, scores)
        filtered_count += len(bboxes) - len(keep)
        if len(bboxes) - len(keep) > 0:
            print(name, category, bboxes, keep) 
        for index in keep:
            merge_result.append(dict(
                name=name,
                category=category,
                bbox=bboxes[index],
                score=scores[index]
            ))

print('Filter:', filtered_count)

with open('./0_try/0121_crop_nms0.1.json', 'w') as f:
    json.dump(merge_result, f)
