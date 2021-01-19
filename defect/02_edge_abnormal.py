import json
import cv2
import numpy as np
import os
from tqdm import tqdm

OFFSET_THRESHOLD = 4
EDGE_ISSUE_THICKNESS = 8
CLOSE_DISTANCE = 8

root_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/'
edge_folder = root_folder + 'edge_infos/'
img_folder = root_folder + 'train_imgs/'
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

def read_edge_info(img_file):
    edge_file = edge_folder + img_file[:-3] + 'json'
    with open(edge_file, 'r') as f:
        info = json.load(f)
    return info

def get_edge_img(img_file):
    img_file = img_folder + img_file
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    v1 = cv2.Canny(img, 80, 150)
    edge_img = v1.astype(np.bool)
    return edge_img

def get_gt_boxes(img_file, category=1):
    boxes = []
    for anno in anno_dict[img_file]:
        if anno['category'] == category:
            boxes.append(anno['bbox'])
    return boxes

def merge_box(b1, b2):
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def get_offset(indexes):
    if len(indexes) == 0:
        return 0
    offset = indexes[0] - 2
    if offset > EDGE_ISSUE_THICKNESS:
        offset = 0
    return offset

################################# 上边 #####################################
def get_up_boxes(edge_info, edge_img):

    # 1. UP: 从左向右排查
    up_line = edge_info['lines']['up']
    up_p = np.poly1d(up_line)
    start_x = int(np.round(edge_info['corners']['up_left'][0]))
    end_x = int(np.round(edge_info['corners']['up_right'][0]))
    issue_boxes = []
    issue_left, issue_bottom, issue_top = None, None, None
    for current_x in range(start_x, end_x+1):
        expect_y = int(np.round(up_p(current_x)))
        indexes = np.where(edge_img[expect_y-2:, current_x])[0]
        offset = max(get_offset(indexes), get_offset(indexes[1:]), get_offset(indexes[2:]))
        real_y = expect_y + offset
        if offset <= OFFSET_THRESHOLD or current_x == end_x: # 没问题 或者 到最后了
            if issue_left is not None:
                issue_right = current_x
                issue_bottom = max(issue_bottom, real_y)
                issue_top = min(issue_top, real_y)
                issue_boxes.append([issue_left, issue_top, issue_right, issue_bottom])
                issue_left = None
        else:
            if issue_left is None:
                issue_left = current_x
                issue_top = min(real_y, expect_y)
                issue_bottom = max(real_y, expect_y)
            else:
                issue_top = min(issue_top, real_y)
                issue_bottom = max(issue_bottom, real_y)
    
    # 2. UP: merge 相邻 boxes
    merged_boxes = []
    prev = None
    for curr in issue_boxes:
        if prev is None:
            prev = curr
            continue
        if curr[0] - prev[2] <= 5:
            prev = merge_box(prev, curr)
        else:
            merged_boxes.append(prev)
            prev = curr
    if prev is not None:
        merged_boxes.append(prev)
    
    # 3. UP: 剔除小boxes
    filtered_boxes = []
    for box in merged_boxes:
        if box[2] - box[0] < 5:
            continue
        filtered_boxes.append(box)
    
    # 4. UP: 左右增长
    final_up_boxes = []
    for issue_box in filtered_boxes:
        left, top, right, bottom = issue_box
        current_x = left
        right_count = 0
        while current_x > start_x+CLOSE_DISTANCE:
            current_x -= 1
            expect_y = int(np.round(up_p(current_x)))
            indexes = np.where(edge_img[expect_y-2:bottom+1, current_x])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False
            if right_count == 3:
                break
        issue_left = current_x

        current_x = right
        right_count = 0
        while current_x < end_x-CLOSE_DISTANCE:
            current_x += 1
            expect_y = int(np.round(up_p(current_x)))
            indexes = np.where(edge_img[expect_y-2:bottom+1, current_x])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_x, expect_y, issue_box)
            if right_count == 2:
                break
        issue_right = current_x

        final_up_boxes.append([issue_left, issue_box[1], issue_right, issue_box[3]+3])

    return final_up_boxes



################################# 下边 #####################################
def get_down_boxes(edge_info, edge_img):

    # 1. DOWN: 从左向右排查
    down_line = edge_info['lines']['down']
    down_p = np.poly1d(down_line)
    start_x = int(np.round(edge_info['corners']['down_left'][0]))
    end_x = int(np.round(edge_info['corners']['down_right'][0]))
    issue_boxes = []
    issue_left, issue_bottom, issue_top = None, None, None
    for current_x in range(start_x, end_x+1):
        expect_y = int(np.round(down_p(current_x)))
        indexes = np.where(edge_img[:expect_y+3, current_x][::-1])[0]
        offset = max(get_offset(indexes), get_offset(indexes[1:]), get_offset(indexes[2:]))
        real_y = expect_y - offset
        if offset <= OFFSET_THRESHOLD or current_x == end_x: # 没问题 或者 到最后了
            if issue_left is not None:
                issue_right = current_x
                issue_bottom = max(issue_bottom, real_y)
                issue_top = min(issue_top, real_y)
                issue_boxes.append([issue_left, issue_top, issue_right, issue_bottom])
                issue_left = None
        else:
            if issue_left is None:
                issue_left = current_x
                issue_top = min(real_y, expect_y)
                issue_bottom = max(real_y, expect_y)
            else:
                issue_top = min(issue_top, real_y)
                issue_bottom = max(issue_bottom, real_y)
    
    # 2. DOWN: merge 相邻 boxes
    merged_boxes = []
    prev = None
    for curr in issue_boxes:
        if prev is None:
            prev = curr
            continue
        if curr[0] - prev[2] <= 5:
            prev = merge_box(prev, curr)
        else:
            merged_boxes.append(prev)
            prev = curr
    if prev is not None:
        merged_boxes.append(prev)
    
    # 3. DOWN: 剔除小boxes
    filtered_boxes = []
    for box in merged_boxes:
        if box[2] - box[0] < 5:
            continue
        filtered_boxes.append(box)
    
    # 4. DOWN: 左右增长
    final_down_boxes = []
    for issue_box in filtered_boxes:
        left, top, right, bottom = issue_box
        current_x = left
        right_count = 0
        while current_x > start_x+CLOSE_DISTANCE:
            current_x -= 1
            expect_y = int(np.round(down_p(current_x)))
            indexes = np.where(edge_img[top:expect_y+3, current_x][::-1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_x, expect_y, issue_box)
            if right_count == 3:
                break
        issue_left = current_x

        current_x = right
        right_count = 0
        while current_x < end_x-CLOSE_DISTANCE:
            current_x += 1
            expect_y = int(np.round(down_p(current_x)))
            indexes = np.where(edge_img[top:expect_y+3, current_x][::-1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_x, expect_y, issue_box)
            if right_count == 2:
                break
        issue_right = current_x

        final_down_boxes.append([issue_left, issue_box[1]-3, issue_right, issue_box[3]])

    return final_down_boxes

################################# 左边 #####################################
def get_left_boxes(edge_info, edge_img):

    # 1. LEFT: 从上向下排查
    left_line = edge_info['lines']['left']
    left_p = np.poly1d(left_line)
    start_y = int(np.round(edge_info['corners']['up_left'][1]))
    end_y = int(np.round(edge_info['corners']['down_left'][1]))
    issue_boxes = []
    issue_top, issue_left, issue_right = None, None, None
    for current_y in range(start_y, end_y+1):
        expect_x = int(np.round(left_p(current_y)))
        indexes = np.where(edge_img[current_y, expect_x-2:])[0]
        offset = max(get_offset(indexes), get_offset(indexes[1:]), get_offset(indexes[2:]))
        real_x = expect_x + offset
        if offset <= OFFSET_THRESHOLD or current_y == end_y: # 没问题 或者 到最后了
            if issue_top is not None:
                issue_bottom = current_y
                issue_right = max(issue_right, real_x)
                issue_left = min(issue_left, real_x)
                issue_boxes.append([issue_left, issue_top, issue_right, issue_bottom])
                issue_top = None
        else:
            if issue_top is None:
                issue_top = current_y
                issue_left = min(expect_x, real_x)
                issue_right = max(expect_x, real_x)
            else:
                issue_left = min(issue_left, real_x)
                issue_right = max(issue_right, real_x)
    
    # 2. LEFT: merge 相邻 boxes
    merged_boxes = []
    prev = None
    for curr in issue_boxes:
        if prev is None:
            prev = curr
            continue
        if curr[1] - prev[3] <= 5:
            prev = merge_box(prev, curr)
        else:
            merged_boxes.append(prev)
            prev = curr
    if prev is not None:
        merged_boxes.append(prev)
    
    # 3. LEFT: 剔除小boxes
    filtered_boxes = []
    for box in merged_boxes:
        if box[3] - box[1] < 5:
            continue
        filtered_boxes.append(box)
    
    # 4. LEFT: 上下增长
    final_left_boxes = []
    for issue_box in filtered_boxes:
        left, top, right, bottom = issue_box
        current_y = top
        right_count = 0
        while current_y > start_y+CLOSE_DISTANCE:
            current_y -= 1
            expect_x = int(np.round(left_p(current_y)))
            indexes = np.where(edge_img[current_y, expect_x-2:right+1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_y, expect_x, issue_box)
            if right_count == 3:
                break
        issue_top = current_y

        current_y = bottom
        right_count = 0
        while current_y < end_y-CLOSE_DISTANCE:
            current_y += 1
            expect_x = int(np.round(left_p(current_y)))
            indexes = np.where(edge_img[current_y, expect_x-2:right+1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (start_y, end_y, current_y, expect_x, issue_box)
            if right_count == 2:
                break
        issue_bottom = current_y

        final_left_boxes.append([issue_box[0], issue_top, issue_box[3]+3, issue_bottom])
    
    return final_left_boxes

################################# 右边 #####################################
def get_right_boxes(edge_info, edge_img):
    # 1. RIGHT: 从上到下排查
    right_line = edge_info['lines']['right']
    right_p = np.poly1d(right_line)
    start_y = int(np.round(edge_info['corners']['up_right'][1]))
    end_y = int(np.round(edge_info['corners']['down_right'][1]))
    issue_boxes = []
    issue_top, issue_left, issue_right = None, None, None
    for current_y in range(start_y, end_y+1):
        expect_x = int(np.round(right_p(current_y)))
        indexes = np.where(edge_img[current_y, :expect_x+3][::-1])[0]
        offset = max(get_offset(indexes), get_offset(indexes[1:]), get_offset(indexes[2:]))
        real_x = expect_x - offset
        if offset <= OFFSET_THRESHOLD or current_y == end_y: # 没问题 或者 到最后了
            if issue_top is not None:
                issue_bottom = current_y
                issue_right = max(issue_right, real_x)
                issue_left = min(issue_left, real_x)
                issue_boxes.append([issue_left, issue_top, issue_right, issue_bottom])
                issue_top = None
        else:
            if issue_top is None:
                issue_top = current_y
                issue_left = min(real_x, expect_x)
                issue_right = max(real_x, expect_x)
            else:
                issue_left = min(issue_left, real_x)
                issue_right = max(issue_right, real_x)
    
    # 2. RIGHT: merge 相邻 boxes
    merged_boxes = []
    prev = None
    for curr in issue_boxes:
        if prev is None:
            prev = curr
            continue
        if curr[1] - prev[3] <= 5:
            prev = merge_box(prev, curr)
        else:
            merged_boxes.append(prev)
            prev = curr
    if prev is not None:
        merged_boxes.append(prev)
    
    # 3. RIGHT: 剔除小boxes
    filtered_boxes = []
    for box in merged_boxes:
        if box[3] - box[1] < 5:
            continue
        filtered_boxes.append(box)
    
    # 4. RIGHT: 上下增长
    final_right_boxes = []
    for issue_box in filtered_boxes:
        left, top, right, bottom = issue_box
        current_y = top
        right_count = 0
        while current_y > start_y+CLOSE_DISTANCE:
            current_y -= 1
            expect_x = int(np.round(right_p(current_y)))
            indexes = np.where(edge_img[current_y, left:expect_x+3][::-1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_y, expect_x, issue_box)
            if right_count == 3:
                break
        issue_top = current_y

        current_y = bottom
        right_count = 0
        while current_y < end_y-CLOSE_DISTANCE:
            current_y += 1
            expect_x = int(np.round(right_p(current_y)))
            indexes = np.where(edge_img[current_y, left:expect_x+3][::-1])[0]
            if len(indexes) != 0:
                offset = indexes[0] - 2
                if offset == 0 and len(indexes) == 1:
                    right_count += 1
                else:
                    right_count = 0
            else:
                assert False, (current_y, expect_x, issue_box)
            if right_count == 2:
                break
        issue_bottom = current_y

        final_right_boxes.append([issue_box[0]-3, issue_top, issue_box[2], issue_bottom])
    
    return final_right_boxes

def iou(box1, box2):
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    area1 = (r1-l1)*(b1-t1)
    area2 = (r2-l2)*(b2-t2)
    max_l = max(l1, l2)
    min_r = min(r1, r2)
    max_t = max(t1, t2)
    min_b = min(b1, b2)
    if min_r <= max_l or min_b <= max_t:
        return 0
    area_i = (min_r - max_l) * (min_b - max_t)
    return area_i / (area1+area2-area_i)

def evaluate(gt_boxes, dt_boxes):
    if not gt_boxes and not dt_boxes:
        return True
    if not gt_boxes and dt_boxes:
        print('FP:', dt_boxes)
        return False
    if gt_boxes and not dt_boxes:
        print('FN:', gt_boxes)
        return False
    iou_matrix = []
    for gt_box in gt_boxes:
        metric = []
        for dt_box in dt_boxes:
            metric.append(iou(gt_box, dt_box))
        iou_matrix.append(metric)
    iou_matrix = np.array(iou_matrix)
    correct_matrix = iou_matrix > 0.5
    dt_num_for_gt = correct_matrix.sum(axis=1)
    FN_indexes = np.where(dt_num_for_gt == 0)
    if len(FN_indexes) > 0:
        for i in FN_indexes:
            print('FN:', gt_boxes[i])
    gt_num_for_dt = correct_matrix.sum(axis=0)
    FP_indexes = np.where(gt_num_for_dt == 0)
    if len(FP_indexes) > 0:
        for i in FP_indexes:
            print('FP:', dt_boxes[i])
    return len(FN_indexes) == 0 and len(FP_indexes) == 0


def get_issue_boxes(img_file):
    edge_info = read_edge_info(img_file)
    edge_img = get_edge_img(img_file)
    gt_boxes = get_gt_boxes(img_file)

    up_boxes = get_up_boxes(edge_info, edge_img)
    down_boxes = get_down_boxes(edge_info, edge_img)
    left_boxes = get_left_boxes(edge_info, edge_img)
    right_boxes = get_right_boxes(edge_info, edge_img)

    up_left_box = []
    if up_boxes and left_boxes and iou(up_boxes[0], left_boxes[0]) > 0:
        up_left_box = [merge_box(up_boxes.pop(0), left_boxes.pop(0))]
    up_right_box = []
    if up_boxes and right_boxes and iou(up_boxes[-1], right_boxes[0]) > 0:
        up_right_box = [merge_box(up_boxes.pop(), right_boxes.pop(0))]
    down_left_box = []
    if down_boxes and left_boxes and iou(down_boxes[0], left_boxes[-1]) > 0:
        down_left_box = [merge_box(down_boxes.pop(0), left_boxes.pop())]
    down_right_box = []
    if down_boxes and right_boxes and iou(down_boxes[-1], right_boxes[-1]) > 0:
        down_right_box = [merge_box(down_boxes.pop(), right_boxes.pop())]
    
    edge_issue_boxes = up_boxes + down_boxes + left_boxes + right_boxes

    assert evaluate(gt_boxes, edge_issue_boxes)


if __name__ == '__main__':
    img_file_list = os.listdir(img_folder)
    start = 0
    for i, img_file in enumerate(img_file_list):
        if i < start:
            continue
        print(i, img_file)
        get_issue_boxes(img_file)

