import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import json

@DATASETS.register_module()
class DefectDataset(CustomDataset):

    CLASSES = ("边异常", "角异常", "白色点瑕疵", "浅色块瑕疵", "深色点块瑕疵", "光圈瑕疵")

    def load_annotations(self, json_file):
        with open(json_file, 'rb') as f:
            merged_infos = json.load(f)
        
        anno_infos = merged_infos['anno']
    
        info_dict = {}
        # convert annotations to middle format
        for origin_info in anno_infos:
            filename = origin_info['name']
            if filename not in info_dict:
                info_dict[filename] = dict(
                    filename=origin_info['name'], 
                    height=origin_info['image_height'], 
                    width=origin_info['image_width'],
                    ann=dict(
                        bboxes=[],
                        labels=[],
                    ),
                )
            info_dict[filename]['ann']['bboxes'].append(origin_info['bbox'])
            info_dict[filename]['ann']['labels'].append(origin_info['category']-1)
        
        data_infos = []
        for info in info_dict.values():
            info['ann']['bboxes'] = np.array(info['ann']['bboxes'], dtype=np.float32)
            info['ann']['labels'] = np.array(info['ann']['labels'], dtype=np.long)
            data_infos.append(info)
        
        img_infos = merged_infos['img']
        for img_name, img_height, img_width in img_infos:
            if img_name not in info_dict:
                data_infos.append(dict(
                    filename = img_name,
                    height = img_height,
                    width = img_width,
                    anno = dict(
                        bboxes = np.zeros([0, 4], dtype=np.float32),
                        labels = np.zeros([0], dtype=np.long)
                    )
                ))

        return data_infos