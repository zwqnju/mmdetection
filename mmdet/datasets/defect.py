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
            origin_infos = json.load(f)
    
        info_dict = {}
        # convert annotations to middle format
        for origin_info in origin_infos:
            filename = origin_info['name']
            if filename not in info_dict:
                info_dict[filename] = dict(
                    filename=origin_info['name'], 
                    width=origin_info['image_height'], 
                    height=origin_info['image_width'],
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

        return data_infos