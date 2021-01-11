import sys
import os
import json
import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

def generate_submit_list(model, img_folder):
    submit_result = []
    for filename in tqdm(os.listdir(img_folder)):
        img = os.path.join(img_folder, filename)
        result = inference_detector(model, img)
        for cls_id, det_boxes in enumerate(result):
            for det_box in det_boxes:
                submit_result.append(dict(
                    name=filename,
                    category=cls_id+1,
                    bbox=det_box[0:4].astype(int).tolist(),
                    score=float(det_box[4]),
                ))
    return submit_result

if __name__ == '__main__':

    config_file = sys.argv[1]
    checkpoint_file = sys.argv[2]
    submit_file = sys.argv[3]

    model = init_detector(
        config_file, 
        checkpoint_file, 
        device='cuda:0'
    )

    submit_result = generate_submit_list(model, model.cfg.data.test.img_prefix)
    with open(submit_file, 'w') as f:
        json.dump(submit_result, f)