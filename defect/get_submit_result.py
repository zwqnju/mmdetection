import sys
import os
import json
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from tqdm import tqdm

def generate_submit_list(model, img_folder):
    submit_result = []
    for filename in tqdm(os.listdir(img_folder)):
        img = mmcv.imread(os.path.join(img_folder, filename))
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
    model_file = sys.argv[2]
    submit_file = sys.argv[3]

    cfg = Config.fromfile(config_file)
    cfg.load_from = model_file
    cfg.gpu_ids = range(1)
    print(f'Config:\n{cfg.pretty_text}')

    model = build_detector(
        cfg.model, 
        train_cfg=cfg.train_cfg, 
        test_cfg=cfg.test_cfg
    )
    model.cfg = cfg

    submit_result = generate_submit_list(model, cfg.data.test.img_prefix)
    with open(submit_file, 'w') as f:
        json.dump(submit_result, f)