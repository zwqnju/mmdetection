import json

root_folder = '/home/allride/workspace/2021/Detection/dataset/tile_round1_train_20201231/'
anno_file = root_folder + 'train_annos.json'

with open(anno_file, 'r') as f:
    annos = json.load(f)

thick_list = []

for anno in annos:
    if anno['category'] == 1:
        print(anno['name'])
        exit()
        left, top, right, bottom = anno['bbox']
        thick_list.append(min(right-left, bottom - top))