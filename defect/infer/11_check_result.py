import numpy as np
from tqdm import tqdm
import json

result_file = './0_try/0121_initial_nms0.1.json'
dst_result_file = './0_try/0121_initial_nms0.1_highscore.json'
thresh = 0.1

if __name__ == '__main__':
    with open(result_file, 'r') as f:
        src_results = json.load(f)

    dst_results = []
    low_count = 0

    for src_result in tqdm(src_results):
        score = src_result['score']
        if score < thresh:
            low_count += 1
            print(src_result)
        else:
            dst_results.append(src_result)

    print(low_count)
    
    with open(dst_result_file, 'w') as f:
        json.dump(dst_results, f)