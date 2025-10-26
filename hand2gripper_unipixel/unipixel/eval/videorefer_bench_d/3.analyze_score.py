# Modified from https://github.com/DAMO-NLP-SG/VideoRefer/blob/main/videorefer/eval/videorefer_bench_d/3.analyze_score.py

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
args = parser.parse_args()

with open(args.input_file) as f:
    data = json.load(f)

out_dict = dict()
score_all = 0

type_list = ['Subject Correspondence', 'Appearance Description', 'Temporal Description', 'Hallucination Detection']
for tp in type_list:
    cnt, score = 0, 0

    for i, d in enumerate(data):
        if tp not in d:
            continue
        cnt += 1
        score += d[tp]

    print(f'{tp} ({cnt}): {score / cnt}')
    score_all += score / cnt
    out_dict[tp] = dict(count=cnt, score=score / cnt)

print(f'All: {score_all / 4}')
out_dict['All'] = dict(score=score_all / 4)

name, ext = os.path.splitext(args.input_file)
with open(f'{name}_score{ext}', 'w') as f:
    json.dump(out_dict, f, indent=4)
