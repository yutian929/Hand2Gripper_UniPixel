# Modified from https://github.com/DAMO-NLP-SG/VideoRefer/blob/main/videorefer/eval/videorefer_bench_d/2.extract_re.py

import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
args = parser.parse_args()

data = json.load(open(args.input_file))
final_data = []
err = 0
for d in data:
    try:
        input_string = d['gpt']
        pattern = r'\d+\.\s+(.*?):\s+([\d.]+)'
        matches = re.findall(pattern, input_string)

        result_dict = {description: float(score) for description, score in matches}
        final_data.append(dict(d, **result_dict))
    except Exception:
        err += 1

if err > 0:
    print('####error num: ', err)

with open(args.input_file, 'w') as f:
    f.write(json.dumps(final_data))
