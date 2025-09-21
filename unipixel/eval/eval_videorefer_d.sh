#!/bin/bash

set -e

output_file=$1

gpt_output_file=$1/gpt4o_mini.json

python unipixel/eval/videorefer_bench_d/1.eval_gpt_4o_mini.py $output_file $gpt_output_file
python unipixel/eval/videorefer_bench_d/2.extract_re.py $gpt_output_file
python unipixel/eval/videorefer_bench_d/3.analyze_score.py $gpt_output_file

# gpt_output_file=$1/gpt4o.json

# python unipixel/eval/videorefer_bench_d/1.eval_gpt_4o.py $output_file $gpt_output_file
# python unipixel/eval/videorefer_bench_d/2.extract_re.py $gpt_output_file
# python unipixel/eval/videorefer_bench_d/3.analyze_score.py $gpt_output_file
