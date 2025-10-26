# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse

import nncore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert nncore.is_dir(args.pred_path)

    log_file = nncore.join(args.pred_path, 'metrics_pixelqa.log')
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    pred_paths = nncore.ls(args.pred_path, ext='json', join_path=True)
    pred_paths = [p for p in pred_paths if 'output' in p]
    nncore.log(f'Total number of files: {len(pred_paths)}\n')

    preds = nncore.flatten([nncore.load(p) for p in pred_paths])

    j_values = nncore.flatten([p['j_values'] for p in preds])
    f_values = nncore.flatten([p['f_values'] for p in preds])

    avg_j = sum(j_values) / len(j_values)
    avg_f = sum(f_values) / len(f_values)

    nncore.log(f'J: {avg_j * 100:.5f}')
    nncore.log(f'F: {avg_f * 100:.5f}')
    nncore.log(f'J&F: {(avg_j + avg_f) / 2 * 100:.5f}')
