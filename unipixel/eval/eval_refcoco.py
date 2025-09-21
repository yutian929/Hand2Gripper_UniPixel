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

    log_file = nncore.join(args.pred_path, 'metrics.log')
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    pred_paths = nncore.ls(args.pred_path, ext='json', join_path=True)
    pred_paths = [p for p in pred_paths if 'output' in p]
    nncore.log(f'Total number of files: {len(pred_paths)}\n')

    preds = nncore.flatten([nncore.load(p) for p in pred_paths])

    cum_inter = sum(p['inter'] for p in preds)
    cum_union = sum(p['union'] for p in preds)
    ciou = cum_inter / cum_union
    giou = sum(p['iou'] for p in preds) / len(preds)

    box_acc = sum(1 if p.get('hit') else 0 for p in preds) / len(preds)

    nncore.log(f'Cum Inter: {cum_inter}')
    nncore.log(f'Cum Union: {cum_union}')
    nncore.log(f'CIoU: {ciou * 100:.5f}')
    nncore.log(f'GIoU: {giou * 100:.5f}')
    nncore.log(f'BBox Acc: {box_acc * 100:.5f}')
