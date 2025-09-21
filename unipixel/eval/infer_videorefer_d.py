# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import io

import nncore
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image

import imageio.v3 as iio
import matplotlib.pyplot as plt
from unipixel.constants import MEM_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_masks, process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_frames_with_inds_keep
from unipixel.utils.transforms import get_sam2_transform


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--split')
    parser.add_argument('--model_path')
    parser.add_argument('--res_pred_path')
    parser.add_argument('--vis_pred_path')
    parser.add_argument('--single_frame_mode', action='store_true')
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--sample_frames', type=int, default=16)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dump', type=int, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.res_pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.res_pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model(args.model_path, image_size=args.image_size, device=args.device)
    device = next(model.parameters()).device

    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    annos = DATASETS.get(args.dataset).load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        question, caption = anno['question'], anno['caption']

        masks = anno['single_frame_masks'] if args.single_frame_mode else anno['masks']

        print()
        print(anno['video_path'])
        print(len(anno['all_frame_inds']), anno['frame_idx'], anno['all_frame_inds'])
        print(question)

        frames, paths, inds = load_frames_with_inds_keep(
            anno['video_path'],
            anno['all_frame_inds'],
            anno['frame_idx'],
            sample_frames=args.sample_frames,
            sample_type='uniform',
            sample_for_llm_only=False,
            num_threads=args.num_threads)

        frame_size = frames.shape[1:3]

        # all samples should be videos
        assert len(paths) > 1

        assert anno['frame_idx'] in anno['all_frame_inds'], (anno['frame_idx'], anno['all_frame_inds'])
        prompt_frame_idx = inds.index(anno['all_frame_inds'].index(anno['frame_idx']))

        label_mask = process_masks(dict(mask_type='rle', masks=masks), frame_size, inds)
        label_mask = torch.stack([torch.stack(o) for o in label_mask]).transpose(0, 1)
        assert label_mask.shape[2:] == frame_size
        cache_mask = label_mask.clone()
        refer_mask = label_mask.clone()
        label_mask = T.resize(label_mask, (model.config.sam2_image_size, model.config.sam2_image_size))
        label_mask = label_mask > 0

        # ensure refer mask has the correct number of frames
        num_objs = refer_mask.size(1)
        if refer_mask.size(0) % 2 != 0:
            refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
        refer_mask = refer_mask.flatten(1)
        refer_mask = F.max_pool1d(refer_mask.transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
        refer_mask = refer_mask.view(-1, num_objs, *frame_size)

        prompt = question

        vids = []
        for obj_idx in range(num_objs):
            tids = (refer_mask[:, obj_idx].any(dim=(-1, -2)).nonzero()[:, 0] * 2 + 1).tolist()
            vids.append(tids)

        prefix = f'Here is a video with {len(paths)} frames denoted as <1> to <{len(paths)}>. The highlighted regions are as follows:\n'

        assert len(vids) == 1
        tids = vids[0]
        prefix += '[0]: ' + ' '.join([f'<{tid}>-<{tid + 1}> ' + MEM_TOKEN for tid in tids]) + '\n'

        assert '<region>' not in prompt and '<object' not in prompt, prompt

        prompt = prefix + prompt
        print(prompt)
        print(caption)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': paths,
                'num_threads': args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * int(args.sample_frames / len(paths))
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos = process_vision_info(messages)

        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        # spatial patch size set to 14, spatial merge size set to 2
        refer_mask = T.resize(refer_mask, (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
        refer_mask = F.max_pool2d(refer_mask, kernel_size=28, stride=28)
        refer_mask = refer_mask > 0

        # assert refer_mask.any(), 'refer mask is empty'
        if not refer_mask.any():
            print('[WARNING] refer mask is empty')

        assert refer_mask.size(0) == data['video_grid_thw'][0][0]
        assert refer_mask.size(2) == data['video_grid_thw'][0][1] // 2
        assert refer_mask.size(3) == data['video_grid_thw'][0][2] // 2

        data['refer_mask'] = [refer_mask.to(device)]

        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]

        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]

        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        print(response)

        dump['pred'] = response

        # dummy output mask
        out_mask = cache_mask.transpose(0, 1)
        assert out_mask.size(1) == len(inds)

        if args.dump > 0 and i % args.dump == 0:
            nncore.mkdir(args.vis_pred_path)

            gif_path = nncore.join(args.vis_pred_path, f"{args.index}_{i}_{anno['vid'].replace('/', '_')}.gif")

            imgs = []
            for idx in range(len(inds)):
                buffer = io.BytesIO()
                plt.figure()
                plt.title(response)
                plt.imshow(Image.fromarray(frames[idx].numpy()))
                for obj_id in range(out_mask.size(0)):
                    show_mask(out_mask[obj_id, idx], plt.gca(), obj_id=obj_id)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(buffer, format='jpg')
                plt.close()
                buffer.seek(0)
                imgs.append(iio.imread(buffer))

            iio.imwrite(gif_path, imgs, duration=100, loop=0)

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
