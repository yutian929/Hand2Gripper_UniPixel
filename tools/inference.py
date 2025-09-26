# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import io

import imageio.v3 as iio
import nncore

from unipixel.dataset.utils import process_vision_info
from unipixel.eval.visualizer import Visualizer, random_color
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_video
from unipixel.utils.transforms import get_sam2_transform

BANNER = r"""
=================================================================================
   __    __    __   __    __    ______     __   ___   ___   _______    __
  |  |  |  |  |  \ |  |  |  |  |   _  \   |  |  \  \ /  /  |   ____|  |  |
  |  |  |  |  |   \|  |  |  |  |  |_)  |  |  |   \  V  /   |  |__     |  |
  |  |  |  |  |  . `  |  |  |  |   ___/   |  |    >   <    |   __|    |  |
  |  `--'  |  |  |\   |  |  |  |  |       |  |   /  .  \   |  |____   |  `----.
   \______/   |__| \__|  |__|  | _|       |__|  /__/ \__\  |_______|  |_______|

=================================================================================
"""

INFO = """
\033[1;36m   Examples:\033[0m 1. Please segment the tallest giraffe.
             2. Where is the nearest sheep? Please provide the segmentation mask.
             3. Why is the boy crying? Please provide the segmentation mask and explain why.
             4. Who shooted the ball? Please answer the question and provide the segmentation mask.
             5. Please segment the object according to the description: <a-long-description>

\033[1;32m Model Path:\033[0m {}
\033[1;32m Media Path:\033[0m {}
\033[1;32m     Prompt:\033[0m {}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('media_path')
    parser.add_argument('prompt')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model_path', default='PolyU-ChenLab/UniPixel-3B')
    parser.add_argument('--sample_frames', type=int, default=16)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print(BANNER + INFO.format(args.model_path, args.media_path, args.prompt))

    model, processor = build_model(args.model_path, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device

    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    if any(args.media_path.endswith(k) for k in ('jpg', 'png')):
        frames, images = load_image(args.media_path), [args.media_path]
    else:
        frames, images = load_video(args.media_path, sample_frames=args.sample_frames)

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28
        }, {
            'type': 'text',
            'text': args.prompt
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)

    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]

    output_ids = model.generate(
        **data.to(device),
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
    print(f'\n\033[1;32m   Response:\033[0m {response}')

    if len(model.seg) >= 1:
        colors = [random_color(rgb=True, maximum=1) for _ in range(model.seg[0].size(0))]

        imgs = []
        for i in range(frames.size(0)):
            vis = Visualizer(frames[i].numpy())

            for obj_idx in range(model.seg[0].size(0)):
                fig = vis.draw_binary_mask_with_number(
                    model.seg[0][obj_idx, i].bool().numpy(), color=colors[obj_idx], alpha=0.3)

            buffer = io.BytesIO()
            fig.save(buffer)
            buffer.seek(0)
            img = iio.imread(buffer)
            imgs.append(img)

        nncore.mkdir(args.output_dir)

        if len(imgs) > 1:
            path = nncore.join(args.output_dir, f'{nncore.pure_name(args.media_path)}.gif')
            print(f'\033[1;32mOutput Path:\033[0m {path}')
            iio.imwrite(path, imgs, duration=100, loop=0)
        else:
            path = nncore.join(args.output_dir, f'{nncore.pure_name(args.media_path)}.png')
            print(f'\033[1;32mOutput Path:\033[0m {path}')
            iio.imwrite(path, img)
