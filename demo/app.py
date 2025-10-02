# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause license.

import os
import re
import uuid
from functools import partial

import gradio as gr
import imageio.v3 as iio
import spaces
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image

from unipixel.constants import MEM_TOKEN, SEG_TOKEN
from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_video
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import draw_mask, sample_color

PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

MODEL = 'PolyU-ChenLab/UniPixel-3B'

TITLE = 'UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning'

HEADER = """
<p align="center" style="margin: 1em 0 2em;"><img width="280" src="https://raw.githubusercontent.com/PolyU-ChenLab/UniPixel/refs/heads/main/.github/logo.png"></p>
<h3 align="center">Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning</h3>
<div style="display: flex; justify-content: center; gap: 5px;">
    <a href="https://arxiv.org/abs/2509.18094" target="_blank"><img src="https://img.shields.io/badge/arXiv-2509.18094-red"></a>
    <a href="https://polyu-chenlab.github.io/unipixel/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
    <a href="https://huggingface.co/collections/PolyU-ChenLab/unipixel-68cf7137013455e5b15962e8" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
    <a href="https://huggingface.co/datasets/PolyU-ChenLab/UniPixel-SFT-1M" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
    <a href="https://github.com/PolyU-ChenLab/UniPixel/blob/main/README.md" target="_blank"><img src="https://img.shields.io/badge/License-BSD--3--Clause-purple"></a>
    <a href="https://github.com/PolyU-ChenLab/UniPixel" target="_blank"><img src="https://img.shields.io/github/stars/PolyU-ChenLab/UniPixel"></a>
</div>
<p style="margin-top: 1em;">UniPixel is a unified MLLM for pixel-level vision-language understanding. It flexibly supports a variety of fine-grained tasks, including image/video segmentation, regional understanding, and a novel PixelQA task that jointly requires object-centric referring, segmentation, and question-answering in videos. Please open an <a href="https://github.com/PolyU-ChenLab/UniPixel/issues/new" target="_blank">issue</a> if you meet any problems.</p>
"""

# https://github.com/gradio-app/gradio/pull/10552
JS = """
function init() {
    if (window.innerWidth >= 1536) {
        document.querySelector('main').style.maxWidth = '1536px'
    }
}
"""

model, processor = build_model(MODEL)
device = next(model.parameters()).device

sam2_transform = get_sam2_transform(model.config.sam2_image_size)

colors = sample_color()
color_map = {f'Target {i + 1}': f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for i, c in enumerate(colors * 255)}
color_map_light = {
    f'Target {i + 1}': f'#{int(c[0] * 127.5 + 127.5):02x}{int(c[1] * 127.5 + 127.5):02x}{int(c[2] * 127.5 + 127.5):02x}'
    for i, c in enumerate(colors)
}


def enable_btns():
    return (gr.Button(interactive=True), ) * 4


def disable_btns():
    return (gr.Button(interactive=False), ) * 4


def reset_seg():
    return 16, gr.Button(interactive=False)


def reset_reg():
    return 1, gr.Button(interactive=False)


def update_region(blob):
    if blob['background'] is None or not blob['layers'][0].any():
        return

    region = blob['background'].copy()
    region[blob['layers'][0][:, :, -1] == 0] = [0, 0, 0, 0]

    return region


def update_video(video, prompt_idx):
    if video is None:
        return

    _, images = load_video(video, sample_frames=16)
    path = images[prompt_idx - 1]

    return path


@spaces.GPU
def infer_seg(media, query, sample_frames=16, media_type=None):
    if not media:
        gr.Warning('Please upload an image or a video.')
        return None, None, None

    if not query:
        gr.Warning('Please provide a text prompt.')
        return None, None, None

    if any(media.endswith(k) for k in ('jpg', 'png')):
        frames, images = load_image(media), [media]
    else:
        frames, images = load_video(media, sample_frames=sample_frames)

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * int(sample_frames / len(images))
        }, {
            'type': 'text',
            'text': query
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
    response = response.replace(f' {SEG_TOKEN}', SEG_TOKEN).replace(f'{SEG_TOKEN} ', SEG_TOKEN)

    entities = []
    for i, m in enumerate(re.finditer(re.escape(SEG_TOKEN), response)):
        entities.append(dict(entity=f'Target {i + 1}', start=m.start(), end=m.end()))

    answer = dict(text=response, entities=entities)

    imgs = draw_mask(frames, model.seg, colors=colors)

    path = f"/tmp/{uuid.uuid4().hex}.{'gif' if len(imgs) > 1 else 'png'}"
    iio.imwrite(path, imgs, duration=100, loop=0)

    if media_type == 'image':
        if len(model.seg) >= 1:
            masks = media, [(m[0, 0].numpy(), f'Target {i + 1}') for i, m in enumerate(model.seg)]
        else:
            masks = None
    else:
        masks = path

    return answer, masks, path


infer_seg_image = partial(infer_seg, media_type='image')
infer_seg_video = partial(infer_seg, media_type='video')


@spaces.GPU
def infer_reg(blob, query, prompt_idx=1, video=None):
    if blob['background'] is None:
        gr.Warning('Please upload an image or a video.')
        return

    if not blob['layers'][0].any():
        gr.Warning('Please provide a mask prompt.')
        return

    if not query:
        gr.Warning('Please provide a text prompt.')
        return

    if video is None:
        frames = torch.from_numpy(blob['background'][:, :, :3]).unsqueeze(0)
        images = [Image.fromarray(blob['background'], mode='RGBA')]
    else:
        frames, images = load_video(video, sample_frames=16)

    frame_size = frames.shape[1:3]

    mask = torch.from_numpy(blob['layers'][0][:, :, -1]).unsqueeze(0) > 0

    refer_mask = torch.zeros(frames.size(0), 1, *frame_size)
    refer_mask[prompt_idx - 1] = mask

    if refer_mask.size(0) % 2 != 0:
        refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
    refer_mask = refer_mask.flatten(1)
    refer_mask = F.max_pool1d(refer_mask.transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
    refer_mask = refer_mask.view(-1, 1, *frame_size)

    if video is None:
        prefix = f'Here is an image with the following highlighted regions:\n[0]: <{prompt_idx}> {MEM_TOKEN}\n'
    else:
        prefix = f'Here is a video with {len(images)} frames denoted as <1> to <{len(images)}>. The highlighted regions are as follows:\n[0]: <{prompt_idx}>-<{prompt_idx + 1}> {MEM_TOKEN}\n'

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * int(16 / len(images))
        }, {
            'type': 'text',
            'text': prefix + query
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)

    refer_mask = T.resize(refer_mask, (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
    refer_mask = F.max_pool2d(refer_mask, kernel_size=28, stride=28)
    refer_mask = refer_mask > 0

    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    data['refer_mask'] = [refer_mask]

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
    response = response.replace(' [0]', '[0]').replace('[0] ', '[0]').replace('[0]', '<REGION>')

    entities = []
    for m in re.finditer(re.escape('<REGION>'), response):
        entities.append(dict(entity='region', start=m.start(), end=m.end(), color="#f85050"))

    answer = dict(text=response, entities=entities)

    return answer


def build_demo():
    with gr.Blocks(title=TITLE, js=JS) as demo:
        gr.HTML(HEADER)

        with gr.Tab('Image Segmentation'):
            download_btn_1 = gr.DownloadButton(label='📦 Download', interactive=False, render=False)
            msk_1 = gr.AnnotatedImage(label='Segmentation Results', color_map=color_map, render=False)
            ans_1 = gr.HighlightedText(
                label='Model Response', color_map=color_map_light, show_inline_category=False, render=False)

            with gr.Row():
                with gr.Column():
                    media_1 = gr.Image(type='filepath')

                    sample_frames_1 = gr.Slider(1, 32, value=16, step=1, visible=False)

                    query_1 = gr.Textbox(label='Text Prompt', placeholder='Please segment the...')

                    with gr.Row():
                        random_btn_1 = gr.Button(value='🔮 Random', visible=False)

                        reset_btn_1 = gr.ClearButton([media_1, query_1, msk_1, ans_1], value='🗑️ Reset')
                        reset_btn_1.click(reset_seg, None, [sample_frames_1, download_btn_1])

                        download_btn_1.render()

                        submit_btn_1 = gr.Button(value='🚀 Submit', variant='primary')
                with gr.Column():
                    msk_1.render()
                    ans_1.render()

            ctx_1 = submit_btn_1.click(disable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])
            ctx_1 = ctx_1.then(infer_seg_image, [media_1, query_1, sample_frames_1], [ans_1, msk_1, download_btn_1])
            ctx_1.then(enable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])

        with gr.Tab('Video Segmentation'):
            download_btn_2 = gr.DownloadButton(label='📦 Download', interactive=False, render=False)
            msk_2 = gr.Image(label='Segmentation Results', render=False)
            ans_2 = gr.HighlightedText(
                label='Model Response', color_map=color_map_light, show_inline_category=False, render=False)

            with gr.Row():
                with gr.Column():
                    media_2 = gr.Video()

                    with gr.Accordion(label='Hyperparameters', open=False):
                        sample_frames_2 = gr.Slider(
                            1,
                            32,
                            value=16,
                            step=1,
                            interactive=True,
                            label='Sample Frames',
                            info='The number of frames to sample from a video (Default: 16)')

                    query_2 = gr.Textbox(label='Text Prompt', placeholder='Please segment the...')

                    with gr.Row():
                        random_btn_2 = gr.Button(value='🔮 Random', visible=False)

                        reset_btn_2 = gr.ClearButton([media_2, query_2, msk_2, ans_2], value='🗑️ Reset')
                        reset_btn_2.click(reset_seg, None, [sample_frames_2, download_btn_2])

                        download_btn_2.render()

                        submit_btn_2 = gr.Button(value='🚀 Submit', variant='primary')
                with gr.Column():
                    msk_2.render()
                    ans_2.render()

            ctx_2 = submit_btn_2.click(disable_btns, None, [random_btn_2, reset_btn_2, download_btn_2, submit_btn_2])
            ctx_2 = ctx_2.then(infer_seg_video, [media_2, query_2, sample_frames_2], [ans_2, msk_2, download_btn_2])
            ctx_2.then(enable_btns, None, [random_btn_2, reset_btn_2, download_btn_2, submit_btn_2])

        with gr.Tab('Image Regional Understanding'):
            download_btn_3 = gr.DownloadButton(visible=False)
            msk_3 = gr.Image(label='Highlighted Region', render=False)
            ans_3 = gr.HighlightedText(label='Model Response', show_inline_category=False, render=False)

            with gr.Row():
                with gr.Column():
                    media_3 = gr.ImageEditor(
                        label='Image & Mask Prompt',
                        brush=gr.Brush(colors=["#ff000080"], color_mode='fixed'),
                        transforms=None,
                        layers=False)
                    media_3.change(update_region, media_3, msk_3)

                    prompt_frame_index_3 = gr.Slider(1, 16, value=1, step=1, visible=False)

                    query_3 = gr.Textbox(label='Text Prompt', placeholder='Please describe the highlighted region...')

                    with gr.Row():
                        random_btn_3 = gr.Button(value='🔮 Random', visible=False)

                        reset_btn_3 = gr.ClearButton([media_3, query_3, msk_3, ans_3], value='🗑️ Reset')
                        reset_btn_3.click(reset_reg, None, [prompt_frame_index_3, download_btn_3])

                        submit_btn_3 = gr.Button(value='🚀 Submit', variant='primary')
                with gr.Column():
                    msk_3.render()
                    ans_3.render()

            ctx_3 = submit_btn_3.click(disable_btns, None, [random_btn_3, reset_btn_3, download_btn_3, submit_btn_3])
            ctx_3 = ctx_3.then(infer_reg, [media_3, query_3, prompt_frame_index_3], ans_3)
            ctx_3.then(enable_btns, None, [random_btn_3, reset_btn_3, download_btn_3, submit_btn_3])

        with gr.Tab('Video Regional Understanding'):
            download_btn_4 = gr.DownloadButton(visible=False)
            prompt_frame_index_4 = gr.Slider(
                1,
                16,
                value=1,
                step=1,
                interactive=True,
                label='Prompt Frame Index',
                info='The index of the frame that includes mask prompts (Default: 1)',
                render=False)
            msk_4 = gr.ImageEditor(
                label='Mask Prompt',
                brush=gr.Brush(colors=['#ff000080'], color_mode='fixed'),
                transforms=None,
                layers=False,
                render=False)
            ans_4 = gr.HighlightedText(label='Model Response', show_inline_category=False, render=False)

            with gr.Row():
                with gr.Column():
                    media_4 = gr.Video()
                    media_4.change(update_video, [media_4, prompt_frame_index_4], msk_4)

                    with gr.Accordion(label='Hyperparameters', open=False):
                        prompt_frame_index_4.render()
                        prompt_frame_index_4.change(update_video, [media_4, prompt_frame_index_4], msk_4)

                    query_4 = gr.Textbox(label='Text Prompt', placeholder='Please describe the highlighted region...')

                    with gr.Row():
                        random_btn_4 = gr.Button(value='🔮 Random', visible=False)

                        reset_btn_4 = gr.ClearButton([media_4, query_4, msk_4, ans_4], value='🗑️ Reset')
                        reset_btn_4.click(reset_reg, None, [prompt_frame_index_4, download_btn_4])

                        submit_btn_4 = gr.Button(value='🚀 Submit', variant='primary')
                with gr.Column():
                    msk_4.render()
                    ans_4.render()

            ctx_4 = submit_btn_4.click(disable_btns, None, [random_btn_4, reset_btn_4, download_btn_4, submit_btn_4])
            ctx_4 = ctx_4.then(infer_reg, [msk_4, query_4, prompt_frame_index_4, media_4], ans_4)
            ctx_4.then(enable_btns, None, [random_btn_4, reset_btn_4, download_btn_4, submit_btn_4])

    return demo


if __name__ == '__main__':
    demo = build_demo()

    demo.queue()
    demo.launch(server_name='0.0.0.0')
