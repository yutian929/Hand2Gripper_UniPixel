# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy

import nncore
from torch.utils.data import Dataset

from unipixel.dataset.hybrid import DATASETS


class MultimodalDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args, repeat=1):
        super().__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno['conversations'][1]['value'].split(' '))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if data_args.min_video_len >= 0 and anno.get('duration', float('inf')) < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and anno.get('duration', 0) > data_args.max_video_len:
                continue
            annos.append(anno)

        self.annos = annos
        self.raw_length = len(raw_annos)
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.repeat = repeat

    def __len__(self):
        return len(self.annos) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.annos)

        anno = copy.deepcopy(self.annos[idx])

        assert not ('image' in anno and 'video' in anno), anno

        assert anno['conversations'][0]['from'] == 'human'
        init_prompt = anno['conversations'][0]['value']

        for key in ('<image>', '<video>'):
            if init_prompt.startswith(f'{key}\n'):
                init_prompt = init_prompt[len(f'{key}\n'):]
            if init_prompt.endswith(f'\n{key}'):
                init_prompt = init_prompt[:-len(f'\n{key}')]

        if 'image' in anno:
            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'image',
                    'image': nncore.join(self.IMAGE_ROOT, anno['image']),
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 2048 * 28 * 28
                }, {
                    'type': 'text',
                    'text': init_prompt
                }]
            }]
        elif 'video' in anno:
            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'video',
                    'video': nncore.join(self.VIDEO_ROOT, anno['video']),
                    'num_threads': self.data_args.num_threads,
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 256 * 28 * 28,
                    'max_frames': int(self.data_args.sample_frames.split(',')[-1]),
                    'fps': 2
                }, {
                    'type': 'text',
                    'text': init_prompt
                }]
            }]
        else:
            messages = [{'role': 'user', 'content': init_prompt}]

        for conv in anno['conversations'][1:]:
            assert conv['from'] in ('human', 'gpt')
            role = 'user' if conv['from'] == 'human' else 'assistant'
            messages.append({'role': role, 'content': conv['value']})

        meta = dict(messages=messages)
        return meta


@DATASETS.register(name='llava_instruct_665k_videogpt_plus_576k')
class LlavaVideoGPTPlusDataset(MultimodalDataset):

    ANNO_PATH = 'data/general/llava_v1_5_mix665k_with_videogpt_plus_576k.json'

    IMAGE_ROOT = 'data/llava_instruct'
    VIDEO_ROOT = 'data/videogpt_plus'

    SOURCE = 'llava_instruct_665k_videogpt_plus_576k'

    @classmethod
    def load_annos(self):
        annos = nncore.load(self.ANNO_PATH)

        for anno in annos:
            assert not ('image' in anno and 'video' in anno), anno
            if 'image' in anno:
                anno['source'] = 'llava_instruct_665k'
                anno['data_type'] = 'multimodal'
            elif 'video' in anno:
                anno['source'] = 'videogpt_plus_576k'
                anno['data_type'] = 'multimodal'
            else:
                anno['source'] = 'llava_instruct_665k'
                anno['data_type'] = 'text'

        return annos
