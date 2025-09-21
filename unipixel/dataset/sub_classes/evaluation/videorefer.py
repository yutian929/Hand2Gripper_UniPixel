# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import re

import nncore
from torch.utils.data import Dataset

from unipixel.constants import REF_TOKEN, SEG_TOKEN
from unipixel.dataset.hybrid import DATASETS


@DATASETS.register(name='videorefer_bench_q')
class VideoReferBenchQDataset(Dataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-Bench/VideoRefer-Bench-Q.json'

    DATA_ROOT = 'data/videorefer/VideoRefer-Bench/videos'

    SOURCE = 'videorefer_bench_q'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            question = raw_anno['Question'].strip() + '\nOptions:'
            for opt in raw_anno['options']:
                question += f'\n{opt}'
            mem_question = question
            question += '\nPlease only give the best option.'

            assert '<video>' not in question, question
            assert question.count('<region>') == len(raw_anno['annotation']), raw_anno

            # obj_frame_inds: num_objs * num_frames_per_obj
            # all_frame_inds: num_frames
            obj_frame_inds = [sorted([int(k) for k in o.keys()]) for o in raw_anno['annotation']]
            all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

            # masks: num_objs * num_sampled_frames * 1 (one rle per object)
            masks = []
            for obj_anno in raw_anno['annotation']:
                obj_masks = []
                for i in all_frame_inds:
                    frm_mask = obj_anno[str(i)]['segmentation'] if str(i) in obj_anno else None
                    obj_masks.append([frm_mask])
                masks.append(obj_masks)

            # single_frame_masks: num_objs * num_sampled_frames * 1 (one rle per object)
            single_frame_masks = []
            for obj_anno in raw_anno['annotation']:
                obj_masks = []
                for i in all_frame_inds:
                    frm_mask = obj_anno[str(i)]['segmentation'] if str(i) == raw_anno['frame_idx'] else None
                    obj_masks.append([frm_mask])
                single_frame_masks.append(obj_masks)

            if '<object' in mem_question:
                matches = re.findall(r'<object\d+><region>', mem_question)
                mem_response = ''
                for match in matches:
                    # '<object0><region>' -> '0' '<object0>'
                    oid, obj = match[7:-9], match[:-8]
                    assert oid == str(int(oid)), match
                    mem_question = mem_question.replace(match, f'[{oid}] {REF_TOKEN}').replace(obj, f'[{oid}]')
                    mem_response += f' [{oid}] {SEG_TOKEN}'
                mem_response = mem_response.strip()
                assert '<region>' not in mem_question, mem_question
            else:
                oid = 0
                mem_question = mem_question.replace('<region>', f'[{oid}] {REF_TOKEN}')
                mem_response = f'[{oid}] {SEG_TOKEN}'

            anno = dict(
                source=self.SOURCE,
                data_type='region_{}',
                video_path=nncore.join(self.DATA_ROOT, raw_anno['video']),
                vid=nncore.pure_name(raw_anno['video']),
                frame_idx=int(raw_anno['frame_idx']),
                obj_frame_inds=obj_frame_inds,
                all_frame_inds=all_frame_inds,
                mem_question=mem_question,
                mem_response=mem_response,
                question=question,
                options=raw_anno['options'],
                ans=raw_anno['Answer'][1],
                mask_type='rle',
                masks=masks,
                single_frame_masks=single_frame_masks,
                task=raw_anno['type'])

            annos.append(anno)

        return annos


@DATASETS.register(name='videorefer_bench_d')
class VideoReferBenchDDataset(Dataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-Bench/VideoRefer-Bench-D.json'

    DATA_ROOT = 'data/videorefer/VideoRefer-Bench/videos/Panda-70M-part'

    SOURCE = 'videorefer_bench_d'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            question = 'Please give a detailed description of the highlighted object [0] in the video.'

            # obj_frame_inds: num_objs * num_frames_per_obj
            # all_frame_inds: num_frames
            obj_frame_inds = [sorted([int(k) for k in o.keys()]) for o in raw_anno['annotation']]
            all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

            # masks: num_objs * num_sampled_frames * 1 (one rle per object)
            masks = []
            for obj_anno in raw_anno['annotation']:
                obj_masks = []
                for i in all_frame_inds:
                    frm_mask = obj_anno[str(i)]['segmentation'] if str(i) in obj_anno else None
                    obj_masks.append([frm_mask])
                masks.append(obj_masks)

            # single_frame_masks: num_objs * num_sampled_frames * 1 (one rle per object)
            single_frame_masks = []
            for obj_anno in raw_anno['annotation']:
                obj_masks = []
                for i in all_frame_inds:
                    frm_mask = obj_anno[str(i)]['segmentation'] if str(i) == raw_anno['frame_idx'] else None
                    obj_masks.append([frm_mask])
                single_frame_masks.append(obj_masks)

            anno = dict(
                source=self.SOURCE,
                data_type='region_{}',
                video_path=nncore.join(self.DATA_ROOT, raw_anno['video']),
                vid=nncore.pure_name(raw_anno['video']),
                frame_idx=int(raw_anno['frame_idx']),
                obj_frame_inds=obj_frame_inds,
                all_frame_inds=all_frame_inds,
                question=question,
                caption=raw_anno['caption'],
                masks=masks,
                single_frame_masks=single_frame_masks)

            annos.append(anno)

        return annos
