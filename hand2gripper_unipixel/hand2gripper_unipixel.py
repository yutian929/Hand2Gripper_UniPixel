# unipixel_infer.py
# Copyright (c) 2025.
# 封装 UniPixel 推理 + 简单示例：本地图片 -> 数手并分割手

import os
import imageio.v3 as iio
import torch
import cv2
import torchvision
import nncore
from typing import Union
import numpy as np

from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_video
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import draw_mask


class UniPixel:
    """
    统一封装的 UniPixel 推理类。
    - 支持单图 / 视频（示例里用单图）
    - 支持 SAM2 掩膜可视化
    """

    def __init__(
        self,
        model_path: str = "PolyU-ChenLab/UniPixel-3B",
        device: str = "auto",
        dtype: str = "bfloat16",
        sample_frames: int = 16
    ) -> None:
        self.model_path = model_path
        self.device_req = device
        self.dtype = dtype
        self.sample_frames = sample_frames

        self.model, self.processor = build_model(self.model_path, device=self.device_req, dtype=self.dtype)
        self.device = next(self.model.parameters()).device
        self.sam2_transform = get_sam2_transform(self.model.config.sam2_image_size)

    def _build_messages(self, media_paths, prompt: str, is_image: bool):
        """
        构造多模态消息。保持与原脚本一致的 'video' 格式以兼容 process_vision_info。
        """
        # 为了兼容原实现，这里即使是单张图片也走 'video' 字段（单帧）
        num_imgs = len(media_paths)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": media_paths,
                    "min_pixels": 128 * 28 * 28,
                    "max_pixels": 256 * 28 * 28 * max(1, int(self.sample_frames / max(1, num_imgs))),
                },
                {"type": "text", "text": prompt},
            ],
        }]
        return messages

    @torch.inference_mode()
    def infer(self, media_path: Union[str, np.ndarray], prompt: str):
        """
        统一推理入口：
        - 读取图片/视频
        - 构造消息与输入张量
        - 运行 generate
        - 如果模型产生了 seg 掩膜，返回可视化并保存
        """
        if isinstance(media_path, np.ndarray):
            # 直接传入了图片数组
            frames = torch.from_numpy(np.array(media_path)).unsqueeze(0)  # Tensor，形状 (1,H,W,3)
            is_image = True
            tmp_file_path = "/tmp/unipixel_input_image.png"
            cv2.imwrite(tmp_file_path, media_path)
            media_list = [tmp_file_path]
        else:
            is_image = any(media_path.lower().endswith(k) for k in (".jpg", ".jpeg", ".png"))
            if is_image:
                frames, media_list = load_image(media_path), [media_path]          # frames: (T,H,W,3) 或 (H,W,3)；load_image 返回 Tensor-like
            else:
                frames, media_list = load_video(media_path, sample_frames=self.sample_frames)

        # 保证 frames 形状为 (T,H,W,3)
        if frames.ndim == 3:
            frames = frames[None, ...]  # 单帧

        messages = self._build_messages(media_list, prompt, is_image=is_image)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors="pt", **kwargs)

        # SAM2 需要的帧与原图尺寸信息
        data["frames"] = [self.sam2_transform(frames).to(self.model.sam2.dtype)]
        # 注意：frame_size 为 (H, W)
        data["frame_size"] = [frames.shape[1:3]]

        output_ids = self.model.generate(
            **data.to(self.device),
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512,
        )

        # 解码文本回答
        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if len(output_ids) > 0 and output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)
        # 处理分割结果（如果有）
        seg_image = None
        seg_mask = None
        if hasattr(self.model, "seg") and len(self.model.seg) >= 1:
            seg_mask = self.model.seg  # list[Tensor]
            seg_imgs = draw_mask(frames, seg_mask)

            seg_image = cv2.cvtColor(seg_imgs[0], cv2.COLOR_RGBA2BGR)
            seg_mask = seg_mask[0].cpu().numpy()
            if seg_mask.ndim == 4:
                seg_mask = np.squeeze(np.squeeze(seg_mask)).astype(bool)
            elif seg_mask.ndim == 3:
                seg_mask = np.squeeze(seg_mask).astype(bool)
            else:
                seg_mask = seg_mask.astype(bool)
            
        return {
            "response": response,
            "seg_image": seg_image,
            "seg_mask": seg_mask,   # 内存中的掩膜（可能是 list 或 ndarray）
        }


if __name__ == "__main__":
    """
    最小测试：读取本地图片，询问“有几只手，并把它们分割出来”，保存可视化结果到 outputs/
    使用前请把 media_path 改成你的本地图片路径。
    """
    media_path = "/home/yutian/projs/Hand2Gripper/P01_01_frame_0000091256.jpg"
    prompt = "图片里有几只手？请直接回答阿拉伯数字，并把所有手分割出来，且提供分割掩膜。"

    up = UniPixel(
        model_path="PolyU-ChenLab/UniPixel-3B",
        device="auto",
        dtype="bfloat16",
        sample_frames=16
    )
    
    result = up.infer(cv2.imread(media_path), prompt)

    cv2.imwrite("seg_image.png", result["seg_image"])
    cv2.imwrite("seg_mask.png", result["seg_mask"].astype(np.uint8) * 255)
    
    print("\n=== UniPixel 回答 ===")
    print(result["response"])
