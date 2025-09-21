#!/bin/bash

set -e

command -v npu-smi &>/dev/null && nproc=$(npu-smi info -l | grep "NPU ID" | wc -l) || nproc=$(nvidia-smi --list-gpus | wc -l)

export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((nproc-1)))
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"
export NCCL_TIMEOUT=600

echo -e "\e[1;32mDevice Count:\e[0m $nproc ($CUDA_VISIBLE_DEVICES)"

stage1_ckpt_path="work_dirs/${nproc}p_stage1_1e"

torchrun --nproc_per_node $nproc unipixel/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path model_zoo/Qwen2.5-VL-3B-Instruct \
    --base_model qwen2_5_vl \
    --conv_type chatml \
    --sam2_config configs/sam2.1_hiera_b+ \
    --sam2_checkpoint model_zoo/sam2.1/sam2.1_hiera_base_plus.pt \
    --sam2_image_size 768 \
    --sam2_apply_postprocessing False \
    --sam2_inference_mode False \
    --sam2_hidden_tokens 2 \
    --sam2_batch_mode False \
    --sam2_enable_decoder False \
    --lora_enable False \
    --tuning_modules ref \
    --datasets videorefer_short_caption_ref,inst_it_image_short_caption_raw_ref \
    --sample_frames 8 \
    --sample_type random \
    --max_conv_turns 3 \
    --max_video_frames 500 \
    --max_video_len 300 \
    --max_num_words 500 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $((256/nproc)) \
    --output_dir $stage1_ckpt_path \
    --save_full_model False \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --tf32 True \
    --bf16 True \
    --report_to wandb

stage2_ckpt_path="work_dirs/${nproc}p_stage2_1e"

torchrun --nproc_per_node $nproc unipixel/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path model_zoo/Qwen2.5-VL-3B-Instruct \
    --base_model qwen2_5_vl \
    --conv_type chatml \
    --sam2_config configs/sam2.1_hiera_b+ \
    --sam2_checkpoint model_zoo/sam2.1/sam2.1_hiera_base_plus.pt \
    --sam2_image_size 768 \
    --sam2_apply_postprocessing False \
    --sam2_inference_mode False \
    --sam2_hidden_tokens 2 \
    --sam2_batch_mode False \
    --sam2_enable_decoder True \
    --lora_enable False \
    --tuning_modules seg \
    --datasets refcoco:5,refcoco+:5,refcocog:5,refclef:5,ref_youtube_vos:3 \
    --sample_frames 8 \
    --sample_type random \
    --max_conv_turns 3 \
    --max_video_frames 500 \
    --max_video_len 300 \
    --max_num_words 500 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $((256/nproc)) \
    --output_dir $stage2_ckpt_path \
    --save_full_model False \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --tf32 True \
    --bf16 True \
    --report_to wandb

stage3_ckpt_path="work_dirs/${nproc}p_stage3_1e"

torchrun --nproc_per_node $nproc unipixel/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path model_zoo/Qwen2.5-VL-3B-Instruct \
    --partial_checkpoint $stage1_ckpt_path,$stage2_ckpt_path \
    --base_model qwen2_5_vl \
    --conv_type chatml \
    --sam2_config configs/sam2.1_hiera_b+ \
    --sam2_checkpoint model_zoo/sam2.1/sam2.1_hiera_base_plus.pt \
    --sam2_image_size 768 \
    --sam2_apply_postprocessing False \
    --sam2_inference_mode False \
    --sam2_hidden_tokens 2 \
    --sam2_batch_mode False \
    --sam2_enable_decoder True \
    --sam2_lr 5e-6 \
    --lora_enable True \
    --lora_type qkvo_all \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --tuning_modules embedding,ref,ref_enc,msk,seg,sam2 \
    --datasets revos:5,mevis:5,lvvis:3,ref_youtube_vos:5,ref_davis17:10,ref_sav:3,groundmore:3,vicas:3,reason_seg:10,ade20k:3,cocostuff:3,mapillary:3,paco_lvis:3,pascal_part:3,refcoco:10,refcoco+:10,refcocog:10,refclef:10,videorefer_detailed_caption:5,videorefer_qa:5,inst_it_video_qa_raw:5,osprey_conversation:5,osprey_detail_description:5,osprey_pos_neg:5,videorefer_qa_mem:3,inst_it_video_qa_raw_mem:3,llava_instruct_665k_videogpt_plus_576k \
    --sample_frames 8 \
    --sample_type random \
    --sample_objects 5 \
    --num_threads 1 \
    --max_conv_turns 3 \
    --max_video_frames 500 \
    --max_video_len 300 \
    --max_num_words 200 \
    --max_num_tokens 40960 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $((32/nproc)) \
    --output_dir $stage3_ckpt_path \
    --save_full_model True \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 500 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --tf32 True \
    --bf16 True \
    --report_to wandb

bash scripts/auto_eval.sh $stage3_ckpt_path
