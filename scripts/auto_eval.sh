#!/bin/bash

set -e

command -v npu-smi &>/dev/null && nproc=$(npu-smi info -l | grep "NPU ID" | wc -l) || nproc=$(nvidia-smi --list-gpus | wc -l)

export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((nproc-1)))
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

ckpt_path=$1

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_q.py \
        --dataset videorefer_bench_q \
        --split test \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/videorefer_bench_q_single \
        --vis_pred_path $ckpt_path/videorefer_bench_q_single_vis \
        --single_frame_mode \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/videorefer_bench_q_single

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_q.py \
        --dataset videorefer_bench_q \
        --split test \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/videorefer_bench_q \
        --vis_pred_path $ckpt_path/videorefer_bench_q_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/videorefer_bench_q

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_d.py \
        --dataset videorefer_bench_d \
        --split test \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/videorefer_bench_d_single \
        --vis_pred_path $ckpt_path/videorefer_bench_d_single_vis \
        --single_frame_mode \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_d.py \
        --dataset videorefer_bench_d \
        --split test \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/videorefer_bench_d \
        --vis_pred_path $ckpt_path/videorefer_bench_d_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_pixelqa.py \
        --prompt_type point \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/pixelqa_point \
        --vis_pred_path $ckpt_path/pixelqa_point_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/pixelqa_point
python unipixel/eval/eval_pixelqa.py $ckpt_path/pixelqa_point

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_pixelqa.py \
        --prompt_type box \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/pixelqa_box \
        --vis_pred_path $ckpt_path/pixelqa_box_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/pixelqa_box
python unipixel/eval/eval_pixelqa.py $ckpt_path/pixelqa_box

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_pixelqa.py \
        --prompt_type mix \
        --model_path $ckpt_path \
        --res_pred_path $ckpt_path/pixelqa_mix \
        --vis_pred_path $ckpt_path/pixelqa_mix_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/pixelqa_mix
python unipixel/eval/eval_pixelqa.py $ckpt_path/pixelqa_mix

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset revos \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/revos_seg \
        --vis_pred_path $ckpt_path/revos_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_revos.py $ckpt_path/revos_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset mevis \
        --split valid_u \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/mevis_val_u_seg \
        --vis_pred_path $ckpt_path/mevis_val_u_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py mevis $ckpt_path/mevis_val_u_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset mevis \
        --split valid \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/mevis_val_seg \
        --vis_pred_path $ckpt_path/mevis_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_youtube_vos \
        --split valid \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_youtube_vos_val_seg \
        --vis_pred_path $ckpt_path/ref_youtube_vos_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_davis17 \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_davis17_seg \
        --vis_pred_path $ckpt_path/ref_davis17_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py ref_davis17 $ckpt_path/ref_davis17_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_sav_eval \
        --split valid \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_sav_seg \
        --vis_pred_path $ckpt_path/ref_sav_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py ref_sav $ckpt_path/ref_sav_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset groundmore \
        --split test \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/groundmore_seg \
        --vis_pred_path $ckpt_path/groundmore_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_groundmore.py $ckpt_path/groundmore_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset reason_seg \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/reason_seg_val_seg \
        --vis_pred_path $ckpt_path/reason_seg_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/reason_seg_val_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset reason_seg \
        --split test \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/reason_seg_test_seg \
        --vis_pred_path $ckpt_path/reason_seg_test_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/reason_seg_test_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco_val_seg \
        --vis_pred_path $ckpt_path/refcoco_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco_val_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco \
        --split testA \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco_testA_seg \
        --vis_pred_path $ckpt_path/refcoco_testA_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco_testA_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco \
        --split testB \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco_testB_seg \
        --vis_pred_path $ckpt_path/refcoco_testB_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco_testB_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco+ \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco+_val_seg \
        --vis_pred_path $ckpt_path/refcoco+_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco+_val_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco+ \
        --split testA \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco+_testA_seg \
        --vis_pred_path $ckpt_path/refcoco+_testA_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco+_testA_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcoco+ \
        --split testB \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcoco+_testB_seg \
        --vis_pred_path $ckpt_path/refcoco+_testB_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcoco+_testB_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcocog \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcocog_val_seg \
        --vis_pred_path $ckpt_path/refcocog_val_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcocog_val_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset refcocog \
        --split test \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/refcocog_test_seg \
        --vis_pred_path $ckpt_path/refcocog_test_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_refcoco.py $ckpt_path/refcocog_test_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_general.py \
        --dataset mvbench \
        --split test \
        --model_path $ckpt_path \
        --pred_path $ckpt_path/mvbench \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/mvbench
