#!/bin/bash
# zeroshot.sh  —  更新版（集成模块B: Prompt Router）
# 用法：bash zeroshot.sh <path/to/dataset>
# 例如：bash zeroshot.sh data/breast_tumors

DATASET=$1
# [INFO] Best hyperopt config: {'vbeta': 1.0, 'vvar': 0.1, 'vlayer': 7, 'top_k': 1, 'router_temperature': 0.03, 'mean_score': 0.10265470392505213, 'mean_coarse_dice': 0.10972601689908461, 'mean_box_iou': 0.08615497365230967, 'num_samples': 40}
# ---------------------------------------------------------------
# Step 1: 生成 saliency map（已集成模块B Prompt Router）
# 新增参数说明：
#   --use-router         启用 Prompt Router，不加则退化为原始平均 prompt
#   --prompt-bank-dir    JSON prompt bank 文件目录
#   --router-top-k       每张图像选 top-k 个最相关 prompt（建议 k=3）
#   --router-temperature softmax 温度，越小权重越集中（默认 0.07）
# ---------------------------------------------------------------
# python saliency_maps/generate_saliency_maps.py \
#   --input-path breast_tumors/test_images \
#   --gt-path breast_tumors/test_masks \
#   --val-path breast_tumors/val_images \
#   --val-mask-path breast_tumors/val_masks \
#   --output-path output_saliency/test_images \
#   --model-name BiomedCLIP \
#   --use-router \
#   --router-prompt-dir generated_prompts \
#   --overwrite \
#   --vvar 0.1 \
#   --vbeta 1.0 \
#   --vlayer 7 \
#   --use-contrastive
#   --hyper-opt \
#   --save-stage-metrics \
#   --save-router-debug \

# #   --use-local-fusion \
# #  --fusion-alpha-m2ib 0.6 \
# #  --fusion-alpha-local 0.4 \
# #  --pv-weight-modality 0.2 \
# #  --pv-weight-pathology 0.5 \
# #  --pv-weight-morphology 0.3
# ---------------------------------------------------------------
# Step 2: 后处理 saliency map → coarse mask（不变）
# ---------------------------------------------------------------
python postprocessing/postprocess_saliency_maps.py \
    --input-path ${DATASET}/test_images \
    --output-path coarse_outputs/${DATASET}/test_masks \
    --sal-path output_saliency/test_images \
    --postprocess kmeans \
    --filter

# # ---------------------------------------------------------------
# # Step 3: SAM 精炼（不变）
# # ---------------------------------------------------------------
python segment-anything/prompt_sam.py \
    --input ${DATASET}/test_images \
    --mask-input coarse_outputs/${DATASET}/test_masks \
    --output sam_outputs/${DATASET}/test_masks \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes

python evaluation/eval.py \
--gt_path ${DATASET}/test_masks \
--seg_path sam_outputs/${DATASET}/test_masks