"""
generate_saliency_maps_patch.py
================================
本文件不是完整脚本，而是对原 generate_saliency_maps.py 的
【分段修改说明】。每一节说明：
  - 修改位置（函数名 / 行号范围描述）
  - 原代码片段
  - 新代码片段
  - 修改原因

请按顺序将这些改动应用到你自己的 generate_saliency_maps.py。
"""

# ==========================================================================
# 修改 1：在文件顶部 import 区域，新增导入 PromptRouter
# --------------------------------------------------------------------------
# 原代码（import 区域末尾附近）：
#
#   from text_prompts import breast_tumor_P2_prompts, ...
#
# 改为：在原有 import 之后添加一行
# ==========================================================================

NEW_IMPORT = """
# ---- 模块B: Prompt Router ----
from prompt_router import PromptRouter
"""

# ==========================================================================
# 修改 2：在 argparse 参数定义区域，新增 --prompt-bank-dir 和 --router-top-k
# --------------------------------------------------------------------------
# 原代码（parser.add_argument 区域）：
#
#   parser.add_argument('--model-name', ...)
#   parser.add_argument('--finetuned', ...)
#   ...
#
# 在最后一个 add_argument 之后追加：
# ==========================================================================

NEW_ARGS = """
    # ---- 模块B 新增参数 ----
    parser.add_argument(
        '--prompt-bank-dir',
        type=str,
        default='generated_prompts',
        help='存放 JSON prompt bank 文件的目录路径。'
             '例如 generated_prompts/ 下有 '
             'benign_breast_prompt_bank.json 等文件。'
    )
    parser.add_argument(
        '--router-top-k',
        type=int,
        default=3,
        help='Prompt Router 每张图像选取的 top-k prompt 数量。'
             '论文建议 k=3，不宜过大（>5 会引入噪声）。'
    )
    parser.add_argument(
        '--router-temperature',
        type=float,
        default=0.07,
        help='Prompt Router softmax 温度 τ，控制权重集中程度。'
             '与 CLIP 原始温度对齐，默认 0.07。'
    )
    parser.add_argument(
        '--use-router',
        action='store_true',
        help='是否启用 Prompt Router（模块B）。'
             '不加此参数则退化为原始平均 prompt 行为。'
    )
"""

# ==========================================================================
# 修改 3：在模型加载完成后，初始化 PromptRouter
# --------------------------------------------------------------------------
# 定位：原代码中 model, preprocess, tokenizer 加载完毕之后，
#       进入图像循环之前（通常在 main() 函数里）。
#
# 原代码示意（伪代码）：
#   model, _, preprocess = open_clip.create_model_and_transforms(...)
#   tokenizer = open_clip.get_tokenizer(...)
#   # ... 加载模型权重 ...
#
#   for img_path in image_paths:   # ← 图像循环开始
#       ...
#
# 在图像循环开始之前插入：
# ==========================================================================

ROUTER_INIT_CODE = """
    # ================================================================
    # 模块B：初始化 Prompt Router
    # 说明：文本特征预计算只做一次，所有图像共用，不重复编码。
    # ================================================================
    if args.use_router:
        import glob
        import os

        # 自动扫描 prompt bank 目录下所有 JSON 文件
        bank_paths = sorted(
            glob.glob(os.path.join(args.prompt_bank_dir, "*.json"))
        )
        if len(bank_paths) == 0:
            raise FileNotFoundError(
                f"在 {args.prompt_bank_dir} 下未找到任何 .json 文件，"
                f"请检查 --prompt-bank-dir 参数。"
            )
        print(f"[PromptRouter] 发现 {len(bank_paths)} 个 prompt bank 文件：")
        for p in bank_paths:
            print(f"    {p}")

        router = PromptRouter(
            model=model,
            tokenizer=tokenizer,
            prompt_bank_paths=bank_paths,
            top_k=args.router_top_k,
            temperature=args.router_temperature,
            device=device,
            encode_text_normalized=True,   # BiomedCLIP 需要归一化
        )
        print(f"[PromptRouter] 初始化完成，top_k={args.router_top_k}，"
              f"temperature={args.router_temperature}")
    else:
        router = None
"""

# ==========================================================================
# 修改 4：在图像循环内部，替换文本特征计算逻辑
# --------------------------------------------------------------------------
# 定位：原代码在循环体内，对每张图像编码文本并得到 Z_text，
#       大致如下：
#
#   # --- 原始代码 ---
#   text_tokens = tokenizer(text_prompts).to(device)
#   with torch.no_grad():
#       Z_text = model.encode_text(text_tokens)        # (N_prompts, D)
#       Z_text = F.normalize(Z_text, dim=-1)
#       Z_text = Z_text.mean(dim=0, keepdim=True)      # 简单平均 → (1, D)
#
# 替换为：
# ==========================================================================

REPLACE_TEXT_FEATURE_CODE = """
        # ================================================================
        # 模块B：Prompt Router 动态选取文本特征
        # ----------------------------------------------------------------
        # 原始做法：对所有 prompt 取平均嵌入（与图像内容无关）
        # 新做法：先用图像特征做 top-k 路由，再 softmax 加权融合
        # ================================================================
        if router is not None:
            # 步骤1：先单独编码图像，得到图像特征（用于路由打分）
            with torch.no_grad():
                # 注意：这里用的是 BiomedCLIP 的 encode_image
                # 返回 CLS token 特征，形状 (1, D)
                img_feature_for_routing = model.encode_image(
                    image_tensor.unsqueeze(0).to(device)
                )  # (1, D)
                img_feature_for_routing = F.normalize(
                    img_feature_for_routing, dim=-1
                )  # L2 归一化，与文本特征空间对齐

            # 步骤2：Router 根据图像特征选 top-k prompt 并融合
            # 输入: (1, D)  →  输出: (D,)
            Z_text = router.route(img_feature_for_routing)  # (D,)
            Z_text = Z_text.unsqueeze(0)  # (1, D)，与原代码输出形状一致

            # （可选）调试：打印这张图选了哪些 prompt
            # top_prompts = router.get_top_k_prompts(img_feature_for_routing)
            # for p in top_prompts:
            #     print(f"  rank{p['rank']}: [{p['type']}] {p['text'][:60]}...")

        else:
            # ---- 退化模式：原始平均 prompt（保持向后兼容）----
            text_tokens = tokenizer(text_prompts).to(device)
            with torch.no_grad():
                Z_text = model.encode_text(text_tokens)      # (N, D)
                Z_text = F.normalize(Z_text, dim=-1)
                Z_text = Z_text.mean(dim=0, keepdim=True)   # (1, D)
        # ================================================================
        # 后续代码不变：Z_text 直接传入 M2IB 生成 saliency map
        # ================================================================
"""

# ==========================================================================
# 修改 5：更新 zeroshot.sh，新增 Router 参数
# --------------------------------------------------------------------------
# 在 generate_saliency_maps.py 的调用行末尾追加：
#
#   --use-router \
#   --prompt-bank-dir generated_prompts \
#   --router-top-k 3 \
#   --router-temperature 0.07 \
# ==========================================================================

NEW_ZEROSHOT_SH = """
#!/bin/bash
# zeroshot.sh  —  更新版（加入模块B Prompt Router）

DATASET=$1

python saliency_maps/generate_saliency_maps.py \\
    --input-path ${DATASET}/images \\
    --output-path saliency_map_outputs/${DATASET}/masks \\
    --val-path ${DATASET}/val_images \\
    --model-name BiomedCLIP \\
    --finetuned \\
    --hyper-opt \\
    --use-router \\
    --prompt-bank-dir generated_prompts \\
    --router-top-k 3 \\
    --router-temperature 0.07

python postprocessing/postprocess_saliency_maps.py \\
    --input-path ${DATASET}/images \\
    --output-path coarse_outputs/${DATASET}/masks \\
    --sal-path saliency_map_outputs/${DATASET}/masks \\
    --postprocess kmeans \\
    --filter

python segment-anything/prompt_sam.py \\
    --input ${DATASET}/images \\
    --mask-input coarse_outputs/${DATASET}/masks \\
    --output sam_outputs/${DATASET}/masks \\
    --model-type vit_h \\
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \\
    --prompts boxes
"""

print("Patch 说明文件生成完毕，请按修改1~5依次应用到你的代码中。")
