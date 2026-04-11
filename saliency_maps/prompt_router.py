import json
from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn.functional as F
from collections.abc import Mapping


class PromptRouter:
    def __init__(
        self,
        model,
        tokenizer,
        prompt_bank_paths: List[Union[str, Path]],
        top_k: int = 3,
        temperature: float = 0.07,
        device: str = "cuda",
        encode_text_normalized: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.temperature = temperature
        self.device = device
        self.encode_text_normalized = encode_text_normalized

        self.prompts: List[str] = []
        self.prompt_meta: List[dict] = []

        print(f"[PromptRouter] 发现 {len(prompt_bank_paths)} 个 prompt bank 文件：")
        for path in prompt_bank_paths:
            print(f"    {path}")
            self._load_bank(path)

        if len(self.prompts) == 0:
            raise ValueError("prompt bank 为空，请检查 JSON 文件路径！")

        print(
            f"[PromptRouter] 共加载 {len(self.prompts)} 条 prompt，来自 {len(prompt_bank_paths)} 个文件。"
        )

        self.model.eval()
        self._text_features: torch.Tensor = self._precompute_text_features()

    def _move_tokens_to_device(self, tokens):
        if hasattr(tokens, "to"):
            return tokens.to(self.device)

        if isinstance(tokens, Mapping):
            return {k: v.to(self.device) for k, v in tokens.items()}

        if torch.is_tensor(tokens):
            return tokens.to(self.device)

        raise TypeError(f"不支持的 tokenizer 输出类型: {type(tokens)}")

    def _tokenize_batch(self, batch_texts):
        try:
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return self._move_tokens_to_device(tokens)
        except TypeError:
            pass

        tokens = self.tokenizer(batch_texts)
        return self._move_tokens_to_device(tokens)

    def _encode_text_batch(self, batch_texts):
        tokens = self._tokenize_batch(batch_texts)

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                if isinstance(tokens, dict) or hasattr(tokens, "keys"):
                    feat = self.model.get_text_features(**tokens)
                else:
                    feat = self.model.get_text_features(tokens)
            elif hasattr(self.model, "encode_text"):
                feat = self.model.encode_text(tokens)
            elif hasattr(self.model, "text_model"):
                if isinstance(tokens, dict) or hasattr(tokens, "keys"):
                    outputs = self.model.text_model(**tokens)
                else:
                    outputs = self.model.text_model(tokens)

                if hasattr(outputs, "last_hidden_state"):
                    feat = outputs.last_hidden_state[:, 0, :]
                else:
                    feat = outputs[0][:, 0, :]
            else:
                raise AttributeError(
                    "当前 model 既没有 get_text_features / encode_text / text_model，无法编码文本"
                )

        if self.encode_text_normalized:
            feat = F.normalize(feat, dim=-1)

        return feat

    def encode_image_feature(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                try:
                    feat = self.model.get_image_features(pixel_values=image_tensor)
                except TypeError:
                    feat = self.model.get_image_features(image_tensor)
            elif hasattr(self.model, "encode_image"):
                feat = self.model.encode_image(image_tensor)
            elif hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(image_tensor, output_hidden_states=True)
                if hasattr(outputs, "last_hidden_state"):
                    feat = outputs.last_hidden_state[:, 0, :]
                else:
                    feat = outputs[0][:, 0, :]
            else:
                raise AttributeError(
                    "当前 model 既没有 get_image_features / encode_image / vision_model，无法编码图像"
                )

        feat = F.normalize(feat, dim=-1)
        return feat

    def _load_bank(self, path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 prompt bank 文件：{path}")

        with open(path, "r", encoding="utf-8") as f:
            bank = json.load(f)

        for item in bank.get("prompts", []):
            text = item.get("text")
            if not text:
                continue
            self.prompts.append(text)
            self.prompt_meta.append(item)

    def _precompute_text_features(self) -> torch.Tensor:
        all_features = []
        batch_size = 64

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(self.prompts), batch_size):
                batch_texts = self.prompts[i: i + batch_size]
                feat = self._encode_text_batch(batch_texts)
                all_features.append(feat.cpu())

        text_feats = torch.cat(all_features, dim=0)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats.to(self.device)

    def _get_top_k(self, image_feature: torch.Tensor, k: Optional[int] = None):
        if image_feature.dim() == 1:
            image_feature = image_feature.unsqueeze(0)

        image_feature = image_feature.to(self.device)
        image_feature = F.normalize(image_feature, dim=-1)

        with torch.no_grad():
            similarities = (image_feature @ self._text_features.T).squeeze(0)
            k = min(k or self.top_k, len(self.prompts))
            topk_scores, topk_indices = torch.topk(similarities, k=k)

        return topk_scores, topk_indices

    def route(self, image_feature: torch.Tensor) -> torch.Tensor:
        topk_scores, topk_indices = self._get_top_k(image_feature, k=self.top_k)

        with torch.no_grad():
            weights = F.softmax(topk_scores / self.temperature, dim=0)
            top_text_feats = self._text_features[topk_indices]
            fused = (weights.unsqueeze(-1) * top_text_feats).sum(dim=0)
            fused = F.normalize(fused, dim=-1)

        return fused

    def route_batch(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.to(self.device)
        image_features = F.normalize(image_features, dim=-1)

        with torch.no_grad():
            similarities = image_features @ self._text_features.T
            k = min(self.top_k, len(self.prompts))
            topk_scores, topk_indices = torch.topk(similarities, k=k, dim=1)
            weights = F.softmax(topk_scores / self.temperature, dim=1)
            top_text_feats = self._text_features[topk_indices]
            fused = (weights.unsqueeze(-1) * top_text_feats).sum(dim=1)
            fused = F.normalize(fused, dim=-1)

        return fused

    def get_top_k_prompt_features(self, image_feature: torch.Tensor, k: int = None):
        topk_scores, topk_indices = self._get_top_k(image_feature, k=k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(topk_scores.cpu().tolist(), topk_indices.cpu().tolist()), start=1
        ):
            meta = self.prompt_meta[idx]
            results.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "index": idx,
                    "id": meta.get("id", str(idx)),
                    "type": meta.get("type", "unknown"),
                    "text": self.prompts[idx],
                    "feature": self._text_features[idx].detach().clone(),
                }
            )
        return results

    def get_top_k_prompts(self, image_feature: torch.Tensor, k: int = None):
        topk = self.get_top_k_prompt_features(image_feature, k=k)
        return [
            {
                "rank": item["rank"],
                "score": item["score"],
                "text": item["text"],
                "type": item["type"],
                "id": item["id"],
            }
            for item in topk
        ]