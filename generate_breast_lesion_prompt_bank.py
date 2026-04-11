#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate lesion-centric breast ultrasound prompt banks via DeepSeek API.

Key fix vs original version
----------------------------
Prompts now use *descriptive* caption style to match BiomedCLIP's PubMed
pre-training distribution (e.g. "An ultrasound image showing …") instead of
imperative/command style ("Segment the …").  BiomedCLIP was trained on
PubMed figure captions that describe what is visible, so imperative prompts
produce poor text embeddings and break the IBA saliency signal.

Usage
-----
export DEEPSEEK_API_KEY="YOUR_KEY"
python generate_breast_lesion_prompt_bank.py --subtype generic
python generate_breast_lesion_prompt_bank.py --subtype benign
python generate_breast_lesion_prompt_bank.py --subtype malignant
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

TARGET_COUNTS = {
    "lesion_generic": 4,
    "subtype_aware": 4,
    "morphology_aware": 4,
}

ALLOWED_TYPES = set(TARGET_COUNTS.keys())
SLOT_KEYS = ["modality", "anatomy", "pathology", "morphology", "location"]

TYPE_PREFIX = {
    "lesion_generic": "lesion",
    "subtype_aware": "subtype",
    "morphology_aware": "morph",
}

GLOBAL_BANNED_TERMS = {
    "breast tissue",
    "fibroglandular tissue",
    "breast parenchyma",
    "surrounding soft tissue",
    "adjacent breast tissue",
    "subcutaneous fat",
    "subcutaneous fat layer",
    "pectoralis muscle",
    "ductal structures",
    "cooper's ligaments",
    "nipple-areolar complex",
    "visible structures",
    "visible structure",
    "anatomical regions",
    "anatomical region",
    "region of interest",
    "retroareolar",
    "upper outer quadrant",
    "upper inner quadrant",
    "lower outer quadrant",
    "lower inner quadrant",
    "skin line",
    # FIX: also ban imperative verbs so fallback stays descriptive
    "segment the",
    "identify the",
    "outline the",
    "delineate the",
    "detect the",
    "locate the",
}

GENERIC_BANNED_TERMS = {
    "malignant",
    "benign",
    "spiculated",
    "angular",
    "invasive",
    "aggressive",
    "posterior shadowing",
    "shadowing",
    "non-parallel",
    "ill-defined",
    "irregular margins",
    "spiculated borders",
    "taller-than-wide",
    "circumscribed",
    "posterior enhancement",
    "smooth margins",
    "oval",
    "round",
    "parallel orientation",
}

BENIGN_MORPHOLOGY_POOL = [
    "oval shape",
    "round shape",
    "circumscribed margins",
    "smooth margins",
    "parallel orientation",
    "posterior enhancement",
    "well-defined margins",
    "wider-than-tall appearance",
]

MALIGNANT_MORPHOLOGY_POOL = [
    "irregular shape",
    "spiculated margins",
    "angular margins",
    "non-parallel orientation",
    "posterior shadowing",
    "ill-defined margins",
    "non-circumscribed borders",
    "taller-than-wide appearance",
]

GENERIC_PATHOLOGY_POOL = [
    "breast lesion",
    "breast mass",
    "tumor region",
    "focal lesion",
]

BENIGN_PATHOLOGY_POOL = [
    "benign breast lesion",
    "benign breast mass",
    "benign tumor region",
    "benign focal lesion",
]

MALIGNANT_PATHOLOGY_POOL = [
    "malignant breast lesion",
    "malignant breast mass",
    "malignant tumor region",
    "malignant focal lesion",
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = normalize_space(
        text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    )
    if text and text[-1] not in ".!?":
        text += "."
    return text


def dedupe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_any_term(text: str, terms: set) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def subtype_config(subtype: str) -> Dict[str, Any]:
    if subtype == "benign":
        return {
            "bank_name": "benign_breast_lesion_prompt_bank",
            "pathology_pool": BENIGN_PATHOLOGY_POOL,
            "morphology_pool": BENIGN_MORPHOLOGY_POOL,
            "pathology_hint": "benign breast lesion / benign breast mass / benign tumor region",
            "morphology_hint": "oval shape, round shape, circumscribed margins, smooth margins, parallel orientation, posterior enhancement",
            "neutral_only": False,
        }
    if subtype == "malignant":
        return {
            "bank_name": "malignant_breast_lesion_prompt_bank",
            "pathology_pool": MALIGNANT_PATHOLOGY_POOL,
            "morphology_pool": MALIGNANT_MORPHOLOGY_POOL,
            "pathology_hint": "malignant breast lesion / malignant breast mass / malignant tumor region",
            "morphology_hint": "irregular shape, spiculated margins, angular margins, non-parallel orientation, posterior shadowing",
            "neutral_only": False,
        }
    return {
        "bank_name": "generic_breast_lesion_prompt_bank",
        "pathology_pool": GENERIC_PATHOLOGY_POOL,
        "morphology_pool": [],
        "pathology_hint": "breast lesion / breast mass / tumor region / focal lesion",
        "morphology_hint": "neutral only; no benign-specific or malignant-specific morphology",
        "neutral_only": True,
    }


def build_system_prompt() -> str:
    return """
You are a medical imaging caption writer generating prompt banks for breast ultrasound tumor segmentation.

Return valid JSON only.

The output JSON must follow this schema:
{
  "bank_name": "string",
  "subtype": "generic|benign|malignant",
  "modality": "breast_ultrasound",
  "prompts": [
    {
      "id": "string",
      "text": "string",
      "type": "lesion_generic|subtype_aware|morphology_aware",
      "slots": {
        "modality": "string or null",
        "anatomy": null,
        "pathology": "string or null",
        "morphology": "string or null",
        "location": null
      }
    }
  ]
}

CRITICAL STYLE RULE:
Every prompt text MUST be a descriptive caption that describes what is visible in the image,
in the style of a PubMed figure caption (e.g. "A breast ultrasound image showing a hypoechoic
mass with irregular margins.").
DO NOT use imperative/command style (e.g. do NOT write "Segment the...", "Identify the...",
"Detect the...", "Outline the...", "Delineate the...").
The text must start with "A breast ultrasound" or "An ultrasound image of the breast" or similar.

Other rules:
1. English only.
2. Each text must be a single descriptive sentence.
3. Every prompt must describe a lesion, mass, or tumor only.
4. Do NOT describe normal anatomy, tissue layers, muscles, ducts, ligaments.
5. Do NOT use location-specific terms such as quadrants or retroareolar region.
6. No BI-RADS, no pathology-report language, no treatment, no prognosis.
7. Keep prompts image-grounded and localizing.
8. Return JSON only.
""".strip()


def build_user_prompt(subtype: str) -> str:
    cfg = subtype_config(subtype)
    return f"""
Generate one structured prompt bank in JSON format for breast ultrasound tumor segmentation.

Target subtype: {subtype}

Target counts by type:
- lesion_generic: {TARGET_COUNTS["lesion_generic"]}
- subtype_aware: {TARGET_COUNTS["subtype_aware"]}
- morphology_aware: {TARGET_COUNTS["morphology_aware"]}

Subtype constraints:
- pathology emphasis: {cfg["pathology_hint"]}
- morphology emphasis: {cfg["morphology_hint"]}

Important constraints:
- EVERY prompt text must be a descriptive PubMed-style caption, NOT imperative.
  CORRECT: "A breast ultrasound image demonstrating a hypoechoic focal mass."
  WRONG:   "Segment the breast lesion in this ultrasound image."
- every prompt must describe a lesion / mass / tumor
- do not mention breast tissue, parenchyma, pectoralis muscle, ducts, ligaments
- do not mention quadrant location phrases
- if subtype is generic, keep prompts neutral and avoid benign-specific or malignant-specific cues
- fill each prompt with: id, text, type, slots.modality, slots.pathology, slots.morphology
  slots.anatomy = null, slots.location = null

Return one JSON object only.
""".strip()


def http_post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def deepseek_generate(
    subtype: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(subtype)},
        ],
        "temperature": temperature,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = http_post_json(API_URL, headers, payload)
            content = resp["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("Empty content returned by API.")
            return json.loads(content)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            last_error = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"DeepSeek API failed after {max_retries} retries: {last_error}")


def make_prompt_text(
    type_name: str,
    pathology: Optional[str],
    morphology: Optional[str],
) -> str:
    """
    FIX: always produce descriptive captions, never imperative commands.
    """
    if type_name == "morphology_aware" and morphology:
        if pathology:
            return normalize_text(
                f"A breast ultrasound image showing a {morphology} {pathology}"
            )
        return normalize_text(
            f"A breast ultrasound image showing a lesion with {morphology}"
        )

    if pathology:
        return normalize_text(
            f"A breast ultrasound image showing a {pathology}"
        )

    return normalize_text(
        "A breast ultrasound image showing a hypoechoic focal mass"
    )


# --------------------------------------------------------------------------
# Fallback bank – hardcoded descriptive prompts used when API is unavailable
# --------------------------------------------------------------------------
def fallback_bank(subtype: str) -> Dict[str, Any]:
    cfg = subtype_config(subtype)

    if subtype == "generic":
        lesion_generic_texts = [
            "A breast ultrasound image showing a hypoechoic focal lesion.",
            "An ultrasound scan of the breast demonstrating a well-defined mass.",
            "A breast ultrasound image revealing a focal tumor region with distinct borders.",
            "An ultrasound image of the breast showing a solid hypoechoic mass.",
        ]
        subtype_aware_texts = [
            "A breast ultrasound image showing a focal lesion within the breast parenchyma.",
            "An ultrasound scan demonstrating a rounded breast mass.",
            "A breast ultrasound image showing a solid focal mass with internal echoes.",
            "An ultrasound image of the breast revealing an echo-poor lesion.",
        ]
        morphology_texts = [
            "A breast ultrasound image showing a localized solid-appearing mass.",
            "An ultrasound scan of the breast with a visible focal hypoechoic lesion.",
            "A breast ultrasound image demonstrating a focal abnormal mass with defined edges.",
            "An ultrasound image of the breast showing a solid echo-poor lesion.",
        ]

    elif subtype == "benign":
        lesion_generic_texts = [
            "A breast ultrasound image showing a benign hypoechoic focal lesion.",
            "An ultrasound scan of the breast demonstrating a benign well-defined mass.",
            "A breast ultrasound image revealing a benign oval mass with smooth borders.",
            "An ultrasound image of the breast showing a benign solid mass.",
        ]
        subtype_aware_texts = [
            "A breast ultrasound image showing a benign breast lesion with circumscribed margins.",
            "An ultrasound scan demonstrating a benign rounded breast mass.",
            "A breast ultrasound image showing a benign oval mass with posterior enhancement.",
            "An ultrasound image of the breast revealing a benign well-circumscribed lesion.",
        ]
        morphology_texts = [
            "A breast ultrasound image showing an oval benign mass with smooth well-defined margins.",
            "An ultrasound scan of the breast with a round benign lesion and posterior enhancement.",
            "A breast ultrasound image demonstrating a benign mass with circumscribed smooth margins.",
            "An ultrasound image of the breast showing a wider-than-tall benign mass.",
        ]

    else:  # malignant
        lesion_generic_texts = [
            "A breast ultrasound image showing a malignant hypoechoic focal lesion.",
            "An ultrasound scan of the breast demonstrating a malignant irregular mass.",
            "A breast ultrasound image revealing a malignant tumor region with ill-defined borders.",
            "An ultrasound image of the breast showing a malignant solid mass.",
        ]
        subtype_aware_texts = [
            "A breast ultrasound image showing a malignant breast lesion with non-circumscribed margins.",
            "An ultrasound scan demonstrating a malignant irregularly shaped breast mass.",
            "A breast ultrasound image showing a malignant mass with posterior shadowing.",
            "An ultrasound image of the breast revealing a malignant ill-defined lesion.",
        ]
        morphology_texts = [
            "A breast ultrasound image showing a malignant mass with spiculated irregular margins.",
            "An ultrasound scan of the breast with a taller-than-wide malignant lesion.",
            "A breast ultrasound image demonstrating a malignant mass with angular non-circumscribed borders.",
            "An ultrasound image of the breast showing a malignant mass with posterior shadowing.",
        ]

    prompts: List[Dict[str, Any]] = []

    def add_prompt_group(type_name: str, texts: List[str]) -> None:
        for idx, text in enumerate(texts, 1):
            prompts.append(
                {
                    "id": f"{subtype}_{TYPE_PREFIX[type_name]}_{idx:02d}",
                    "text": normalize_text(text),
                    "type": type_name,
                    "slots": {
                        "modality": "breast ultrasound",
                        "anatomy": None,
                        "pathology": None,
                        "morphology": None,
                        "location": None,
                    },
                }
            )

    add_prompt_group("lesion_generic", lesion_generic_texts)
    add_prompt_group("subtype_aware", subtype_aware_texts)
    add_prompt_group("morphology_aware", morphology_texts)

    for p in prompts:
        low = p["text"].lower()
        if "benign" in low:
            p["slots"]["pathology"] = "benign breast lesion"
        elif "malignant" in low:
            p["slots"]["pathology"] = "malignant breast lesion"
        elif "tumor" in low:
            p["slots"]["pathology"] = "tumor region"
        elif "mass" in low:
            p["slots"]["pathology"] = "breast mass"
        else:
            p["slots"]["pathology"] = "breast lesion"

        morph_terms = [
            "oval", "round", "circumscribed", "smooth", "irregular",
            "spiculated", "angular", "non-parallel", "localized", "solid",
            "posterior enhancement", "posterior shadowing", "ill-defined",
            "taller-than-wide", "wider-than-tall",
        ]
        found = [m for m in morph_terms if m in low]
        p["slots"]["morphology"] = ", ".join(found) if found else None

    return {
        "bank_name": cfg["bank_name"],
        "subtype": subtype,
        "modality": "breast_ultrasound",
        "prompts": prompts,
    }


def ensure_slots(obj: Dict[str, Any]) -> Dict[str, Any]:
    slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    fixed: Dict[str, Optional[str]] = {}
    for key in SLOT_KEYS:
        val = slots.get(key)
        if isinstance(val, str):
            val = normalize_space(val)
            if not val:
                val = None
        elif val is not None:
            val = str(val)
        fixed[key] = val

    fixed["modality"] = "breast ultrasound"
    fixed["anatomy"] = None
    fixed["location"] = None
    return fixed


def sanitize_type(value: Any) -> str:
    s = str(value).strip().lower()
    aliases = {
        "generic": "lesion_generic",
        "lesion": "lesion_generic",
        "lesion_generic": "lesion_generic",
        "subtype": "subtype_aware",
        "subtype_aware": "subtype_aware",
        "pathology": "subtype_aware",
        "pathology_aware": "subtype_aware",
        "morphology": "morphology_aware",
        "morphology_aware": "morphology_aware",
    }
    s = aliases.get(s, s)
    return s if s in ALLOWED_TYPES else "lesion_generic"


def validate_text(text: str) -> bool:
    if not text:
        return False
    n = len(text.split())
    if n < 6 or n > 30:
        return False
    # FIX: reject imperative prompts that slipped through
    imperative_starters = (
        "segment ", "identify ", "outline ", "delineate ",
        "detect ", "locate ", "mark ", "find ",
    )
    if text.lower().startswith(imperative_starters):
        return False
    return True


def make_id(subtype: str, type_name: str, n: int) -> str:
    return f"{subtype}_{TYPE_PREFIX[type_name]}_{n:02d}"


def text_to_slots(text: str, subtype: str, type_name: str) -> Dict[str, Optional[str]]:
    low = text.lower()

    pathology = None
    morphology = None

    if subtype == "benign":
        if "mass" in low:
            pathology = "benign breast mass"
        elif "tumor" in low:
            pathology = "benign tumor region"
        else:
            pathology = "benign breast lesion"
    elif subtype == "malignant":
        if "mass" in low:
            pathology = "malignant breast mass"
        elif "tumor" in low:
            pathology = "malignant tumor region"
        else:
            pathology = "malignant breast lesion"
    else:
        if "mass" in low:
            pathology = "breast mass"
        elif "tumor" in low:
            pathology = "tumor region"
        else:
            pathology = "breast lesion"

    morph_map = [
        "oval shape", "round shape", "circumscribed margins", "smooth margins",
        "parallel orientation", "posterior enhancement", "well-defined margins",
        "wider-than-tall appearance", "irregular shape", "spiculated margins",
        "angular margins", "non-parallel orientation", "posterior shadowing",
        "ill-defined margins", "non-circumscribed borders", "taller-than-wide appearance",
        "localized", "solid-appearing",
    ]
    found = [m for m in morph_map if m.split()[0] in low]
    morphology = ", ".join(found) if found else None

    if type_name == "lesion_generic":
        morphology = None
        if subtype == "generic":
            pathology = "breast lesion"

    return {
        "modality": "breast ultrasound",
        "anatomy": None,
        "pathology": pathology,
        "morphology": morphology,
        "location": None,
    }


def postprocess_bank(raw: Dict[str, Any], subtype: str) -> Dict[str, Any]:
    cfg = subtype_config(subtype)
    raw_prompts = raw.get("prompts", [])
    if not isinstance(raw_prompts, list):
        raw_prompts = []

    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGET_COUNTS}
    seen = set()

    for obj in raw_prompts:
        if not isinstance(obj, dict):
            continue

        type_name = sanitize_type(obj.get("type"))
        slots = ensure_slots(obj)

        raw_text = obj.get("text")
        if isinstance(raw_text, str):
            text = normalize_text(raw_text)
        else:
            text = make_prompt_text(
                type_name=type_name,
                pathology=slots.get("pathology"),
                morphology=slots.get("morphology"),
            )

        if not validate_text(text):
            continue
        if contains_any_term(text, GLOBAL_BANNED_TERMS):
            continue
        if subtype == "generic" and contains_any_term(text, GENERIC_BANNED_TERMS):
            continue

        low = text.lower()
        if not any(token in low for token in ["lesion", "mass", "tumor"]):
            continue

        slots = text_to_slots(text, subtype, type_name)

        key = dedupe_key(text)
        if not key or key in seen:
            continue
        seen.add(key)

        grouped[type_name].append(
            {
                "id": "",
                "text": text,
                "type": type_name,
                "slots": slots,
            }
        )

    fallback = fallback_bank(subtype)
    fallback_grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGET_COUNTS}
    for p in fallback["prompts"]:
        fallback_grouped[p["type"]].append(p)

    repaired_prompts: List[Dict[str, Any]] = []

    for type_name, need in TARGET_COUNTS.items():
        cur = grouped[type_name][:need]

        if len(cur) < need:
            for fb in fallback_grouped[type_name]:
                fb_key = dedupe_key(fb["text"])
                if fb_key in seen:
                    continue
                seen.add(fb_key)
                cur.append(
                    {
                        "id": "",
                        "text": fb["text"],
                        "type": fb["type"],
                        "slots": fb["slots"],
                    }
                )
                if len(cur) >= need:
                    break

        for idx, item in enumerate(cur[:need], 1):
            item["id"] = make_id(subtype, type_name, idx)
            repaired_prompts.append(item)

    return {
        "bank_name": cfg["bank_name"],
        "subtype": subtype,
        "modality": "breast_ultrasound",
        "prompts": repaired_prompts,
    }


def export_json(bank: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bank, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_py(bank: Dict[str, Any], out_path: Path) -> None:
    subtype = bank["subtype"]
    variable_name = f"{subtype}_breast_lesion_prompt_bank"

    lines = [
        f"# Auto-generated lesion-centric prompt bank for subtype={subtype}",
        "",
        f"{variable_name} = {json.dumps(bank, indent=2, ensure_ascii=False)}",
        "",
        "# Convenience accessors",
        f"{variable_name}_texts = [p['text'] for p in {variable_name}['prompts']]",
        f"{variable_name}_pathology_phrases = [p['slots']['pathology'] for p in {variable_name}['prompts'] if p['slots']['pathology']]",
        f"{variable_name}_morphology_phrases = [p['slots']['morphology'] for p in {variable_name}['prompts'] if p['slots']['morphology']]",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subtype",
        choices=["generic", "benign", "malignant"],
        required=True,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out_dir", default="generated_prompts")
    parser.add_argument("--use_fallback_only", action="store_true")
    args = parser.parse_args()

    if args.use_fallback_only:
        bank = fallback_bank(args.subtype)
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY is not set.", file=sys.stderr)
            return 1

        try:
            raw = deepseek_generate(
                subtype=args.subtype,
                model=args.model,
                api_key=api_key,
                temperature=args.temperature,
            )
            bank = postprocess_bank(raw, args.subtype)
        except Exception as e:
            print(f"[WARN] API generation failed, using fallback instead: {e}", file=sys.stderr)
            bank = fallback_bank(args.subtype)

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.subtype}_breast_lesion_prompt_bank.json"
    py_path = out_dir / f"{args.subtype}_breast_lesion_prompt_bank.py"

    export_json(bank, json_path)
    export_py(bank, py_path)

    print(f"[OK] Saved JSON: {json_path}")
    print(f"[OK] Saved PY  : {py_path}")
    print(f"[OK] Total prompts: {len(bank['prompts'])}")

    by_type = {}
    for p in bank["prompts"]:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    print(f"[OK] By type: {json.dumps(by_type, ensure_ascii=False)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())