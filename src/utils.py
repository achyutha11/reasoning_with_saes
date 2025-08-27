import os
import json
import re
import math
import numpy as np
# from math_grader import strip_string
from math_verify import parse, verify, LatexExtractionConfig

DATASET_MAP = {
    "MATH-500": {
        "args": ("HuggingFaceH4/MATH-500", "test"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "AIME2024": {
        "args": ("HuggingFaceH4/aime_2024", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "gpqa": {
        "args": ("hendrydong/gpqa_diamond_mc", "test"),
        "question_key": "problem",
        "answer_key": "solution"
    },
    "gsm8k": {
        "args": ("skrishna/gsm8k_only_answer", "test"),
        "question_key": "text",
        "answer_key": "label"
    },
    "openr1-math": {
        "args": ("open-r1/OpenR1-Math-220k", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "AIME2025": {
        "args": ("yentinglin/aime_2025", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "MMLU-Pro-math": {
        "args": ("TIGER-Lab/MMLU-Pro", "test"),
        "options_key": "options",
        "question_key": "question",
        "answer_key": "answer"
    }
}

MODEL_MAP   = {
    "deepseek-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-llama3-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "qwq-32b": "Qwen/QwQ-32B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "deepseek-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "phi4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    "nemotron-7b": "nvidia/OpenMath-Nemotron-7B",
}


def verify_answer(pred: str, ref: str) -> bool:

    # ── patterns & threshold ─────────────────────────────────────────────────
    BASE_N_RE    = re.compile(r"^\(?([0-9A-Za-z]+)\)?_\{(\d+)\}$")
    EXP_RE       = re.compile(r"\^\{(\d+)\}")
    MAX_SAFE_EXP = 10_000

    # ── normalize inputs ─────────────────────────────────────────────────────
    if pred is None or ref is None:
        return False
    p = pred.strip()
    r = ref.strip()

    # ── 1) base-N literal in prediction ─────────────────────────────────────
    m = BASE_N_RE.match(p)
    if m:
        return m.group(1) == r

    # ── 2) base-N literal in reference ──────────────────────────────────────
    m = BASE_N_RE.match(r)
    if m:
        return m.group(1) == p

    # ── 3) huge-exponent guard ───────────────────────────────────────────────
    exps = [int(e) for e in EXP_RE.findall(p)]
    if exps and max(exps) > MAX_SAFE_EXP:
        return p.replace(" ", "") == r.replace(" ", "")

    # ── 4) fallback to math_verify ──────────────────────────────────────────
    wrap = lambda s: f"\\({s}\\)"
    cfg  = LatexExtractionConfig()
    try:
        g_node = parse(wrap(r), extraction_config=[cfg])
        p_node = parse(wrap(p), extraction_config=[cfg])
        return verify(g_node, p_node, float_rounding=2)
    except Exception:
        return False

def extract_answer(text):
    if text is None:
        return None
    # Step 1: Remove everything that is not a number, letter, ".", or "-"
    # text = re.sub(r'[^0-9a-zA-Z{}\\.\-]', '', text)
    # Try extracting from 'boxed' first
    boxed_matches = extract_boxed(text)
    if boxed_matches:
        extracted_answer = boxed_matches[-1][1:-1]
        return extracted_answer

    # Fallback: extract any numbers
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if not numbers:
        return None

    try:
        extracted_number = float(numbers[-1])
        # Guard against infinity
        if math.isinf(extracted_number):
            return None

        return numbers[-1]
    except (ValueError, OverflowError):
        return None

def extract_boxed(text):
    pattern = re.compile(r'boxed\{')
    matches = []
    stack = []

    i = 0
    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break

        start = match.end() - 1  # Position at the first `{`
        stack.append(start)
        i = start + 1
        count = 1  # To track `{}` pairs

        while i < len(text) and stack:
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:  # Found a matching closing `}`
                    start = stack.pop()
                    matches.append(text[start:i+1])
                    break
            i += 1

    return matches
