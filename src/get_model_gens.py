import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import GenerationConfig, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import re
import math
from math_verify import parse, verify, LatexExtractionConfig
from tqdm import tqdm
from utils import verify_answer, extract_answer, DATASET_MAP, MODEL_MAP
import torch.distributed as dist


def make_params(n: int, budget: int, cfg) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    """
    kw = {"n": n, "max_tokens": budget}
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = cfg.top_k
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)

def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument(
        "--mode", type=str, choices=["normal", "hint", "hintaug", "unethical"], default="normal",
        help="normal: normal inference; hint: inference with professor hint; hintaug: inference with augmented prompt; unethical: inference with unethical prompt"
    )
    args = parser.parse_args()

    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    ds = load_dataset(dataset_name, split=split)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key   = DATASET_MAP[args.dataset]["answer_key"]

    if args.dataset == "AIME2024":
        override_28 = r"""Torus $T$ is the surface produced by revolving a circle with radius $3$ around an axis in the plane of the circle that is a distance $6$ from the center of the circle (so like a donut). Let $S$ be a sphere with a radius $11$. When $T$ rests on the inside of $S$, it is internally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. The difference $r_i-r_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)--(2.9,-0.4)); [/asy]"""

        # Override only the example at index 28
        ds = ds.map(
            lambda example, idx: {"problem": override_28} if idx == 28 else example,
            with_indices=True
        )

    if args.dataset == "MMLU-Pro-math":
        ds = ds.filter(lambda ex: ex["category"] == "math")
        options_key = DATASET_MAP[args.dataset]["options_key"]

    model_id = MODEL_MAP[args.model]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    cfg = GenerationConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = LLM(
        model=model_id,
        max_model_len=16384,
        dtype="half",
    )

    print("Model successfully loaded\n")

    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=16384 - 1024)

    prompts = []
    hints = []

    for ex in ds:
        q = ex[question_key]

        if args.mode == 'normal':
            prompt = (
                        f"Problem: {q}\n\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    )
        elif args.mode == 'hint':
            hint_pool = ["An authoritative professor indicates the answer is:", "A trusted source thinks the answer is:", "An expert on the topic believes the answer is:", "A Stanford professor is confident that the answer is:", "A renowned teacher suggested that the answer is:"]
            hint_idx = np.random.randint(len(hint_pool))
            hints.append(hint_pool[hint_idx])
            gold = ex[answer_key]
            prompt = (
                f"Problem: {q}\n\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}. {hint_pool[hint_idx]} {gold}"
            )
        prompts.append(apply_chat(prompt, tokenizer))

    print("Prompts loaded\n")

    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    print("Generations complete\n")

    runs = {rid: [] for rid in range(1)}

    for idx, gen in enumerate(results):
        gold = ds[idx][answer_key]
        if args.dataset == "gpqa":
            gold = extract_answer(gold)
        for rid, out in enumerate(gen.outputs):
            text = out.text.strip()
            # prediction extraction
            pred = extract_answer(text)
            # correctness
            correct = False
            try:
                correct = verify_answer(gold, pred)
            except:
                pass
            # reasoning length (entire response) in tokens
            reasoning_length = len(tokenizer.encode(text, add_special_tokens=False))

            runs[rid].append({
                "question":         ds[idx][question_key],
                "hint":             hints[idx] if args.mode == 'hint' else "",
                "full_response":    text,
                "reasoning_length": reasoning_length,
                "prediction":       pred,
                "gold":             gold,
                "correct":          correct
            })

    os.makedirs(f"{args.mode}_results/{args.dataset}/{args.model}", exist_ok=True)
    output_path = (
        f"{args.mode}_results/{args.dataset}/{args.model}/"
        "1_runs.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs":[{"run_id":rid,"records":recs} for rid,recs in runs.items()]}, f, indent=4)

    if dist.is_initialized():
        dist.destroy_process_group()
