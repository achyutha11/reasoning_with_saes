from prompt import format_prompt_aqua, format_prompt_trivia
from model import load_model
from sae import load_sae, get_sae_acts
from datasets import load_aqua, load_trivia
import torch
from tqdm import tqdm
from scipy.stats import trim_mean
import numpy as np


def get_reasoning_features(trivia_examples=250, k=10, model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', sae_release="llama_scope_r1_distill", sae_id= "l25r_400m_slimpajama_400m_openr1_math", sae_layer=25):
    # Todo: make these customizable
    model, tokenizer = load_model(model_name)
    sae = load_sae(sae_release, sae_id)

    aqua_ds = load_aqua()
    trivia_ds = load_trivia()

    aqua_means = torch.zeros((aqua_ds.num_rows, sae.cfg.d_sae))

    for index, question in enumerate(tqdm(aqua_ds)):
        prompt = format_prompt_aqua(question, reasoning=False, include_options=False)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        aqua_means[index, :] = get_sae_acts(inputs, sae_layer)

    trivia_means = torch.zeros((trivia_examples, sae.cfg.d_sae))

    for index, question in enumerate(tqdm(trivia_ds[:trivia_examples]['question'])):
        prompt = format_prompt_trivia(question, reasoning=False)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        trivia_means[index, :] = get_sae_acts(inputs, sae_layer)

    mean_aqua = trim_mean(aqua_means.detach(), proportiontocut=0.05, axis=0)
    mean_trivia = trim_mean(trivia_means.detach(), proportiontocut=0.05, axis=0)

    epsilon = 1e-6
    percentage_increase = 100 * (mean_aqua - mean_trivia) / (mean_trivia + epsilon)

    valid = (mean_trivia > 0.01) & (mean_aqua > 0.1)
    filtered_percentage_increase = percentage_increase[valid]
    valid_indices = np.where(valid)[0]

    ranked_order = np.argsort(-filtered_percentage_increase)
    ranked_feature_indices = valid_indices[ranked_order]
    reasoning_feats = []

    for i in range(k):
        idx = ranked_feature_indices[i]
        reasoning_feats.append(idx)

    return reasoning_feats
