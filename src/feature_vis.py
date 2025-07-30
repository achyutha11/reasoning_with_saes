from prompt import format_prompt_aqua, format_prompt_trivia, MCQ_ANSWER_PROMPT
from model import load_model
from sae import load_sae, get_sae_acts
from get_datasets import load_aqua, load_trivia
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import trim_mean
import numpy as np
from functools import partial
import string
import pickle
import gc

def get_binned_acts_batch(dataloader, model, sae, sae_layer, feature):

    tensor_list = []

    with torch.no_grad():
        for batch_inputs in tqdm(dataloader):
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            attention_mask = batch_inputs['attention_mask'].cpu()

            result = get_sae_acts(model, sae, batch_inputs, sae_layer, agg='none')
            feature_acts = result[:, :, feature].cpu()

            binned_batch = bin_token_activations_batch(feature_acts, attention_mask)
            tensor_list.append(binned_batch)

    # Clear memory
    del batch_inputs, result, feature_acts, attention_mask
    torch.cuda.empty_cache()
    gc.collect()

    return torch.cat(tensor_list, dim=0)

def bin_token_activations_batch(activation_tensor, attention_mask, num_bins=20):

    B, T = activation_tensor.shape
    binned = torch.zeros(B, num_bins)
    counts = torch.zeros(B, num_bins)

    for b in range(B):

        valid_indices = attention_mask[b].nonzero(as_tuple=True)[0]
        n_valid = len(valid_indices)

        bin_idx = torch.linspace(0, num_bins, steps=n_valid, dtype=torch.long)
        bin_idx = torch.clamp(bin_idx, max=num_bins - 1)

        for i, token_pos in enumerate(valid_indices):
            idx = bin_idx[i].item()
            binned[b, idx] += activation_tensor[b, token_pos]
            counts[b, idx] += 1

    counts = torch.clamp(counts, min=1)
    return binned / counts
