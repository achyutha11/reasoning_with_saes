from prompt import format_prompt_aqua, format_prompt_trivia
from model import load_model
from sae import load_sae, get_sae_acts
from datasets import load_aqua, load_trivia
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import trim_mean
import numpy as np
from functools import partial

class IndexedPromptDataset(Dataset):
    def __init__(self, num_examples):
        self.indices = list(range(num_examples))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

def collate_tokenized(batch_indices, tokenized):
    return {k: v[batch_indices] for k, v in tokenized.items()}

def get_ds_saes(sae, layer, prompts, model, collate_fn, batch_size=8, agg='mean'):

    dataset = IndexedPromptDataset(len(prompts))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    num_feats = sae.cfg.d_sae
    sae_mat = torch.zeros(len(prompts), num_feats)

    with torch.no_grad():
        for i, batch_inputs in enumerate(tqdm(dataloader)):
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            batch_feats = get_sae_acts(batch_inputs, layer=layer, agg=agg)
            start = i * batch_size
            end = start + batch_feats.shape[0]
            sae_mat[start:end] = batch_feats.cpu()

    # torch.cuda.empty_cache()
    # gc.collect()

    return sae_mat

def get_reasoning_features(trivia_examples=250, k=10, model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', sae_release="llama_scope_r1_distill", sae_id= "l25r_400m_slimpajama_400m_openr1_math", sae_layer=25):

    model, tokenizer = load_model(model_name)
    sae = load_sae(sae_release, sae_id)

    aqua_ds = load_aqua()
    trivia_ds = load_trivia()

    aq_prompts = [format_prompt_aqua(q, reasoning=False, include_options=False) for q in aqua_ds['question']]
    aq_tokenized = tokenizer(aq_prompts, return_tensors='pt', padding=True, truncation=True)
    aq_collate_fn = partial(collate_tokenized, tokenized=aq_tokenized)

    aqua_means = get_ds_saes(sae, sae_layer, aq_prompts, model, collate_fn=aq_collate_fn, agg='mean')

    tr_prompts = [format_prompt_trivia(q, reasoning=False) for q in trivia_ds[:trivia_examples]['question']]
    tr_tokenized = tokenizer(tr_prompts, return_tensors='pt', padding=True, truncation=True)
    tr_collate_fn = partial(collate_tokenized, tokenized=tr_tokenized)

    trivia_means = get_ds_saes(sae, sae_layer, tr_prompts, model, collate_fn=tr_collate_fn, agg='mean')

    aqua_means = trim_mean(aqua_means.detach(), proportiontocut=0.05, axis=0)
    trivia_means = trim_mean(trivia_means.detach(), proportiontocut=0.05, axis=0)

    epsilon = 1e-6
    percentage_increase = 100 * (aqua_means - trivia_means) / (trivia_means + epsilon)

    valid = (trivia_means > 0.01) & (aqua_means > 0.1)
    filtered_percentage_increase = percentage_increase[valid]
    valid_indices = np.where(valid)[0]

    ranked_order = np.argsort(-filtered_percentage_increase)
    ranked_feature_indices = valid_indices[ranked_order]
    reasoning_feats = []

    for i in range(k):
        idx = ranked_feature_indices[i]
        reasoning_feats.append(idx)

    return reasoning_feats

if __name__ == "__main__":
    result = get_reasoning_features()
    print(result)
