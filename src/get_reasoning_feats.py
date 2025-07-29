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
            batch_feats = get_sae_acts(model, sae, batch_inputs, layer=layer, agg=agg)
            start = i * batch_size
            end = start + batch_feats.shape[0]
            sae_mat[start:end] = batch_feats.cpu()

    return sae_mat

class TokenizedPromptDataset(Dataset):
    def __init__(self, tokenized_inputs, queries):
        self.input_ids = tokenized_inputs['input_ids']
        self.attn_masks = tokenized_inputs['attention_mask']
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'query': self.queries[idx]
        }

def collate_tokenized_for_cot(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    queries = [item['query'] for item in batch]
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, queries

def get_cot_batch(ds, batch_size, tokenizer, model, collate_fn):
    tokenized = tokenizer(
        [format_prompt_aqua(q, reasoning=False, include_options=True) for q in ds],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=8192
    )

    dataset = TokenizedPromptDataset(tokenized, ds)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_preds = []
    all_generations = []

    for batch_inputs, queries in tqdm(dataloader):

        input_ids = batch_inputs['input_ids'].to(model.device)
        attention_mask = batch_inputs['attention_mask'].to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        decoded = ["".join(out.split('[/INST]')[1:]) for out in decoded]
        all_generations.extend(decoded)

        answer_prompts = [text + MCQ_ANSWER_PROMPT for text in decoded]
        answer_inputs = tokenizer(answer_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)

        with torch.inference_mode():
            out = model(**answer_inputs)

        for i, query in enumerate(queries):
            options = query['options']
            letters = list(string.ascii_uppercase)[:len(options)]
            valid_ids = tokenizer.convert_tokens_to_ids(letters)
            logits = out.logits[i, -1, valid_ids]
            pred_idx = torch.argmax(logits).item()
            all_preds.append(letters[pred_idx])

    return all_preds, all_generations

def feature_extraction(baseline_mean, exp_mean, k=10, epsilon=1e-6):

    percentage_increase = 100 * (exp_mean - baseline_mean) / (baseline_mean + epsilon)

    valid = (baseline_mean > 0.01) & (exp_mean > 0.1)
    filtered_percentage_increase = percentage_increase[valid]
    valid_indices = np.where(valid)[0]

    ranked_order = np.argsort(-filtered_percentage_increase)
    ranked_feature_indices = valid_indices[ranked_order]
    feats = []

    for i in range(k):
        idx = ranked_feature_indices[i]
        feats.append(idx)

    return feats


def get_reasoning_features(k=10, model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', sae_release="llama_scope_r1_distill", sae_id= "l25r_400m_slimpajama_400m_openr1_math", sae_layer=25):

    model, tokenizer = load_model(model_name)
    sae = load_sae(sae_release, sae_id)

    aqua_ds = load_aqua()

    preds, gens = get_cot_batch(aqua_ds, 16, tokenizer, model, collate_tokenized_for_cot)

    aq_prompts = [format_prompt_aqua(query, reasoning=False, include_options=False) for query in aqua_ds]
    aq_tokenized = tokenizer(aq_prompts, return_tensors='pt', padding=True, truncation=True)
    aq_collate_fn = partial(collate_tokenized, tokenized=aq_tokenized)
    query_means = get_ds_saes(sae, sae_layer, aq_prompts, model, collate_fn=aq_collate_fn, agg='mean')

    cot_tokenized = tokenizer(gens, return_tensors='pt', padding=True, truncation=True)
    cot_collate_fn = partial(collate_tokenized, tokenized=cot_tokenized)
    cot_means = get_ds_saes(sae, sae_layer, gens, model, collate_fn=cot_collate_fn, agg='mean')

    answers = ["The correct answer is (" + i + ")" for i in preds]
    ans_tokenized = tokenizer(answers, return_tensors='pt', padding=True, truncation=True)
    ans_collate_fn = partial(collate_tokenized, tokenized=ans_tokenized)
    ans_means = get_ds_saes(sae, sae_layer, answers, model, collate_fn=ans_collate_fn, agg='mean')

    query_means = trim_mean(query_means.detach(), proportiontocut=0.05, axis=0)
    cot_means = trim_mean(cot_means.detach(), proportiontocut=0.05, axis=0)
    ans_means = trim_mean(ans_means.detach(), proportiontocut=0.05, axis=0)

    cot_features = feature_extraction(query_means, cot_means)
    ans_features = feature_extraction(query_means, ans_means)

    return cot_features, ans_features

if __name__ == "__main__":
    cot_features, ans_features = get_reasoning_features()
    print(cot_features)
    print(ans_features)
