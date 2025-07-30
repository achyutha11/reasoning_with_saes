from sae_lens import SAE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_sae(release, sae_id):
    sae = SAE.from_pretrained(release, sae_id)
    sae = sae.to(device)
    return sae

def get_sae_acts(model, sae, input_batch, layer, agg='mean'):

    activation_dict = {}

    def hook_fn(module, input, output):
        activation_dict["hidden"] = output

    hook = model.model.layers[layer].register_forward_hook(hook_fn)

    model.eval()

    with torch.no_grad():
        _ = model(**input_batch)

    hook.remove()

    hidden_states = activation_dict['hidden']
    raw_feats = sae.encode(hidden_states)

    if agg == 'mean':
        mask = input_batch['attention_mask'].unsqueeze(-1)
        masked_feats = raw_feats * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        result = masked_feats.sum(dim=1) / lengths
    elif agg == 'last':
        last_token_idxs = input_batch['attention_mask'].sum(dim=1) - 1
        batch_indices = torch.arange(raw_feats.size(0), device=raw_feats.device)
        result = raw_feats[batch_indices, last_token_idxs]
    elif agg == 'none':
        mask = input_batch['attention_mask'].unsqueeze(-1)
        result = raw_feats * mask

    del hidden_states, raw_feats

    return result
