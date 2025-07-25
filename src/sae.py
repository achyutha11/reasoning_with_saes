from sae_lens import SAE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_sae(release, sae_id):
    release = "llama_scope_r1_distill"
    sae_id = "l25r_400m_slimpajama_400m_openr1_math"
    sae, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(release, sae_id)
    sae = sae.to(device)
    return sae, cfg_dict

def get_sae_acts(model, sae, text, layer, agg='mean'):
    model.eval()
    with torch.no_grad():
        outputs = model(**text, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    raw_feats = sae.encode(hidden_states[layer])
    if agg == 'mean':
        return raw_feats[0].mean(axis=0)
    elif agg == 'last':
        return raw_feats[0, -1]
