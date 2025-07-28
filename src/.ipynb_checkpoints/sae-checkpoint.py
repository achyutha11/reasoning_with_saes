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
        result= raw_feats.mean(dim=1)
    elif agg == 'last':
        result = raw_feats[:, -1]

    del hidden_states, raw_feats

    return result
