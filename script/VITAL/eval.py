from eval_utils.eval_vae import *
from eval_utils.eval_clip_ts2txt import *
from eval_utils.eval_clip_txt2ts import *
import torch
from data import get_features3d, get_features

# df = df_train.iloc[:1,].copy()
def vital_infer(df, model, config_dict,
                K = 100, top = 10, distances = [0, 5e-4, 7.5e-4, 1e-3, 2e-3]):
    
    model.eval()

    if config_dict['3d']:
        ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = config_dict['text_col_ls'])
    else:
        ts_f, tx_f, _ = get_features(df, config_dict)
    logits_ls = []
    x_hat_ls = []
    distance_ls = []
    # sample K Zs for each distance
    for distance in distances:
        for _ in range(K):
            _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f)
            z = model.ts_encoder.reparameterization(z_mean, z_log_var + distance)
            
            if config_dict['3d']:
                logits = model.clip(z, tx_f_ls)
            else:
                logits = model.clip(z, tx_f)

            logits = torch.diag(logits).reshape(-1, 1)
            logits_ls.append(logits)
            x_hat = model.ts_decoder(z, x_mean, x_std)
            x_hat_ls.append(x_hat)
            distance_ls.append(distance)# append distance for reference

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)
    
    # find the topk softmax probabilities, and their corresponding x_hats
    top_probs = torch.topk(softmax_probs, top, dim=1)[0]
    top_indices = torch.topk(softmax_probs, top, dim=1)[1]
    top_indices = top_indices[0].cpu().detach().numpy().tolist()
    top_ts_hats = [x_hat_ls[idx] for idx in top_indices]
    top_distances = [distance_ls[idx] for idx in top_indices]

    return top_probs, top_ts_hats, top_distances
