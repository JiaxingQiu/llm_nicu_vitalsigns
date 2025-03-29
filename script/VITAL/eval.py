from eval_utils.eval_vae import *
from eval_utils.eval_clip_ts2txt import *
from eval_utils.eval_clip_txt2ts import *
import torch
from data import get_features3d, get_features

# df = df_train.iloc[:1,].copy()
def vital_infer(df, model, config_dict,
                text_col = 'text', text_col_ls = ['demo', 'cl_event', 'ts_description'],
                K = 100, top = 8, 
                distance_ratios = [0, 1, 50, 100]):
    
    model.eval()
    ts = df.loc[:,'1':'300'].values

    if config_dict['3d']:
        ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_col_ls)
        ts_f = ts_f.to(model.device)
        tx_f_ls = [tx_f.to(model.device) for tx_f in tx_f_ls]
    else:
        ts_f, tx_f, _ = get_features(df, config_dict, text_col = text_col)
        ts_f = ts_f.to(model.device)
        tx_f = tx_f.to(model.device)
    logits_ls = []
    x_hat_ls = []
    distance_ratio_ls = []
    # sample K Zs for each distance
    for distance_ratio in distance_ratios:
        for _ in range(K):
            _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f)
            z = model.ts_encoder.reparameterization(z_mean, z_log_var*distance_ratio)
            
            if config_dict['3d']:
                logits = model.clip(z, tx_f_ls)
            else:
                logits = model.clip(z, tx_f)

            logits = torch.diag(logits).reshape(-1, 1)
            logits_ls.append(logits)
            x_hat = model.ts_decoder(z, x_mean, x_std)
            x_hat_ls.append(x_hat)
            distance_ratio_ls.append(distance_ratio)# append distance for reference

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)
    
    # find the topk softmax probabilities, and their corresponding x_hats
    top_probs = torch.topk(softmax_probs, top, dim=1)[0]
    top_indices = torch.topk(softmax_probs, top, dim=1)[1]
    top_indices = top_indices[0].cpu().detach().numpy().tolist()
    top_ts_hats = [x_hat_ls[idx] for idx in top_indices]
    top_distance_ratios = [distance_ratio_ls[idx] for idx in top_indices]

    return ts, top_probs, top_ts_hats, top_distance_ratios



def plot_vital_reconstructions(ts, top_probs, top_ts_hats, top_distance_ratios):
    if ts.shape[0] == 1:
        ts = ts[0]
    n_reconstructions = len(top_ts_hats)  # total number of subplots
    n_cols = 4
    n_rows = (n_reconstructions + n_cols - 1) // n_cols  # ceiling division for number of rows
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    fig.suptitle('Original Signal vs Reconstructions with Different Noise Levels', fontsize=14)
    axs = axs.flatten()

    # Plot original signal in first subplot
    axs[0].plot(ts, 'b-', label='Original')
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].set_ylim(50, 200)
    axs[0].grid(True)
    axs[0].legend()

    # plot the reconstructions
    for i in range(1, n_reconstructions):
        ts_hat = top_ts_hats[i]
        prob = top_probs[0][i].item()
        distance_ratio = top_distance_ratios[i]
        # Plot reconstruction
        if ts_hat.dim() == 2:
            ts_hat = ts_hat[0]
        axs[i].plot(ts_hat.cpu().detach().numpy(), 'r-', label='Reconstruction')
        axs[i].plot(ts, 'b--', alpha=0.5, label='Original')
        axs[i].set_title(f'var distance ratio={distance_ratio:.1e}\nprob={prob:.3f}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')
        axs[i].set_ylim(50, 200)
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
