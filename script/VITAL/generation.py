import matplotlib.pyplot as plt
import numpy as np
import torch
from config import *
from data import get_features3d

def interpolate_ts_tx(df, model, config_dict, text_cols, w, label = False):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    raw_ts = ts_f[0].detach().cpu().numpy()
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _, x_mean, x_std = model.ts_encoder(ts_f)
    if not model.variational:
        ts_emb = ts_emb_mean
    
    ts_hat_ls = {}
    simi = {}
    l1 = {}
    l2 = {}
    for txid in range(len(tx_f_ls)):
        # ----- text_embeddings -----
        tx_emb = model.text_encoder(tx_f_ls[txid])
        # ----- interpolation -----
        ts_emb_inter = (1-w)*ts_emb + w*tx_emb # interpolation of ts_emb and tx_emb
        # ----- decoder -----
        if model.concat_embeddings:
            emb = torch.cat([ts_emb_inter, tx_emb], dim=1)
        else:
            emb = ts_emb_inter
        ts_hat = model.ts_decoder(emb, x_mean, x_std)
        # # plot the ts_hat
        # plt.plot(ts_hat[0].detach().cpu().numpy())
        # plt.show()
        if label:
            text_condition = df[text_cols[txid]].iloc[0] # string version of text condition
        else:
            text_condition = text_cols[txid]

        ts_hat_ls[text_condition] = ts_hat[0].detach().cpu().numpy()
        logits = torch.matmul(ts_emb_inter, tx_emb.T) 
        simi[text_condition] = torch.diag(logits).detach().cpu().numpy()# torch.exp()
        l2[text_condition] = torch.norm(ts_emb_inter - tx_emb, dim=1, p=2).detach().cpu().numpy()  
        l1[text_condition] = torch.norm(ts_emb_inter - tx_emb, dim=1, p=1).detach().cpu().numpy()
        
    ts2tx_distances = {'simi': simi, 'l1': l1, 'l2': l2}
    
    return ts2tx_distances, ts_hat_ls, raw_ts



def plot_interpolate_ts_tx_ws(df, model, config_dict, text_cols, w_values=None, label = True):
    """
    Plot interpolated time series for multiple w values in a grid layout.
    Creates a separate figure for each text condition.
    
    Args:
        df: Input DataFrame
        model: VITAL model
        config_dict: Configuration dictionary
        text_cols: List of text conditions
        w_values: List of interpolation weights (default: np.arange(0, 1.1, 0.1))
    """
    if w_values is None:
        w_values = np.arange(0, 1.1, 0.1)
    
    # Get ground truth reconstruction
    _, ts_hats, raw_ts = interpolate_ts_tx(df, model, config_dict, 
                                         text_cols = ['text'], # ground truth text condition
                                         w=0, # reconstruction only on ground truth text condition
                                         label = False)
    
    # Collect all results first
    all_results = {}
    for w in w_values:
        dists, ts_hats, _ = interpolate_ts_tx(df, model, config_dict, text_cols, w=w, label=label)
        all_results[w] = {'dists': dists, 'ts_hats': ts_hats}
    
    # Create a figure for each text condition
    for text_condition in list(dists['simi'].keys()):
        # Calculate grid dimensions for this text condition
        n_cols = 5
        n_rows = int(np.ceil(len(w_values) / n_cols))
        
        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        fig.suptitle(f'Augmenting towards: {text_condition}', fontsize=18)
        
        # Flatten axes array if needed
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        
        # Plot for each w value
        for idx, w in enumerate(w_values):
            row = idx // n_cols
            col = idx % n_cols
            
            # Get results for this w value
            dists = all_results[w]['dists']
            ts_hats = all_results[w]['ts_hats']
            ts_hat = ts_hats[text_condition]
            
            # Plot reconstruction and ground truth
            axs[row, col].plot(ts_hat, 'r-', label='Augmented') #np.round(ts_hat)
            axs[row, col].plot(raw_ts, 'b--', alpha=0.5, label='Raw')
            
            # Add distance metrics to title
            simi_val = float(dists['simi'][text_condition]) if isinstance(dists['simi'][text_condition], np.ndarray) else dists['simi'][text_condition]
            l1_val = float(dists['l1'][text_condition]) if isinstance(dists['l1'][text_condition], np.ndarray) else dists['l1'][text_condition]
            l2_val = float(dists['l2'][text_condition]) if isinstance(dists['l2'][text_condition], np.ndarray) else dists['l2'][text_condition]
            
            w_title = f'w = {w:.1f}' if w != 0 else 'w = 0 (reconstruction)'
            title = f'{w_title}\nCosine: {simi_val:.3f}\nL2: {l2_val:.3f}' # \nL1: {l1_val:.3f}
            axs[row, col].set_title(title, pad=15, fontsize=10)
            axs[row, col].set_xlabel('Time Steps', fontsize=9)
            axs[row, col].set_ylabel('Value', fontsize=9)
            axs[row, col].grid(True, linestyle='--', alpha=0.3)
            axs[row, col].legend()
            
            # Remove top and right spines
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
        
        # Hide any empty subplots
        for idx in range(len(w_values), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
