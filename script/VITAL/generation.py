import matplotlib.pyplot as plt
import numpy as np
import torch
from config import *
from data import get_features3d


def interpolate_ts_tx(df, model, config_dict, text_cols, w):

    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _, x_mean, x_std = model.ts_encoder(ts_f)
    if not model.variational:
        ts_emb = ts_emb_mean

    ts_hat_ls = {}
    for txid in range(len(tx_f_ls)):
        # ----- text_embeddings -----
        tx_emb = model.text_encoder(tx_f_ls[txid])
        # ----- interpolation -----
        ts_emb_inter = (1-w)*ts_emb + w*tx_emb # interpolation of ts_emb and tx_emb
        # ----- decoder -----
        if model.concat_embeddings:
            emb = torch.cat([ts_emb_inter, tx_emb], dim=1)
        else:
            emb = ts_emb_inter # torch.cat([ts_emb_inter, x_mean, x_std], dim=1)
        ts_hat = model.ts_decoder(emb, x_mean, x_std)
        text_condition = df[text_cols[txid]]
        ts_hat_ls[text_cols[txid]] = list(zip(text_condition, ts_hat))
    
    return ts_hat_ls


def eval_augmented_properties(df, model, config_dict, type, w, y_cols = None):
    model.eval()
    df_augmented_all = pd.DataFrame()
    if y_cols is None:
        y_cols = config_dict['txt2ts_y_cols']
    for y_col in y_cols:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=False).copy()
            
            # add new text conditions
            for j in range(len(y_levels)):
                if type == "conditional":
                    modified_text = df_level['text'].values[0]
                    for level in y_levels:
                        if level in modified_text:
                            modified_text = modified_text.replace(level, y_levels[j])
                    df_level['text' + str(j)] = modified_text
                elif type == "marginal":
                    df_level['text' + str(j)] = y_levels[j]
                else:
                    raise ValueError("type must be either 'marginal' or 'conditional'")
                
            # Augment the time series with the given text conditions
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            ts_hat_ls = interpolate_ts_tx(df_level, model, config_dict, text_cols, w)

            # Calculate the math properties of the generated time series
            df_prop_all = pd.DataFrame()
            for text_col, pairs in ts_hat_ls.items():
                df_prop = pd.DataFrame(pairs, columns=['aug_text', 'ts_hat'])
                properties = df_prop['ts_hat'].apply(lambda x: get_all_properties(x.detach().cpu().numpy()))
                df_prop['properties'] = properties
                df_prop['text_col'] = text_col
                df_prop['index'] = df_level.index # saved original index
                df_prop_all = pd.concat([df_prop_all, df_prop])
            
            df_prop_all['org_y_col'] = y_col
            df_prop_all['org_y_level'] = y_levels[i]
            df_augmented_all = pd.concat([df_augmented_all, df_prop_all])

    return df_augmented_all


def viz_augmented_properties(df_augmented, org_y_level, y_col):
    """Visualize augmented properties and return the figure.
    
    Args:
        df_augmented: DataFrame with augmented data
        org_y_level: Original y-level to filter
        y_col: Column name to analyze
        
    Returns:
        fig, ax: matplotlib figure and axis objects
        metric: string indicating which metric was analyzed
    """
    y_levels = list(df_augmented[df_augmented['org_y_col'] == y_col]['org_y_level'].unique())
    if 'trend' in y_levels[0].lower():
        metric = 'trend'
    elif 'seasonal' in y_levels[0].lower():
        metric = 'seasonality'
    elif 'shift' in y_levels[0].lower():
        metric = 'shift'
    elif 'variability' in y_levels[0].lower():
        metric = 'variability'
    else:
        raise ValueError(f"Invalid org_y_level: {y_levels[0]}")
    
    stats_ls = {}
    for y_level in y_levels:
        filtered_df = df_augmented[
            (df_augmented['org_y_level'].str.contains(org_y_level, case=False)) & 
            (df_augmented['aug_text'].str.contains(y_level, case=False))
        ]
        if metric == 'seasonality':
            stats_ls[y_level] = filtered_df['properties'].apply(lambda x: x[metric]['entropy'])
        else:
            stats_ls[y_level] = filtered_df['properties'].apply(lambda x: x[metric])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # plot the histogram of the statistics
    for i in range(len(y_levels)):
        ax.hist(stats_ls[y_levels[i]], bins=20, alpha=0.5, label=y_levels[i])
    ax.set_xlabel(metric + ' statistic')
    ax.legend()
    
    return fig, ax, metric


# augment a single time series with given text conditions
def interpolate_ts_tx_single(df, model, config_dict, text_cols, w, label = False):
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
            emb = ts_emb_inter # torch.cat([ts_emb_inter, x_mean, x_std], dim=1)
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
    _, ts_hats, raw_ts = interpolate_ts_tx_single(df, model, config_dict, 
                                         text_cols = ['text'], # ground truth text condition
                                         w=0, # reconstruction only on ground truth text condition
                                         label = False)
    
    # Collect all results first
    all_results = {}
    for w in w_values:
        dists, ts_hats, _ = interpolate_ts_tx_single(df, model, config_dict, text_cols, w=w, label=label)
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


# visualize the transition of timeseries from raw to instructed augmentation
def viz_generation_marginal(df, model, config_dict, tid=0, w_values = np.arange(0.4, 1.2, 0.2)):
    model.eval()
    w_values = np.concatenate([[0], w_values])
    for y_col in config_dict['txt2ts_y_cols']:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=True).iloc[[tid]].copy()
            print(df_level[y_col])
            for j in range(len(y_levels)):
                df_level['text' + str(j)] = y_levels[j]
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            plot_interpolate_ts_tx_ws(df_level, model, config_dict, text_cols, w_values = w_values, label = True)

def viz_generation_conditional(df, model, config_dict, tid=0, w_values = np.arange(0.4, 1.2, 0.2)):
    model.eval()
    w_values = np.concatenate([[0], w_values])
    for y_col in config_dict['txt2ts_y_cols']:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=True).iloc[[tid]].copy()
            print(df_level[y_col])
            for j in range(len(y_levels)):
                # find the substring in raw_text(df_level['text'].values[0]) that is the same as any of the elemnet in y_levels
                # replace the substring y_levels[j] 
                modified_text = df_level['text'].values[0]
                for level in y_levels:
                    if level in modified_text:
                        modified_text = modified_text.replace(level, y_levels[j])
                df_level['text' + str(j)] = modified_text
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            plot_interpolate_ts_tx_ws(df_level, model, config_dict, text_cols, w_values = w_values, label = True)
