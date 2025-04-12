import torch
from data import get_features3d, get_features
from config import *
from eval_utils.eval_vae import *
from eval_utils.eval_clip_ts2txt import *
from eval_utils.eval_clip_txt2ts import *
from eval_utils.eval_conditions import *



def vital_contrast_infer(df, model, config_dict,
                         text_cols = ['text', 'text1', 'text2'], # i.e text1 text2 columns as 'low consecutive increases' and 'moderate consecutive increases'
                         var_ratio = 2,
                         K=1000):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    ts_f_repeated = ts_f.repeat(K, 1)  # Adjust dimensions as needed
    tx_f_ls_repeated = [tx_f.repeat(K, 1) for tx_f in tx_f_ls]

    # Forward pass for the batch
    _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f_repeated)
    z = model.ts_encoder.reparameterization(z_mean, z_log_var * var_ratio)
    ts_hat = model.ts_decoder(z, x_mean, x_std)
        
    logits_ls = []
    # get the logits for each text (condition)
    for tx_f in tx_f_ls_repeated:
        tx_embedded = model.text_encoder(tx_f)
        logits = model.clip(z, tx_embedded)
        logits = torch.diag(logits).reshape(-1, 1)
        logits_ls.append(logits)

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # each row corresponds to a z, each column corresponds to the probability of a text condition given z

    text_conditioned_ts_hat = {}
    text_conditioned_ts_hat_probs = {}
    # For each text condition, get the time series where it has the highest probability
    for i, text_col in enumerate(text_cols):
        # Get the probabilities for this text condition
        probs = softmax_probs[:, i]
        # Get the indices where this text condition has the highest probability
        mask = torch.argmax(softmax_probs, dim=1) == i
        # Store the time series and their corresponding probabilities
        text_col_string = df[text_col].iloc[0]
        text_conditioned_ts_hat[text_col_string] = ts_hat[mask]
        text_conditioned_ts_hat_probs[text_col_string] = probs[mask]

    return text_conditioned_ts_hat, text_conditioned_ts_hat_probs

def vital1_contrast_infer(df, model, config_dict,
                         text_cols = ['text', 'text1', 'text2'], # i.e text1 text2 columns as 'low consecutive increases' and 'moderate consecutive increases'
                         var_ratio = 2,
                         K=100):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    ts_f_repeated = ts_f.repeat(K, 1)  # Adjust dimensions as needed
    tx_f_ls_repeated = [tx_f.repeat(K, 1) for tx_f in tx_f_ls]

    # Forward pass for the batch
    _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f_repeated)
    z = model.ts_encoder.reparameterization(z_mean, z_log_var * var_ratio)
    
    ts_hat_ls = []
    logits_ls = []
    # get the logits for each text (condition)
    for tx_f in tx_f_ls_repeated:
        tx_embedded = model.text_encoder(tx_f)
        
        z_tx_embedded = torch.cat([z, tx_embedded], dim=1)
        ts_hat = model.ts_decoder(z_tx_embedded, x_mean, x_std)
        
        logits = model.clip(z, tx_embedded)
        logits = torch.diag(logits).reshape(-1, 1)
        logits_ls.append(logits)
        ts_hat_ls.append(ts_hat)

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # each row corresponds to a z, each column corresponds to the probability of a text condition given z

    text_conditioned_ts_hat = {}
    text_conditioned_ts_hat_probs = {}
    # For each text condition, get the time series where it has the highest probability
    for i, text_col in enumerate(text_cols):
        ts_hat = ts_hat_ls[i]
        # Get the probabilities for this text condition
        probs = softmax_probs[:, i]
        # Get the indices where this text condition has the highest probability
        mask = torch.argmax(softmax_probs, dim=1) == i
        # Store the time series and their corresponding probabilities
        text_col_string = df[text_col].iloc[0]
        text_conditioned_ts_hat[text_col_string] = ts_hat[mask]
        text_conditioned_ts_hat_probs[text_col_string] = probs[mask]

    return text_conditioned_ts_hat, text_conditioned_ts_hat_probs

def vital3_contrast_infer(df, model, config_dict,
                         text_cols = ['text', 'text1', 'text2'], # i.e text1 text2 columns as 'low consecutive increases' and 'moderate consecutive increases'
                         var_ratio = 2,
                         K=100):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    ts_f_repeated = ts_f.repeat(K, 1)  # Adjust dimensions as needed
    tx_f_ls_repeated = [tx_f.repeat(K, 1) for tx_f in tx_f_ls]

    # Forward pass for the batch
    _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f_repeated)
    z = model.ts_encoder.reparameterization(z_mean, z_log_var, var_ratio)
    
    ts_hat_ls = []
    logits_ls = []
    # get the logits for each text (condition)
    for tx_f in tx_f_ls_repeated:
        tx_embedded = model.text_encoder(tx_f)
        
        z_tx_embedded = torch.cat([z, tx_embedded], dim=1)
        ts_hat = model.ts_decoder(z_tx_embedded, x_mean, x_std)
        
        logits = model.clip(z, tx_embedded)
        logits = torch.diag(logits).reshape(-1, 1)
        logits_ls.append(logits)
        ts_hat_ls.append(ts_hat)

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # each row corresponds to a z, each column corresponds to the probability of a text condition given z

    text_conditioned_ts_hat = {}
    text_conditioned_ts_hat_probs = {}
    # For each text condition, get the time series where it has the highest probability
    for i, text_col in enumerate(text_cols):
        ts_hat = ts_hat_ls[i]
        # Get the probabilities for this text condition
        probs = softmax_probs[:, i]
        # Get the indices where this text condition has the highest probability
        mask = torch.argmax(softmax_probs, dim=1) == i
        # Store the time series and their corresponding probabilities
        text_col_string = df[text_col].iloc[0]
        text_conditioned_ts_hat[text_col_string] = ts_hat[mask]
        text_conditioned_ts_hat_probs[text_col_string] = probs[mask]

    return text_conditioned_ts_hat, text_conditioned_ts_hat_probs

def plot_vital_contrast_reconstructions(raw_ts, text_conditioned_ts_hat, text_conditioned_ts_hat_probs, n=3, title=''):
    """
    Plot the top n time series reconstructions for each text condition.
    
    Args:
        raw_ts (torch.Tensor): Original time series
        text_conditioned_ts_hat (dict): Dictionary mapping text conditions to their corresponding time series
        text_conditioned_ts_hat_probs (dict): Dictionary mapping text conditions to their probabilities
        n (int): Number of top reconstructions to plot for each condition
        title (str): Title for the plot
    """
    n_conditions = len(text_conditioned_ts_hat)
    n_cols = n
    n_rows = n_conditions
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    fig.suptitle(title, fontsize=18)
    
    # If only one condition, make axs 2D
    if n_conditions == 1:
        axs = axs.reshape(1, -1)
    
    # Plot reconstructions for each text condition
    for i, (text_condition, ts_hats) in enumerate(text_conditioned_ts_hat.items()):
        probs = text_conditioned_ts_hat_probs[text_condition]
        
        # Sort by probability and get top n
        sorted_indices = torch.argsort(probs, descending=True)
        top_n_indices = sorted_indices[:n]
        top_n_ts_hats = ts_hats[top_n_indices]
        top_n_probs = probs[top_n_indices]
        
        # Plot each of the top n reconstructions
        for j in range(n):
            if j < len(top_n_ts_hats):  # Check if we have enough reconstructions
                ts_hat = top_n_ts_hats[j]
                prob = top_n_probs[j].item()
                
                # Plot reconstruction
                if ts_hat.dim() == 2:
                    ts_hat = ts_hat[0]
                axs[i, j].plot(ts_hat.cpu().detach().numpy(), 'r-', label='Reconstruction')
                axs[i, j].plot(raw_ts, 'b--', alpha=0.5, label='Original')
                axs[i, j].set_title(f'{text_condition}\nprob={prob:.3f}')
                axs[i, j].set_xlabel('Time')
                axs[i, j].set_ylabel('Value')
                axs[i, j].set_ylim(50, 200)  # Adjust y-limits as needed
                axs[i, j].grid(True)
                axs[i, j].legend()
            else:
                # Hide empty subplots
                axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def cal_embeddings_distances(df, text_cols, model, config_dict):
    
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _, _, _ = model.ts_encoder(ts_f)
        

    # ----- ts_embeddings distances to tx_embeddings -----
    simi = {}
    l1 = {}
    l2 = {}
    for txid in range(len(tx_f_ls)):
        tx_emb = model.text_encoder(tx_f_ls[txid])
        
        # calculate dot product similarity between all ts and this txt_emb
        logits = torch.matmul(ts_emb_mean, tx_emb.T) 
        simi[df[text_cols[txid]].iloc[0]] = torch.exp(torch.diag(logits))
        
        # calculate L1/L2 norm distance between all ts and this txt_emb
        l2[df[text_cols[txid]].iloc[0]] = torch.norm(ts_emb_mean - tx_emb, dim=1, p=2)
        l1[df[text_cols[txid]].iloc[0]] = torch.norm(ts_emb_mean - tx_emb, dim=1, p=1)
    # # nicer print dict
    # for key, value in simi.items():
    #     print(f'{key}: {value}')
    # for key, value in l1.items():
    #     print(f'{key}: {value}')
    # for key, value in l2.items():
    #     print(f'{key}: {value}')
    ts2tx_distances = {'simi': simi, 'l1': l1, 'l2': l2}

        
    # ----- pairwise similary / l2/ l1 distances between all embeddings (concate ts and tx_unqiue) -----
    tx_emb = None
    for txid in range(len(tx_f_ls)):
        tx_emb_1 = model.text_encoder(tx_f_ls[txid])[0].reshape(1,-1) # [1, tx_emb_dim]
        if tx_emb is None:
            tx_emb = tx_emb_1
        else:
            tx_emb = torch.cat([tx_emb, tx_emb_1], dim=0)

    # concate txt embeddings (one for each text) and all ts embeddings
    all_emb = torch.cat([tx_emb, ts_emb_mean], dim=0)
    # calculate pairwise similary / l2/ l1 distance between all embeddings
    simi_mat = torch.exp(torch.matmul(all_emb, all_emb.T))
    l1_dist_mat = torch.cdist(all_emb, all_emb, p=1) # L1 norm
    l2_dist_mat = torch.cdist(all_emb, all_emb, p=2) # L2 norm

    pairwise_distances = {'simi': simi_mat, 'l1': l1_dist_mat, 'l2': l2_dist_mat}

    return pairwise_distances, ts2tx_distances

def plot_embeddings_graph(adj_mat):
    # Create a network graph
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # zero out lower than 50 percentile
    adj_mat[adj_mat < np.percentile(adj_mat, 50)] = 0
    np.fill_diagonal(adj_mat, 0)  # Remove self-loops

    # Create and draw network
    G = nx.from_numpy_array(adj_mat)
    pos = nx.spring_layout(G, k=0.5, iterations=5000)

    plt.figure(figsize=(6, 4))

    # Create color list with three groups
    n_nodes = len(G.nodes())
    n_remaining = n_nodes - 2
    half_remaining = n_remaining // 2

    # Draw edges first
    nx.draw_networkx_edges(G, pos,
                          edge_color='grey',
                          width=0.1)

    # Draw first two nodes as triangles
    first_two_nodes = list(G.nodes())[:2]
    first_two_colors = ['darkgreen', 'blue']
    nx.draw_networkx_nodes(G, pos,
                          nodelist=first_two_nodes,
                          node_color=first_two_colors,
                          node_shape='^',
                          node_size=100)

    # Draw remaining nodes as circles, split into two color groups
    remaining_nodes_first_half = list(G.nodes())[2:2+half_remaining]
    remaining_nodes_second_half = list(G.nodes())[2+half_remaining:]
    
    nx.draw_networkx_nodes(G, pos,
                          nodelist=remaining_nodes_first_half,
                          node_color='darkgreen',
                          node_size=10)
    
    nx.draw_networkx_nodes(G, pos,
                          nodelist=remaining_nodes_second_half,
                          node_color='blue',
                          node_size=10)
    
    # Add node indices as labels
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=3)
    
    plt.show()

















# df = df_train.iloc[:1,].copy()
def vital_infer(df, model, config_dict,
                text_col = 'text', text_col_ls = ['demo', 'cl_event', 'ts_description'],
                K = 100, # at each given distance ratio, sample K Zs
                top = 10, # get the top number of reconstructions
                distance_ratios = [0, 1, 50, 100],
                threshold = None,
                poptop=False):
    
    if threshold is None:
        threshold = 1/top
    topp_above_threshold = False
    while not topp_above_threshold:

        model.eval()
        ts = df.loc[:,'1':'300'].values

        if config_dict['3d']:
            ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_col_ls)
            ts_f = ts_f.to(device)
            tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
        else:
            ts_f, tx_f, _ = get_features(df, config_dict, text_col = text_col)
            ts_f = ts_f.to(device)
            tx_f = tx_f.to(device)

        # create batch
        ts_f_repeated = ts_f.repeat(K, 1)  # Adjust dimensions as needed
        if config_dict['3d']:
            tx_f_ls_repeated = [tx_f.repeat(K, 1) for tx_f in tx_f_ls]
        else:
            tx_f_repeated = tx_f.repeat(K, 1)


        logits_ls = []
        x_hat_ls = []
        distance_ratio_ls = []

        # sample K Zs for each distance
        for distance_ratio in distance_ratios:
            # Forward pass for the batch
            _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f_repeated)
            z = model.ts_encoder.reparameterization(z_mean, z_log_var * distance_ratio)

            if config_dict['3d']:
                tx_embedded = model.text_encoder(tx_f_ls_repeated)
                logits = model.clip(z, tx_embedded)
            else:
                tx_embedded = model.text_encoder(tx_f_repeated)
                logits = model.clip(z, tx_embedded)
            
            logits = torch.diag(logits)#.reshape(-1, 1)
            logits_ls.append(logits)
            x_hat = model.ts_decoder(z, x_mean, x_std)
            x_hat_ls.append(x_hat)
            distance_ratio_ls.extend([distance_ratio] * K)
            
            # Clear unnecessary tensors
            torch.cuda.empty_cache()
            # for _ in range(K):
            #     _, z_mean, z_log_var, x_mean, x_std = model.ts_encoder(ts_f)
            #     z = model.ts_encoder.reparameterization(z_mean, z_log_var*distance_ratio)
                
            #     if config_dict['3d']:
            #         tx_embedded = model.text_encoder(tx_f_ls)
            #         logits = model.clip(z, tx_embedded)
            #     else:
            #         tx_embedded = model.text_encoder(tx_f)
            #         logits = model.clip(z, tx_embedded)

            #     logits = torch.diag(logits).reshape(-1, 1)
            #     logits_ls.append(logits)
            #     x_hat = model.ts_decoder(z, x_mean, x_std)
            #     x_hat_ls.append(x_hat)
            #     distance_ratio_ls.append(distance_ratio)# append distance for reference

        # get the softmax probabilities
        logits_ls = torch.cat(logits_ls, dim=0)
        exp_preds = torch.exp(logits_ls)
        softmax_probs = exp_preds / exp_preds.sum(dim=0, keepdim=True)
        

        # find the topk softmax probabilities, and their corresponding x_hats
        top_probs, top_indices = torch.topk(softmax_probs, top, dim=0)  # Get top probabilities
        top_indices = top_indices.cpu().detach().numpy().tolist()

        # Get corresponding x_hats and distance ratios
        x_hat_ls = torch.cat(x_hat_ls, dim=0)  # Concatenate all x_hats
        top_ts_hats = [x_hat_ls[idx] for idx in top_indices]
        top_distance_ratios = [distance_ratio_ls[idx] for idx in top_indices]
        
        if poptop:
            # pop the first one (in case of outliers)
            top_probs = top_probs[1:]
            top_ts_hats = top_ts_hats[1:]
            top_distance_ratios = top_distance_ratios[1:]

        if top_probs[0] > threshold:# and top_probs[0][0] < threshold+0.1:
            topp_above_threshold = True
    
    return ts, top_probs, top_ts_hats, top_distance_ratios




def plot_vital_reconstructions(ts, top_probs, top_ts_hats, top_distance_ratios, keep=3, title=''):
    
    keep = min(keep, len(top_probs))
    top_probs = top_probs[:keep]
    top_ts_hats = top_ts_hats[:keep]
    top_distance_ratios = top_distance_ratios[:keep]

    if ts.shape[0] == 1:
        ts = ts[0]
    n_reconstructions = len(top_ts_hats)  # total number of subplots
    n_cols = 4
    n_rows = (n_reconstructions + n_cols - 1) // n_cols  # ceiling division for number of rows
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    fig.suptitle(title, fontsize=18)
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
    for i in range(1, n_reconstructions+1):
        ts_hat = top_ts_hats[i-1]
        prob = top_probs[i-1].item()
        distance_ratio = top_distance_ratios[i-1]
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



def net_emb(df, model, config_dict, top=100, text_levels = ['High amount of consecutive increases.', 'Low amount of consecutive increases.'] ):
    
    df_ls = []
    for i in range(len(text_levels)):
        df_sub = df[df['text'] == text_levels[i]].reset_index(drop=True)
        df_sub = df_sub.iloc[range(top)].copy()
        df_ls.append(df_sub)
    df = pd.concat(df_ls, ignore_index=True)

    text_cols = []
    for i in range(len(text_levels)):
        df['text'+str(i)] = text_levels[i]
        text_cols.append('text'+str(i))

    pairwise_distances, ts2tx_distances = cal_embeddings_distances(df, text_cols, model, config_dict)


    adj_mat = pairwise_distances['l2'].detach().cpu().numpy()
    adj_mat = 1/(adj_mat+1e-8)
    plot_embeddings_graph(adj_mat)


    adj_mat = pairwise_distances['simi'].detach().cpu().numpy()
    plot_embeddings_graph(adj_mat)

    return pairwise_distances, ts2tx_distances
