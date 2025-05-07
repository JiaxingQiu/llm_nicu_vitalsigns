# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


from tslearn.metrics import dtw, lcss
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt
import pandas as pd
from generation import interpolate_ts_tx

def calculate_distances_parallel(ref_ts_list, aug_ts_list, n_jobs=None, standardize=True):
    """
    Calculate DTW and LCSS distance matrices between reference and augmented time series in parallel.
    
    Args:
        ref_ts_list (list): List of reference time series
        aug_ts_list (list): List of augmented time series
        n_jobs (int): Number of parallel jobs. If None, uses 70% of available CPUs.
        
    Returns:
        tuple: (dtw_matrix, lcss_matrix, dtw_summaries, lcss_summaries)
        where each matrix has shape (n_ref, n_aug) and summaries are lists of length n_aug
    """
    if n_jobs is None:
        n_jobs = max(1, int(multiprocessing.cpu_count() * 0.7))
    
    n_ref = len(ref_ts_list)
    n_aug = len(aug_ts_list)
    dtw_matrix = np.zeros((n_ref, n_aug))
    lcss_matrix = np.zeros((n_ref, n_aug))
    
    def process_pair(args, standardize=standardize):
        i, j, ref_ts, aug_ts = args # ref_ts and aug_ts are 1-d numpy arrays
        if standardize:
            ref_ts = (ref_ts - ref_ts.mean()) / ref_ts.std()
            aug_ts = (aug_ts - aug_ts.mean()) / aug_ts.std()
        dtw_dist = dtw(ref_ts, aug_ts)
        lcss_dist = lcss(ref_ts, aug_ts)
        return i, j, dtw_dist, lcss_dist
    
    # Prepare arguments for parallel processing
    args_list = []
    for i, ref_ts in enumerate(ref_ts_list):
        for j, aug_ts in enumerate(aug_ts_list):
            args_list.append((i, j, ref_ts, aug_ts))
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(args) for args in tqdm(args_list, desc="Calculating distances")
    )
    # # Process in parallel with batch size to reduce overhead
    # batch_size = 500  # Adjust this based on your system's memory
    # results = []
    
    # # Calculate total number of batches for progress bar
    # n_batches = (len(args_list) + batch_size - 1) // batch_size
    # parallel = Parallel(n_jobs=n_jobs, batch_size=1, prefer='processes')  # Send 10 tasks to each worker at a time
    
    # # Process in batches with progress bar
    # for batch_start in tqdm(range(0, len(args_list), batch_size), 
    #                       total=n_batches,
    #                       desc="Calculating distances"):
    #     batch_end = min(batch_start + batch_size, len(args_list))
    #     batch = args_list[batch_start:batch_end]
        
    #     # Process batch in parallel
    #     batch_results = parallel(
    #         delayed(process_pair)(args) for args in batch
    #     )
    #     results.extend(batch_results)
        
    #     # Force garbage collection after each batch
    #     gc.collect()
    
    # Fill the matrices
    for i, j, dtw_dist, lcss_dist in results:
        dtw_matrix[i, j] = dtw_dist
        lcss_matrix[i, j] = lcss_dist
    
    # Create summaries for each augmented time series
    dtw_summaries = []
    lcss_summaries = []
    
    for j in range(n_aug):
        # DTW summary for j-th augmented time series
        dtw_summary = {
            'mean': np.round(dtw_matrix[:, j].mean(), 4),
            'std': np.round(dtw_matrix[:, j].std(), 4),
            'min': np.round(dtw_matrix[:, j].min(), 4),
            'max': np.round(dtw_matrix[:, j].max(), 4),
            'q5': np.round(np.quantile(dtw_matrix[:, j], 0.05), 4),
            'q25': np.round(np.quantile(dtw_matrix[:, j], 0.25), 4),
            'q50': np.round(np.quantile(dtw_matrix[:, j], 0.5), 4),
            'q75': np.round(np.quantile(dtw_matrix[:, j], 0.75), 4),
            'q95': np.round(np.quantile(dtw_matrix[:, j], 0.95), 4)
        }
        dtw_summaries.append(dtw_summary)
        
        # LCSS summary for j-th augmented time series
        lcss_summary = {
            'mean': np.round(lcss_matrix[:, j].mean(), 4),
            'std': np.round(lcss_matrix[:, j].std(), 4),
            'min': np.round(lcss_matrix[:, j].min(), 4),
            'max': np.round(lcss_matrix[:, j].max(), 4),
            'q5': np.round(np.quantile(lcss_matrix[:, j], 0.05), 4),
            'q25': np.round(np.quantile(lcss_matrix[:, j], 0.25), 4),
            'q50': np.round(np.quantile(lcss_matrix[:, j], 0.5), 4),
            'q75': np.round(np.quantile(lcss_matrix[:, j], 0.75), 4),
            'q95': np.round(np.quantile(lcss_matrix[:, j], 0.95), 4)
        }
        lcss_summaries.append(lcss_summary)
    
    return dtw_matrix, lcss_matrix, dtw_summaries, lcss_summaries


def _stratified_bootstrap(group, b):
    n = len(group)
    replace = n < b
    return group.sample(n=b, replace=replace, random_state=333)

def eval_ts_distances(df, # df can be df_train / df_test
                      model, config_dict, w,  y_col, 
                      conditions = None, # a list of tuples of (y_col, y_level) to filter the df (should not filter y_col)
                      b=200):
    
    if conditions is not None:
        for condition in conditions:
            df = df[df[condition[0]] == condition[1]]

    df = (
        df.groupby(y_col, group_keys=False)
        .apply(_stratified_bootstrap, b=b)
        .reset_index(drop=True)
    )

    model.eval()
    df_dists = pd.DataFrame()
    y_levels = list(df[y_col].unique())

    # for each reference text
    for i in range(len(y_levels)):
        # agument data with original reference text y_levels[i]
        ref_text = y_levels[i]
        df_level = df[df[y_col] == ref_text].reset_index(drop=False).copy() #
        # agument towards new text conditions 
        for j in range(len(y_levels)):
            df_level['text' + str(j)] = y_levels[j] # only do marginal augmentation
        # only augment towards NEW text conditions except original aug_text
        new_text_cols = ['text' + str(j) for j in range(len(y_levels)) if df_level['text' + str(j)][0] != ref_text]
        # augment towards new text conditions with w!
        ts_hat_ls = interpolate_ts_tx(df_level, model, config_dict, new_text_cols, w) # 
        

        aug_df_all = pd.DataFrame()
        for _, pairs in ts_hat_ls.items():
            
            # augmented time series
            aug_df = pd.DataFrame(pairs, columns=['aug_text', 'ts_hat'])
            aug_text = aug_df['aug_text'].unique()[0] # # augment towards text
            aug_df['ts_hat'] = aug_df['ts_hat'].apply(lambda x: x.cpu().detach().numpy())
            aug_ts_list = [aug_df['ts_hat'][i] for i in range(len(aug_df))]
            
            # target time series
            tgt_df = df[df[y_col] == aug_text]
            tgt_ts_list = [tgt_df[[str(i+1) for i in range(config_dict['seq_length'])]].to_numpy()[i] for i in range(len(tgt_df))]
            
            # orginal time series (reference time series)
            ref_df = df[df[y_col] == ref_text]   
            ref_ts_list = [ref_df[[str(i+1) for i in range(config_dict['seq_length'])]].to_numpy()[i] for i in range(len(ref_df))]

            _, _, dtw_aug2tgt, lcss_aug2tgt = calculate_distances_parallel(aug_ts_list, tgt_ts_list)
            _, _, dtw_ref2tgt, lcss_ref2tgt = calculate_distances_parallel(ref_ts_list, tgt_ts_list)
            

            aug_df['dtw_aug2tgt'] = dtw_aug2tgt
            aug_df['lcss_aug2tgt'] = lcss_aug2tgt
            aug_df['dtw_ref2tgt'] = dtw_ref2tgt
            aug_df['lcss_ref2tgt'] = lcss_ref2tgt
            aug_df_all = pd.concat([aug_df_all, aug_df], ignore_index=True)

        aug_df_all['ref_y_col'] = y_col
        aug_df_all['ref_y_level'] = ref_text 
        df_dists = pd.concat([df_dists, aug_df_all], ignore_index=True)
   
    return df_dists




def eng_dists(df_dist, 
              ref_y_level, 
              aug_y_level, 
              metrics = ['lcss', 'dtw'], 
              plot = False):
    import matplotlib.pyplot as plt
   
    df = df_dist[
                    (df_dist['ref_y_level'].str.contains(ref_y_level, case=False)) & 
                    (df_dist['aug_text'].str.contains(aug_y_level, case=False))
                ]
    for metric in metrics:
        df['aug_'+metric] = df[metric+'_aug2tgt'].apply(lambda x: x['q50'])
        df['ref_'+metric] = df[metric+'_ref2tgt'].apply(lambda x: x['q50'])
        df['diff_'+metric] = df['aug_'+metric] - df['ref_'+metric]

    if plot:
        for metric in metrics:
            plt.hist(df['aug_'+metric], bins=20, alpha=0.5, label='augmented')
            plt.hist(df['ref_'+metric], bins=20, alpha=0.5, label='reference')
            plt.title(f'{metric} similarity')
            plt.legend()
            plt.show()
    
    return df




def eng_dists_multiple(df_dists, base_aug_dict, metric='lcss'):
    df_ls = []
    for _, pairs in base_aug_dict.items():
        n_cols = 4  # Always use 4 columns
        n_pairs = len(pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        # Create figure with 3 rows per pair: boxplot of differences, boxplot of values, and histogram
        fig, axes = plt.subplots(3 * n_rows, n_cols, figsize=(n_cols*5, 8*n_rows))
        
        # Handle single row case properly
        if n_rows == 1:
            axes = axes.reshape(3, n_cols)  # Make it 2D array if only one row
        else:
            axes = axes.reshape(3 * n_rows, n_cols)  # Ensure proper shape for multiple rows
        
        for idx, (ref, aug) in enumerate(pairs):
            row = idx // n_cols
            col = idx % n_cols
            
            df = eng_dists(df_dists, ref, aug, metrics = [metric])
            df['ref'] = ref
            df['aug'] = aug
            df_ls.append(df)

            # first row boxplot 'diff_'+metric + horizontal line at 0
            diff_df = pd.DataFrame()
            diff_df[metric] = df['diff_'+metric]
            box = axes[2*row, col].boxplot(
                [diff_df[col] for col in diff_df.columns],
                labels=diff_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, metric in enumerate(diff_df.columns):
                color = 'black' #if metric == aug_matrix else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            axes[2*row, col].set_ylim(-1, 1)
            axes[2*row, col].axhline(0, color='black', linewidth=0.5)
            axes[2*row, col].set_title(f'{ref}\n→\n{aug}', pad=20)
            if col == 0:  # Only add ylabel to first subplot
                axes[2*row, col].set_ylabel('similarity increase')
            
            # second row boxplot 'aug_'+metric + 'ref_'+metric + horizontal line at 'ref_'+metric .median()
            box_df = pd.DataFrame()
            box_df['augmented'] = df['aug_' + metric]
            box_df['reference'] = df['ref_' + metric]
            box = axes[2*row+1, col].boxplot(
                [box_df[col] for col in box_df.columns],
                labels=box_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, x in enumerate(box_df.columns):
                color = 'black' if x == 'augmented' else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            axes[2*row+1, col].set_ylim(-0.1, 1)
            axes[2*row+1, col].axhline(box_df['reference'].median(), color='black', linewidth=0.5)
            if col == 0:  # Only add ylabel to first subplot
                axes[2*row+1, col].set_ylabel(metric + ' similarity')
            
            # --- third row histogram ---
            aug_stat = df['aug_'+metric].values.tolist()
            bas_stat = df['ref_'+metric].values.tolist()
            
            # Calculate shared bins based on combined data range
            all_data = aug_stat + bas_stat
            bins = np.linspace(min(all_data), max(all_data), 20)
            
            axes[2*row+2, col].hist(aug_stat, bins=bins, alpha=0.5, label='Augmented')
            axes[2*row+2, col].hist(bas_stat, bins=bins, alpha=0.5, label='Base')
            axes[2*row+2, col].set_xlabel(metric + ' similarity')
            if col == 0:
                axes[2*row+2, col].set_ylabel('Count')
            axes[2*row+2, col].legend()
        
        # Hide any empty subplots
        total_plots = n_rows * n_cols
        for idx in range(n_pairs, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            axes[2*row, col].axis('off')
            axes[2*row+1, col].axis('off')
            axes[2*row+2, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    res_df = pd.concat(df_ls, ignore_index=True)
    return res_df





