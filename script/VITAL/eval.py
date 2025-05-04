from eval_utils.eval_vae import *
from eval_utils.eval_clip_ts2txt import *
from eval_utils.eval_clip_txt2ts import *
from eval_utils.eval_math import *
from eval_utils.eval_embedding import *


from generation import interpolate_ts_tx
# import time
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt



# ----------------- Math statistics evaluation on synthetic data --------------------------------
# calculate the math properties of the generated time series
def eval_math_properties(df, model, config_dict, type, w, y_cols = None):
    model.eval()
    df_augmented = pd.DataFrame()
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
                # start = time.time()
                # properties = df_prop['ts_hat'].apply(lambda x: get_all_stats(x.detach().cpu().numpy()))
                # end = time.time()
                # print(f"Time to compute properties for {len(df_prop)} samples: {end - start:.4f} seconds")
                # start = time.time()
                n_cores = min(max(1, int(multiprocessing.cpu_count() * 0.7)), 10)
                properties = Parallel(n_jobs=n_cores, verbose=1)(
                    delayed(get_all_stats)(x.detach().cpu().numpy()) for x in df_prop['ts_hat']
                )
                # end = time.time()
                # print(f"Time to compute properties for {len(df_prop)} samples (parallel): {end - start:.4f} seconds")


                df_prop['properties'] = properties
                df_prop['text_col'] = text_col
                df_prop['index'] = df_level.index # saved original index
                df_prop_all = pd.concat([df_prop_all, df_prop])
            
            df_prop_all['org_y_col'] = y_col
            df_prop_all['org_y_level'] = y_levels[i]
            df_augmented = pd.concat([df_augmented, df_prop_all])

    return df_augmented


# engineer the math properties from the math properties of the augmented time serie (df_augmented)
def eng_math_diff(df_augmented, base_y_level, augm_y_level, metrics = ['trend', 'curvature', 'seasonality', 'shift', 'variability']):

    df_metrics = None
    for metric in metrics:
        aug_df = df_augmented[
                    (df_augmented['org_y_level'].str.contains(base_y_level, case=False)) & 
                    (df_augmented['aug_text'].str.contains(augm_y_level, case=False))
                ]
        aug_df['aug_'+metric] = aug_df['properties'].apply(lambda x: x[metric])
        bas_df = df_augmented[
                    (df_augmented['org_y_level'].str.contains(base_y_level, case=False)) & 
                    (df_augmented['aug_text'].str.contains(base_y_level, case=False))
                ]
        bas_df['bas_'+metric] = bas_df['properties'].apply(lambda x: x[metric])
        scl = bas_df['bas_'+metric].std()

        # merge the two dataframes on the index column
        aug_df = aug_df.loc[:, ['index', 'aug_'+metric]]
        bas_df = bas_df.loc[:, ['index', 'bas_'+metric]]
        # assert two df has same nrow
        assert len(aug_df) == len(bas_df)
        df = aug_df.merge(bas_df, on = 'index', how = 'left')
        df['diff_'+metric] = df['aug_'+metric] - df['bas_'+metric]
        df['diff_'+metric] = df['diff_'+metric] / scl# / df['diff_'+metric].abs().max()
        df = df.loc[:, ['index', 'diff_'+metric, 'aug_'+metric, 'bas_'+metric]]
        if df_metrics is None:
            df_metrics = df
        else: 
            df_metrics  = df_metrics.merge(df, on = 'index', how = 'left')

    return df_metrics



def eng_math_diff_multiple(df_augmented, base_aug_dict, metrics = ['trend', 'curvature', 'seasonality', 'shift', 'variability']):
    df_ls = []
    for aug_matrix, pairs in base_aug_dict.items():
        n_pairs = len(pairs)
        # Create figure with 2 rows: one for boxplots, one for histograms
        fig, axes = plt.subplots(2, n_pairs, figsize=(5*n_pairs, 6))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)  # Make it 2D array if only one column
        
        for col, (base, aug) in enumerate(pairs):
            df = eng_math_diff(df_augmented, base, aug, metrics = metrics)
            df['base'] = base
            df['aug'] = aug
            df['aug_matrix'] = aug_matrix
            df_ls.append(df)

            # --- Boxplot in first row ---
            diff_df = pd.DataFrame()
            for metric in metrics:
                diff_df[metric] = df['diff_'+metric]
            
            box = axes[0, col].boxplot(
                [diff_df[col] for col in diff_df.columns],
                labels=diff_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, metric in enumerate(diff_df.columns):
                color = 'black' if metric == aug_matrix else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            axes[0, col].set_ylim(-10, 10)
            axes[0, col].axhline(0, color='black', linewidth=0.5)
            axes[0, col].set_title(f'{base}\n→\n{aug}', pad=20)
            axes[0, col].set_xlabel('Statistics')
            if col == 0:  # Only add ylabel to first subplot
                axes[0, col].set_ylabel('Difference')
            
            # --- Histogram in second row ---
            aug_stat = df['aug_'+aug_matrix].values.tolist()
            bas_stat = df['bas_'+aug_matrix].values.tolist()
            axes[1, col].hist(aug_stat, bins=20, alpha=0.5, label='Augmented')
            axes[1, col].hist(bas_stat, bins=20, alpha=0.5, label='Base')
            axes[1, col].set_xlabel(aug_matrix)
            if col == 0:
                axes[1, col].set_ylabel('Count')
            axes[1, col].legend()
        
        plt.tight_layout()
        plt.show()
    res_df = pd.concat(df_ls)
    return res_df


