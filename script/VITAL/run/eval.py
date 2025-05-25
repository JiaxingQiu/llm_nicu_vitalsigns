if 'meta' not in locals():
    meta = None
if 'configs' not in locals():
    configs = None

# ---------------------------------------  Math eval ---------------------------------------
# Math properties (applicable to quantitative time series attributes)
if math:
    filename = output_dir+'/df_stats_all'+suffix+'.pt.gz'
    if overwrite or (not os.path.exists(filename)):
        # calculate the properties of the generated time series
        df = df_eval.sample(1000)
        df_stats_all = pd.DataFrame()
        for aug_type in ['conditional', 'marginal']:
            df_stats = eval_math_properties(df, model, config_dict, aug_type = aug_type, w = w, 
                                            meta = meta, configs = configs)
            df_stats['aug_type'] = aug_type
            df_stats_all = pd.concat([df_stats_all, df_stats], ignore_index=True)
        
        # Save using PyTorch with gzip compression
        buffer = io.BytesIO()
        torch.save(df_stats_all, buffer)
        with gzip.open(filename, 'wb') as f:
            f.write(buffer.getvalue())
    else:
        # Load compressed file
        with gzip.open(filename, 'rb') as f:
            buffer = io.BytesIO(f.read())
            df_stats_all = torch.load(buffer, map_location=device, weights_only=False)


# ---------------------------------------  TS distance eval ---------------------------------------
# point-wise distance (to syn-gt only()
if config_dict.get('text_config') and 'text_pairs' in config_dict['text_config']:
    if config_dict['text_config']['gt']:
        filename = output_dir+'/df_pw_dists_all'+suffix+'.pt.gz'
        if overwrite or (not os.path.exists(filename)):
            df_pw_dists_all = pd.DataFrame()
            for aug_type in ['conditional', 'marginal']:
                df_pw_dists = eval_pw_dist(df_eval, 
                                            model, 
                                            config_dict, 
                                            w, 
                                            aug_type=aug_type, 
                                            meta = meta, configs = configs)
                df_pw_dists['aug_type'] = aug_type
                df_pw_dists_all = pd.concat([df_pw_dists_all, df_pw_dists], ignore_index=True)

            
            buffer = io.BytesIO()
            torch.save(df_pw_dists_all, buffer)
            with gzip.open(filename, 'wb') as f: f.write(buffer.getvalue())
        else:
            # Load compressed file
            with gzip.open(filename, 'rb') as f:
                buffer = io.BytesIO(f.read())
                df_pw_dists_all = torch.load(buffer, map_location=device, weights_only=False)



# TS distance (to both quantitative and qualitative attributes)
if ts_dist:
    filename = output_dir+'/df_dists_all'+suffix+'.pt.gz'
    if overwrite or (not os.path.exists(filename)):
        df_dists_ls = []
        for args in args_ls:
            df_dists = pd.DataFrame()
            for y_col in args.keys():
                print(y_col)
                df = pd.DataFrame()
                for aug_type in ['conditional', 'marginal']:
                    df_type = eval_ts_similarity(df_eval,
                                                model, 
                                                config_dict, 
                                                w = w,
                                                y_col = y_col,
                                                conditions = args[y_col],
                                                b = 500 if aug_type == 'marginal' else 200, 
                                                aug_type = aug_type, 
                                                meta = meta, configs = configs)  
                    df_type['aug_type'] = aug_type
                    df = pd.concat([df, df_type], ignore_index=True)
                df['attr'] = y_col
                df_dists = pd.concat([df_dists, df], ignore_index=True)
            df_dists_ls.append(df_dists)

        buffer = io.BytesIO()
        torch.save(df_dists_ls, buffer)
        with gzip.open(filename, 'wb') as f: f.write(buffer.getvalue())
    else:
        # Load compressed file
        with gzip.open(filename, 'rb') as f:
            buffer = io.BytesIO(f.read())
            df_dists_ls = torch.load(buffer, map_location=device, weights_only=False)
            
            

# ---------------------------------------  RaTS eval ---------------------------------------
# RaTS from classifiers (to both quantitative and qualitative attritbutes)
if rats:
    if meta is None:
        vital_model = model # defined as vital
    
    df_eval_rats = df_eval if len(df_eval) <= 15000 else df_eval.sample(15000)
        
    filename = output_dir+'/df_rats_all'+suffix+'.pt.gz'
    if overwrite or (not os.path.exists(filename)):
        df_rats_ls = []
        for args in args_ls:
            df_rats = pd.DataFrame()
            for y_col in args.keys():
                df = pd.DataFrame()
                for aug_type in ['conditional', 'marginal']:
                    df_type, _ = eval_ts_classifier(df_eval_rats, vital_model, config_dict,
                                                    w = w, y_col = y_col, conditions = args[y_col], aug_type = aug_type, meta = meta, configs = configs)
                    df = pd.concat([df, df_type], ignore_index=True)
                df_rats = pd.concat([df_rats, df], ignore_index=True)
            df_rats_ls.append(df_rats)

        buffer = io.BytesIO()
        torch.save(df_rats_ls, buffer)
        with gzip.open(filename, 'wb') as f: f.write(buffer.getvalue())
    else:
        # Load compressed file
        with gzip.open(filename, 'rb') as f:
            buffer = io.BytesIO(f.read())
            df_rats_ls = torch.load(buffer, map_location=device, weights_only=False)



# define a function to summarize the scores
def summarize_scores(df_all, aug_type= 'conditional'):
    df_conditional = df_all[df_all['aug_type'] == aug_type]

    stats_table = df_conditional.groupby('metric')['score'].agg(
        mean='mean',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q50=lambda x: x.quantile(0.50),
        q75=lambda x: x.quantile(0.75)
    ).round(3)

    stats_table['final_score'] = stats_table.apply(
        lambda row: f"{row['q50']:.2f} [{row['q25']:.2f}, {row['q75']:.2f}]",
        axis=1
    )

    final_score_row = stats_table['final_score'].to_frame().T
    final_score_row.index = ['final_score']
    # Rename columns if necessary
    if 'mse' in final_score_row.columns:
        final_score_row = final_score_row.rename(columns={
            'mse': 'Point-wise MSE ↓',
            'mae': 'Point-wise MAE ↓',
            'delta_dtw': 'DTW similarity improvement ↑',
            'RaTS': 'RaTS ↑'
        })
    else:
        final_score_row = final_score_row.rename(columns={
            'delta_dtw': 'DTW similarity improvement ↑',
            'RaTS': 'RaTS ↑'
        })
    if 'delta_lcss' in final_score_row.columns:
        final_score_row = final_score_row.rename(columns={
            'delta_lcss': 'LCSS similarity improvement ↑'
        })

    # Reorder columns (only keep those that exist)
    desired_order = [
        'Point-wise MSE ↓',
        'Point-wise MAE ↓',
        'DTW similarity improvement ↑',
        'LCSS similarity improvement ↑',
        'RaTS ↑'
    ]
    final_score_row = final_score_row[[col for col in desired_order if col in final_score_row.columns]]
    return final_score_row
