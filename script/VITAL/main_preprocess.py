import pandas as pd
from sklearn.model_selection import train_test_split

# ---- ready original dataframes ----
# Train Data
df = pd.read_excel('../../data/PAS Challenge HR Data.xlsx', engine="calamine")
df.columns = df.columns.astype(str)
df_y = pd.read_excel('../../data/PAS Challenge Outcome Data.xlsx', engine="calamine")[['VitalID', 'DiedNICU', 'DeathAge']]
df_demo = pd.read_excel('../../data/PAS Challenge Demographic Data.xlsx', engine="calamine")
df_x = pd.read_excel('../../data/PAS Challenge Model Data.xlsx', engine="calamine")
df = df.merge(df_x[['VitalID', 'VitalTime', 'Age']], on=['VitalID', 'VitalTime'], how='left')
df = label_death7d(df, df_y, id_col='VitalID')
df = df.merge(df_demo, on='VitalID', how='left')
df_desc = generate_descriptions_parallel(ts_df = df.loc[:, '1':'300'], id_df = df.loc[:, ['VitalID', 'VitalTime']])
df = df.merge(df_desc, on=['VitalID', 'VitalTime'], how='left')
df = text_gen_input_column(df, config_dict['text_config'])
df['rowid'] = df.index.to_series() 
df_train = df

# Test Data
df_y_test = pd.read_excel('../../data/Test Data/Test Demographic Key.xlsx', sheet_name=0, engine="calamine")
df_test = pd.read_excel('../../data/Test Data/Test HR Data.xlsx', sheet_name=0, engine="calamine") # test hr with description
df_test.columns = df_test.columns.astype(str)
df_test = label_death7d(df_test, df_y_test, id_col='TestID')
df_demo_test = pd.read_excel('../../data/Test Data/Test Demographic Data.xlsx', sheet_name=0, engine="calamine")
df_test = df_test.merge(df_demo_test, on='TestID', how='left')
df_test['rowid'] = df_test.index.to_series()
df_test['VitalTime'] = df_test['Age']*24*60*60 # convert to second since birth
df_test['VitalTime'] = df_test['VitalTime'].astype(int)
rename_dict = {'TestID': 'VitalID'}
df_test = df_test.rename(columns=rename_dict)

df_desc_test = generate_descriptions_parallel(ts_df = df_test.loc[:, '1':'300'], id_df = df_test.loc[:, ['VitalID', 'VitalTime']])
df_test = df_test.merge(df_desc_test, on=['VitalID', 'VitalTime'], how='left')
df_test = text_gen_input_column(df_test, config_dict['text_config'])
df_test_org = df_test[df.columns]
df_test, df_leftout = train_test_split(df_test_org, test_size=0.5, stratify=df_test_org[config_dict['y_col']], random_state=config_dict['random_state']) 


# df_train, df_test = train_test_split(df_train, test_size=0.2, stratify=df_train[config_dict['y_col']], random_state=config_dict['random_state']) 
# ---- downsample negative class(es) ----
if config_dict['downsample']:
    df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
    df_test = downsample_neg_levels(df_test, config_dict, config_dict['random_state'])

# ---- augment + balance train data----
# target_event_rate = len(df_test[df_test[config_dict['y_col']]==config_dict['y_levels'][0]])/len(df_test)
if config_dict['ts_aug']:
    df_train = augment_balance_data(df_train, 
                                    config_dict['y_levels'], 
                                    config_dict['y_col'], 
                                    config_dict, 
                                    pretrained_model_path='./pretrained/hr_vae_linear_medium.pth')
    df_test['augid'] = 0

# ---- created masked subsequences of each time series ----
if config_dict['ts_subseq']:
    df_train = subseq_raw_df(df_train,config_dict)
    df_test = subseq_raw_df(df_test,config_dict)
    
    # default fill na with preset global mean
    df_train_na = df_train.loc[:, '1':'300'].copy()
    df_test_na = df_test.loc[:, '1':'300'].copy()

    if config_dict['ts_local_normalize']:
        # Fill NaN with row-wise means using np.nanmean
        df_train_na = df_train_na.apply(lambda x: x.fillna(np.nanmean(x)), axis=1)
        df_test_na = df_test_na.apply(lambda x: x.fillna(np.nanmean(x)), axis=1)
    else:
        # Fill NaN with global preset mean
        df_train_na = df_train_na.fillna(config_dict['ts_normalize_mean'])
        df_test_na = df_test_na.fillna(config_dict['ts_normalize_mean'])
    
    df_train.loc[:, '1':'300'] = df_train_na
    df_test.loc[:, '1':'300'] = df_test_na
    # remove df_train_na and df_test_na from memory
    del df_train_na, df_test_na

# ---- block or not ----
if not config_dict['block_target']:
    df_train['label'] = df_train.index.to_series()
    df_test['label'] = df_test.index.to_series()
else:
    if 'subid' in df_train.columns:
        if config_dict['ts_augsub']:
            # = get a subsequence then augment it multiple times
            df_train['label'] = df_train['rowid'].astype(int)*100 + df_train['subid'].astype(int)
            df_test['label'] = df_test['rowid'].astype(int)*100 + df_test['subid'].astype(int)
        else:
            # get unqiue subsequences for each augmentation
            df_train['label'] = df_train['rowid'].astype(int)*10000 + df_train['augid'].astype(int)*100 + df_train['subid'].astype(int)
            df_test['label'] = df_test['rowid'].astype(int)*10000 + df_test['augid'].astype(int)*100 + df_test['subid'].astype(int)

    else:
        df_train['label'] = df_train['rowid'].astype(int)
        df_test['label'] = df_test['rowid'].astype(int)


# ---- ready eval inputs ----
n_levels = len(config_dict['y_levels'])
mtype = "3d" if config_dict['3d'] else "2d"

for i in range(n_levels):
    df_train[f'true{i+1}'] = df_train[config_dict['y_col']].apply(lambda x: 1 if x == config_dict['y_levels'][i] else 0)
    df_train[f'text{i+1}'] = config_dict['y_pred_levels'][i]
for i in range(n_levels):
    df_test[f'true{i+1}'] = df_test[config_dict['y_col']].apply(lambda x: 1 if x == config_dict['y_levels'][i] else 0)
    df_test[f'text{i+1}'] = config_dict['y_pred_levels'][i]

evalinputs_train = EvalInputs(df_train, 
                            config_dict['text_encoder_name'], 
                            normalize_mean = config_dict['ts_normalize_mean'], 
                            normalize_std = config_dict['ts_normalize_std'],
                            y_true_cols = [f'true{i+1}' for i in range(n_levels)],
                            y_pred_cols = [f'text{i+1}' for i in range(n_levels)],
                            y_pred_cols_ls = config_dict['y_pred_cols_ls'],
                            mtype=mtype)
evalinputs_test = EvalInputs(df_test, 
                            config_dict['text_encoder_name'], 
                            normalize_mean = config_dict['ts_normalize_mean'], 
                            normalize_std = config_dict['ts_normalize_std'],
                            y_true_cols = [f'true{i+1}' for i in range(n_levels)],
                            y_pred_cols = [f'text{i+1}' for i in range(n_levels)],
                            y_pred_cols_ls = config_dict['y_pred_cols_ls'],
                            mtype=mtype)


print(df_train[config_dict['y_col']].value_counts())
print(df_test[config_dict['y_col']].value_counts())



output_dir = './results/'+model_name
model_path = output_dir+'/model.pth' 
eval_path = output_dir+'/evals.pth'
config_path = output_dir+'/config.pth'


if overwrite:
    # ------------------------- ready dataloaders ------------------------- 
    if config_dict['3d']:
        ts_f_train, tx_f_train_ls, labels_train = get_features3d(df_train,
                                                                config_dict['text_encoder_name'], 
                                                                config_dict['ts_normalize_mean'],
                                                                config_dict['ts_normalize_std'],
                                                                global_norm = config_dict['ts_global_normalize'],
                                                                local_norm = config_dict['ts_local_normalize'],
                                                                text_col_ls = config_dict['text_col_ls'])
        train_dataloader = VITAL3DDataset(ts_f_train, tx_f_train_ls, labels_train).dataloader(batch_size=config_dict['batch_size'])
        ts_f_test, tx_f_test_ls, labels_test = get_features3d(df_test, 
                                                              config_dict['text_encoder_name'], 
                                                              config_dict['ts_normalize_mean'],
                                                              config_dict['ts_normalize_std'],
                                                              global_norm = config_dict['ts_global_normalize'],
                                                              local_norm = config_dict['ts_local_normalize'],
                                                              text_col_ls = config_dict['text_col_ls'])
        test_dataloader = VITAL3DDataset(ts_f_test, tx_f_test_ls, labels_test).dataloader(batch_size=config_dict['batch_size'])
    else: 
        ts_f_train, tx_f_train, labels_train = get_features(df_train,
                                                            config_dict['text_encoder_name'], 
                                                            config_dict['ts_normalize_mean'],
                                                            config_dict['ts_normalize_std'],
                                                            global_norm = config_dict['ts_global_normalize'],
                                                            local_norm = config_dict['ts_local_normalize'])
        train_dataloader = VITALDataset(ts_f_train, tx_f_train, labels_train).dataloader(batch_size=config_dict['batch_size'])
        ts_f_test, tx_f_test, labels_test = get_features(df_test,
                                                         config_dict['text_encoder_name'], 
                                                         config_dict['ts_normalize_mean'],
                                                         config_dict['ts_normalize_std'],
                                                         global_norm = config_dict['ts_global_normalize'],
                                                         local_norm = config_dict['ts_local_normalize'])
        test_dataloader = VITALDataset(ts_f_test, tx_f_test, labels_test).dataloader(batch_size=config_dict['batch_size'])
 