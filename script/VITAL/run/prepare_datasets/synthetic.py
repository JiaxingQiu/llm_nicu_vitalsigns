
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data/synthetic/data.csv.zip', compression='zip')
df.columns = df.columns.astype(str)
df['text'] = df['ts_description']
df = df.reset_index(drop=True)
if 'text_pairs' in config_dict['text_config'] : # use mixture of attributes instead of single attributes 
    if config_dict['text_config']['gt']:
        # mixture with ground truth
        df_train, df_test, df_left = gt_train_test_left(df, config_dict)
        
    else:
        df = mix_w_counter(df, config_dict, n = config_dict['text_config']['n'], plot=False)
        df = df.reset_index(drop=True)
        df = add_y_col(df, config_dict)
        print(df.text.value_counts())
        #  split: 70% train, 20% test, 10% left
        df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df[config_dict['y_col']], random_state=config_dict['random_state'])
        df_test, df_left = train_test_split(df_temp, test_size=1/3,stratify=df_temp[config_dict['y_col']],random_state=config_dict['random_state'])
        # downsample negative levels
        if config_dict['downsample']:
            df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
            df_test = downsample_neg_levels(df_test, config_dict, config_dict['random_state'])
            df_left = downsample_neg_levels(df_left, config_dict, config_dict['random_state'])

df_train['label'] = df_train.index.to_series()
df_test['label'] = df_test.index.to_series()
df_left['label'] = df_left.index.to_series()

# ---- prepare target matrix ----
if len(config_dict['custom_target_cols']) > 0:
    target_train = gen_target(df_train, config_dict['custom_target_cols'])
    target_test = gen_target(df_test, config_dict['custom_target_cols'])
    target_left = gen_target(df_left, config_dict['custom_target_cols'])
else:
    target_train = None
    target_test = None
    target_left = None

print('\n\nfinal distribution of text prediction')
print(df_train['text'].value_counts())
print(df_test['text'].value_counts())
print(df_left['text'].value_counts())
