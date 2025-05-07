
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data/air_quality/air_quality_kdd.csv.zip', compression='zip')
df.columns = df.columns.astype(str)
df['text'] = df['ts_description']
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[config_dict['y_col']], random_state=config_dict['random_state']) 

if config_dict['downsample']:
    df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
    df_test = downsample_neg_levels(df_test, config_dict, config_dict['random_state'])

df_train['label'] = df_train.index.to_series()
df_test['label'] = df_test.index.to_series()

# ---- prepare target matrix ----
if len(config_dict['custom_target_cols']) > 0:
    target_train = gen_target(df_train, config_dict['custom_target_cols'])
    target_test = gen_target(df_test, config_dict['custom_target_cols'])
else:
    target_train = None
    target_test = None

print('\n\nfinal distribution of text prediction')
print(df_train['text'].value_counts())
print(df_test['text'].value_counts())
