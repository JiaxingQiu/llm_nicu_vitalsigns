# ------------------------------------------------------------------------------------------------
# prepare dataset and arguments for training
# ------------------------------------------------------------------------------------------------
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






# ------------------------------------------------------------------------------------------------
# prepare arguments for evaluation
# ------------------------------------------------------------------------------------------------

df_eval = df_left #df_test if 'df_left' not in locals() else df_left
w = 0.8 # stength of augmentation
suffix = ''#'_test' if 'df_left' not in locals() else ''

# Matrices
math = True
ts_dist = True
rats = True

# argument dictionary used for ts_dist and rats
args0 = {'segment1': None,
        'segment2': None,
        'segment3': None,
        'segment4':None
        }

args1 = {'segment1': [('segment2', 'No seasonal pattern.'), ('segment3', 'No sharp shifts.'), ('segment4', 'The time series exhibits low variability.')],
        'segment2': [('segment1', 'No trend.'), ('segment3', 'No sharp shifts.'), ('segment4', 'The time series exhibits low variability.')],
        'segment3': [('segment1', 'No trend.'), ('segment2', 'No seasonal pattern.'), ('segment4', 'The time series exhibits low variability.')],
        'segment4': [('segment1', 'No trend.'), ('segment2', 'No seasonal pattern.'), ('segment3', 'No sharp shifts.')]
        }
args_ls = [args0, args1]

# Define the base augmentation pairs used in math and ts_dist
base_aug_dict = {'trend': [('No trend.', 'The time series shows upward linear trend.'), 
                            ('The time series shows downward linear trend.', 'The time series shows upward linear trend.'),
                            ('No trend.', 'The time series shows downward linear trend.'), 
                            ('The time series shows upward linear trend.', 'The time series shows downward linear trend.')],
                    'curvature': [('No trend.', 'The time series shows upward quadratic trend.'),
                                  ('The time series shows upward linear trend.', 'The time series shows upward quadratic trend.'),
                                  ('No trend.', 'The time series shows downward quadratic trend.'),
                                  ('The time series shows downward linear trend.', 'The time series shows downward quadratic trend.')],
                    'seasonality': [('No seasonal pattern.', 'The time series exhibits a seasonal pattern.'),
                                ('The time series exhibits a seasonal pattern.', 'No seasonal pattern.')],
                    'shift': [('No sharp shifts.', 'The mean of the time series shifts upwards.'),
                            ('The mean of the time series shifts downwards.', 'The mean of the time series shifts upwards.'),
                            ('No sharp shifts.', 'The mean of the time series shifts downwards.'),
                            ('The mean of the time series shifts upwards.', 'The mean of the time series shifts downwards.')],
                    'variability': [("The time series exhibits low variability.", "The time series exhibits high variability."),
                                ('The time series exhibits high variability.', "The time series exhibits low variability.")]
                    }


