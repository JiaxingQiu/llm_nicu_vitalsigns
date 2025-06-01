if 'model_name' not in locals(): model_name = 'nicu_at' 


text_config = {
    'cl': {
        'die7d': True,
        'fio2': False
    },
    'demo': {
        'ga_bwt': True,
        'gre': False, # gender_race_ethnicity
        'apgar_mage': False
    },
    'ts': {
        # dynamic
        'ts_event': {
            'sumb': True, # sum_brady
            'sumd': False, # sum_desat
            'simple': False,
            'full': False,
            'event1': True},
        # static and categorical
        'succ_inc': True,
        'succ_unc': False,
        'histogram': True
    },
    'split': False
}


# text_config['cl']['die7d'] = True # udpate text_config here if needed
# text_config['ts']['succ_unc'] = False

update_config(
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'description_succ_inc',
    y_levels = ['High amount of consecutive increases.', 'Moderate amount of consecutive increases.', 'Low amount of consecutive increases.'],
    y_pred_levels = ['High amount of consecutive increases.', 'Moderate  amount of consecutive increases.', 'Low amount of consecutive increases.'],
    # txt2ts
    txt2ts_y_cols = ['description_succ_inc', 'description_histogram', 'description_ts_event_binary'],# 'description_succ_unc', 
    
    
    
    # Data settings
    text_col = 'ts_description',
    downsample = True,
    downsample_size = 20000,
    downsample_levels = ['High amount of consecutive increases.', 'Moderate amount of consecutive increases.', 'Low amount of consecutive increases.'],
    custom_target_cols = ['description_succ_inc', 'description_histogram', 'description_ts_event_binary', 'label'], #  description_ts_event_binary 'description_succ_unc', 
    ts_global_normalize = True, 
    
    # Model settings
    model_name = model_name,
    
    # Train settings
    # init_lr = 1e-5,
    
    # Text configuration
    text_config = text_config
)
config_dict = get_config_dict()
