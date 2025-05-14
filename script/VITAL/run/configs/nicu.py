
model_name = 'nicu3' 
text_config['cl']['die7d'] = True # udpate text_config here if needed
text_config['ts']['succ_unc'] = False

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
    downsample_size = 15000,
    downsample_levels = ['High amount of consecutive increases.', 'Moderate amount of consecutive increases.', 'Low amount of consecutive increases.'],
    custom_target_cols = ['description_succ_inc', 'description_histogram', 'description_ts_event_binary', 'label'], #  description_ts_event_binary 'description_succ_unc', 
    
    
    # Model settings
    model_name = model_name,
    variational = False,
    
    # Train settings
    init_lr = 1e-5,
    patience = 500,
    num_saves = 8,
    alpha = 1/100,
    
    # Text configuration
    text_config = text_config
)
config_dict = get_config_dict()
