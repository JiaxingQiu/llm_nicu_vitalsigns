# --- prepare saving paths ------------------------------------------------------------
output_dir = './results/'+config_dict['model_name']
model_path = output_dir+'/model.pth' 
config_path = output_dir+'/config.pth'


# if encoder and decoder not defined otherwise, use the default ones
if 'ts_encoder' not in locals():
    ts_encoder = None
if 'ts_decoder' not in locals():
    ts_decoder = None
if 'text_encoder' not in locals():
    text_encoder = None

# customize model
if overwrite or not os.path.exists(model_path):  
    
    # ------------------------- initialize model -------------------------
    if config_dict['3d']:
        model = VITAL3D(
                    ts_dim=ts_f_dim.shape[1],
                    text_dim=tx_f_dim_ls[0].shape[1],
                    n_text=len(tx_f_dim_ls),
                    output_dim=config_dict['embedded_dim'],
                    ts_encoder = ts_encoder,
                    ts_decoder = ts_decoder,
                    clip_mu = config_dict['clip_mu']
                )
    else:
        model = VITAL(
                    ts_dim=ts_f_dim.shape[1],
                    text_dim=tx_f_dim.shape[1],
                    output_dim=config_dict['embedded_dim'],
                    ts_encoder=ts_encoder,
                    ts_decoder=ts_decoder,
                    text_encoder = text_encoder,
                    clip_mu = config_dict['clip_mu'],
                    variational = config_dict['variational'],
                    gen_w_src_text = config_dict['gen_w_src_text']
                )
    update_config(model_init = model)
    config_dict = get_config_dict()
    train_eval_metrics_ts2txt_list = []
    test_eval_metrics_ts2txt_list = []
    train_eval_metrics_txt2ts_list = []
    test_eval_metrics_txt2ts_list = []
    train_losses = []
    test_losses = []
    # ------------------------- ready output directory -------------------------
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    config_dict['model_init'] = model
    torch.save(config_dict, config_path)
    # overwrite = False # reset overwrite to False
else:
    config_dict = torch.load(config_path, map_location=torch.device(device), weights_only=False)
    model = config_dict['model_init']
    print(nn_summary(model))
    model.device = device
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=False))
    

    