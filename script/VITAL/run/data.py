# --- model name as the saving path ----------------------------------------------------------
attr_suffix = '' if text_based else '_at'
model_name = ''.join([dataset_name, attr_suffix, suffix])

# --- configs, default attribute-based version ----------------------------------------
if dataset_name == 'syn_gt':
    exec(open('run/configs/synthetic_w_gt.py', 'r').read())
elif dataset_name == "syn":
    exec(open('run/configs/synthetic.py', 'r').read())
elif dataset_name == "air":
    exec(open('run/configs/air_quality.py', 'r').read())
elif dataset_name == "nicu":
    exec(open('run/configs/nicu.py', 'r').read())
if text_based:
    update_config(custom_target_cols = ['label'])
    config_dict = get_config_dict()
if 'open_vocab' in locals():
    update_config(open_vocab = open_vocab)
    config_dict = get_config_dict()
    
# --- prepare train, test, left dataframes --------------------------------------------
if dataset_name == 'syn_gt':
    exec(open('run/prepare_datasets/synthetic.py', 'r').read())
elif dataset_name == "syn":
    exec(open('run/prepare_datasets/synthetic.py', 'r').read())
elif dataset_name == "air":
    exec(open('run/prepare_datasets/air_quality.py', 'r').read())
elif dataset_name == "nicu":
    exec(open('run/prepare_datasets/nicu.py', 'r').read())

# --- prepare tensor model inputs -----------------------------------------------------
with open('run/inputs.py', 'r') as file:
    exec(file.read())


