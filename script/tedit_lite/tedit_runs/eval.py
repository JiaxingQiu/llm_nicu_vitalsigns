from tedit_data import *
from tedit_generation import *
import sys, os
if vital_path not in sys.path: 
    sys.path.append(vital_path)
from eval import *

# prepare meta, model, configs, output_dir
_, meta = TEditDataset(df_train, df_test, df_left, config_dict, split="test").get_loader(batch_size=128)
model, configs = load_model(meta, dataset_name=dataset_name, mdl_name = tedit_mdl)
output_dir = configs['train']['output_folder'] 

# run the same evaluation script on vital model
exec(open(os.path.join(vital_path, 'run/eval.py')).read())

# rename file "df_rats_all.pt.gz" under output_dir to "df_rats_all{suffix}.pt.gz"


# ... existing code ...