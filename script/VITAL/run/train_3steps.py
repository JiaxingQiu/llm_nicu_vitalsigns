# # run this script to train clip first then decoder.
config_dict_org = config_dict.copy() # maintain the original config

# 1. train clip first (for once)
if not os.path.exists(config_dict_org['output_dir']+'/model_clip.pth'):
    config_dict['train_type'] = 'clip'
    config_dict['num_saves'] = 1
    config_dict['es_patience'] = 100 # early stopping patience
    with open('run/train.py', 'r') as file:
        exec(file.read())
    torch.save(model.state_dict(), config_dict_org['output_dir']+'/model_clip.pth')
else:
    state_dict = torch.load(config_dict_org['output_dir'] + '/model_clip.pth', map_location=torch.device(device))
    ts_encoder_state = {k.replace('ts_encoder.', ''): v for k, v in state_dict.items() if k.startswith('ts_encoder.')}
    text_encoder_state = {k.replace('text_encoder.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder.')}
    model.ts_encoder.load_state_dict(ts_encoder_state)
    model.text_encoder.load_state_dict(text_encoder_state)
    
# 2. train decoder only
for param in model.ts_encoder.parameters():
    param.requires_grad = False
for param in model.text_encoder.parameters():
    param.requires_grad = False
config_dict['train_type'] = 'joint'
config_dict['num_saves'] = 1
config_dict['num_epochs'] = 10000
config_dict['target_ratio'] = None
config_dict['init_lr'] = 1e-4
config_dict['es_patience'] = 500 
with open('run/train.py', 'r') as file:
    exec(file.read())

# 3. train jointly
for param in model.ts_encoder.parameters():
    param.requires_grad = True
for param in model.text_encoder.parameters():
    param.requires_grad = True
config_dict['train_type'] = 'joint'
config_dict['target_ratio'] = config_dict_org['target_ratio']
config_dict['num_saves'] = 1
config_dict['num_epochs'] = 2000
config_dict['es_patience'] = 500 
with open('run/train.py', 'r') as file:
    exec(file.read())

config_dict = config_dict_org.copy()