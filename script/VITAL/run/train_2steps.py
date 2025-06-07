# # run this script to train clip first then decoder.

# # train clip first
# config_dict['train_type'] = 'clip'
# config_dict['num_saves'] = 1
# config_dict['es_patience'] = 100 # early stopping patience
# with open('run/train.py', 'r') as file:
#     exec(file.read())


# # train decoder only
# for param in model.ts_encoder.parameters():
#     param.requires_grad = False
# for param in model.text_encoder.parameters():
#     param.requires_grad = False
# print("Checking trainable parameters:")
# for name, param in model.named_parameters():
#     if param.requires_grad: print(f"{name}")

# config_dict['train_type'] = 'joint' # 'vae' works the same
# config_dict['num_saves'] = 10
# config_dict['num_epochs'] = 10000
# config_dict['target_ratio'] = None
# config_dict['init_lr'] = 1e-4
# config_dict['es_patience'] = 1000 
# config_dict['patience'] = 200 
# with open('run/train.py', 'r') as file:
#     exec(file.read())


# train clip first
config_dict['train_type'] = 'clip'
config_dict['num_saves'] = 1
config_dict['es_patience'] = 100 # early stopping patience
with open('run/train.py', 'r') as file:
    exec(file.read())

config_dict['train_type'] = 'joint'
config_dict['num_saves'] = 2
config_dict['es_patience'] = 10000 
with open('run/train.py', 'r') as file:
    exec(file.read()) 
