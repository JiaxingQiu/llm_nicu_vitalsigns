# run this script to train clip first then decoder.

# train clip first
config_dict['train_type'] = 'clip'
config_dict['num_saves'] = 8
config_dict['num_epochs'] = 500
config_dict['init_lr'] = 1e-4
config_dict['patience'] = 100
with open('run/train.py', 'r') as file:
    exec(file.read())


# train decoder only
for param in model.ts_encoder.parameters():
    param.requires_grad = False
for param in model.text_encoder.parameters():
    param.requires_grad = False
# for param in model.ts_encoder.logvar_layer.parameters():
#     param.requires_grad = True
print("Checking trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad: print(f"{name}")

config_dict['train_type'] = 'joint' # 'vae' works the same
config_dict['num_saves'] = 10
config_dict['num_epochs'] = 1000
config_dict['init_lr'] = 1e-4
config_dict['patience'] = 500
with open('run/train.py', 'r') as file:
    exec(file.read())