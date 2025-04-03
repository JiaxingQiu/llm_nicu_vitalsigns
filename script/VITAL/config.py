# configurations goes here
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        'sumb': True, # sum_brady
        'sumd': False, # sum_desat
        'simple': True,
        'full': False,
        'event1': True,
        'succ_inc': True,
        'succ_unc': True,
        'histogram': True
    },
    'split': False
}

# Initialize the config dictionary with default values
config_dict = {
    # Device
    'device': device,
    'random_state': 333,

    # ts2txt
    'text_col_ls': ['demo', 'cl_event', 'ts_description'], #['cl_event', 'ts_description', 'demo_ga', 'demo_weight', 'demo_apgar', 'demo_mother']
    'y_col': 'cl_event', # column name for the classification outcome, ie. 'cl_event'
    'y_levels': ['This infant will die in 7 days. ', 'This infant will survive. '],
    'y_pred_levels': ['will die in 7 days', 'will survive'],
    'y_pred_cols_ls': None,
    # txt2ts
    'sub_text_col': 'ts_description', 


    # Data settings
    'downsample': True,
    'downsample_levels': ['This infant will survive. '],
    'downsample_size': 1000,
    'ts_aug': False,
    'ts_aug_max_size': None,
    'balance': True,
    'block_target': True, # only block or diagonal two types of clip targets
    'ts_subseq': False,
    'ts_subseq_n': 1,
    'ts_subseq_min_length_ratio':1/6,
    'ts_subseq_max_length_ratio': 2/3,
    'ts_subseq_step_size_ratio': 1/30,
    'ts_augsub': False,

    # Data loader settings
    'batch_size': 2048,
    'text_encoder_name': 'sentence-transformers/all-mpnet-base-v2',
    'ts_encoder_name': 'hr_vae_linear_medium',
    'ts_global_normalize': False, 
    'ts_normalize_mean': 150, # global normalization mean
    'ts_normalize_std': 20, # global normalization std
    'ts_local_normalize': False, # shared with ts_subseq settings to fill na
    

    # Model settings
    'model_name': 'hey_you_forget_to_name_your_model',
    '3d': True,
    'embedded_dim': 128,
    'model_init': None,
    
    # Training settings
    'init_lr': 0.0001,
    'patience': 50,
    'num_saves': 200,
    'num_epochs': 50,
    
    # Text configuration
    'text_config': text_config
}


def set_seed(seed: int = 333) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    import random
    import numpy as np
    import torch
    from sklearn.utils import check_random_state
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Scikit-learn
    check_random_state(seed)
    
    print(f"Random seed set to {seed}")

# Usage
set_seed(config_dict['random_state'])  # or any other seed value

def update_config(**kwargs):
    """Update configuration with new values"""
    config_dict.update(kwargs)
    return config_dict


def get_config_dict():
    """Get current configuration"""
    
    base_config = config_dict
        
    # Create a copy to avoid modifying the original
    config = base_config.copy()
    
    # Validate y_levels and y_pred_levels match
    assert len(config['y_levels']) == len(config['y_pred_levels'])
    n_levels = len(config['y_levels'])

    # Generate y_pred_cols_ls for 3D models
    if config['3d']:
        pred_cols = [f'text{i+1}' for i in range(n_levels)]
        y_pred_cols_ls = []
        for text_col in pred_cols:
            new_cols = [text_col if x == config['y_col'] else x 
                       for x in config['text_col_ls']]
            y_pred_cols_ls.append(new_cols)
    else:
        y_pred_cols_ls = None

    config['y_pred_cols_ls'] = y_pred_cols_ls

    return config


# Usage example:
"""
# Update specific values
update_config(
    model_name='new_model_name',
    batch_size=64,
    ts_aug=True
)

# Get current config
current_config = get_config_dict()
"""

text_encoders = [
    # BERT-base
    'bert-base-uncased',
    'bert-base-cased',
    'bert-large-uncased',
    'bert-large-cased',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'allenai/scibert_scivocab_uncased',
    'dmis-lab/biobert-base-cased-v1.2',
    # Standard RoBERTa
    'roberta-base',             # 12-layer
    'roberta-large',            # 24-layer
    'allenai/biomed_roberta_base',  # Biomedical domain
    'roberta-base-openai-detector', # OpenAI's version
    'microsoft/roberta-base-openai-detector',  # Microsoft's version
    # DistilBERT
    'distilbert-base-uncased',  # General purpose, uncased
    'distilbert-base-cased',    # General purpose, cased
    'distilroberta-base',       # Distilled version of RoBERTa
    'distilbert-base-multilingual-cased',  # Multilingual version
    # MPNet
    'microsoft/mpnet-base',     # Base model
    'microsoft/mpnet-large',    # Large model
    # Sentence Transformers
    'sentence-transformers/all-mpnet-base-v2',  # Optimized for sentence embeddings
    'sentence-transformers/paraphrase-mpnet-base-v2',  # Optimized for paraphrase detection
    'sentence-transformers/all-MiniLM-L6-v2',  # Optimized for sentence embeddings
    'sentence-transformers/all-MiniLM-L12-v2'  # Optimized for sentence embeddings
]

ts_encoders = [
    'hr_vae_linear_medium',
    'sp_vae_linear_medium'
]


def get_true_components(config):
    """Get all components that are True in the config."""
    true_components = []
    
    # Flatten nested dictionary and get True values
    for category, params in config.items():
        if category == 'split':
            if params:
                true_components += ['split']
        else:
            true_components += [
                key for key, value in params.items() 
                if value is True
            ]
    
    return true_components