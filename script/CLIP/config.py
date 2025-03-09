# configurations goes here
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
set_seed(333)  # or any other seed value

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
        'event1': False,
        'succ_inc': True,
        'histogram': True
    }
}

# Initialize the config dictionary with default values
config_dict = {
    # Device
    'device': device,
    
    # Data settings
    'batch_size': 128,
    'text_encoder_name': 'sentence-transformers/all-mpnet-base-v2',
    'ts_encoder_name': 'hr_vae_linear_medium',
    'ts_aug': False,
    'ts_normalize': True,
    'ts_encode': True,
    'block_target': True,
    'balance': False,
    
    # Model settings
    'model_name': 'hey_you_forget_to_name_your_model',
    'embedded_dim': 128,
    'init_lr': 0.0001,
    'patience': 50,
    
    # Training settings
    'num_saves': 200,
    'num_epochs': 50,
    'loss_type': 'block_diagonal',
    'txt_ls': None,
    
    # Text configuration
    'text_config': text_config
}

def update_config(**kwargs):
    """Update configuration with new values"""
    config_dict.update(kwargs)
    return config_dict

def get_config_dict():
    """Get current configuration"""
    return config_dict.copy()

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
        true_components += [
            key for key, value in params.items() 
            if value is True
        ]
    
    return true_components