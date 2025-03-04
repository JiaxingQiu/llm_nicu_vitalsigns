# configurations goes here
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data
batch_size = 128
text_encoder_name = 'sentence-transformers/all-mpnet-base-v2' # select from below
ts_encoder_name = 'hr_vae_linear_medium'

# model
overwrite = False
model_name = 'hey_you_forget_to_name_your_model'
embedded_dim = 128 # dim to project ts and text to, and get logits
init_lr = 0.0001 # initial learning rate
patience = 50 # patience for learning rate decay

# training
num_saves = 200 # total epochs will be num_saves * num_epochs
num_epochs = 50
loss_type = 'block_diagonal'
train_losses=[]
test_losses=[]
train_eval_metrics_list = []
test_eval_metrics_list = []

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
        'event1': False
    }
}

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