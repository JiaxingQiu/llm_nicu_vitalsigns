# configurations goes here

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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