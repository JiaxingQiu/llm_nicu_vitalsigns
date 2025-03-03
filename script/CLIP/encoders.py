from config import *
import torch
import torch.nn as nn
#%pip install sentence_transformers==3.0.1
#%pip install xformers

class TXTEncoder():
    def __init__(self, model_name):
       
        # model_name = 'all-mpnet-base-v2', 'all-MiniLM-L6-v2', etc
        self.model_name = model_name
        self.device = device
        

    def encode_text_list(self, text_list):
        # sentence transformer
        if self.model_name in ['sentence-transformers/all-mpnet-base-v2',  # Optimized for sentence embeddings
                          'sentence-transformers/paraphrase-mpnet-base-v2',  # Optimized for paraphrase detection
                          'sentence-transformers/all-MiniLM-L6-v2',  # Optimized for sentence embeddings
                          'sentence-transformers/all-MiniLM-L12-v2']:
            
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name, trust_remote_code=True).to(device)
            output = model.encode(text_list) # tensor shape: [obs, embed_dim]
            return output
        
        # BERT-base
        elif self.model_name in ['bert-base-uncased',
                            'bert-base-cased',
                            'bert-large-uncased',
                            'bert-large-cased',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                            'allenai/scibert_scivocab_uncased',
                            'dmis-lab/biobert-base-cased-v1.2']:

            from transformers import BertTokenizer, BertModel
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertModel.from_pretrained(self.model_name)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True,truncation=True, max_length=512)
            output = model(**encoded_input)
            return output.pooler_output #output.last_hidden_state[:, 0, :]
            
        # RoBERTa
        elif self.model_name in ['roberta-base',             # 12-layer
                            'roberta-large',            # 24-layer
                            'allenai/biomed_roberta_base',  # Biomedical domain
                            'roberta-base-openai-detector', # OpenAI's version
                            'microsoft/roberta-base-openai-detector']:

            from transformers import RobertaTokenizer, RobertaModel
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            model = RobertaModel.from_pretrained(self.model_name)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            output = model(**encoded_input)
            return output.pooler_output #output.last_hidden_state[:, 0, :]
            
        
        # DistilBERT
        elif self.model_name in ['distilbert-base-uncased',  # General purpose, uncased
                            'distilbert-base-cased',    # General purpose, cased
                            'distilroberta-base',       # Distilled version of RoBERTa
                            'distilbert-base-multilingual-cased']:
            from transformers import DistilBertTokenizer, DistilBertModel
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            model = DistilBertModel.from_pretrained(self.model_name)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            output = model(**encoded_input)
            return output.last_hidden_state.mean(dim=1) # output.last_hidden_state[:, 0, :]
            
        # MPNet
        elif self.model_name in ['microsoft/mpnet-base',     # Base model
                            'microsoft/mpnet-large']:
            from transformers import AutoTokenizer, MPNetModel
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = MPNetModel.from_pretrained(self.model_name)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            output = model(**encoded_input)
            return output.pooler_output 
        

class TSEncoder():
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = device
        
        # load pretrained TS encoder
        if model_name == 'hr_vae_linear_medium':
            self.model = VAE_Linear_Medium().to(device)
            self.model.load_state_dict(torch.load('./pretrained/'+model_name+'.pth' ))
        if model_name == 'sp_vae_linear_medium':
            self.model = VAE_Linear_Medium().to(device)
            self.model.load_state_dict(torch.load('./pretrained/'+model_name+'.pth' ))
        

class VAE_Linear_Medium(nn.Module):
    def __init__(self, seq_len=300):
        super(VAE_Linear_Medium, self).__init__()
        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(128, 32)
        self.logvar_layer = nn.Linear(128, 32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, seq_len)
        )
     
    def encode(self, x):
        # Linear encoding
        x = self.encoder(x)
        
        # Get latent parameters
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z):
        # Linear decoding
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


