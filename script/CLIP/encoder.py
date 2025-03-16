from config import *
import torch
import torch.nn as nn

# ------- pretrained encoders -------
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
            model = model.to(device)
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
            model = model.to(device)
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
            model = model.to(device)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            output = model(**encoded_input)
            return output.last_hidden_state.mean(dim=1) # output.last_hidden_state[:, 0, :]
            
        # MPNet
        elif self.model_name in ['microsoft/mpnet-base',     # Base model
                            'microsoft/mpnet-large']:
            from transformers import AutoTokenizer, MPNetModel
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = MPNetModel.from_pretrained(self.model_name)
            model = model.to(device) 
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


# ------- custom ts encoders -------
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            # First transformation
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),  # Dropout after activation
            
            # Second transformation
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout)   # Dropout before residual connection
        )
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.activation(x + self.proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm attention
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended
        
        # Pre-norm MLP
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=8, dropout=0.1):
        super().__init__()
        
        layers = [
            # Initial projection
            nn.Linear(ts_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ]
        
        # Add residual blocks
        for i in range(num_blocks):
            layers.append(
                ResidualBlock(
                    hidden_dim, 
                    hidden_dim,  # Keep dimension constant
                    dropout=dropout*(1+i/num_blocks)
                )
            )
        
        # Final projection
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# resnet_encoder = ResNetEncoder(ts_dim=300, output_dim=128)
# model = GeneralBinaryClassifier(resnet_encoder)
# model = CLIPModel(ts_encoder=resnet_encoder, text_encoder=None)



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x) 

class CNNEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, num_channels=[64, 64, 128, 256], kernel_size=5, dropout=0.2):
        """
        CNN encoder for time series.
        
        Args:
            ts_dim (int): Input time series length
            hidden_dim (int): Final hidden dimension
            num_channels (list): Number of channels for each conv layer
            kernel_size (int): Kernel size for conv layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        layers = [Lambda(lambda x: x.unsqueeze(1))]  # Add channel dimension
        in_channels = 1
        
        # Add conv blocks
        for out_channels in num_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        layers.append(nn.Flatten())
        
        # Calculate output dimension
        with torch.no_grad():
            x = torch.zeros(2, ts_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# # CNN Encoder
# cnn = CNNEncoder(
#     ts_dim=300,
#     output_dim=128,
#     num_channels=[32, 64, 128],  # Three conv layers
#     kernel_size=5,
#     dropout=0.2
# )
# model = GeneralBinaryClassifier(cnn)

class MLPEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_hidden_layers=6, dropout=0.2):
        """
        Multi-layer perceptron encoder.
        
        Args:
            ts_dim (int): Input time series length
            output_dim (int): Output dimension
            hidden_dim (int): Hidden dimension
            num_hidden_layers (int): Number of hidden layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        layers = []
        
        if num_hidden_layers > 0:
            # First hidden layer (input projection)
            layers.extend([
                nn.Linear(ts_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
        
            # Hidden layers
            if num_hidden_layers > 1:
                for _ in range(num_hidden_layers - 1):
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    ])
        else:
            hidden_dim = ts_dim
        
        # Final projection - extend instead of append for list of layers
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# # Linear Encoder
# mlp = MLPEncoder(
#     ts_dim=300,
#     output_dim=128,   
#     hidden_dim=128,
#     num_hidden_layers=6,
#     dropout=0.2
# )
# # Create classifiers
# model = GeneralBinaryClassifier(mlp)

class LSTMEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.1, bidirectional=False):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=ts_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional in final dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, ts_dim)
        # If input is (batch_size, ts_dim), unsqueeze to add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Get LSTM output
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state from all layers
        if self.lstm.bidirectional:
            # Concatenate forward and backward last hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
            
        # Project to final dimension
        output = self.projection(hidden)
        
        return output

# Usage:
# lstm_encoder = LSTMEncoder(ts_dim=300, output_dim=128)
# model = GeneralBinaryClassifier(lstm_encoder)
# model = CLIPModel(ts_encoder=lstm_encoder, text_encoder=None)



class MultiLSTMEncoder(nn.Module):
    def __init__(self, 
                 ts_dim, 
                 output_dim, 
                 hidden_dims=[128, 256, 64],  # Multiple LSTM sizes
                 num_layers=2, 
                 dropout=0.1, 
                 bidirectional=False):
        super().__init__()
        
        # Create multiple LSTM modules
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=ts_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            ) for hidden_dim in hidden_dims
        ])
        
        # Total dimension from all LSTMs
        total_lstm_dim = sum(hidden_dim * 2 if bidirectional else hidden_dim 
                           for hidden_dim in hidden_dims)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(total_lstm_dim, total_lstm_dim // 2),
            nn.BatchNorm1d(total_lstm_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(total_lstm_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        self.bidirectional = bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, ts_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Process through each LSTM
        lstm_outputs = []
        for lstm in self.lstms:
            _, (hidden, _) = lstm(x)
            
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1]
                
            lstm_outputs.append(hidden)
        
        # Concatenate all LSTM outputs
        combined = torch.cat(lstm_outputs, dim=1)
        
        # Project to final dimension
        output = self.projection(combined)
        
        return output

# Usage:
# multi_lstm_encoder = MultiLSTMEncoder(
#     ts_dim=300, 
#     output_dim=128,
#     hidden_dims=[128, 256, 64],  # Three LSTMs with different sizes
#     num_layers=2
# )
# model = GeneralBinaryClassifier(multi_lstm_encoder)
# model = CLIPModel(ts_encoder=multi_lstm_encoder, text_encoder=None)


