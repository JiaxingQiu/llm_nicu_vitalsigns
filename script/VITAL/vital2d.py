import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from config import *
from model_utils.encoder import *
from model_utils.decoder import *

# VITAL model (2d)
class VITAL(nn.Module):
    def __init__(self, 
                 ts_dim: int, 
                 text_dim: int, 
                 output_dim: int,
                 temperature: float = 0.01,
                 beta: float = 1.0,
                 ts_encoder_custom = None,
                 text_encoder_type = 'mlp',
                 ts_decoder = None,
                 variational = True,
                 clip_mu = False,
                 concat_embeddings = False):
        """Initialize VITAL model
        
        Args:
            ts_dim: Input time series dimension
            text_dim: Input text embedding dimension
            output_dim: Latent space dimension
            temperature: Temperature for similarity scaling
            beta: Weight for VAE loss
            ts_encoder: Optional custom time series encoder
            text_encoder: Optional custom text encoder
            ts_decoder: Optional custom time series decoder
        """

        super().__init__()
        
        # ts_encoder
        self.ts_encoder = TSVAEEncoder(ts_dim, output_dim, encoder_layers = ts_encoder_custom) #default encoder
        
        # text_encoder
        self.text_encoder = TextEncoder(text_dim, output_dim, text_encoder_type = text_encoder_type)
        
        # ts_decoder
        if concat_embeddings:
            decode_dim = 2*output_dim
        else:
            decode_dim = output_dim
        if ts_decoder is None:
            self.ts_decoder = TSVAEDecoder(ts_dim = ts_dim, output_dim = decode_dim) #default decoder
        elif isinstance(ts_decoder, type):
            self.ts_decoder = ts_decoder(ts_dim = ts_dim, output_dim = decode_dim)
        else:
            self.ts_decoder = ts_decoder
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.device = device
        self.beta = beta
        self.clip_mu = clip_mu
        self.concat_embeddings = concat_embeddings
        self.variational = variational
        self.to(device)
        print(nn_summary(self))
    
    def clip(self, ts_embedded, text_embedded):
        # ts_embedded = F.normalize(ts_embedded, dim=1)
        # text_embedded = F.normalize(text_embedded, dim=1)
        logits = torch.matmul(ts_embedded, text_embedded.T) #* torch.exp(self.temperature)
        return logits
    
    def forward(self, ts, text_features):
        """Forward pass of the VITAL model.
        
        Args:
            ts: Globally normalized time series of shape [batch_size, ts_dim]
            text_features: Text features from pretrained encoder of shape [batch_size, text_dim]
                
        Returns:
            tuple:
                - logits: Similarity scores of shape [batch_size, batch_size]
                - ts_hat: Reconstructed time series of shape [batch_size, ts_dim]
                - mean: Latent mean of shape [batch_size, output_dim]
                - log_var: Latent log variance of shape [batch_size, output_dim]
        """

        # ---- VAE encoder ----
        z, mean, log_var, x_mean, x_std = self.ts_encoder(ts) # ts in raw scale
        if not self.variational: # if not variational (AE instead of VAE), use the mean as the latent variable, 
            z = mean
        
        # --- Text encoder forward pass ---
        text_embedded = self.text_encoder(text_features)
        
        # --- CLIP forward pass ---
        if self.clip_mu:
            logits = self.clip(mean, text_embedded)
        else:
            logits = self.clip(z, text_embedded)
        
        # --- VAE decoder forward pass ---
        if self.concat_embeddings:
            # concate z and text_embedded 
            decode_input = torch.cat([z, text_embedded], dim=1)
        else:
            decode_input = z
        ts_hat = self.ts_decoder(decode_input, x_mean, x_std)

        return logits, ts_hat, mean, log_var

class LocalNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        # Compute mean and std along feature dimension
        mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std = x.std(dim=1, keepdim=True)    # [batch_size, 1]
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        return x_norm, mean, std

class TSVAEEncoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, encoder_layers = None):
        """Time series encoder with progressive architecture
        
        Args:
            ts_dim (int): Input time series dimension
            output_dim (int): Output embedding dimension
        """
        super().__init__()
        self.local_norm = LocalNorm()
        if encoder_layers is None:
            # default encoder layers
            self.encoder_layers = nn.Sequential(
                MultiLSTMEncoder(
                    ts_dim=ts_dim, 
                    output_dim=256,
                    hidden_dims=[512, 512, 256, 256],  # LSTMs with different sizes
                    num_layers=2,
                    dropout=0,
                    bidirectional=False,
                    mask=0  # mask 0 with 0 to suppress the effect of masked values
                ),
                nn.LayerNorm(256),  # Add LayerNorm at the end
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(512),
                nn.Linear(512, output_dim),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(output_dim)
            )
        else:
            self.encoder_layers = encoder_layers # pass an instance of custom encoder layers from classes in the encoder module
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(output_dim, output_dim)
        self.logvar_layer = nn.Linear(output_dim, output_dim)
    
    def reparameterization(self, mean, log_var, ep=1):
        var = ep * torch.exp(0.5 * log_var) # slower than using log_var directly
        # var = log_var
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon  # Using var directly
        # z = F.softmax(z, dim=1) # This variable follows a Dirichlet distribution
        return z
    
    def forward(self, x):
        
        _, x_mean, x_std =  self.local_norm(x)
        #  ---- encode -----
        x_encoded = self.encoder_layers(x)
        mean = self.mean_layer(x_encoded)
        mean = F.normalize(mean, dim=1) 
        
        log_var = self.logvar_layer(x_encoded)

        #  ---- reparameterization -----
        z = self.reparameterization(mean, log_var)
        # z = F.normalize(z, dim=1)
        
        return z, mean, log_var, x_mean, x_std

# default ts vae decoder
class TSVAEDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, decoder_layers = None):
        super().__init__()
        if decoder_layers is None:
            self.decoder = nn.Sequential(
                nn.Linear(output_dim, 256),
                nn.LeakyReLU(0.2),
                # nn.Linear(256, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 512),
                # nn.LeakyReLU(0.2),
                # nn.Linear(512, 256),
                # nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, ts_dim)
            )
        else:
            self.decoder = decoder_layers
    
    def forward(self, z, x_mean, x_std):
        x_hat = self.decoder(z)
        # scale back to raw scale
        x_hat = x_hat * x_std + x_mean
        # round to 0 decimal places
        # x_hat = torch.round(x_hat)
        return x_hat

class TextEncoder(nn.Module):
    def __init__(self, text_dim: int, output_dim: int, text_encoder_type = 'mlp'):
        """Text encoder that can use either MLP or CNN architecture.
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
            text_encoder_type (str): Type of encoder to use ('mlp' or 'cnn')
        """
        super().__init__()
        
        if text_encoder_type == 'mlp':
            self.text_encoder = TextEncoderMLP(text_dim, output_dim)
        elif text_encoder_type == 'cnn':
            self.text_encoder = TextEncoderCNN(output_dim)
        else:
            raise ValueError(f"Unknown text encoder type: {text_encoder_type}")
        
    def forward(self, text_features):
        tx_emb = self.text_encoder(text_features)
        tx_emb = F.normalize(tx_emb, dim=1)
        return tx_emb

# # test text encoder
# # Initialize text encoder
# text_dim = 768  # typical BERT dimension
# output_dim = 10
# text_encoder = TextEncoder(text_dim, output_dim)
# # Create a single random vector
# single_vector = torch.randn(text_dim)
# # Create a batch of repeated vectors
# batch_size = 4
# repeated_batch = single_vector.unsqueeze(0).repeat(batch_size, 1)
# # Create a mixed batch where first element is single_vector and rest are random
# mixed_batch = torch.cat([
#     single_vector.unsqueeze(0),  # First element is single_vector
#     torch.randn(batch_size - 1, text_dim)  # Rest are random
# ], dim=0)
# # Process single vector
# single_output = text_encoder(single_vector.unsqueeze(0))[0]  # Take first element since encoder expects batch
# # Process repeated batch
# repeated_output = text_encoder(repeated_batch)
# # Process mixed batch
# mixed_output = text_encoder(mixed_batch)
# print(single_output)
# print(repeated_output[0])
# print(mixed_output[0])