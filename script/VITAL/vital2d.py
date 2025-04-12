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
                 ts_encoder = None,
                 text_encoder = None,
                 ts_decoder = None,
                 clip_mu = True,
                 concat_embeddings = True):
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
        
        # Handle ts_encoder
        if ts_encoder is None:
            self.ts_encoder = TSVAEEncoder(ts_dim, output_dim) #default encoder
        elif isinstance(ts_encoder, type):
            self.ts_encoder = ts_encoder(ts_dim, output_dim)
        else:
            self.ts_encoder = ts_encoder
        
        # Handle text_encoder with attention
        if text_encoder is None:
            self.text_encoder = TextEncoder(text_dim, output_dim)
        elif isinstance(text_encoder, type):
            self.text_encoder = text_encoder(text_dim, output_dim)
        else:
            self.text_encoder = text_encoder
        
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
    def __init__(self, ts_dim: int, output_dim: int):
        """Time series encoder with progressive architecture
        
        Args:
            ts_dim (int): Input time series dimension
            output_dim (int): Output embedding dimension
        """
        super().__init__()
        self.local_norm = LocalNorm()
        
        self.encoder_layers = nn.Sequential(
            MultiLSTMEncoder(
                ts_dim=ts_dim, 
                output_dim=256,
                hidden_dims=[512, 512, 256, 256],  # LSTMs with different sizes
                num_layers=2,
                mask=0  # mask 0 with 0 to suppress the effect of masked values
            ),
            nn.LayerNorm(256),  # Add LayerNorm at the end
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2), 
            # nn.LayerNorm(512),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2), 
            # nn.LayerNorm(512),
            nn.Linear(512, output_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(output_dim)
        )
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
    def __init__(self, ts_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, ts_dim)
        )
    
    def forward(self, z, x_mean, x_std):
        x_hat = self.decoder(z)
        # scale back to raw scale
        x_hat = x_hat * x_std + x_mean
        # round to 0 decimal places
        # x_hat = torch.round(x_hat)
        return x_hat

# # default text encoder
# class TextEncoder(nn.Module):
#     def __init__(self, 
#                  text_dim: int, 
#                  output_dim: int,
#                  dropout: float = 0.0):
#         """Text encoder using transformer blocks
        
#         Args:
#             text_dim (int): Input text embedding dimension
#             output_dim (int): Output embedding dimension
#         """
#         super().__init__()
#         self.encoder = nn.Sequential(
#             # Initial projection
#             nn.Linear(text_dim, 512),
#             # nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(dropout),
            
#             # Transformer blocks
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
            
#             # Final projection
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, text_features):
#         """Forward pass
        
#         Args:
#             text_features (torch.Tensor): Input tensor of shape [batch_size, text_dim]
            
#         Returns:
#             torch.Tensor: Encoded text of shape [batch_size, output_dim]
#         """

#         tx_emb = self.encoder(text_features)
#         tx_emb = F.normalize(tx_emb, dim=1)
#         return tx_emb


# Simple text encoder
class TextEncoder(nn.Module):
    def __init__(self, 
                 text_dim: int, 
                 output_dim: int):
        super().__init__()
        
        # Simple MLP with 3 layers
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, text_features):
        tx_emb = self.encoder(text_features)
        tx_emb = F.normalize(tx_emb, dim=1)
        return tx_emb

# ts vae encoder wrapper for custom ts encoders
class TSVAEEncoderWrapper(nn.Module):
    def __init__(self, ts_encoder_layers, output_dim=None):
        super().__init__()
        self.local_norm = LocalNorm()

        # ts_encoder_layers must be a instance of a class defined similarly as belows
        self.encoder_layers = ts_encoder_layers
        
        # Get the hidden dimension from the encoder's output
        hidden_dim = ts_encoder_layers.output_dim
        if output_dim is None:
            output_dim = hidden_dim
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)


    def reparameterization(self, mean, log_var, lam=1):
        var = lam * torch.exp(0.5 * log_var) # calculate the variance from the log_var 
        # var = log_var
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon  # Using var directly
        # z = F.softmax(z,dim=1) # This variable follows a Dirichlet distribution
        return z
    
    
    def forward(self, x):
        _, x_mean, x_std = self.local_norm(x)

        #  ---- encode -----
        x_encoded = self.encoder_layers(x)
        mean = self.mean_layer(x_encoded)
        log_var = self.logvar_layer(x_encoded)

        #  ---- reparameterization -----
        z = self.reparameterization(mean, log_var)
        return z, mean, log_var, x_mean, x_std

# usage:
"""
lstm_encoder = MultiLSTMEncoder(
    ts_dim=300, 
    output_dim=128,
    hidden_dims=[128, 256, 64],  # Three LSTMs with different sizes
    num_layers=2
)
ts_encoder = TSVAEEncoderWrapper(lstm_encoder)
for batch_idx, (ts_features, text_features, labels) in enumerate(train_dataloader):

    ts_features = ts_features.to(device)
    print(ts_encoder(ts_features))
    break
"""


# ts vae decoder wrapper for custom ts decoders
class TSVAEDecoderWrapper(nn.Module):
    def __init__(self, ts_decoder):
        super().__init__()
        self.decoder = ts_decoder
        
    def forward(self, z, x_mean, x_std):
        x_hat = self.decoder(z)
        x_hat = x_hat * x_std + x_mean
        # x_hat = torch.round(x_hat)
        return x_hat



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