import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from config import *
from model_utils.encoder import *
from model_utils.decoder import *

# Verbally Instructed Time series Augmentation Learning (VITAL)  (3d)
class VITAL3D(nn.Module):
    def __init__(self, 
                 ts_dim: int, 
                 text_dim: int, 
                 n_text: int,
                 output_dim: int,
                 temperature: float = 0.01,
                 ts_encoder = None,
                 text_encoder = None,
                 ts_decoder = None):
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
            self.text_encoder = TextEncoderWithAttention(text_dim, n_text, output_dim)
        elif isinstance(text_encoder, type):
            self.text_encoder = text_encoder(text_dim, n_text, output_dim)
        else:
            self.text_encoder = text_encoder
        
        if ts_decoder is None:
            self.ts_decoder = TSVAEDecoder(ts_dim = ts_dim, output_dim = output_dim) #default decoder
        elif isinstance(ts_decoder, type):
            self.ts_decoder = ts_decoder(ts_dim = ts_dim, output_dim = output_dim)
        else:
            self.ts_decoder = ts_decoder
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.device = device
        self.to(device)
        print(nn_summary(self))

    def forward(self, ts, text_features_list):
        """Forward pass of the VITAL model.
        
        Args:
            ts (torch.Tensor): Globally normalized time series of shape [batch_size, ts_dim].
            text_features_list (List[torch.Tensor]): List of text features from a pretrained encoder.
                Each tensor in the list has shape [batch_size, text_dim].
                
        Returns:
            tuple:
                - logits (torch.Tensor): Similarity scores of shape [batch_size, batch_size]
                - x_hat (torch.Tensor): Reconstructed time series of shape [batch_size, ts_dim]
                - mean (torch.Tensor): Latent mean of shape [batch_size, output_dim]
                - log_var (torch.Tensor): Latent log variance of shape [batch_size, output_dim]
        """

        # ---- VAE encoder ----
        # Encode time series
        z, mean, log_var, x_mean, x_std = self.ts_encoder(ts) # ts in raw scale

        # --- CLIP forward pass ---
        ts_embedded = F.normalize(z, dim=1)
        text_embedded, _ = self.text_encoder(text_features_list)
        text_embedded = F.normalize(text_embedded, dim=1)
        logits = torch.matmul(ts_embedded, text_embedded.T) * torch.exp(self.temperature)

        # --- VAE decoder forward pass ---
        ts_hat = self.ts_decoder(z, x_mean, x_std) # ts_hat in raw scale
        
        return logits, ts_hat, mean, log_var

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
                 ts_decoder = None):
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
        
        if ts_decoder is None:
            self.ts_decoder = TSVAEDecoder(ts_dim = ts_dim, output_dim = output_dim) #default decoder
        elif isinstance(ts_decoder, type):
            self.ts_decoder = ts_decoder(ts_dim = ts_dim, output_dim = output_dim)
        else:
            self.ts_decoder = ts_decoder
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.device = device
        self.to(device)
        print(nn_summary(self))
        self.beta = beta
    
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

        # --- CLIP forward pass ---
        ts_embedded = F.normalize(z, dim=1)
        text_embedded = self.text_encoder(text_features)
        text_embedded = F.normalize(text_embedded, dim=1)
        logits = torch.matmul(ts_embedded, text_embedded.T) * torch.exp(self.temperature)

        # --- VAE decoder forward pass ---
        ts_hat = self.ts_decoder(z, x_mean, x_std)

        return logits, ts_hat, mean, log_var


class TSVAEEncoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int):
        """Time series encoder with progressive architecture
        
        Args:
            ts_dim (int): Input time series dimension
            output_dim (int): Output embedding dimension
        """
        super().__init__()
        self.encoder_layers = MultiLSTMEncoder(
                            ts_dim=ts_dim, 
                            output_dim=output_dim,
                            hidden_dims=[256, 256, 256],  # LSTMs with different sizes
                            num_layers=2,
                            mask=0# mask 0 with 0 to suppress the effect of masked values
                        )
        # Latent mean and variance 
        self.mean_layer = nn.Linear(output_dim, output_dim)
        self.logvar_layer = nn.Linear(output_dim, output_dim)
    
    def reparameterization(self, mean, log_var):
        # var = torch.exp(0.5 * log_var) # slower than using log_var directly
        var = log_var
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon  # Using var directly
        # z = F.softmax(z,dim=1) # This variable follows a Dirichlet distribution
        return z
    
    def forward(self, x):
        x_mean = 0
        x_std = 1

        #  ---- encode -----
        x_encoded = self.encoder_layers(x)
        mean = self.mean_layer(x_encoded)
        log_var = self.logvar_layer(x_encoded)

        #  ---- reparameterization -----
        # z = self.reparameterization(mean, log_var)
        z = x_encoded
        
        return z, mean, log_var, x_mean, x_std
  


# # default ts vae encoder
# class LocalNorm(nn.Module):
#     def __init__(self, eps=1e-5):
#         super().__init__()
#         self.eps = eps
    
#     def forward(self, x):
#         # Compute mean and std along feature dimension
#         mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1]
#         std = x.std(dim=1, keepdim=True)    # [batch_size, 1]
        
#         # Normalize
#         x_norm = (x - mean) / (std + self.eps)
        
#         return x_norm, mean, std
# class TSVAEEncoder(nn.Module):
#     def __init__(self, ts_dim: int, output_dim: int):
#         """Time series encoder with progressive architecture
        
#         Args:
#             ts_dim (int): Input time series dimension
#             output_dim (int): Output embedding dimension
#         """
#         super().__init__()


#         # Simple normalization without learnable parameters
#         self.local_norm = LocalNorm()

#         self.encoder_layers = nn.Sequential(
#             nn.Linear(ts_dim, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.2)
#         )
#         # Latent mean and variance 
#         self.mean_layer = nn.Linear(128, output_dim)
#         self.logvar_layer = nn.Linear(128, output_dim)
    
#     def reparameterization(self, mean, log_var):
#         # var = torch.exp(0.5 * log_var) # slower than using log_var directly
#         var = log_var
#         epsilon = torch.randn_like(var).to(device)      
#         z = mean + var*epsilon  # Using var directly
#         # z = F.softmax(z,dim=1) # This variable follows a Dirichlet distribution
#         return z
    
#     def forward(self, x):
#         x, x_mean, x_std = self.local_norm(x)

#         #  ---- encode -----
#         x_encoded = self.encoder_layers(x)
#         mean = self.mean_layer(x_encoded)
#         log_var = self.logvar_layer(x_encoded)

#         #  ---- reparameterization -----
#         z = self.reparameterization(mean, log_var)
        
#         return z, mean, log_var, x_mean, x_std

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
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, ts_dim)
        )
    
    def forward(self, z, x_mean, x_std):
        x_hat = self.decoder(z)
        # scale back to raw scale
        x_hat = x_hat * x_std + x_mean
        return x_hat

# default text encoder
class TextEncoder(nn.Module):
    def __init__(self, 
                 text_dim: int, 
                 output_dim: int):
        """Text encoder using transformer blocks
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            # Initial projection
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Transformer blocks
            TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=0.1),
            TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=0.1),
            TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=0.1),
            
            # Final projection
            nn.Linear(512, output_dim)
        )

    def forward(self, text_features):
        """Forward pass
        
        Args:
            text_features (torch.Tensor): Input tensor of shape [batch_size, text_dim]
            
        Returns:
            torch.Tensor: Encoded text of shape [batch_size, output_dim]
        """
        return self.encoder(text_features)

# default text encoder with attention on text splits
class TextEncoderWithAttention(nn.Module):
    def __init__(self, text_dim, n_text, output_dim, encoder_config=None):
        super().__init__()

        # Default encoder configurations
        self.encoder_config = {
            '5': {
                'hidden_dim': 128,
                'n_heads': 1,
                'n_layers': 1,
                'dropout': 0.1
            },
            '4': {
                'hidden_dim': 256,
                'n_heads': 1,
                'n_layers': 1,
                'dropout': 0.1
            },
            '3': {
                'hidden_dim': 256,
                'n_heads': 2,
                'n_layers': 1,
                'dropout': 0.1
            },
            '2': {
                'hidden_dim': 1024,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.1
            },
            '1': {
                'hidden_dim': 1024,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1
            }
        } if encoder_config is None else encoder_config
        
        # Select configuration based on n_text
        self.config = self.encoder_config[str(min(n_text, 5))]

        # Create n_text copies of the encoder
        self.text_encoders = nn.ModuleList([
            self._single_text_encoder(text_dim, output_dim, self.config) 
            for _ in range(n_text)
        ])
        
        # Attention mechanism to combine text features
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
    
    def _single_text_encoder(self, text_dim, output_dim, config, hidden_dim=256):
        layers = [
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        ]
        # Add transformer blocks
        for _ in range(config['n_layers']):
            layers.append(
                TransformerBlock(
                    dim=hidden_dim,
                    hidden_dim=config['hidden_dim'],
                    num_heads=config['n_heads'],
                    dropout=config['dropout']
                )
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    

    def forward(self, text_features):
        # Encode each text input
        encoded_texts = []
        for i, text in enumerate(text_features):
            encoded = self.text_encoders[i](text)  # [batch_size, output_dim]
            encoded_texts.append(encoded)
        
        # Stack encoded texts [batch_size, n_text, output_dim]
        encoded_texts = torch.stack(encoded_texts, dim=1)
        
        # Expand query for batch size
        query = self.query.expand(encoded_texts.size(0), -1, -1)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=encoded_texts,  # [batch_size, n_text, output_dim]
            value=encoded_texts  # [batch_size, n_text, output_dim]
        )
        
        # Remove the query dimension
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output, attention_weights
    
# # Initialize
# text_encoder = TextEncoderWithAttention(
#     text_dim=768,    # BERT embedding size
#     n_text=3,        # Number of text inputs
#     output_dim=128   # Final output dimension
# )

# # Forward pass
# text_features = [text1, text2, text3]  # List of text tensors
# combined_output, attention_weights = text_encoder(text_features)

# # Visualize attention weights if needed
# import matplotlib.pyplot as plt
# plt.imshow(attention_weights[0].detach().cpu())
# plt.colorbar()
# plt.show()



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

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Convert log_var to std
        epsilon = torch.randn_like(std).to(device)
        z = mean + std * epsilon
        return z
    
    def forward(self, x):
        x, x_mean, x_std = self.local_norm(x)

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
        return x_hat
