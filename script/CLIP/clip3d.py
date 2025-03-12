import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from config import *
from data import *
from encoder import *



class CLIP3DModel(nn.Module):
    def __init__(self, 
                 ts_dim, 
                 text_dim, 
                 n_text,
                 output_dim=128,
                 temperature=0.01,
                 ts_encoder=None,
                 text_encoder=None):
        super().__init__()
        
        # Handle ts_encoder
        if ts_encoder is None:
            self.W_ts = self._default_ts_encoder(ts_dim, output_dim)
        elif isinstance(ts_encoder, type):
            self.W_ts = ts_encoder(ts_dim, output_dim)
        else:
            self.W_ts = ts_encoder
        
        # Handle text_encoder with attention
        if text_encoder is None:
            self.W_t = TextEncoderWithAttention(text_dim, n_text, output_dim)
        elif isinstance(text_encoder, type):
            self.W_t = text_encoder(text_dim, n_text, output_dim)
        else:
            self.W_t = text_encoder
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.device = device
        self.to(device)
        print(nn_summary(self))
    
    def _default_ts_encoder(self, ts_dim, output_dim):
        """Default time series encoder"""
        return nn.Sequential(
            nn.Linear(ts_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            ResidualBlock(128, 256, dropout=0.1),
            ResidualBlock(128, 256, dropout=0.1),
            ResidualBlock(128, 512, dropout=0.2),
            ResidualBlock(128, 512, dropout=0.2),
            ResidualBlock(128, 1024, dropout=0.3),
            ResidualBlock(128, 1024, dropout=0.3),
            ResidualBlock(128, 2048, dropout=0.4),
            ResidualBlock(128, 2048, dropout=0.4),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, ts_features, text_features_list, return_attn_weights=False):
        """
        Args:
            ts_features: tensor of shape [batch_size, ts_dim]
            text_features_list: list of n_text tensors, each of shape [batch_size, text_dim]
        Returns:
            logits: similarity scores
            ts_embedded: normalized time series embeddings
            text_embedded: normalized combined text embeddings
            attention_weights: attention weights for text combination (not return during training)
        """
        # Encode time series
        ts_embedded = F.normalize(self.W_ts(ts_features), dim=1)
        
        # Encode and combine text features with attention
        text_embedded, attention_weights = self.W_t(text_features_list)
        text_embedded = F.normalize(text_embedded, dim=1)
        
        # Calculate similarity
        logits = torch.matmul(ts_embedded, text_embedded.T) * torch.exp(self.temperature)
        
        if return_attn_weights:
            return logits, ts_embedded, text_embedded, attention_weights
        else:
            return logits, ts_embedded, text_embedded

# # Initialize model
# model = CLIP3DModel(
#     ts_dim=300,
#     text_dim=768,
#     n_text=3,
#     output_dim=128
# )

# # Forward pass
# ts_features = torch.randn(32, 300)  # [batch_size, ts_dim]
# text_features = [
#     torch.randn(32, 768),  # text1 [batch_size, text_dim]
#     torch.randn(32, 768),  # text2
#     torch.randn(32, 768)   # text3
# ]

# logits, ts_emb, text_emb, attn_weights = model(ts_features, text_features)

# # Visualize attention weights
# import matplotlib.pyplot as plt
# plt.imshow(attn_weights[0].detach().cpu())
# plt.title('Attention Weights')
# plt.colorbar()
# plt.show()

# custom_config = {
#     'small': {'hidden_dim': 256, 'n_heads': 2, 'n_layers': 1, 'dropout': 0.1},
#     'medium': {'hidden_dim': 512, 'n_heads': 4, 'n_layers': 2, 'dropout': 0.1},
#     'large': {'hidden_dim': 1024, 'n_heads': 8, 'n_layers': 3, 'dropout': 0.1}
# }
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
                'hidden_dim': 512,
                'n_heads': 4,
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
        self.config = self.encoder_config[str(n_text)]

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
        
    
    def _single_text_encoder(self, text_dim, output_dim, config):
        layers = [
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        ]
        # Add transformer blocks
        for _ in range(config['n_layers']):
            layers.append(
                TransformerBlock(
                    dim=512,
                    hidden_dim=config['hidden_dim'],
                    num_heads=config['n_heads'],
                    dropout=config['dropout']
                )
            )
        layers.append(nn.Linear(512, output_dim))
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