# from config import *
import torch
import torch.nn as nn
from .helper import CNNEncoder

class MultiCNNEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, kernel_sizes=[80, 50, 20, 5], hidden_num_channel=16, dropout=0.0):
        """
        Multi-resolution CNN encoder with attention mechanism.
        
        Args:
            ts_dim (int): Input time series length
            output_dim (int): Output embedding dimension
            kernel_sizes (list): Different kernel sizes for multi-resolution analysis
            hidden_num_channel (int): Number of channels in CNN layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        # Create CNNs for different resolutions
        self.cnns = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.cnns.append(CNNEncoder(ts_dim, 
                                      output_dim,
                                      num_channels=[hidden_num_channel],
                                      kernel_size=kernel_size, 
                                      dropout=dropout))
        
        # Attention mechanism to combine different resolutions
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Get embeddings from each CNN
        cnn_embeddings = []
        for cnn in self.cnns:
            embedding = cnn(x)  # [batch_size, output_dim]
            cnn_embeddings.append(embedding)
        
        # Stack embeddings from different resolutions [batch_size, n_kernels, output_dim]
        cnn_embeddings = torch.stack(cnn_embeddings, dim=1)
        
        # Apply layer normalization
        cnn_embeddings = self.layer_norm(cnn_embeddings)
        
        # Expand query for batch size
        query = self.query.expand(cnn_embeddings.size(0), -1, -1)
        
        # Apply attention to combine different resolutions
        attended_output, _ = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=cnn_embeddings,  # [batch_size, n_kernels, output_dim]
            value=cnn_embeddings  # [batch_size, n_kernels, output_dim]
        )
        
        # Remove the query dimension and get final embedding
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output

# old default encoder layers
# MultiCNNEncoder(ts_dim = ts_dim,
#                 output_dim=output_dim,
#                 kernel_sizes=[150, 100, 50, 10],
#                 hidden_num_channel=16,
#                 dropout=0)


# ------- custom text encoder_layers (for VITAL embedding generation) -------
class IdenticalEncoder(nn.Module):
    def __init__(self, text_dim: int, output_dim: int):
        super().__init__()
        if text_dim != output_dim:
            raise ValueError(
                f"Identity mapping requires text_dim == output_dim "
                f"(got {text_dim} vs {output_dim})."
            )
    def forward(self, text_features):
        return text_features


class TextEncoderMultiCNN(nn.Module):
    def __init__(self, 
                 text_dim: int,
                 output_dim: int,
                 kernel_sizes=[500, 250, 100, 50],  # Different context windows
                 hidden_num_channel=16,
                 dropout=0.0):
        """
        Multi-resolution CNN encoder for text features with attention mechanism.
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
            kernel_sizes (list): Different kernel sizes for multi-resolution analysis
            hidden_num_channel (int): Number of channels in CNN layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Create CNNs for different resolutions
        self.cnns = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.cnns.append(
                TextEncoderCNN(
                    text_dim=text_dim,
                    output_dim=output_dim,
                    num_channels=[hidden_num_channel],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        
        # Attention mechanism to combine different resolutions
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, text_features):
        # Get embeddings from each CNN
        cnn_embeddings = []
        for cnn in self.cnns:
            embedding = cnn(text_features)  # [batch_size, output_dim]
            cnn_embeddings.append(embedding)
        
        # Stack embeddings from different resolutions [batch_size, n_kernels, output_dim]
        cnn_embeddings = torch.stack(cnn_embeddings, dim=1)
        
        # Apply layer normalization
        cnn_embeddings = self.layer_norm(cnn_embeddings)
        
        # Expand query for batch size
        query = self.query.expand(cnn_embeddings.size(0), -1, -1)
        
        # Apply attention to combine different resolutions
        attended_output, _ = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=cnn_embeddings,  # [batch_size, n_kernels, output_dim]
            value=cnn_embeddings  # [batch_size, n_kernels, output_dim]
        )
        
        # Remove the query dimension and get final embedding
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output

# Usage example:
# old default text encoder layer 
# TextEncoderMultiCNN(text_dim = text_dim,
#                     output_dim=output_dim,
#                     kernel_sizes=[768, 384, 192],  # Different context windows
#                     hidden_num_channel=16,
#                     dropout=0.0)

class TextEncoderAttention(nn.Module):
    """
    A minimal text encoder that summarizes a sequence of token embeddings
    with a single trainable–query multi-head attention layer.

    Args
    ----
    text_dim   : int  – dimension of the incoming token embeddings
    output_dim : int  – dimension of the fixed-length text representation
    num_heads  : int  – number of attention heads (≥ 1)
    dropout    : float
    """

    def __init__(
        self,
        text_dim: int,
        output_dim: int,
        num_heads: int = 1, # single head
        dropout: float = 0.0,
    ):
        super().__init__()

        # Single-layer (self-)attention that pools the whole sequence.
        self.attn = nn.MultiheadAttention(
            embed_dim=text_dim,  # Use text_dim directly
            num_heads=num_heads,
            batch_first=True,  # (B, L, D) I/O layout
        )

        # Single trainable query that "asks" the sequence for information.
        self.query = nn.Parameter(torch.randn(1, 1, text_dim))  # Match text_dim

        # Project to output dimension after attention
        self.output_proj = nn.Linear(text_dim, output_dim)
        self.gelu = nn.GELU()

        # Normalization & dropout for a touch of stability.
        # self.norm = nn.LayerNorm(text_dim)  # Normalize text_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text_features : Tensor, shape (batch, text_dim) or (batch, seq_len, text_dim)
            Token embeddings (e.g. from a pretrained word / sentence encoder).
            If 2D, will be treated as a single-token sequence.

        Returns
        -------
        pooled_repr : Tensor, shape (batch, output_dim)
            Fixed-length embedding summarizing the whole sequence.
        """
        # Handle 2D input by adding sequence length dimension
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # (B, 1, D)

        # Normalize input tokens
        tokens = text_features # self.norm(text_features)  # (B, L, D)

        # Expand the shared query to the current batch size.
        q = self.query.expand(tokens.size(0), -1, -1)  # (B, 1, D)

        # Attention pooling.
        pooled, _ = self.attn(query=q, key=tokens, value=tokens)  # (B, 1, D)
        pooled = pooled.squeeze(1)                               # (B, D)

        # Project to output dimension
        pooled = self.output_proj(pooled)  # (B, output_dim)
        pooled = self.gelu(pooled)
        pooled = self.dropout(pooled)

        return pooled


# usage:
# text_encoder = TextEncoderAttention(
#     text_dim=768,  # e.g., BERT embedding dimension
#     output_dim=128,
#     num_heads=4,
#     dropout=0.1
# )
