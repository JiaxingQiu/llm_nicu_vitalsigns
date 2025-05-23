import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ResidualBlock, AddChannelDim
import math

# ------- custom ts decoder_layers -------
class ResNetDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        
        layers = [
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ts_dim),
            nn.LeakyReLU(0.2)
        ])

        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
# # old default decoder layers
# nn.Sequential(
#     nn.Linear(output_dim, 256), # +2 2 for x_mean and x_std
#     nn.LeakyReLU(0.2),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(0.2),
#     nn.Linear(256, ts_dim)
# )

# class ResNetDecoder(nn.Module):
#     def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=8, dropout=0.1):
#         super().__init__()
#         self.ts_dim = ts_dim
#         self.output_dim = output_dim

#         layers = [
#             # Initial projection
#             nn.Linear(output_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout)
#         ]
        
#         # Add residual blocks
#         for i in range(num_blocks):
#             layers.append(
#                 ResidualBlock(
#                     hidden_dim, 
#                     hidden_dim,  # Keep dimension constant
#                     dropout=dropout*(1+i/num_blocks)
#                 )
#             )
        
#         # Final projection
#         layers.extend([
#             nn.Linear(hidden_dim, ts_dim),
#             nn.BatchNorm1d(ts_dim)
#         ])
        
#         self.encoder = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.encoder(x)


class CNNDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, num_channels=[64, 64, 128, 256], kernel_size=5, dropout=0.2):
        """
        CNN decoder for time series.
        
        Args:
            ts_dim (int): Input time series length
            hidden_dim (int): Final hidden dimension
            num_channels (list): Number of channels for each conv layer
            kernel_size (int): Kernel size for conv layers
            dropout (float): Dropout rate
        """
        super().__init__()

        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        layers = [AddChannelDim()]  # Add channel dimension
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
            x = torch.zeros(2, output_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, ts_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

class MLPDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_hidden_layers=6, dropout=0.2):
        """
        Multi-layer perceptron decoder.
        
        Args:
            ts_dim (int): Input time series length
            output_dim (int): Output dimension
            hidden_dim (int): Hidden dimension
            num_hidden_layers (int): Number of hidden layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        layers = []
        
        if num_hidden_layers > 0:
            # First hidden layer (input projection)
            layers.extend([
                nn.Linear(output_dim, hidden_dim),
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
            hidden_dim = output_dim
        
        # Final projection - extend instead of append for list of layers
        layers.extend([
            nn.Linear(hidden_dim, ts_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ])
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class LSTMDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.1, bidirectional=False):
        super().__init__()


        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
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


class MultiLSTMDecoder(nn.Module):
    def __init__(self, 
                 ts_dim, 
                 output_dim, 
                 hidden_dims=[128, 256, 64],  # Multiple LSTM sizes
                 num_layers=2, 
                 dropout=0.1, 
                 bidirectional=False, 
                 mask=-1): # it is the indicator of masked values in the time series (i.e. -1), default to -1
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        # Create multiple LSTM modules
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=output_dim,
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
            nn.Linear(total_lstm_dim // 2, ts_dim),
            nn.BatchNorm1d(ts_dim)
        )
        
        self.bidirectional = bidirectional
        self.mask = mask
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, ts_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if self.mask is not None:
            # Create mask for values != -1
            mask = (x != self.mask).float()
            # Apply mask to input
            x = x * mask # zero out masked values
            
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


class TransformerDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, 
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """Transformer-based decoder for time series reconstruction.
        
        Args:
            ts_dim: Input time series dimension
            output_dim: Latent space dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Project latent vector to transformer input dimension
        self.input_projection = nn.Linear(output_dim, ts_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(ts_dim, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ts_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(ts_dim, ts_dim)
        
    def forward(self, z):
        # # Project latent vector to transformer input dimension
        x = self.input_projection(z)  # [batch_size, ts_dim]
        # x = z
        # Add sequence dimension and repeat for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, ts_dim]
        # Add positional encoding
        x = self.pos_encoder(x)
        # Create memory (same as input for auto-regressive generation)
        memory = x
        # Generate output sequence
        output = self.transformer_decoder(x, memory)
        # Project back to original dimension
        output = self.output_projection(output.squeeze(1))
        return output

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to a tensor of shape
    [batch_size, seq_len, d_model]. d_model = the number of features per position
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                   # [max_len, d_model]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)         # even indices
        pe[:, 1::2] = torch.cos(position * div_term)         # odd  indices
        pe = pe.unsqueeze(0)                                 # [1, max_len, d_model]

        self.register_buffer("pe", pe)                       # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]                         # broadcast on batch dim
        return self.dropout(x)

# old default decoder layers
# TransformerDecoder(ts_dim = ts_dim, 
#                   output_dim = output_dim, 
#                   nhead = 4,
#                   num_layers = 6,
#                   dim_feedforward = 512,
#                   dropout = 0.0)

class AttentionDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, 
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Simple attention-based decoder for time series reconstruction.
        
        Args:
            ts_dim: Input time series dimension
            output_dim: Latent space dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        
        # Multi-head attention for reconstruction
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Trainable query for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, ts_dim)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        k = v = self.norm(z.unsqueeze(1))          # [B,1,E]
        q = self.query.expand(k.size(0), -1, -1)   # [B,1,E]

        attn_out, _ = self.attention(q, k, v)      # [B,1,E]
        out = self.output_proj(attn_out.squeeze(1))  # [B,E] ->[B,ts_dim]
        return out


class TransformerDecoderTXTS(nn.Module):
    """
    Decoder that takes
      • ts_emb  → target  (tgt)
      • txt_emb → memory
    With positional encoding.
    
    Args:
        ts_dim: original time-series dimension
        output_dim: shared embedding size of ts_emb / txt_emb
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: dimension of feedforward network
        dropout: dropout probability
        project_input: whether to project output_dim to ts_dim at the beginning
    """

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding size of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        project_input: bool = False,  # whether to project output_dim to ts_dim at the beginning
    ):
        super().__init__()
        hidden_dim = output_dim

        self.project_input = project_input
        if project_input:
            self.input_projection = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim
            
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
            
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, ts_dim)
        
    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Accept [B,E] or [B,L,E] and return [B,L,E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        """
        ts_emb : [B,E] time-series embeddings  (tgt)
        txt_emb: [B,E] text embeddings        (memory)
        """
        tgt = self._as_sequence(ts_emb) # [B, 1, E]
        memory = self._as_sequence(txt_emb) # [B, 1, E]
        
        # Project input if needed
        if self.project_input:
            tgt = self.input_projection(tgt)
            memory = self.input_projection(memory)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)

        dec_out = self.decoder(tgt=tgt, memory=memory) # [B, 1, E]
        ts_hat = self.output_projection(dec_out) # ts_hat: [B, 1, ts_dim]
        return ts_hat

class TransformerDecoderAuto(nn.Module):
    """
    Autoregressive decoder that takes:
      • ts_emb + txt_emb → memory (concatenated and projected)
      • ts → target (raw time series for autoregressive prediction)
    
    Args:
        ts_dim: original time-series dimension
        output_dim: shared embedding size of ts_emb / txt_emb
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: dimension of feedforward network
        dropout: dropout probability
    """

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding dim of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ts_dim = ts_dim  # Store ts_dim for output projection
        self.memory_feature_dim = output_dim  # Use output_dim as the feature dimension
        
        self.memory_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.memory_query = nn.Parameter(torch.randn(1, output_dim))  # [1, E]
        self.memory_projection = nn.Linear(output_dim, 1)  # [B, T, E] → [B, T, 1]

        # Register causal mask once with diagonal=1 to allow current position
        self.register_buffer(
            "strict_causal_mask",
            torch.triu(torch.ones(5000, 5000), diagonal=1).bool()  # allow current position, block future
        )
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=1,  # Decoder runs in 1D (scalar) space
            nhead=1,    # Must be 1 if d_model=1
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(1, 1)


    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is [B, 1, E] if originally [B, E]"""
        return x.unsqueeze(1) if x.dim() == 2 else x


    def forward(self, ts_emb: torch.Tensor,
                    txt_emb: torch.Tensor,
                    ts:      torch.Tensor):
        
        B, T_full = ts.shape
        if T_full < 2:
            raise ValueError("Need at least 2 timesteps for teacher forcing")

        # ---------- 1.  SHIFT  ----------
        # Prepend ⟨SOS⟩  → length T  (teacher forcing)
        sos = ts[:, :1]         # [B,1]
        ts_in  = torch.cat([sos, ts[:, :-1]], 1)            # [B, T-1 + 1] inputs
        T = ts_in.size(1) # T = T_full now

        # ---------- 2.  MEMORY ----------
        ts_emb = self._as_sequence(ts_emb)         # [B,1,E]
        txt_emb = self._as_sequence(txt_emb)       # [B,1,E]
        memory  = torch.cat([ts_emb, txt_emb], 1)  # [B,2,E]
        
        # Use attention to project memory to [B,T,1]
        memory_query = self.memory_query.unsqueeze(0).expand(B, T, -1)  # [B,T,E]
        
        memory_attn, _ = self.memory_attention(query=memory_query, key=memory, value=memory)  # [B,T,E]
        memory = self.memory_projection(memory_attn)  # [B,T,1]

        # ---------- 3.  TGT EMBEDDING ----------
        tgt = ts_in.unsqueeze(-1)                  # [B,T,1]
        
        # ---------- 4.  CAUSAL MASK ----------
        # allow self, block future
        causal_mask = self.strict_causal_mask[:T, :T].to(ts.device)

        # ---------- 5.  DECODER ----------
        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B,T,1]
        ts_hat  = self.output_projection(dec_out).squeeze(-1)      # [B,T]

        return ts_hat        # return targets so loss can be computed outside

