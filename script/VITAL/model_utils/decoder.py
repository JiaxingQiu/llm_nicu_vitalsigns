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

# class TransformerDecoderAuto(nn.Module):
#     """
#     Autoregressive decoder that takes:
#       • ts_emb + txt_emb → memory (concatenated and projected)
#       • ts → target (raw time series for autoregressive prediction)
    
#     Args:
#         ts_dim: original time-series dimension
#         output_dim: shared embedding size of ts_emb / txt_emb
#         nhead: number of attention heads
#         num_layers: number of transformer layers
#         dim_feedforward: dimension of feedforward network
#         dropout: dropout probability
#     """

#     def __init__(
#         self,
#         ts_dim: int,           # original time-series dimension
#         output_dim: int,          # shared embedding dim of ts_emb / txt_emb
#         nhead: int = 8,
#         num_layers: int = 6,
#         dim_feedforward: int = 2048,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
        
#         self.ts_dim = ts_dim  # time series length
#         self.output_dim = output_dim  # embeddings' dimension
#         self.d_model = nhead * 2 # if set to 1, decoder runs in 1D (scalar) space (i.e. univariate time series)
        

#         # self.memory_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=nhead, dropout=dropout, batch_first=True)
#         self.memory_query = nn.Parameter(torch.randn(1, output_dim))  # [1, E]
#         self.mem_proj = nn.Linear(output_dim * 2, self.d_model)  # [B, T, E] → [B, T, d_model]
#         self.tgt_proj = nn.Linear(1, self.d_model)  # project tgt to d_model

       
#        # Register causal mask once with diagonal=1 to allow current position
#         self.register_buffer("strict_causal_mask", torch.triu(torch.ones(5000, 5000), diagonal=1).bool())
        
#         self.decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.d_model,  # Decoder runs in 1D (scalar) space
#             nhead=nhead,    # Must be 1 if d_model=1
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
#         self.output_projection = nn.Linear(self.d_model, 1)


#     @staticmethod
#     def _as_sequence(x: torch.Tensor) -> torch.Tensor:
#         """Ensure input is [B, 1, E] if originally [B, E]"""
#         return x.unsqueeze(1) if x.dim() == 2 else x


#     def forward(self, ts_emb: torch.Tensor,
#                     txt_emb: torch.Tensor,
#                     ts:      torch.Tensor):
        
#         B, T_full = ts.shape
#         if T_full < 2:
#             raise ValueError("Need at least 2 timesteps for teacher forcing")

#         # ---------- 1.  SHIFT  ----------
#         sos = ts[:, :1]         # [B,1] Prepend ⟨SOS⟩
#         ts_in  = torch.cat([sos, ts[:, :-1]], 1)            # [B, T]
#         T = ts_in.size(1) # T = T_full

#         # ---------- 2.  MEMORY ----------
#         # Repeat ts/text embeddings across time steps
#         ts_emb_tiled  = ts_emb.unsqueeze(1).repeat(1, T, 1)   # [B,T,E]
#         txt_emb_tiled = txt_emb.unsqueeze(1).repeat(1, T, 1)  # [B,T,E]

#         memory_cat = torch.cat([ts_emb_tiled, txt_emb_tiled], dim=2)  # [B,T,2E]
#         memory     = self.mem_proj(memory_cat)                        # [B,T,d_model]
        
#         # ---------- 3.  TGT EMBEDDING ----------
#         tgt = self.tgt_proj(ts_in.unsqueeze(-1))                  # [B,T,d_model]
        
#         # ---------- 4.  CAUSAL MASK ----------
#         # allow self, block future
#         causal_mask = self.strict_causal_mask[:T, :T].to(ts.device)

#         # ---------- 5.  DECODER ----------
#         dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B,T,d_model]
#         ts_hat  = self.output_projection(dec_out).squeeze(-1)      # [B,T]

#         return ts_hat 
    
#     def _compute_memory(self, ts_emb: torch.Tensor,
#                         txt_emb: torch.Tensor,
#                         T_out: int) -> torch.Tensor:
#         B = ts_emb.size(0)
#         ts_emb_tiled  = ts_emb.unsqueeze(1).repeat(1, T_out, 1)   # [B,T,E]
#         txt_emb_tiled = txt_emb.unsqueeze(1).repeat(1, T_out, 1)  # [B,T,E]
#         memory_cat = torch.cat([ts_emb_tiled, txt_emb_tiled], dim=2)  # [B,T,2E]
#         memory     = self.mem_proj(memory_cat)                        # [B,T,d_model]
#         return memory

    
#     @torch.no_grad()
#     def forecast(self,
#                  ts_emb:   torch.Tensor,     # [B, E]
#                  txt_emb:  torch.Tensor,     # [B, E]
#                  ts_ctx:   torch.Tensor,     # [B, T_ctx]   observed part
#                  horizon:  int = 1) -> torch.Tensor:
#         """
#         Public function forautoregressive forecasting method

#         Returns tensor of shape [B, T_ctx + horizon] = original context + predictions.
#         """
#         device = ts_ctx.device
#         seq    = ts_ctx.clone()          # running buffer  (doesn't alter caller's tensor)

#         for _ in range(horizon):
#             T_cur   = seq.size(1)                        # length so far
#             memory  = self._compute_memory(ts_emb, txt_emb, T_cur)
#             sos     = seq[:, :1]                         # prepend first value
#             ts_in   = torch.cat([sos, seq[:, :-1]], 1)   # teacher-forcing shift
#             tgt     = self.tgt_proj(ts_in.unsqueeze(-1)) # [B,T_cur,d_model]

#             causal_mask = self.strict_causal_mask[:T_cur, :T_cur].to(device)
#             dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B,T_cur,d_model]
#             ts_hat  = self.output_projection(dec_out).squeeze(-1)      # [B,T_cur]

#             next_val = ts_hat[:, -1:]                # prediction for step T_cur
#             seq      = torch.cat([seq, next_val], 1) # append and continue

#         return seq


class TransformerDecoderPreAuto(nn.Module):

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding dim of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        k = 10, # number of condition tokens per embedding
    ):
        super().__init__()
        self.k = k
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        self.d_model = nhead * 4

        # ---------- Conditioning projections ----------
        self.start_token = nn.Linear(output_dim * 2, 1)
        self.cond_proj = nn.Sequential(nn.Linear(output_dim, k * self.d_model), nn.GELU())
        self.mem_proj  = nn.Sequential(nn.Linear(output_dim * 2, self.d_model), nn.GELU())
        self.tgt_proj  = nn.Sequential(nn.Linear(1, self.d_model), nn.GELU())

        # ---------- Decoder stack ----------
        self.register_buffer("strict_causal_mask", torch.triu(torch.ones(5000,5000),1).bool())
        self.decoder_layer = nn.TransformerDecoderLayer(
                                d_model=self.d_model, 
                                nhead=nhead,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout,
                                batch_first=True)
        self.decoder  = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(self.d_model, 1)
        
    def forward(self, ts_emb:torch.Tensor, txt_emb:torch.Tensor, ts:torch.Tensor):
        """
        Returns ts_hat of shape [B, T] (same length as input series).
        """
        B, T_full = ts.shape
        if T_full < 1:
            raise ValueError("need at least one step")

        # 1) teacher-forcing shift
        # sos   = ts[:, :1]                       # [B,1] raw first value
        sos = self.start_token(torch.cat([ts_emb, txt_emb], dim=1))  # [B,1]
        ts_in = torch.cat([sos, ts[:, :-1]], 1) # [B,T]
        T     = ts_in.size(1)                   # T_full

        # 2) build per-time-step memory (same as before, but for T+2k)
        T_plus = T + 2 * self.k
        ts_emb_t = ts_emb.unsqueeze(1).repeat(1, T_plus, 1)    # [B,T+2k,E]
        txt_emb_t= txt_emb.unsqueeze(1).repeat(1, T_plus, 1)   # [B,T+2k,E]
        memory_cat = torch.cat([ts_emb_t, txt_emb_t], 2)       # [B,T+2k,2E]
        memory     = self.mem_proj(memory_cat)                 # [B,T+2k,d]

        # 3) prepend 2k condition tokens to the decoder input
        cond_ts  = self.cond_proj(ts_emb).view(B, self.k, self.d_model)        # [B,k,d]
        cond_txt = self.cond_proj(txt_emb).view(B, self.k, self.d_model)        # [B,k,d]
        tgt_ts   = self.tgt_proj(ts_in.unsqueeze(-1))          # [B,T,d]

        tgt = torch.cat([cond_ts, cond_txt, tgt_ts], 1)        # [B,T+2k,d]

        # 4) causal mask (allow reading cond tokens + past)
        causal_mask = self.strict_causal_mask[:T_plus, :T_plus].to(ts.device)

        # 5) decode
        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)   # [B,T+2k,d]
        ts_hat  = self.output_projection(dec_out[:, 2*self.k:]).squeeze(-1)  # drop cond tokens → [B,T]

        return ts_hat
    
    @torch.no_grad()
    def forecast(
        self,
        txt_emb:  torch.Tensor,            # [B, E]  text embedding
        horizon:  int,                     # how many steps to produce
        ts_emb:   torch.Tensor| None = None,            # [B, E]  ts embedding
        ts_ctx:   torch.Tensor | None = None,  # optional observed prefix [B, T_ctx]
    ) -> torch.Tensor:
        """
        Autoregressively extend `ts_ctx` by `horizon` steps.
        If `ts_ctx` is None or empty, starts from scratch (generation).

        Returns
        -------
        seq : torch.Tensor  shape [B, T_ctx + horizon]
        """
        if ts_emb is None:
            ts_emb = txt_emb.clone()

        device = ts_emb.device
        dtype  = ts_emb.dtype
        B      = ts_emb.size(0)

        # --- initial context -------------------------------------------------
        if ts_ctx is None:
            seq = torch.empty(B, 0, device=device, dtype=dtype)   # start fresh
        else:
            seq = ts_ctx.to(device=device, dtype=dtype).clone()

        # --- autoregressive rollout -----------------------------------------
        for _ in range(horizon):
            T_cur = seq.size(1)

            # build dummy input for forward() : context + placeholder
            if T_cur == 0:
                dummy_ts = torch.zeros(B, 1, device=device, dtype=dtype)
            else:
                dummy_ts = torch.cat([seq, torch.zeros(B, 1, device=device, dtype=dtype)], dim=1)

            # model forward (learned start token handles step-0)
            ts_hat = self(ts_emb, txt_emb, dummy_ts)          # [B, T_cur+1]
            next_val = ts_hat[:, -1:]                         # newest prediction
            seq = torch.cat([seq, next_val], dim=1)           # append

        return seq                                            # [B, T_ctx + horizon]
