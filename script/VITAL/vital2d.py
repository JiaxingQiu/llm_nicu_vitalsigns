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
                 beta: float = 1.0,
                 ts_encoder = None,
                 text_encoder = None,
                 ts_decoder = None,
                 variational = True,
                 clip_mu = False,
                 concat_embeddings = False,
                 gen_w_src_text = False):
        """Initialize VITAL model
        
        Args:
            ts_dim: Input time series dimension
            text_dim: Input text embedding dimension
            output_dim: Latent space dimension
            beta: Weight for VAE loss
            ts_encoder: Optional custom time series encoder
            text_encoder: Optional custom text encoder
            ts_decoder: Optional custom time series decoder
            variational: Whether to use VAE (default: True)
            clip_mu: Whether to use mean for CLIP instead of sampled z (default: False)
            concat_embeddings: Whether to concatenate embeddings before decoding (default: False)
        """

        super().__init__()
        
        # ts_encoder
        self.ts_encoder = TSEncoder(ts_dim, output_dim, encoder_layers = ts_encoder, variational = variational)
        
        # text_encoder
        self.text_encoder = TextEncoder(text_dim, output_dim, encoder_layers = text_encoder)
        
        # ts_decoder
        if concat_embeddings:
            decode_dim = 2*output_dim
        else:
            decode_dim = output_dim
        self.ts_decoder = TSDecoder(ts_dim = ts_dim, output_dim = decode_dim, decoder_layers = ts_decoder)
        
        
        self.device = device
        self.beta = beta
        self.clip_mu = clip_mu
        self.concat_embeddings = concat_embeddings
        self.variational = variational
        self.gen_w_src_text = gen_w_src_text
        self.to(device)
        print(nn_summary(self))
    
    def clip(self, ts_embedded, text_embedded):
        # ts_embedded = F.normalize(ts_embedded, dim=1)
        # text_embedded = F.normalize(text_embedded, dim=1)
        logits = torch.matmul(ts_embedded, text_embedded.T) 
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

        # ---- (V)AE encoder ----
        z, mean, log_var = self.ts_encoder(ts) # ts in raw scale
        
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
        
        ts_hat = self.ts_decoder(decode_input, text_embedded, ts, text_embedded) # during trining, only use text_embedded as source text embedding

        return logits, ts_hat, mean, log_var
    
    def generate(self, w, ts, tx_f_tgt, tx_f_src):
        """
        Generate a time series from a source time series and a target text features.

        Args:
            ts: [B, ts_dim], source time series
            tx_f_tgt: [B, text_dim], target text features embedded by pretrained sentence encoder
            w: [B, 1], interpolation weight
            tx_f_src: [B, text_dim], source text features embedded by pretrained sentence encoder (should be from 'text' columm in the dataframe)
        """
        # tx_emb_tgt
        tx_emb_tgt = self.text_encoder(tx_f_tgt)
        
        # tx_emb_src 
        if self.gen_w_src_text: tx_f_src = tx_f_tgt # overwrite tx_f_src with tx_f_tgt
        tx_emb_src = self.text_encoder(tx_f_src)

        # ts_emb_src
        ts_emb_src, _, _ = self.ts_encoder(ts)
        
        # ts_emb_tgt
        ts_emb_tgt = (1-w)*ts_emb_src + w*tx_emb_tgt # interpolation of ts_emb and tx_emb
        
        ts_hat = self.ts_decoder(ts_emb_tgt, tx_emb_tgt, ts, tx_emb_src) # during generation, use tx_emb_src as source text embedding

        return ts_hat, ts_emb_tgt, tx_emb_tgt, tx_emb_src


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

class TSEncoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, encoder_layers = None, variational: bool = False):
        """Time series encoder with progressive architecture
        
        Args:
            ts_dim (int): Input time series dimension
            output_dim (int): Output embedding dimension
            encoder_layers (nn.Module): Custom encoder layers (default: None)
            variational (bool): Whether to enable VAE sampling (default: True)
        """
        super().__init__()
        self.variational = variational
        self.local_norm = LocalNorm()
        if encoder_layers is None:
            # default encoder layers
            self.encoder_layers = MultiCNNEncoder(ts_dim = ts_dim,
                                                    output_dim=output_dim,
                                                    kernel_sizes=[100, 50, 10],
                                                    hidden_num_channel=16,
                                                    dropout=0.0)
        else:
            self.encoder_layers = encoder_layers # pass an instance of custom encoder layers from classes in the encoder module
        
        # Only create mean and log variance layers if variational is True
        if self.variational:
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
        # _, x_mean, x_std = self.local_norm(x)
        #  ---- encode -----
        x_encoded = self.encoder_layers(x)

        if self.variational:
            mean = self.mean_layer(x_encoded)
            mean = F.normalize(mean, dim=1)
            log_var = self.logvar_layer(x_encoded)
            z = self.reparameterization(mean, log_var)
        else:
            mean = x_encoded
            mean = F.normalize(mean, dim=1)
            log_var = torch.full_like(mean, -1e2)  # effectively 0 variance
            z = mean
        
        return z, mean, log_var

import inspect
class TSDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, decoder_layers = None):
        super().__init__()
        self.ts_dim = ts_dim  # Store ts_dim for zero initialization
        if decoder_layers is None:
            self.decoder = TransformerDecoderTXTS(
                ts_dim     = ts_dim,
                output_dim = output_dim,
                nhead = 8,
                num_layers = 6,
                dim_feedforward = 1024,
                dropout = 0.0)
        else:
            self.decoder = decoder_layers
        
        # ---------- figure out how many positional args it needs --------
        sig = inspect.signature(self.decoder.forward)
        # skip the first ("self") parameter
        required_params_count = 0
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default == inspect.Parameter.empty:
                required_params_count += 1
        self._n_required_args = required_params_count
    
    def forward(self, ts_emb, txt_emb = None, ts = None, src_txt_emb = None):
        if isinstance(self.decoder, TransformerDecoderTXTS2):
            x_hat = self.decoder(ts_emb, txt_emb, src_txt_emb)
        elif isinstance(self.decoder, TransformerDecoderTXTS):
            x_hat = self.decoder(ts_emb, txt_emb)
        elif isinstance(self.decoder, TransformerDecoderPreAuto):
            x_hat = self.decoder(ts_emb, txt_emb, ts)
        else:
            x_hat = self.decoder(ts_emb, txt_emb)
        
        x_hat = x_hat.squeeze(1) if x_hat.dim() == 3 and x_hat.size(1) == 1 else x_hat         # [B, ts_dim]
        
        return x_hat

class TextEncoder(nn.Module):
    def __init__(self, text_dim: int, output_dim: int, encoder_layers = None):
        """Text encoder that can use either MLP or CNN architecture.
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
            text_encoder_type (str): Type of encoder to use ('mlp' or 'cnn')
        """
        super().__init__()
        if encoder_layers is None:
            self.encoder_layers = TextEncoderAttention(
                text_dim=text_dim,
                output_dim=output_dim,
                num_heads=1,
                dropout=0.0
            )   
        else:
            self.encoder_layers = encoder_layers # pass an instance of custom encoder layers from classes in the encoder module
        
    def forward(self, text_features):
        tx_emb = self.encoder_layers(text_features)
        tx_emb = F.normalize(tx_emb, dim=1)
        return tx_emb

