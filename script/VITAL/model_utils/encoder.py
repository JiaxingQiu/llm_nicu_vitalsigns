# from config import *
import torch
import torch.nn as nn

# ------- custom ts encoder_layers (for VITAL embedding generation) -------
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

class ResNetEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        # layers = [
        #     # Initial projection
        #     nn.Linear(ts_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(dropout)
        # ]
        layers = [
            ResidualBlock(
                    ts_dim, 
                    ts_dim,
                    dropout=0.1
                ),
            nn.Linear(ts_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2)
        ])

        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# resnet_encoder = ResNetEncoder(ts_dim=300, output_dim=128)
# model = GeneralBinaryClassifier(resnet_encoder)
# model = CLIPModel(ts_encoder=resnet_encoder, text_encoder=None)



# class Lambda(nn.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
    
#     def forward(self, x):
#         return self.func(x) 

class AddChannelDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.unsqueeze(1)

class CNNEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, 
                 num_channels=[64, 64, 128, 256], 
                 kernel_size=5, 
                 dropout=0.2):
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

        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        # layers = [Lambda(lambda x: x.unsqueeze(1))]  # Add channel dimension
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
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

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
                 bidirectional=False,
                 mask=-1): # it is the indicator of masked values in the time series (i.e. -1), default to -1
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

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

# Usage:
# multi_lstm_encoder = MultiLSTMEncoder(
#     ts_dim=300, 
#     output_dim=128,
#     hidden_dims=[128, 256, 64],  # Three LSTMs with different sizes
#     num_layers=2
# )
# model = GeneralBinaryClassifier(multi_lstm_encoder)
# model = CLIPModel(ts_encoder=multi_lstm_encoder, text_encoder=None)


# # ts vae encoder wrapper for custom ts encoders
# class TSVAEEncoderWrapper(nn.Module):
#     def __init__(self, ts_encoder_layers, output_dim=None):
#         super().__init__()
#         self.local_norm = LocalNorm()

#         # ts_encoder_layers must be a instance of a class defined similarly as belows
#         self.encoder_layers = ts_encoder_layers
        
#         # Get the hidden dimension from the encoder's output
#         hidden_dim = ts_encoder_layers.output_dim
#         if output_dim is None:
#             output_dim = hidden_dim
        
#         # Latent mean and variance 
#         self.mean_layer = nn.Linear(hidden_dim, output_dim)
#         self.logvar_layer = nn.Linear(hidden_dim, output_dim)


#     def reparameterization(self, mean, log_var, lam=1):
#         var = lam * torch.exp(0.5 * log_var) # calculate the variance from the log_var 
#         # var = log_var
#         epsilon = torch.randn_like(var).to(device)      
#         z = mean + var*epsilon  # Using var directly
#         # z = F.softmax(z,dim=1) # This variable follows a Dirichlet distribution
#         return z
    
    
#     def forward(self, x):
#         _, x_mean, x_std = self.local_norm(x)

#         #  ---- encode -----
#         x_encoded = self.encoder_layers(x)
#         mean = self.mean_layer(x_encoded)
#         log_var = self.logvar_layer(x_encoded)

#         #  ---- reparameterization -----
#         z = self.reparameterization(mean, log_var)
#         return z, mean, log_var, x_mean, x_std

# # usage:
# """
# lstm_encoder = MultiLSTMEncoder(
#     ts_dim=300, 
#     output_dim=128,
#     hidden_dims=[128, 256, 64],  # Three LSTMs with different sizes
#     num_layers=2
# )
# ts_encoder = TSVAEEncoderWrapper(lstm_encoder)
# for batch_idx, (ts_features, text_features, labels) in enumerate(train_dataloader):

#     ts_features = ts_features.to(device)
#     print(ts_encoder(ts_features))
#     break
# """


# # ts vae decoder wrapper for custom ts decoders
# class TSVAEDecoderWrapper(nn.Module):
#     def __init__(self, ts_decoder):
#         super().__init__()
#         self.decoder = ts_decoder
        
#     def forward(self, z, x_mean, x_std):
#         x_hat = self.decoder(z)
#         x_hat = x_hat * x_std + x_mean
#         # x_hat = torch.round(x_hat)
#         return x_hat





# ------- custom text encoder_layers (for VITAL embedding generation) -------
# MLP text encoder
class TextEncoderMLP(nn.Module):
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
        return tx_emb


# CNN text encoder
class TextEncoderCNN(nn.Module):
    def __init__(self, 
                 text_dim: int,
                 output_dim: int,
                 num_channels=[64],#, 128, 256
                 kernel_size=50,
                 dropout=0):
        """
        CNN encoder for text features.
        
        Args:
            output_dim (int): Output embedding dimension
            num_channels (list): Number of channels for each conv layer
            kernel_size (int): Kernel size for conv layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
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
            x = torch.zeros(2, text_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, text_features):
        return self.encoder(text_features)

# # default text encoder with attention
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