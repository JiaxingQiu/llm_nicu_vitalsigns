import torch
import torch.nn as nn

# global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)


class VAE_Linear_Medium(nn.Module):
    def __init__(self, seq_len=300, device=device):
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


class VAE_Linear_Large(nn.Module):
    def __init__(self, seq_len=300, device=device):
        super(VAE_Linear_Large, self).__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(128, 64)
        self.logvar_layer = nn.Linear(128, 64)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
             nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
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


class VAE_Linear_Small(nn.Module):
    def __init__(self, seq_len=300, hidden_dim=50, device=device):
        super(VAE_Linear_Small, self).__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 150),
            nn.LeakyReLU(0.2),
            nn.Linear(150, 150),
            nn.LeakyReLU(0.2),
            nn.Linear(150, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, 10)
        self.logvar_layer = nn.Linear(hidden_dim, 10)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 150),
            nn.LeakyReLU(0.2),
            nn.Linear(150, 150),
            nn.LeakyReLU(0.2),
            nn.Linear(150, seq_len)
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



class VAE_LSTM_Encoder_Small(nn.Module):
    def __init__(self, seq_len=300, device=device):
        super(VAE_LSTM_Encoder_Small, self).__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,           # Each time step has 1 feature
            hidden_size=50,  # Number of features in hidden state
            num_layers=2,           # multiple LSTM layer
            batch_first=True        # Input shape: [batch, seq_len, features]
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(50, 256),
            nn.LeakyReLU(0.2),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(0.2),
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
            # nn.Linear(256, 256),
            # nn.LeakyReLU(0.2),
            nn.Linear(256, seq_len)
        )
        
     
    def encode(self, x):
        # Add feature dimension for LSTM
        x = x.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        
        # LSTM encoding
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-1]  # Take last hidden state: [batch_size, hidden_dim]
        
        # Linear encoding
        x = self.encoder_linear(hidden)
        
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


class VAE_LSTM(nn.Module):
    def __init__(self, seq_len=300, hidden_dim=150, device=device):
        super(VAE_LSTM, self).__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,           # Each time step has 1 feature
            hidden_size=hidden_dim,  # Number of features in hidden state
            num_layers=1,           # Single LSTM layer
            batch_first=True        # Input shape: [batch, seq_len, features]
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, 1)
        self.logvar_layer = nn.Linear(hidden_dim, 1)
        
        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,    # Input from previous linear layers
            hidden_size=1,            # Output dimension (1 for time series)
            num_layers=1,
            batch_first=True
        )
     
    def encode(self, x):
        # Add feature dimension for LSTM
        x = x.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        
        # LSTM encoding
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-1]  # Take last hidden state: [batch_size, hidden_dim]
        
        # Linear encoding
        x = self.encoder_linear(hidden)
        
        # Get latent parameters
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z):
        # Linear decoding
        x = self.decoder_linear(z)  # Shape: [batch_size, hidden_dim]
        
        # Repeat hidden state for each time step
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Shape: [batch_size, seq_len, hidden_dim]
        
        # LSTM decoding
        output, _ = self.decoder_lstm(x)  # Shape: [batch_size, seq_len, 1]
        return output.squeeze(-1)  # Shape: [batch_size, seq_len]

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
