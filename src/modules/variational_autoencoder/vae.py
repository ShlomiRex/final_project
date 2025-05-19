"""
From my project: BlendDigits
"""

import torch
import torch.nn as nn
import math
from einops import rearrange

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.to(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.image_width = int(math.sqrt(input_dim))
        self.image_height = int(math.sqrt(input_dim))

        self.__setup_encoder()
        self.__setup_decoder()
    
    def __setup_encoder(self):
        self.enc_fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)

        self.relu = nn.ReLU().to(self.device)

        # Now we have two layers for each vector in latent space (going from hidden_dim to latent_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)  # Mean vector
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)  # Log-variance vector

    def __setup_decoder(self):
        self.dec_fc1 = nn.Linear(self.latent_dim, self.hidden_dim).to(self.device)
        self.dec_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.dec_fc3 = nn.Linear(self.hidden_dim, self.input_dim).to(self.device)
    
    def encode(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)') # Flatten the input
        x = self.relu(self.enc_fc1(x))
        x = self.relu(self.enc_fc2(x))

        mean = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        # Here we don't return x, we return mean and log_var, this is different to AE
        return mean, log_var
    
    def decode(self, latent):
        x = self.relu(self.dec_fc1(latent))
        x = self.relu(self.dec_fc2(x))

        x_hat = torch.sigmoid(self.dec_fc3(x))

        x_hat = rearrange(x_hat, 'b (c h w) -> b c h w', c=1, h=self.image_width, w=self.image_height) # Reshape the output

        return x_hat
    
    def reparameterization(self, mean, var):
        """
        Variance is exponential of log_var
        """
        epsilon = torch.randn_like(var).to(self.device)
        mean = mean.to(self.device)
        var = var.to(self.device)

        z = mean + var * epsilon
        return z


    def forward(self, x):
        assert x.shape[-3:] == (1, 28, 28)

        x.to(self.device)

        # Encode - instead of latent vector we get mean and log_var (look at image!)
        mean, log_var = self.encode(x)

        # Here is the magic of VAE
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        
        # Decode
        x_reconstructed = self.decode(z)

        # Return x hat
        return x_reconstructed, mean, log_var