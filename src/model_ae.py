"""
Acoustic PDM — Neural Network Architectures Module
==================================================
Deep learning Autoencoder architectures for acoustic anomaly detection:
1. Conv2D-AE (Primary Standalone Model & Hybrid Backbone)
2. FC-AE (Fully-Connected Baseline Autoencoder)
3. LSTM-AE (Recurrent Temporal Baseline Autoencoder)

All models operate on normalized context tensor blocks of shape (Batch, 1, 128, 5).
Bottleneck latent dimension: z in R^32.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class Conv2DAutoencoder(nn.Module):
    """
    Convolutional 2D Autoencoder for Acoustic Anomaly Detection.
    
    Architecture:
      Encoder:
        - Conv2D(1 -> 32, 3x3, stride 2, pad 1) + LeakyReLU(0.2)   -> (B, 32, 64, 3)
        - Conv2D(32 -> 64, 3x3, stride 2, pad 1) + LeakyReLU(0.2)  -> (B, 64, 32, 2)
        - Conv2D(64 -> 128, 3x3, stride 2, pad 1) + LeakyReLU(0.2) -> (B, 128, 16, 1)
        - Flatten (128 * 16 * 1 = 2048) -> Linear(2048 -> 32)      -> Latent z in R^32
      Decoder:
        - Linear(32 -> 2048) + LeakyReLU(0.2) -> Unflatten to (B, 128, 16, 1)
        - ConvTranspose2D(128 -> 64, 3x3, stride 2, pad 1, out_pad (1, 1)) -> (B, 64, 32, 2)
        - ConvTranspose2D(64 -> 32, 3x3, stride 2, pad 1, out_pad (1, 0))  -> (B, 32, 64, 3)
        - ConvTranspose2D(32 -> 1, 3x3, stride 2, pad 1, out_pad (1, 0))   -> (B, 1, 128, 5)
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # ── Encoder ──
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_fc = nn.Linear(128 * 16 * 1, latent_dim)
        
        # ── Decoder ──
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation z in R^32."""
        h = self.encoder_conv(x)
        h = torch.flatten(h, start_dim=1)
        z = self.encoder_fc(h)
        return z
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input tensor from latent vector z."""
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 16, 1)
        recon = self.decoder_conv(h)
        return recon
        
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, 1, 128, 5).
            return_latent: Whether to also return the bottleneck latent vector.
            
        Returns:
            Reconstruction of shape (Batch, 1, 128, 5), or (reconstruction, latent).
        """
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon


class FCAutoencoder(nn.Module):
    """
    Fully-Connected Baseline Autoencoder.
    Flattens the 640-dim context block and applies dense multi-layer reduction.
    """
    
    def __init__(self, input_dim: int = 640, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        flat = x.view(b, -1)
        return self.encoder(flat)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        flat_recon = self.decoder(z)
        return flat_recon.view(b, 1, 128, 5)
        
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon


class LSTMAutoencoder(nn.Module):
    """
    Recurrent LSTM Baseline Autoencoder.
    Models the temporal sequence of 5 consecutive time frames (128 Mel features each).
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        context_frames: int = 5,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2
    ):
        super().__init__()
        self.n_mels = n_mels
        self.context_frames = context_frames
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder LSTM: (Batch, 5, 128) -> hidden (Batch, hidden_dim)
        self.encoder_lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder LSTM: (Batch, 5, hidden_dim) -> (Batch, 5, 128)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_dim, n_mels)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, 128, 5) -> permute to (Batch, 5, 128)
        seq = x.squeeze(1).permute(0, 2, 1)
        _, (h_n, _) = self.encoder_lstm(seq)
        # Take last layer's hidden state
        last_h = h_n[-1]
        z = self.encoder_fc(last_h)
        return z
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h = self.decoder_fc(z)  # (B, hidden_dim)
        # Repeat across 5 context frames
        seq_in = h.unsqueeze(1).repeat(1, self.context_frames, 1)  # (B, 5, hidden_dim)
        out, _ = self.decoder_lstm(seq_in)
        recon_seq = self.output_fc(out)  # (B, 5, 128)
        # Permute back to (Batch, 1, 128, 5)
        recon = recon_seq.permute(0, 2, 1).unsqueeze(1)
        return recon
        
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon
