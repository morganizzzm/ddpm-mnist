"""
Neural Network Architecture Components

This module contains the U-Net architecture components used for the
diffusion model, including residual blocks, time embeddings, and the
complete TinyUNet model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.

    Architecture:
        x → Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU → (+skip) → out

    Features:
        - GroupNorm: More stable than BatchNorm for small batches
        - SiLU (Swish): Smooth activation function for diffusion models
        - Skip connection: Helps gradient flow (ResNet-style)

    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) 
            if in_ch != out_ch 
            else nn.Identity()
        )
        
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, in_ch, H, W]
            
        Returns:
            torch.Tensor: Output tensor [B, out_ch, H, W]
        """
        h = F.silu(self.gn1(self.conv1(x)))
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by MLP.

    Creates unique representations for each timestep using sinusoidal functions:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Similar to Transformer positional encodings.

    Args:
        dim (int): Dimension of the time embedding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        """
        Args:
            t (torch.Tensor): Timesteps [B]
            
        Returns:
            torch.Tensor: Time embeddings [B, dim]
        """
        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            torch.arange(half, device=device) * (-math.log(10000) / (half - 1))
        )

        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)


class TinyUNet(nn.Module):
    """
    Lightweight U-Net architecture for MNIST diffusion.

    Architecture:
        Input (28x28) → Encoder → Bottleneck → Decoder → Output (28x28)

    Features:
        - Skip connections between encoder and decoder
        - FiLM conditioning: output = features * (1 + γ(t)) + β(t)
        - Timestep-adaptive processing

    Args:
        base (int): Base number of channels (default: 64)
        time_dim (int): Dimension of time embeddings (default: 64)
        in_ch (int): Input channels (default: 1 for grayscale)
    """

    def __init__(self, base=64, time_dim=64, in_ch=1):
        super().__init__()

        self.tproj = TimeEmbedding(time_dim)

        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base, kernel_size=3, stride=1, padding=1)
        self.down1 = Block(base, base)
        self.pool1 = nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1)
        self.down2 = Block(base, base * 2)
        self.pool2 = nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.mid = Block(base * 2, base * 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base * 2, base * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = Block(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1)
        self.dec2 = Block(base * 2, base)

        self.out = nn.Conv2d(base, in_ch, kernel_size=3, stride=1, padding=1)

        # FiLM conditioning layers
        self.film_down1 = nn.Linear(time_dim, base * 2)
        self.film_down2 = nn.Linear(time_dim, base * 4)
        self.film_mid = nn.Linear(time_dim, base * 4)
        self.film_dec1 = nn.Linear(time_dim, base * 4)
        self.film_dec2 = nn.Linear(time_dim, base * 2)

    def forward(self, x, t):
        """
        Forward pass through the U-Net.
        
        Args:
            x (torch.Tensor): Input tensor [B, in_ch, H, W]
            t (torch.Tensor): Timesteps [B]
            
        Returns:
            torch.Tensor: Output tensor [B, in_ch, H, W]
        """
        temb = self.tproj(t)

        # Encoder
        h = self.in_conv(x)
        
        h1 = self.down1(h)
        film_params1 = self.film_down1(temb)
        gamma1, beta1 = film_params1.chunk(2, dim=1)
        h1 = h1 * (1 + gamma1[:, :, None, None]) + beta1[:, :, None, None]
        h = self.pool1(h1)

        h2 = self.down2(h)
        film_params2 = self.film_down2(temb)
        gamma2, beta2 = film_params2.chunk(2, dim=1)
        h2 = h2 * (1 + gamma2[:, :, None, None]) + beta2[:, :, None, None]
        h = self.pool2(h2)

        # Bottleneck
        h = self.mid(h)
        film_params_mid = self.film_mid(temb)
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        h = h * (1 + gamma_mid[:, :, None, None]) + beta_mid[:, :, None, None]

        # Decoder
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec1(h)
        film_params_dec1 = self.film_dec1(temb)
        gamma_dec1, beta_dec1 = film_params_dec1.chunk(2, dim=1)
        h = h * (1 + gamma_dec1[:, :, None, None]) + beta_dec1[:, :, None, None]

        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2(h)
        film_params_dec2 = self.film_dec2(temb)
        gamma_dec2, beta_dec2 = film_params_dec2.chunk(2, dim=1)
        h = h * (1 + gamma_dec2[:, :, None, None]) + beta_dec2[:, :, None, None]

        return self.out(h)
