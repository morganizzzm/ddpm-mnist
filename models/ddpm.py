"""
Denoising Diffusion Probabilistic Model (DDPM)

Implementation of DDPM as described in "Denoising Diffusion Probabilistic Models"
(Ho et al., 2020) with support for multiple noise schedules and training objectives.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Implements the forward diffusion process (adding noise) and reverse process
    (denoising) with support for multiple training objectives.

    Args:
        model (nn.Module): U-Net backbone for denoising
        timesteps (int): Number of diffusion steps T
        schedule (str): Noise schedule type ('linear', 'cosine', 'sigmoid')
        objective (str): Training objective ('eps', 'x0', 'v')
    """

    def __init__(self, model, timesteps=1000, schedule='linear', objective='eps'):
        super().__init__()
        self.model = model
        self.T = timesteps
        self.objective = objective

        # Generate noise schedule
        betas = self._get_noise_schedule(schedule, timesteps)

        # Pre-compute useful values
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1 - alphas_bar))

        # Posterior parameters
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_bar_prev) / (1 - alphas_bar)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1 - alphas_bar_prev) * torch.sqrt(alphas) / (1 - alphas_bar)
        )
        
        posterior_variance = betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20))
        )

    def _get_noise_schedule(self, schedule, timesteps):
        """
        Generate noise schedule β_t.
        
        Args:
            schedule (str): Schedule type ('linear', 'cosine', 'sigmoid')
            timesteps (int): Number of timesteps
            
        Returns:
            torch.Tensor: Beta values for each timestep
        """
        if schedule == 'linear':
            return torch.linspace(0.0001, 0.02, timesteps)
        
        elif schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_bar = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, timesteps)
            return torch.sigmoid(betas) * (0.02 - 0.0001) + 0.0001
        
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        
        Args:
            x0 (torch.Tensor): Clean images [B, C, H, W]
            t (torch.Tensor): Timesteps [B]
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            tuple: (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        return (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t

    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x_t and predicted x_0."""
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        return (x_t - sqrt_alpha_bar_t * x0) / sqrt_one_minus_alpha_bar_t

    def q_posterior_mean_variance(self, x0, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x0 +
            self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x_t, t):
        """
        Compute predicted mean and variance for p(x_{t-1} | x_t).
        
        Args:
            x_t (torch.Tensor): Noisy images [B, C, H, W]
            t (torch.Tensor): Timesteps [B]
            
        Returns:
            tuple: (model_mean, posterior_variance, posterior_log_variance, pred_x0)
        """
        model_output = self.model(x_t, t)

        # Convert model output to x_0 prediction
        if self.objective == 'eps':
            pred_noise = model_output
            pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == 'x0':
            pred_x0 = model_output
        elif self.objective == 'v':
            v = model_output
            sqrt_alpha_bar_t = self.sqrt_alphas_bar[t][:, None, None, None]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
            pred_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * v
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(pred_x0, x_t, t)
        )

        return model_mean, posterior_variance, posterior_log_variance, pred_x0

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x_t, t)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, shape, device, steps=None):
        """
        Generate samples by running the reverse diffusion process.
        
        Args:
            shape (tuple): Shape of samples (B, C, H, W)
            device (torch.device): Device to generate on
            steps (int, optional): Number of denoising steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        if steps is None:
            steps = self.T

        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(steps)), desc='Sampling', total=steps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)

        return x

    def p_losses(self, x0, t, noise=None, min_snr_gamma=0, objective=None):
        """
        Compute training loss for the diffusion model.
        
        Args:
            x0 (torch.Tensor): Clean images [B, C, H, W]
            t (torch.Tensor): Timesteps [B]
            noise (torch.Tensor, optional): Noise to add
            min_snr_gamma (float): Min-SNR gamma parameter
            objective (str, optional): Override default objective
            
        Returns:
            torch.Tensor: Scalar loss
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_t, _ = self.q_sample(x0, t, noise)
        model_output = self.model(x_t, t)

        obj = objective if objective is not None else self.objective

        # Define target based on objective
        if obj == 'eps':
            target = noise
        elif obj == 'x0':
            target = x0
        elif obj == 'v':
            sqrt_alpha_bar_t = self.sqrt_alphas_bar[t][:, None, None, None]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
            target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x0
        else:
            raise ValueError(f"Unknown objective: {obj}")

        loss = F.mse_loss(model_output, target, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])

        # Apply Min-SNR weighting
        if min_snr_gamma > 0:
            snr = self.alphas_bar[t] / (1 - self.alphas_bar[t])
            
            if obj == 'eps':
                snr_weight = torch.clamp(snr, max=min_snr_gamma)
            elif obj == 'x0':
                snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr
            elif obj == 'v':
                snr_weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1)
            
            loss = loss * snr_weight

        return loss.mean()
