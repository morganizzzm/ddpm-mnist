"""
Generate samples from a trained DDPM model.

This script loads a trained checkpoint and generates sample images.
"""

import argparse
import os
import torch
from torchvision import utils

from models import TinyUNet, DDPM
from utils import get_device


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate samples from trained DDPM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of denoising steps')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--output', type=str, default='samples.png',
                        help='Output filename')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='Use EMA weights if available')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load model on
        
    Returns:
        DDPM: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    base = args.get('base', 64)
    timesteps = args.get('timesteps', 1000)
    schedule = args.get('schedule', 'linear')
    objective = args.get('objective', 'eps')
    
    # Create model
    net = TinyUNet(base=base, time_dim=64, in_ch=1).to(device)
    ddpm = DDPM(
        net,
        timesteps=timesteps,
        schedule=schedule,
        objective=objective
    ).to(device)
    
    # Load weights
    ddpm.load_state_dict(checkpoint['model'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Configuration: base={base}, timesteps={timesteps}, "
          f"schedule={schedule}, objective={objective}")
    
    return ddpm, checkpoint


def apply_ema_weights(model, checkpoint):
    """
    Apply EMA weights to model if available.
    
    Args:
        model (DDPM): Model to apply EMA weights to
        checkpoint (dict): Checkpoint dictionary
    """
    if 'ema' in checkpoint:
        ema_state = checkpoint['ema']
        shadow = ema_state['shadow']
        
        for name, param in model.model.named_parameters():
            if name in shadow:
                param.data = shadow[name]
        
        print("Applied EMA weights")
    else:
        print("No EMA weights found in checkpoint")


def generate_samples(model, num_samples, steps, batch_size, device):
    """
    Generate samples from the model.
    
    Args:
        model (DDPM): Model to generate from
        num_samples (int): Total number of samples
        steps (int): Number of denoising steps
        batch_size (int): Batch size for generation
        device (torch.device): Device to generate on
        
    Returns:
        torch.Tensor: Generated samples
    """
    model.eval()
    samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        print(f"Generating batch {i+1}/{num_batches} "
              f"({current_batch_size} samples)...")
        
        with torch.no_grad():
            batch_samples = model.sample(
                (current_batch_size, 1, 28, 28),
                device=device,
                steps=steps
            )
        
        samples.append(batch_samples.cpu())
    
    return torch.cat(samples, dim=0)


def save_samples(samples, output_path, nrow=8):
    """
    Save samples as an image grid.
    
    Args:
        samples (torch.Tensor): Generated samples
        output_path (str): Path to save image
        nrow (int): Number of samples per row in grid
    """
    # Normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    
    # Create grid
    grid = utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    # Save
    utils.save_image(grid, output_path)
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main generation function."""
    args = parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Apply EMA weights if requested
    if args.use_ema:
        apply_ema_weights(model, checkpoint)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with {args.steps} steps...")
    samples = generate_samples(
        model,
        args.num_samples,
        args.steps,
        args.batch_size,
        device
    )
    
    # Save samples
    nrow = int(args.num_samples ** 0.5)
    save_samples(samples, args.output, nrow=nrow)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
