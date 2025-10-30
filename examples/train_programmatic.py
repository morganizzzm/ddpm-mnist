"""
Example: Training DDPM with custom configuration

This example demonstrates how to use the DDPM implementation
programmatically instead of through the command line.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import TinyUNet, DDPM
from utils import EMA, get_device, evaluate


def main():
    # Configuration
    config = {
        'epochs': 10,
        'batch_size': 256,
        'lr': 2e-4,
        'timesteps': 1000,
        'schedule': 'cosine',
        'objective': 'v',
        'min_snr_gamma': 5.0,
        'use_ema': True,
        'ema_decay': 0.9999,
    }
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_ds = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_subset = Subset(train_ds, list(range(50000)))
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    net = TinyUNet(base=64, time_dim=64, in_ch=1).to(device)
    ddpm = DDPM(
        net,
        timesteps=config['timesteps'],
        schedule=config['schedule'],
        objective=config['objective']
    ).to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config['lr'], weight_decay=0.01)
    ema = EMA(ddpm.model, decay=config['ema_decay']) if config['use_ema'] else None
    
    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"Configuration: {config}\n")
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        ddpm.train()
        epoch_losses = []
        
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            t = torch.randint(0, ddpm.T, (x.size(0),), device=device).long()
            
            # Forward pass
            loss = ddpm.p_losses(
                x, t,
                min_snr_gamma=config['min_snr_gamma'],
                objective=config['objective']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            
            # Update EMA
            if ema:
                ema.update(ddpm.model)
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")
    
    # Generate samples
    print("Generating samples...")
    ddpm.eval()
    
    with torch.no_grad():
        if ema:
            ema.apply_shadow(ddpm.model)
        
        samples = ddpm.sample((16, 1, 28, 28), device=device)
        
        if ema:
            ema.restore(ddpm.model)
    
    print("Samples generated!")
    print("Shape:", samples.shape)
    print("Range:", samples.min().item(), "to", samples.max().item())


if __name__ == "__main__":
    main()
