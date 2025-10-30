"""
Train DDPM on MNIST

Main training script for the Denoising Diffusion Probabilistic Model
on the MNIST dataset.
"""

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from models import TinyUNet,DDPM
from utils import EMA,get_device,evaluate,plot_losses,print_results_table, \
    print_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DDPM on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture
    parser.add_argument(
        '--base',type=int,default=64,
        help='Base number of channels in U-Net'
        )
    parser.add_argument(
        '--timesteps',type=int,default=1000,
        help='Number of diffusion timesteps'
        )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',type=int,default=50,
        help='Number of training epochs'
        )
    parser.add_argument(
        '--batch-size',type=int,default=128,
        help='Batch size for training'
        )
    parser.add_argument(
        '--lr',type=float,default=2e-4,
        help='Learning rate'
        )
    parser.add_argument(
        '--grad-clip',type=float,default=1.0,
        help='Gradient clipping threshold (0 to disable)'
        )

    # Diffusion settings
    parser.add_argument(
        '--schedule',type=str,default='linear',
        choices=['linear','cosine','sigmoid'],
        help='Noise schedule type'
        )
    parser.add_argument(
        '--objective',type=str,default='eps',
        choices=['eps','x0','v'],
        help='Training objective (eps=noise, x0=image, v=velocity)'
        )
    parser.add_argument(
        '--min-snr-gamma',type=float,default=5.0,
        help='Min-SNR gamma parameter (0 to disable)'
        )

    # EMA settings
    parser.add_argument(
        '--use-ema',action='store_true',default=True,
        help='Use Exponential Moving Average'
        )
    parser.add_argument(
        '--no-ema',action='store_false',dest='use_ema',
        help='Disable Exponential Moving Average'
        )
    parser.add_argument(
        '--ema-decay',type=float,default=0.9999,
        help='EMA decay rate'
        )

    # Data settings
    parser.add_argument(
        '--train-size',type=int,default=50000,
        help='Size of training set'
        )
    parser.add_argument(
        '--eval-size',type=int,default=10000,
        help='Size of evaluation set'
        )
    parser.add_argument(
        '--num-workers',type=int,default=0,
        help='Number of data loader workers (use 0 for macOS/MPS)'
        )

    # Checkpointing
    parser.add_argument(
        '--save-every',type=int,default=10,
        help='Save checkpoint every N epochs'
        )
    parser.add_argument(
        '--resume',type=str,default='',
        help='Path to checkpoint to resume from'
        )

    # Output
    parser.add_argument(
        '--outdir',type=str,default='./outputs',
        help='Output directory for samples and checkpoints'
        )

    return parser.parse_args()


def setup_data(args,device):
    """
    Setup data loaders for training, evaluation, and testing.

    Args:
        args: Parsed command-line arguments
        device: Device being used (to determine pin_memory)

    Returns:
        tuple: (train_loader, eval_loader, test_loader)
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ]
    )

    full_train_ds = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_indices = list(range(args.train_size))
    eval_indices = list(
        range(args.train_size,args.train_size + args.eval_size)
        )

    train_subset = Subset(full_train_ds,train_indices)
    eval_subset = Subset(full_train_ds,eval_indices)

    test_ds = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Determine if we should use pin_memory (not supported on MPS)
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader,eval_loader,test_loader


def setup_model(args,device):
    """
    Setup model, optimizer, and EMA.

    Args:
        args: Parsed command-line arguments
        device: Device to use

    Returns:
        tuple: (model, optimizer, ema)
    """
    net = TinyUNet(base=args.base,time_dim=64,in_ch=1).to(device)

    ddpm = DDPM(
        net,
        timesteps=args.timesteps,
        schedule=args.schedule,
        objective=args.objective
    ).to(device)

    total_params = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),lr=args.lr,weight_decay=0.01
        )
    ema = EMA(ddpm.model,decay=args.ema_decay) if args.use_ema else None

    return ddpm,optimizer,ema


def load_checkpoint(args,ddpm,optimizer,ema,device):
    """
    Load checkpoint if resume path is provided.

    Args:
        args: Parsed command-line arguments
        ddpm: DDPM model
        optimizer: Optimizer
        ema: EMA helper
        device: Device

    Returns:
        tuple: (start_epoch, train_losses, eval_losses)
    """
    start_epoch = 1
    train_losses = []
    eval_losses = []

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume,map_location=device)
        ddpm.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses',[])
        eval_losses = checkpoint.get('eval_losses',[])
        if args.use_ema and 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
        print(f"Resumed from epoch {checkpoint['epoch']}")
    elif args.resume:
        print(f"No checkpoint found at {args.resume}, starting from scratch")

    return start_epoch,train_losses,eval_losses


def train_epoch(ddpm,train_loader,optimizer,ema,args,device,epoch):
    """
    Train for one epoch.

    Args:
        ddpm: DDPM model
        train_loader: Training data loader
        optimizer: Optimizer
        ema: EMA helper
        args: Arguments
        device: Device
        epoch: Current epoch number

    Returns:
        float: Average training loss
    """
    ddpm.train()
    epoch_losses = []

    pbar = tqdm(train_loader,desc=f"Epoch {epoch} [Train]",leave=False)
    for x,_ in pbar:
        x = x.to(device)
        t = torch.randint(0,ddpm.T,(x.size(0),),device=device).long()

        loss = ddpm.p_losses(
            x,t,
            min_snr_gamma=args.min_snr_gamma,
            objective=args.objective
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(),args.grad_clip)

        optimizer.step()

        if args.use_ema:
            ema.update(ddpm.model)

        epoch_losses.append(loss.item())
        pbar.set_postfix(
            {
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{sum(epoch_losses) / len(epoch_losses):.4f}'
            }
        )

    return sum(epoch_losses) / len(epoch_losses)


def generate_samples(ddpm,ema,args,device,epoch):
    """
    Generate and save sample images.

    Args:
        ddpm: DDPM model
        ema: EMA helper
        args: Arguments
        device: Device
        epoch: Current epoch number
    """
    ddpm.eval()

    with torch.no_grad():
        if args.use_ema:
            ema.apply_shadow(ddpm.model)

        samples = ddpm.sample((64,1,28,28),device=device,steps=1000)

        if args.use_ema:
            ema.restore(ddpm.model)

        grid = (samples + 1) / 2
        utils.save_image(
            grid,
            os.path.join(args.outdir,f"samples_epoch_{epoch:03d}.png"),
            nrow=8
        )


def save_checkpoint(ddpm,optimizer,ema,epoch,train_losses,eval_losses,args,
                    name):
    """
    Save a checkpoint.

    Args:
        ddpm: DDPM model
        optimizer: Optimizer
        ema: EMA helper
        epoch: Current epoch
        train_losses: Training loss history
        eval_losses: Evaluation loss history
        args: Arguments
        name: Checkpoint filename
    """
    checkpoint_data = {
        'model': ddpm.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'args': vars(args)
    }
    if args.use_ema:
        checkpoint_data['ema'] = ema.state_dict()

    save_path = os.path.join(args.outdir,name)
    torch.save(checkpoint_data,save_path)
    return save_path


def main():
    """Main training function."""
    args = parse_args()

    os.makedirs(args.outdir,exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Setup
    train_loader,eval_loader,test_loader = setup_data(args,device)
    ddpm,optimizer,ema = setup_model(args,device)
    start_epoch,train_losses,eval_losses = load_checkpoint(
        args,ddpm,optimizer,ema,device
    )

    print_config(args)

    # Training loop
    for epoch in tqdm(range(start_epoch,args.epochs + 1),desc="Epochs"):
        t0 = time.time()

        # Train
        avg_train_loss = train_epoch(
            ddpm,train_loader,optimizer,ema,args,device,epoch
        )
        train_losses.append(avg_train_loss)

        # Evaluate
        avg_eval_loss = evaluate(
            ddpm,eval_loader,device,
            ema if args.use_ema else None,
            min_snr_gamma=args.min_snr_gamma,
            objective=args.objective
        )
        eval_losses.append(avg_eval_loss)

        # Log
        tqdm.write(
            f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - "
            f"Eval Loss: {avg_eval_loss:.4f} - Time: {time.time() - t0:.1f}s"
        )

        # Generate samples
        tqdm.write("Generating samples...")
        generate_samples(ddpm,ema,args,device,epoch)

        # Save checkpoints
        if epoch % args.save_every == 0:
            path = save_checkpoint(
                ddpm,optimizer,ema,epoch,train_losses,eval_losses,args,
                f'checkpoint_epoch_{epoch:03d}.pt'
            )
            tqdm.write(f"Saved checkpoint: {path}")

        save_checkpoint(
            ddpm,optimizer,ema,epoch,train_losses,eval_losses,args,
            'last.pt'
        )

    # Final evaluation
    print("\nEvaluating on test set...")
    test_loss = evaluate(
        ddpm,test_loader,device,
        ema if args.use_ema else None,
        min_snr_gamma=args.min_snr_gamma,
        objective=args.objective
    )

    # Save results
    plot_losses(
        train_losses,eval_losses,os.path.join(args.outdir,'loss_curves.png')
        )
    print(
        f"Loss curves saved to {os.path.join(args.outdir,'loss_curves.png')}"
        )

    print_results_table(train_losses[-1],eval_losses[-1],test_loss)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Samples saved to: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()