"""
Training utilities including evaluation, logging, and visualization.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_device():
    """
    Automatically select the best available device.
    
    Returns:
        torch.device: CUDA if available, MPS if on Apple Silicon, else CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def evaluate(model, data_loader, device, ema=None, min_snr_gamma=0, objective='eps'):
    """
    Evaluate model on a dataset.
    
    Args:
        model (DDPM): Model to evaluate
        data_loader (DataLoader): Data loader for evaluation
        device (torch.device): Device to evaluate on
        ema (EMA, optional): EMA helper
        min_snr_gamma (float): Min-SNR gamma parameter
        objective (str): Training objective
        
    Returns:
        float: Average loss over the dataset
    """
    model.eval()
    total_loss = 0
    count = 0

    if ema is not None:
        ema.apply_shadow(model.model)

    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc='Evaluating', leave=False):
            x = x.to(device)
            t = torch.randint(0, model.T, (x.size(0),), device=device).long()
            loss = model.p_losses(x, t, min_snr_gamma=min_snr_gamma, objective=objective)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    if ema is not None:
        ema.restore(model.model)

    return total_loss / count


def plot_losses(train_losses, eval_losses, save_path):
    """
    Plot and save training and evaluation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch
        eval_losses (list): Evaluation losses per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, eval_losses, 'r-', label='Evaluation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_results_table(train_loss, eval_loss, test_loss):
    """
    Print a formatted results table.
    
    Args:
        train_loss (float): Final training loss
        eval_loss (float): Final evaluation loss
        test_loss (float): Test set loss
    """
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Loss':>10}")
    print("-" * 60)
    print(f"{'Training Loss':<20} {train_loss:>10.4f}")
    print(f"{'Evaluation Loss':<20} {eval_loss:>10.4f}")
    print(f"{'Test Loss':<20} {test_loss:>10.4f}")
    print("=" * 60)


def print_config(args):
    """
    Print training configuration in a formatted table.
    
    Args:
        args: Parsed command-line arguments
    """
    print(f"\n{'=' * 60}")
    print("Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Schedule: {args.schedule}")
    print(f"  Objective: {args.objective}")
    print(f"  Min-SNR Gamma: {args.min_snr_gamma if args.min_snr_gamma > 0 else 'Disabled'}")
    print(f"  Gradient Clipping: {args.grad_clip if args.grad_clip > 0 else 'Disabled'}")
    print(f"  EMA: {'Enabled' if args.use_ema else 'Disabled'}")
    if args.use_ema:
        print(f"  EMA Decay: {args.ema_decay}")
    print(f"{'=' * 60}\n")
