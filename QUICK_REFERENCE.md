# 📋 DDPM-MNIST Quick Reference

**Your go-to cheat sheet for common operations**

---

## 🚀 Essential Commands

### Training
```bash
# Basic training (default settings)
python train.py

# Custom configuration
python train.py --epochs 100 --batch-size 256 --lr 2e-4

# Resume from checkpoint
python train.py --resume outputs/last.pt

# Advanced: cosine schedule + velocity prediction
python train.py --schedule cosine --objective v --min-snr-gamma 5
```

### Sample Generation
```bash
# Generate 64 samples
python generate.py --checkpoint outputs/last.pt

# Generate 100 samples with fewer steps (faster)
python generate.py --checkpoint outputs/last.pt --num-samples 100 --steps 100
```

### Git Operations
```bash
# Initial setup
git init
git add .
git commit -m "Initial commit: Professional DDPM implementation"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/ddpm-mnist.git
git push -u origin main

# Making changes
git status
git add .
git commit -m "Descriptive message"
git push
```

---

## 📁 File Organization

```
ddpm-mnist/
├── train.py              # Main training script
├── generate.py           # Sample generation
├── models/               # Neural network code
│   ├── unet.py          # U-Net architecture
│   └── ddpm.py          # Diffusion model
├── utils/                # Helper functions
│   ├── ema.py           # EMA implementation
│   └── training.py      # Training utilities
├── examples/             # Usage examples
├── .github/workflows/    # CI/CD pipeline
└── *.md                  # Documentation
```

---

## ⚙️ Configuration Options

### Model Architecture
- `--base 64` - U-Net base channels
- `--timesteps 1000` - Diffusion steps

### Training
- `--epochs 50` - Training epochs
- `--batch-size 128` - Batch size
- `--lr 2e-4` - Learning rate
- `--grad-clip 1.0` - Gradient clipping

### Diffusion Settings
- `--schedule linear|cosine|sigmoid` - Noise schedule
- `--objective eps|x0|v` - Training objective
- `--min-snr-gamma 5.0` - Min-SNR weighting

### EMA
- `--use-ema` - Enable EMA (default)
- `--no-ema` - Disable EMA
- `--ema-decay 0.9999` - EMA decay rate

---

## 📊 Training Objectives

| Objective | What it predicts | Best for |
|-----------|------------------|----------|
| `eps` | Noise ε | Most stable (default) |
| `x0` | Original image | Direct prediction |
| `v` | Velocity | Few-step sampling |

---

## 📈 Noise Schedules

| Schedule | Characteristics | Best for |
|----------|-----------------|----------|
| `linear` | Simple, stable | Low-res images (default) |
| `cosine` | Smoother | High-res images |
| `sigmoid` | Hybrid | Balanced approach |

---

## 🎯 Common Workflows

### Experiment 1: Quick Test
```bash
python train.py --epochs 10 --batch-size 256
```

### Experiment 2: Best Quality
```bash
python train.py --epochs 20 --schedule cosine --objective v \
  --min-snr-gamma 0.2
```

### Experiment 3: Fast Training
```bash
python train.py --epochs 20 --batch-size 512 --timesteps 500
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
python train.py --batch-size 64  # Reduce batch size
```

### Training Too Slow
```bash
python train.py --num-workers 8  # More data loader workers
python train.py --timesteps 500  # Fewer timesteps
```

### Want Better Samples
```bash
python train.py --use-ema --schedule cosine
```

---

## 📦 Package Structure

Import in your own code:
```python
from models import TinyUNet, DDPM
from utils import EMA, get_device, evaluate

# Create model
device = get_device()
net = TinyUNet(base=64).to(device)
ddpm = DDPM(net, timesteps=1000, schedule='linear')

# Generate samples
samples = ddpm.sample((16, 1, 28, 28), device=device)
```

---

## 🔍 Monitoring Training

### Check Progress
- Watch terminal output for loss values
- Generated samples saved every epoch: `outputs/samples_epoch_*.png`
- Loss curves: `outputs/loss_curves.png`
- Checkpoints: `outputs/checkpoint_epoch_*.pt`

### Best Checkpoint
Always use `outputs/last.pt` for the most recent weights, or specific 
`checkpoint_epoch_XXX.pt` files.

---

## 🎨 Output Files

| File | Description |
|------|-------------|
| `samples_epoch_001.png` | Generated samples (64 images) |
| `checkpoint_epoch_010.pt` | Model checkpoint |
| `last.pt` | Latest checkpoint (for resuming) |
| `loss_curves.png` | Training/eval loss plot |

---

## 💡 Pro Tips

1. **Cosine schedule** often works better than linear
2. **Min-SNR gamma=0.2** is a good default
3. **Resume training** if interrupted (use `--resume outputs/last.pt`)
4. **Velocity objective** gives best results
5. **Generate samples** during training to monitor quality

---

## 🆘 Need Help?

1. **Check Documentation**
   - `README.md` - Detailed usage
   - `PROJECT_OVERVIEW.md` - Architecture details

2. **Code Documentation**
   - Every function has docstrings
   - Comments explain complex logic
   - Type hints for IDE support

3. **Examples**
   - `examples/quick_start.sh` - Bash script
   - `examples/train_programmatic.py` - Python API

---

## 📞 Quick Links

- **Train**: `python train.py --help`
- **Generate**: `python generate.py --help`
- **GitHub**: Follow `GIT_SETUP_GUIDE.md`
- **Contributing**: See `CONTRIBUTING.md`

---

Happy training! 🚀
