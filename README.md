# DDPM for MNIST 🎨

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for generating MNIST digits. This implementation emphasizes code quality, documentation, and ease of use.

<p align="center">
  <img src="/assets/samples.png" alt="Generated MNIST samples" width="400"/>
</p>

## ✨ Features

- 🏗️ **Modular Architecture** - Clean separation of concerns with well-organized modules
- 🎯 **Multiple Training Objectives** - Support for ε-prediction, x₀-prediction, and v-prediction
- 📊 **Flexible Noise Schedules** - Linear, cosine, and sigmoid schedules
- 🎚️ **Min-SNR Weighting** - Improved training stability across timesteps
- 📈 **EMA Support** - Exponential Moving Average for stable, high-quality sampling
- 💾 **Comprehensive Checkpointing** - Save and resume training seamlessly
- 📉 **Training Visualization** - Automatic loss curve plotting and sample generation
- 🔧 **Easy Configuration** - Extensive command-line arguments for customization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ddpm-mnist.git
cd ddpm-mnist

# Install dependencies
pip install -r requirements.txt
```

### Basic Training

```bash
python train.py
```

This will train DDPM with default settings (50 epochs, 1000 timesteps, linear schedule).

### Advanced Training

```bash
python train.py \
  --epochs 100 \
  --batch-size 256 \
  --schedule cosine \
  --objective v \
  --use-ema \
  --outdir ./my_experiment
```

## 📁 Project Structure

```
ddpm-mnist/
│
├── models/
│   ├── __init__.py
│   ├── unet.py          # U-Net architecture with FiLM conditioning
│   └── ddpm.py          # DDPM implementation with noise schedules
│
├── utils/
│   ├── __init__.py
│   ├── ema.py           # Exponential Moving Average
│   └── training.py      # Training utilities and evaluation
│
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── LICENSE             # MIT license
```

## 🎓 Model Architecture

### U-Net with FiLM Conditioning

```
Input (28×28)
    ↓
[Initial Conv] → 64 channels
    ↓
┌─ [Encoder Block 1] (28×28, 64) ───┐
│   ↓ [Downsample]                   │
│  [Encoder Block 2] (14×14, 128) ───┤
│   ↓ [Downsample]                   │
│  [Bottleneck] (7×7, 128)           │
│   ↓ [Upsample]                     │
│  [Decoder Block 1] (14×14, 128) ───┘
│   ↓ [Upsample]
└→ [Decoder Block 2] (28×28, 64)
    ↓
[Output Conv] → 1 channel
```

**Key Components:**
- **Residual Blocks** with GroupNorm and SiLU activation
- **FiLM Conditioning**: `output = features * (1 + γ(t)) + β(t)`
- **Skip Connections** for gradient flow
- **~2M parameters** optimized for 28×28 images

### Time Embedding

Sinusoidal positional encoding (similar to Transformers) followed by a 2-layer MLP:

```python
PE(t, 2i) = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

## ⚙️ Configuration

### Command-Line Arguments

#### Model Architecture
```bash
--base 64              # Base number of U-Net channels
--timesteps 1000       # Number of diffusion timesteps
```

#### Training Hyperparameters
```bash
--epochs 50            # Number of training epochs
--batch-size 128       # Batch size
--lr 2e-4              # Learning rate
--grad-clip 1.0        # Gradient clipping (0 to disable)
```

#### Diffusion Settings
```bash
--schedule linear      # Noise schedule: linear, cosine, sigmoid
--objective eps        # Training objective: eps, x0, v
--min-snr-gamma 5.0    # Min-SNR weighting (0 to disable)
```

#### EMA Settings
```bash
--use-ema              # Enable EMA (default)
--no-ema               # Disable EMA
--ema-decay 0.9999     # EMA decay rate
```

#### Data Settings
```bash
--train-size 50000     # Training set size
--eval-size 10000      # Evaluation set size
--num-workers 4        # DataLoader workers
```

#### Checkpointing
```bash
--save-every 10        # Save checkpoint frequency
--resume path/to/checkpoint.pt
--outdir ./outputs     # Output directory
```

## 🎯 Training Objectives

### 1. Noise Prediction (ε-prediction) [Default]

The model predicts the noise added to the clean image:

```
Loss = ||ε - ε_θ(x_t, t)||²
```

**Pros**: Most stable, commonly used in practice

### 2. Image Prediction (x₀-prediction)

The model directly predicts the original clean image:

```
Loss = ||x_0 - x_θ(x_t, t)||²
```

**Pros**: Can provide better sample quality in some cases

### 3. Velocity Prediction (v-prediction)

The model predicts a velocity parameterization:

```
v = √ᾱ_t * ε - √(1-ᾱ_t) * x_0
Loss = ||v - v_θ(x_t, t)||²
```

**Pros**: Better for few-step sampling, more balanced training

## 📈 Noise Schedules

### Linear Schedule [Default]
```python
β_t increases linearly from 0.0001 to 0.02
```
Simple and effective for low-resolution images.

### Cosine Schedule
```python
α̅_t = cos²((t/T + s)/(1 + s) * π/2)
```
Smoother transitions, better for high-resolution images.

### Sigmoid Schedule
```python
β_t = sigmoid(linspace(-6, 6, T)) * (β_max - β_min) + β_min
```
Hybrid approach combining benefits of both schedules.

## 🎨 Min-SNR Weighting

Improves training stability by capping loss weights at high noise levels:

```python
weight = min(SNR(t), γ) / SNR(t)
where SNR(t) = α̅_t / (1 - α̅_t)
```

Recommended γ values: 1-10 (default: 5)

## 💾 Outputs

Training generates the following in `--outdir`:

```
outputs/
├── samples_epoch_001.png    # Generated samples
├── samples_epoch_002.png
├── ...
├── checkpoint_epoch_010.pt  # Periodic checkpoints
├── checkpoint_epoch_020.pt
├── ...
├── last.pt                  # Latest checkpoint (for resuming)
└── loss_curves.png          # Training/eval loss visualization
```

## 📊 Example Results

Training on MNIST for 50 epochs with default settings:

```
Epoch 50 - Train Loss: 0.0234 - Eval Loss: 0.0241
Model parameters: 2,047,745
Test Loss: 0.0239
```

## 🔄 Resuming Training

To resume from a saved checkpoint:

```bash
python train.py --resume outputs/last.pt
```

The script automatically loads:
- Model weights
- Optimizer state
- EMA parameters
- Training history
- Configuration

## 🧪 Advanced Usage

### Experiment with Different Configurations

```bash
# Cosine schedule with velocity prediction
python train.py --schedule cosine --objective v --epochs 100

# High-capacity model without EMA
python train.py --base 128 --no-ema

# Strong Min-SNR weighting
python train.py --min-snr-gamma 10.0

# Custom data split
python train.py --train-size 55000 --eval-size 5000
```

### Hyperparameter Search

```bash
for lr in 1e-4 2e-4 5e-4; do
  for schedule in linear cosine; do
    python train.py --lr $lr --schedule $schedule --outdir "exp_${lr}_${schedule}"
  done
done
```

## 📚 References

### Papers

1. **Denoising Diffusion Probabilistic Models**  
   Ho et al., NeurIPS 2020  
   [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

2. **Improved Denoising Diffusion Probabilistic Models**  
   Nichol & Dhariwal, ICML 2021  
   [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

3. **Progressive Distillation for Fast Sampling**  
   Salimans & Ho, ICLR 2022  
   [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

4. **Perception Prioritized Training of Diffusion Models**  
   Hang et al., NeurIPS 2023  
   [arXiv:2204.00227](https://arxiv.org/abs/2204.00227)

### Code Inspirations

- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:
- The authors of the original DDPM paper
- The PyTorch team for an excellent framework
- The open-source community for inspiration and guidance

## 📧 Contact

For questions or feedback, please open an issue or reach out to [morganizzzm@gmail.com](mailto:your.email@example.com).

---

<p align="center">
  Made with ❤️ by morganizzzm
</p>

<p align="center">
  ⭐ Star this repo if you find it helpful!
</p>
