# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added
- Initial release of DDPM-MNIST
- Modular architecture with separate models and utils packages
- TinyUNet implementation with FiLM conditioning
- DDPM implementation with multiple noise schedules (linear, cosine, sigmoid)
- Support for multiple training objectives (ε-prediction, x₀-prediction, v-prediction)
- Min-SNR weighting for improved training stability
- Exponential Moving Average (EMA) support
- Comprehensive checkpointing and resuming functionality
- Automatic sample generation during training
- Loss curve visualization
- Command-line interface with extensive configuration options
- Example configurations and usage patterns

### Features
- ~2M parameter U-Net optimized for 28×28 images
- Automatic device selection (CUDA, MPS, CPU)
- Gradient clipping for training stability
- GroupNorm for small batch stability
- SiLU activation functions
- Skip connections in U-Net
- Sinusoidal time embeddings
- AdamW optimizer with weight decay
- Data augmentation support
- Flexible train/eval/test splits


## [Unreleased]

### Planned Features
- Multi-GPU training support
- CIFAR-10 and other dataset support
- Conditional generation
- Classifier-free guidance
- DDIM sampling for faster generation
- FID/IS evaluation metrics
- Jupyter notebook tutorials
- Pre-trained model weights
- Docker support
- Weights & Biases integration

---

For more details on each release, see the [GitHub releases page](https://github.com/yourusername/ddpm-mnist/releases).
