#!/bin/bash

# Quick Start Script for DDPM-MNIST Training
# This script demonstrates basic usage with sensible defaults

echo "=========================================="
echo "DDPM-MNIST Quick Start Training"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import torch; import torchvision; import matplotlib; import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting training with default settings:"
echo "  - 50 epochs"
echo "  - Batch size: 128"
echo "  - Linear schedule"
echo "  - Noise prediction objective"
echo "  - EMA enabled"
echo ""

# Create output directory
mkdir -p outputs

# Run training
python train.py \
    --epochs 50 \
    --batch-size 128 \
    --schedule linear \
    --objective eps \
    --use-ema \
    --outdir ./outputs

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ./outputs/"
echo "  - Generated samples: samples_epoch_*.png"
echo "  - Checkpoints: checkpoint_epoch_*.pt"
echo "  - Loss curves: loss_curves.png"
echo ""
echo "To generate more samples from the trained model:"
echo "  python generate.py --checkpoint outputs/last.pt --num-samples 100"
echo ""
