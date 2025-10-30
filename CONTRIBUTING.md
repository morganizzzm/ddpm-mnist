# Contributing to DDPM-MNIST

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, PyTorch version)
- **Code snippets or error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** - why is this enhancement useful?
- **Detailed description** of the proposed functionality
- **Possible implementation** approach (if you have ideas)

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes** with clear, commented code
3. **Add tests** if applicable
4. **Update documentation** (README, docstrings, etc.)
5. **Follow the existing code style**
6. **Write a clear commit message**

#### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add type hints where helpful
- Write comments for complex logic

#### Example of Good Code Style

```python
def compute_loss(model, batch, timesteps):
    """
    Compute the diffusion loss for a batch.
    
    Args:
        model (DDPM): Diffusion model
        batch (torch.Tensor): Batch of images [B, C, H, W]
        timesteps (torch.Tensor): Timesteps [B]
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Sample noise
    noise = torch.randn_like(batch)
    
    # Forward diffusion
    noisy_batch = model.q_sample(batch, timesteps, noise)
    
    # Predict and compute loss
    prediction = model(noisy_batch, timesteps)
    loss = F.mse_loss(prediction, noise)
    
    return loss
```

### Project Structure

When adding new features, follow the existing structure:

```
models/     - Neural network architectures and diffusion logic
utils/      - Utility functions, helpers, and training tools
train.py    - Main training script
```

### Testing

Before submitting a PR:

1. Ensure your code runs without errors
2. Test on different configurations (if applicable)
3. Verify documentation is up-to-date

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

Examples:
```
Add cosine noise schedule implementation
Fix gradient clipping bug in training loop
Update README with new configuration options
Refactor U-Net architecture for clarity
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ddpm-mnist.git
cd ddpm-mnist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make your changes
git checkout -b my-feature-branch

# Test your changes
python train.py --epochs 1  # Quick sanity check

# Commit and push
git add .
git commit -m "Add my feature"
git push origin my-feature-branch
```

## Questions?

Feel free to open an issue for any questions or clarifications!

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience
- Nationality, personal appearance, race
- Religion, or sexual identity and orientation

### Our Standards

**Examples of encouraged behavior:**
- Being respectful and professional
- Providing constructive feedback
- Focusing on what's best for the community
- Showing empathy towards others

**Examples of unacceptable behavior:**
- Harassment, trolling, or discriminatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Attribution

This Contributing guide is adapted from open-source contribution guidelines.

---

Thank you for contributing! 🚀
