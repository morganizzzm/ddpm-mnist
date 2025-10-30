"""
Exponential Moving Average (EMA) for model parameters.

Provides stable weights for evaluation and sampling by maintaining
a shadow copy of model parameters that are smoothly updated.
"""

import torch


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a shadow copy of model weights updated using:
        θ_shadow ← decay * θ_shadow + (1 - decay) * θ_current

    Args:
        model (nn.Module): Model to track with EMA
        decay (float): EMA decay rate (typical: 0.999 - 0.9999)
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """
        Update EMA parameters after training step.
        
        Args:
            model (nn.Module): Model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        """
        Replace model parameters with EMA parameters.
        Must call restore() afterwards.
        
        Args:
            model (nn.Module): Model to apply EMA weights to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """
        Restore original model parameters.
        
        Args:
            model (nn.Module): Model to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]
        self.original = {}

    def state_dict(self):
        """Save EMA state for checkpointing."""
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
