"""
Stable RTRL Trainer with Lyapunov regularization support.

Features:
- Wraps model and optimizer
- Supports Lyapunov stability penalty
- Expert norm regularization
- Compatible with both MoE and RNN models
"""

import torch
import torch.nn as nn


class StableRTRLTrainer:
    """Trainer wrapper for stable RTRL training."""
    
    def __init__(self, model, lr=3e-3, lyapunov_weight=0.001, expert_norm_weight=0.001, weight_decay=1e-5):
        """
        Initialize trainer.
        
        Args:
            model: RecurrentMoE or ClassicRNN model
            lr: Learning rate
            lyapunov_weight: Weight for Lyapunov stability penalty
            expert_norm_weight: Weight for expert norm regularization
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.lr = lr
        self.lyapunov_weight = lyapunov_weight
        self.expert_norm_weight = expert_norm_weight
        self.weight_decay = weight_decay
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def compute_lyapunov_penalty(self, jacobians):
        """
        Compute Lyapunov stability penalty.
        
        Args:
            jacobians: List of Jacobian matrices from RTRL
            
        Returns:
            Scalar penalty term
        """
        if not jacobians or self.lyapunov_weight == 0:
            return torch.tensor(0.0, device=self.model.parameters().__next__().device)
        
        # Compute product of Jacobians and regularize norm
        penalty = 0.0
        for J in jacobians:
            if J is not None:
                # Regularize singular values to be close to 1
                U, S, Vt = torch.svd(J)
                # Penalize deviation from unit singular values
                penalty = penalty + torch.mean((S - 1.0) ** 2)
        
        return self.lyapunov_weight * penalty / max(len(jacobians), 1)
    
    def compute_expert_norm_penalty(self):
        """
        Compute expert norm regularization penalty.
        
        Returns:
            Scalar penalty term
        """
        if self.expert_norm_weight == 0:
            return torch.tensor(0.0, device=self.model.parameters().__next__().device)
        
        penalty = 0.0
        
        # Regularize expert parameters to have unit norm
        for name, param in self.model.named_parameters():
            if "expert" in name and param.grad is not None:
                penalty = penalty + torch.mean(param ** 2)
        
        return self.expert_norm_weight * penalty
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def step(self):
        """Optimizer step."""
        self.optimizer.step()
    
    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
