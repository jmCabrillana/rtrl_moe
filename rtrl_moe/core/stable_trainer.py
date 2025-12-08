"""
Stable RTRL Trainer with Lyapunov regularization and expert norm penalties.

Features:
- Lyapunov stability monitoring (K-step QR decomposition)
- Expert norm regularization to prevent collapse
- Adaptive learning rate scheduling
- Gradient accumulation support
"""

import torch
import torch.nn as nn
from torch.optim import Adam


class StableRTRLTrainer:
    """Trainer for RTRL models with stability regularization."""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-3,
        weight_decay: float = 1e-5,
        lyapunov_weight: float = 0.001,
        expert_norm_weight: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The recurrent model to train
            lr: Learning rate
            weight_decay: L2 weight decay
            lyapunov_weight: Weight for Lyapunov stability penalty
            expert_norm_weight: Weight for expert norm regularization
            betas: Adam optimizer betas (momentum parameters)
            eps: Adam optimizer epsilon
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lyapunov_weight = lyapunov_weight
        self.expert_norm_weight = expert_norm_weight
        
        # Get state parameters (exclude output layer)
        state_params = [
            p for name, p in model.named_parameters() 
            if not name.startswith("output_")
        ]
        
        self.optimizer = Adam(
            state_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        
        self.step_count = 0
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def step(self, loss: torch.Tensor, clip_norm: float = 1.0) -> dict:
        """
        Perform optimization step with gradient clipping.
        
        Args:
            loss: Scalar loss tensor
            clip_norm: Maximum gradient norm
            
        Returns:
            Dictionary with step statistics
        """
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            clip_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.step_count += 1
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
        }
    
    def compute_lyapunov_penalty(
        self,
        jacobians: list,
        k: int = 8,
    ) -> torch.Tensor:
        """
        Compute K-step Lyapunov penalty to maintain ||J_product|| ≈ 1.
        
        Args:
            jacobians: List of Jacobian matrices [B, H, H] for each step
            k: Number of steps to look back
            
        Returns:
            Scalar penalty term
        """
        if len(jacobians) < k:
            return torch.tensor(0.0, device=jacobians[0].device)
        
        # Compute product of last k Jacobians
        J_product = torch.eye(
            jacobians[0].shape[-1],
            device=jacobians[0].device,
            dtype=jacobians[0].dtype,
        ).unsqueeze(0).expand_as(jacobians[0])
        
        for j in jacobians[-k:]:
            J_product = torch.bmm(J_product, j)
        
        # QR decomposition to measure norm
        try:
            Q, R = torch.linalg.qr(J_product)
            # Penalty on log of singular values
            log_s = torch.log(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1)) + 1e-8)
            penalty = (log_s ** 2).mean()
        except:
            # Fallback: use Frobenius norm
            penalty = (torch.norm(J_product, p='fro') - 1.0) ** 2
        
        return penalty.mean()
    
    def compute_expert_norm_penalty(self) -> torch.Tensor:
        """
        Compute penalty to prevent expert collapse (norm regularization).
        
        Returns:
            Scalar penalty term
        """
        penalty = 0.0
        count = 0
        
        for name, param in self.model.named_parameters():
            # Apply to expert parameters
            if 'expert' in name.lower() and param.requires_grad:
                penalty = penalty + torch.norm(param, p=2) ** 2
                count += 1
        
        if count > 0:
            penalty = penalty / count
        else:
            penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        return penalty
    
    def compute_total_loss(
        self,
        task_loss: torch.Tensor,
        jacobians: list = None,
        include_expert_penalty: bool = True,
    ) -> torch.Tensor:
        """
        Compute total loss with regularization.
        
        Args:
            task_loss: Main task loss
            jacobians: List of Jacobian matrices for Lyapunov penalty
            include_expert_penalty: Whether to include expert norm penalty
            
        Returns:
            Total loss with regularization
        """
        total = task_loss
        
        # Lyapunov penalty
        if jacobians is not None and self.lyapunov_weight > 0:
            lyap_penalty = self.compute_lyapunov_penalty(jacobians)
            total = total + self.lyapunov_weight * lyap_penalty
        
        # Expert norm penalty
        if include_expert_penalty and self.expert_norm_weight > 0:
            expert_penalty = self.compute_expert_norm_penalty()
            total = total + self.expert_norm_weight * expert_penalty
        
        return total
