"""
Simple, stable RNN for counting tasks (AnBn, Haystack).

Key design:
- Full state updates (no sparse gating to break causality)
- Layer normalization for stability
- Simple gated recurrent unit-like dynamics
- Bounded state norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    """Minimal stable recurrent model for sequence tasks."""
    
    def __init__(self, d_model=32, d_in=4, d_out=2):
        super().__init__()
        self.d_model = d_model
        self.d_in = d_in
        self.d_out = d_out
        
        # Input embedding
        self.embed = nn.Linear(d_in, d_model)
        
        # Recurrent cell: h_next = (1-α)h + α*φ(W_h*h + W_x*x)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=True)
        self.gate = nn.Linear(d_model, d_model, bias=True)  # for gating
        self.ln = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_out)
        
    @torch.no_grad()
    def init_state(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
    
    def forward(self, x, h_prev):
        """
        Process one timestep.
        
        Args:
            x: [B, 1, d_in] - one token (one-hot)
            h_prev: [B, d_model] - previous state
            
        Returns:
            logits: [B, d_out] - output logits
            info: dict with routing info (for RTRL compatibility)
            h_next: [B, d_model] - next state
        """
        B = x.shape[0]
        
        # Embed input
        x_emb = self.embed(x.squeeze(1))  # [B, d_model]
        
        # Gated update: h_next = (1-α)*h_prev + α*φ(h_contrib)
        h_contrib = self.W_h(h_prev) + self.W_x(x_emb)
        h_contrib = F.relu(h_contrib)  # Activation
        
        # Gate: controls how much to update
        α = torch.sigmoid(self.gate(h_prev))  # [B, d_model] in [0, 1]
        
        # Update with residual: new state is weighted combination
        h_next = (1 - α) * h_prev + α * h_contrib
        
        # Layer norm for stability
        h_next = self.ln(h_next)
        
        # Output (from final state at the end)
        logits = self.output_proj(h_next)  # [B, d_out]
        
        # Info for RTRL compatibility (full state, no sparse gating)
        info = {
            'idx_experts': torch.tensor([0], device=x.device),
            'idx_slot_write': list(range(self.d_model)),
            'idx_slots_read': list(range(self.d_model))
        }
        
        return logits, info, h_next


def get_active_params_simple(model, info):
    """Extract active parameters for RTRL (everything is active)."""
    state_params = {k: v for k, v in model.named_parameters() 
                    if not k.startswith('output_')}
    # For SimpleRNN, all state params are active (no sparse gating)
    write_indices = list(range(model.d_model))
    read_indices = list(range(model.d_model))
    return state_params, write_indices, read_indices
