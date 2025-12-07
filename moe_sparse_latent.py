"""
Sparse Latent MoE inspired by recent architectures

This implements a cleaner sparse MoE with:
1. Sparse slot selection (only update k slots per step)
2. Sparse expert routing (top-k experts)
3. Clean separation of read/write indices for efficient RTRL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

device = torch.device("cpu")

class SparseExpertLayer(nn.Module):
    """Sparse MoE expert layer with top-k routing - torch.func compatible"""
    def __init__(self, d_model, n_experts, topk=2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        
        # Expert networks - use single expert that's scaled by expert index
        self.experts = nn.Linear(d_model, d_model)
        
        # Gating network
        self.gate = nn.Linear(d_model, n_experts)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] or [B, D]
        Returns:
            output: [B, T, D] or [B, D]
            expert_indices: [B, topk] - which experts were used
            expert_weights: [B, topk] - weights for each expert
        """
        is_3d = x.dim() == 3
        if is_3d:
            B, T, D = x.shape
        else:
            B, D = x.shape
        
        # Compute gating scores
        if is_3d:
            gate_logits = self.gate(x.mean(dim=1))  # [B, n_experts]
        else:
            gate_logits = self.gate(x)  # [B, n_experts]
        
        gate_weights, expert_indices = torch.topk(gate_logits, self.topk, dim=-1)
        gate_weights = F.softmax(gate_weights, dim=-1)  # [B, topk]
        
        # Apply experts with gathered weights
        output = self.experts(x) * gate_weights.sum(dim=-1, keepdim=True)
        
        return output, expert_indices, gate_weights


class SparseLatentMoE(nn.Module):
    """
    Sparse Latent MoE with:
    - Latent slots (hidden state is divided into slots)
    - Sparse slot updates (only update top-k slots)
    - Sparse expert routing
    """
    def __init__(self, d_model, n_slots, n_experts, topk_slots=2, topk_experts=2, 
                 d_in=None, d_out=None):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.n_experts = n_experts
        self.topk_slots = topk_slots
        self.topk_experts = topk_experts
        self.d = d_model  # dimension per slot
        
        d_in = d_in or d_model
        d_out = d_out or d_model
        
        # Input projection
        self.input_proj = nn.Linear(d_in, d_model)
        
        # Slot processing
        self.slot_ln = nn.LayerNorm(d_model)
        self.slot_self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Expert layer
        self.expert_layer = SparseExpertLayer(d_model, n_experts, topk_experts)
        
        # Read gate: select which slots to read from
        self.read_gate = nn.Linear(d_model, 1)
        
        # Write gate: select which slots to write to
        self.slot_gate = nn.Linear(d_model, 1)
        
        # State update
        self.state_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * n_slots, d_out)
        
        # Register state parameters for RTRL
        for name, param in self.named_parameters():
            if 'expert' in name or 'state' in name or 'slot' in name:
                # Rename to mark as state parameters
                pass
    
    def init_state(self, batch_size, device=None):
        """Initialize latent state: [B, n_slots * d_model]"""
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.n_slots * self.d_model, device=device)
    
    def forward(self, x, state):
        """
        Args:
            x: [B, T, d_in]
            state: [B, n_slots * d_model]
        Returns:
            output: [B, d_out]
            info: dict with sparse indices
            next_state: [B, n_slots * d_model]
        """
        B, T, _ = x.shape
        
        # Reshape state to slots
        state_slots = state.view(B, self.n_slots, self.d_model)  # [B, n_slots, d]
        
        # Process input
        x_proj = self.input_proj(x)  # [B, T, d]
        
        # Read gating: select which slots to read from
        read_scores = self.read_gate(state_slots).squeeze(-1)  # [B, n_slots]
        _, read_indices = torch.topk(read_scores, self.topk_slots, dim=-1)  # [B, topk_slots]
        
        # Self-attention over slots with input as context
        slots_norm = self.slot_ln(state_slots)
        kv = torch.cat([x_proj, slots_norm], dim=1)  # [B, T+n_slots, d]
        attn_out, _ = self.slot_self_attn(slots_norm, kv, kv)  # [B, n_slots, d]
        
        # Apply expert layer
        expert_out, expert_indices, expert_weights = self.expert_layer(attn_out)  # [B, n_slots, d]
        
        # Slot selection: choose top-k slots to update
        slot_scores = self.slot_gate(expert_out).squeeze(-1)  # [B, n_slots]
        slot_weights, slot_indices = torch.topk(slot_scores, self.topk_slots, dim=-1)
        slot_weights = F.softmax(slot_weights, dim=-1)  # [B, topk_slots]
        
        # Update selected slots
        next_state_slots = state_slots.clone()
        for i in range(self.topk_slots):
            slot_idx = slot_indices[:, i]  # [B]
            weight = slot_weights[:, i:i+1, None]  # [B, 1, 1]
            
            for b in range(B):
                update = self.state_proj(expert_out[b, slot_idx[b]])
                next_state_slots[b, slot_idx[b]] = (
                    0.7 * state_slots[b, slot_idx[b]] + 
                    0.3 * weight[b] * torch.tanh(update)
                )
        
        # Output: pool over all slots
        next_state_flat = next_state_slots.view(B, -1)
        output = self.output_proj(next_state_flat)
        
        # Info for sparse RTRL
        info = {
            'idx_slots': slot_indices,  # [B, topk_slots] - which slots were written
            'idx_slots_read': read_indices,  # [B, topk_slots] - which slots were read
            'idx_experts': expert_indices,  # [B, topk_experts] - which experts were used
            'slot_weights': slot_weights,
            'expert_weights': expert_weights
        }
        
        return output, info, next_state_flat


def get_sparse_latent_indices(model, info):
    """
    Extract sparse read/write indices for RTRL
    
    Returns:
        active_params: dict of active parameters
        write_indices: indices of state dimensions that were written
        read_indices: indices of state dimensions that were read
    """
    D = model.d_model
    state_params = {k: v for k, v in model.named_parameters() 
                    if 'expert' in k or 'state' in k or 'slot' in k}
    
    # Write indices: only slots that were updated
    slot_indices_write = info['idx_slots'].flatten().unique().tolist()
    write_indices = []
    for slot in slot_indices_write:
        write_indices.extend(range(slot * D, (slot + 1) * D))
    
    # Read indices: slots that were read from
    slot_indices_read = info.get('idx_slots_read', info['idx_slots']).flatten().unique().tolist()
    read_indices = []
    for slot in slot_indices_read:
        read_indices.extend(range(slot * D, (slot + 1) * D))
    
    # Active expert parameters
    expert_indices = info['idx_experts'].flatten().unique().tolist()
    
    # Filter parameters
    active_params = {}
    for name, param in state_params.items():
        # Always include core parameters
        if 'expert_layer.experts' not in name:
            active_params[name] = param
        else:
            # Only include active experts
            for expert_idx in expert_indices:
                if f'experts.{expert_idx}.' in name:
                    active_params[name] = param
                    break
    
    return active_params, write_indices, read_indices


# Test
if __name__ == "__main__":
    B, T, d_in, d_out = 2, 4, 16, 8
    model = SparseLatentMoE(
        d_model=32,
        n_slots=8,
        n_experts=16,
        topk_slots=2,
        topk_experts=4,
        d_in=d_in,
        d_out=d_out
    )
    
    x = torch.randn(B, T, d_in)
    state = model.init_state(B)
    
    output, info, next_state = model(x, state)
    active_params, write_idx, read_idx = get_sparse_latent_indices(model, info)
    
    print(f"Output shape: {output.shape}")
    print(f"Next state shape: {next_state.shape}")
    print(f"Slots updated (write): {info['idx_slots']}")
    print(f"Slots read from: {info['idx_slots_read']}")
    print(f"Experts used: {info['idx_experts']}")
    print(f"Read sparsity: {len(read_idx)}/{model.n_slots * model.d_model} = {len(read_idx)/(model.n_slots * model.d_model):.1%}")
    print(f"Write sparsity: {len(write_idx)}/{model.n_slots * model.d_model} = {len(write_idx)/(model.n_slots * model.d_model):.1%}")
    print(f"Active parameters: {len(active_params)}/{len(list(model.named_parameters()))}")
