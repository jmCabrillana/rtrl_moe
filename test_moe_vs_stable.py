#!/usr/bin/env python3
"""
Compare moe.py vs moe_stable.py convergence + gradient norms (proxy for stability).
"""
import torch
import torch.nn.functional as F
from moe import RecurrentMoE as MoE
from moe_stable import RecurrentMoE as MoEStable
import random
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def generate_task(seq_len=64, num_samples=16):
    """Binary classification: predict if sequence has more 1s or 0s"""
    sequences = []
    targets = []
    for _ in range(num_samples):
        seq = torch.randint(0, 2, (seq_len,))
        target = (seq.sum() > seq_len // 2).long()
        sequences.append(seq)
        targets.append(target)
    return torch.stack(sequences), torch.stack(targets)

device = 'cpu'
D = 64
seq_len = 64

print("=" * 80)
print("Comparing MoE vs MoE-Stable (Gradient Norms = Stability Proxy)")
print("=" * 80)

# Models
model_moe = MoE(d_model=D, n_heads=4, n_slots=8, n_experts=4, topk=2, 
                 d_in=1, d_out=2).to(device)
model_stable = MoEStable(d_model=D, n_heads=4, n_slots=8, n_experts=4,
                         topk=2, topk_read=2, topk_write=2,
                         d_in=1, d_out=2).to(device)

opt_moe = torch.optim.Adam(model_moe.parameters(), lr=1e-3)
opt_stable = torch.optim.Adam(model_stable.parameters(), lr=1e-3)

print(f"{'Epoch':<8} {'MoE Loss':<12} {'MoE Acc':<10} {'MoE ∇norm':<12} {'Stable Loss':<12} {'Stable Acc':<10} {'Stable ∇norm':<12}")
print("-" * 80)

for epoch in range(3):
    seqs, targets = generate_task(seq_len, 16)
    seqs = seqs.to(device).float().unsqueeze(-1)
    targets = targets.to(device)
    
    moe_loss, moe_acc, moe_grad_norm = 0.0, 0.0, 0.0
    stable_loss, stable_acc, stable_grad_norm = 0.0, 0.0, 0.0
    
    for i in range(16):
        x = seqs[i:i+1]
        target = targets[i:i+1]
        
        # Standard MoE
        state_moe = model_moe.init_state(1, device=device)
        for t in range(seq_len):
            y_moe, info_moe, state_moe = model_moe(x[:, t:t+1, :], state_moe)
        
        loss_moe = F.cross_entropy(y_moe, target)
        opt_moe.zero_grad()
        loss_moe.backward()
        gn_moe = torch.nn.utils.clip_grad_norm_(model_moe.parameters(), 1.0)
        opt_moe.step()
        
        moe_loss += loss_moe.item()
        moe_acc += (y_moe.argmax(dim=1) == target).item()
        moe_grad_norm += gn_moe.item()
        
        # MoE-Stable (with Cayley + write selection)
        state_stable = model_stable.init_state(1, device=device)
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            y_stable, info_stable, state_stable = model_stable(x_t, state_stable)
        
        loss_stable = F.cross_entropy(y_stable, target)
        
        opt_stable.zero_grad()
        loss_stable.backward()
        gn_stable = torch.nn.utils.clip_grad_norm_(model_stable.parameters(), 1.0)
        opt_stable.step()
        
        stable_loss += loss_stable.item()
        stable_acc += (y_stable.argmax(dim=1) == target).item()
        stable_grad_norm += gn_stable.item()
    
    print(f"{epoch+1:<8} {moe_loss/16:<12.4f} {moe_acc/16*100:<10.1f} {moe_grad_norm/16:<12.4f} "
          f"{stable_loss/16:<12.4f} {stable_acc/16*100:<10.1f} {stable_grad_norm/16:<12.4f}")

print("=" * 80)
print("✓ Comparison completed!")
print("=" * 80)
