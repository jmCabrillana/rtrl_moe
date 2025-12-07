#!/usr/bin/env python3
"""
Compare moe.py vs moe_stable.py convergence with Lyapunov stability.
"""
import torch
import torch.nn.functional as F
from moe import RecurrentMoE as MoE
from moe_stable import RecurrentMoE as MoEStable, compute_lyapunov_penalty
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
print("Comparing MoE vs MoE-Stable (with Lyapunov Stability)")
print("=" * 80)

# Models
model_moe = MoE(d_model=D, n_heads=4, n_slots=8, n_experts=4, topk=2, 
                 d_in=1, d_out=2).to(device)
model_stable = MoEStable(d_model=D, n_heads=4, n_slots=8, n_experts=4,
                         topk=2, topk_read=2, topk_write=2,
                         d_in=1, d_out=2).to(device)

opt_moe = torch.optim.Adam(model_moe.parameters(), lr=1e-3)
opt_stable = torch.optim.Adam(model_stable.parameters(), lr=1e-3)

print(f"{'Epoch':<8} {'MoE Loss':<12} {'MoE Acc':<10} {'Stable Loss':<14} {'Stable Acc':<12} {'Lyap Penalty':<14}")
print("-" * 80)

for epoch in range(3):
    seqs, targets = generate_task(seq_len, 16)
    seqs = seqs.to(device).float().unsqueeze(-1)
    targets = targets.to(device)
    
    moe_loss, moe_acc = 0.0, 0.0
    stable_loss, stable_acc = 0.0, 0.0
    lyap_penalty_avg = 0.0
    
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
        torch.nn.utils.clip_grad_norm_(model_moe.parameters(), 1.0)
        opt_moe.step()
        
        moe_loss += loss_moe.item()
        moe_acc += (y_moe.argmax(dim=1) == target).item()
        
        # MoE-Stable with Lyapunov
        state_stable = model_stable.init_state(1, device=device)
        x_window = []
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            y_stable, info_stable, state_stable = model_stable(x_t, state_stable)
            x_window.append(x_t)
        
        loss_stable = F.cross_entropy(y_stable, target)
        
        # Compute Lyapunov penalty every few steps
        if i % 4 == 0:
            try:
                x_window_batch = torch.cat(x_window, dim=1)  # [1, seq_len, 1]
                lyap_penalty = compute_lyapunov_penalty(model_stable, state_stable, 
                                                        x_window_batch, K=8, probes=2)
                lyap_penalty_avg += lyap_penalty.item() * 0.001  # Weight
                loss_stable = loss_stable + 0.001 * lyap_penalty
            except:
                pass
        
        opt_stable.zero_grad()
        loss_stable.backward()
        torch.nn.utils.clip_grad_norm_(model_stable.parameters(), 1.0)
        opt_stable.step()
        
        stable_loss += loss_stable.item()
        stable_acc += (y_stable.argmax(dim=1) == target).item()
    
    print(f"{epoch+1:<8} {moe_loss/16:<12.4f} {moe_acc/16*100:<10.1f} "
          f"{stable_loss/16:<14.4f} {stable_acc/16*100:<12.1f} {lyap_penalty_avg/4:<14.4f}")

print("=" * 80)
print("✓ Comparison completed!")
print("=" * 80)
