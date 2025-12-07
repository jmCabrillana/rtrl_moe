#!/usr/bin/env python3
"""
Quick convergence test for moe_stable.py on binary classification task.
"""
import torch
import torch.nn.functional as F
from moe_stable import RecurrentMoE
import random
import numpy as np

# Set seeds
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

# Setup
device = 'cpu'
D = 64
seq_len = 64

print("=" * 70)
print("MoE-Stable Convergence Test (Binary Classification)")
print("=" * 70)

# Model
model = RecurrentMoE(d_model=D, n_heads=4, n_slots=8, n_experts=4,
                     topk=2, topk_read=2, topk_write=2,
                     d_in=1, d_out=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(3):
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    seqs, targets = generate_task(seq_len, 16)
    seqs = seqs.to(device).float().unsqueeze(-1)
    targets = targets.to(device)
    
    for i in range(16):
        x = seqs[i:i+1]
        target = targets[i:i+1]
        
        state = model.init_state(1, device=device)
        
        for t in range(seq_len):
            y, info, state = model(x[:, t:t+1, :], state)
        
        loss = F.cross_entropy(y, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += (y.argmax(dim=1) == target).item()
    
    print(f"Epoch {epoch+1}: Loss={epoch_loss/16:.4f}, Acc={epoch_acc/16*100:.1f}%")

print("=" * 70)
print("✓ Test completed!")
print("=" * 70)
