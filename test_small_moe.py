"""
Test if small MoE (same size as quick_verify) can learn with BPTT.
"""

import torch
import torch.nn as nn
import random
from moe import RecurrentMoE

device = torch.device("cpu")
torch.manual_seed(42)

print("Testing small MoE (d=32, slots=4, experts=4)")

model = RecurrentMoE(d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

def make_seq():
    n = random.randint(1, 3)
    if random.random() < 0.5:
        seq = torch.cat([torch.tensor([[1., 0.]]).repeat(n, 1), torch.tensor([[0., 1.]]).repeat(n, 1)])
        tgt = torch.tensor([1])
    else:
        m = random.choice([k for k in range(1, 4) if k != n])
        seq = torch.cat([torch.tensor([[1., 0.]]).repeat(n, 1), torch.tensor([[0., 1.]]).repeat(m, 1)])
        tgt = torch.tensor([0])
    return tgt.to(device), seq.to(device)

losses = []
correct = []

for step in range(1500):
    tgt, x_seq = make_seq()
    h_t = model.init_state(1, device=device)
    
    x_batch = x_seq.unsqueeze(0)  # [B, T, D]
    y, info, h_next = model(x_batch, h_t)
    
    loss = criterion(y, tgt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    pred = y.argmax(dim=-1).item()
    correct.append(1 if pred == tgt.item() else 0)
    
    if (step + 1) % 150 == 0:
        avg_loss = sum(losses[-150:]) / min(150, len(losses))
        avg_acc = sum(correct[-150:]) / min(150, len(correct)) * 100
        print(f"Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")

early_loss = sum(losses[:150]) / 150
late_loss = sum(losses[-150:]) / 150
early_acc = sum(correct[:150]) / 150 * 100
late_acc = sum(correct[-150:]) / 150 * 100

print(f"\nResults:")
print(f"  Initial: Loss = {early_loss:.4f}, Acc = {early_acc:.1f}%")
print(f"  Final:   Loss = {late_loss:.4f}, Acc = {late_acc:.1f}%")

if late_acc > 80:
    print("✓ Small MoE can learn")
else:
    print(f"✗ Small MoE struggles (final acc={late_acc:.1f}%)")
