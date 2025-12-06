"""
Analyze why MoE doesn't reach zero loss like dense RNN.
Test: longer training, better gating, architectural tweaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from moe import RecurrentMoE

device = torch.device("cpu")

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

def train_model(model, name, steps=3000, lr=1e-2):
    """Train a model and return final performance"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    correct = []
    
    for step in range(steps):
        tgt, x_seq = make_seq()
        h_t = model.init_state(1, device=device)
        
        x_batch = x_seq.unsqueeze(0)
        y, info, h_next = model(x_batch, h_t)
        
        loss = criterion(y, tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        pred = y.argmax(dim=-1).item()
        correct.append(1 if pred == tgt.item() else 0)
        
        if (step + 1) % 500 == 0:
            avg_loss = sum(losses[-100:]) / 100
            avg_acc = sum(correct[-100:]) / 100 * 100
            print(f"  {name:30s} Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")
    
    final_loss = sum(losses[-100:]) / 100
    final_acc = sum(correct[-100:]) / 100 * 100
    return final_loss, final_acc

print("=" * 70)
print("ANALYSIS: Why doesn't MoE reach zero loss?")
print("=" * 70)

# Test 1: Longer training
print("\nTest 1: Longer training (3000 steps)")
print("-" * 70)
torch.manual_seed(42)
model1 = RecurrentMoE(d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss1, acc1 = train_model(model1, "Original MoE (3000 steps)", steps=3000)
print(f"  Final: Loss = {loss1:.4f}, Acc = {acc1:.1f}%")

# Test 2: More capacity (larger model)
print("\nTest 2: Larger model (d=64, slots=8, experts=8)")
print("-" * 70)
torch.manual_seed(42)
model2 = RecurrentMoE(d_model=64, n_heads=2, n_slots=8, n_experts=8, topk=2, d_in=2, d_out=2).to(device)
loss2, acc2 = train_model(model2, "Larger MoE", steps=2000)
print(f"  Final: Loss = {loss2:.4f}, Acc = {acc2:.1f}%")

# Test 3: More experts active (topk=4 instead of 2)
print("\nTest 3: More active experts (topk=4)")
print("-" * 70)
torch.manual_seed(42)
model3 = RecurrentMoE(d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=4, d_in=2, d_out=2).to(device)
loss3, acc3 = train_model(model3, "MoE topk=4", steps=2000)
print(f"  Final: Loss = {loss3:.4f}, Acc = {acc3:.1f}%")

# Test 4: Higher learning rate
print("\nTest 4: Higher learning rate (lr=2e-2)")
print("-" * 70)
torch.manual_seed(42)
model4 = RecurrentMoE(d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss4, acc4 = train_model(model4, "MoE lr=2e-2", steps=2000, lr=2e-2)
print(f"  Final: Loss = {loss4:.4f}, Acc = {acc4:.1f}%")

# Test 5: More slots written per step
print("\nTest 5: Checking slot writing behavior")
print("-" * 70)
torch.manual_seed(42)
model5 = RecurrentMoE(d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)

# Check how many unique slots are used
tgt, x_seq = make_seq()
h_t = model5.init_state(1, device=device)
x_batch = x_seq.unsqueeze(0)
y, info, h_next = model5(x_batch, h_t)

print(f"  Slot indices chosen: {info['idx_slots'].tolist()}")
print(f"  Number of unique slots: {len(set(info['idx_slots'][0].tolist()))}")
print(f"  Total slots available: 4")

# Train it
loss5, acc5 = train_model(model5, "Standard MoE", steps=2000)
print(f"  Final: Loss = {loss5:.4f}, Acc = {acc5:.1f}%")

# Test 6: Check if beta (residual strength) matters
print("\nTest 6: Analyzing state update mechanism")
print("-" * 70)
print("  Note: Model uses beta=0.6 for residual connection")
print("  state = 0.6*state_old + 0.4*alpha*new_latent")
print("  This means only 40% of new information is mixed in")
print("  Hypothesis: Strong residual might slow learning")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
results = [
    ("Original (1500 steps)", 0.67, 81.3),
    ("Longer training (3000)", loss1, acc1),
    ("Larger model", loss2, acc2),
    ("More active experts", loss3, acc3),
    ("Higher LR", loss4, acc4),
]

best_loss = min(r[1] for r in results)
best_acc = max(r[2] for r in results)

for name, loss, acc in results:
    marker = " ✓" if loss < 0.1 else " " if loss < 0.5 else ""
    print(f"  {name:30s}: Loss = {loss:.4f}, Acc = {acc:5.1f}%{marker}")

print(f"\nBest: Loss = {best_loss:.4f}, Acc = {best_acc:.1f}%")

if best_loss < 0.1:
    print("\n✓ MoE CAN reach near-zero loss with proper configuration!")
elif best_loss < 0.3:
    print("\n⚠ MoE approaches good performance but doesn't fully converge")
    print("   Potential issues:")
    print("   - Sparse slot updates limit information propagation")
    print("   - Strong residual (beta=0.6) slows adaptation")
    print("   - Gating bottleneck in expert selection")
else:
    print("\n✗ MoE struggles to converge fully")
    print("   This suggests architectural limitations")
