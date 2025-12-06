"""
Simple test: Can the MoE model learn the a^n b^n task with regular BPTT?
This isolates whether the issue is with the MoE architecture or with RTRL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import RecurrentMoE

device = torch.device("cpu")
torch.manual_seed(42)

print("=" * 60)
print("TEST: Can MoE learn a^n b^n with standard BPTT?")
print("=" * 60)

def make_seq(max_n=4):
    """Generate a^n b^n sequence"""
    n = torch.randint(1, max_n+1, (1,)).item()
    seq = []
    for _ in range(n):
        seq.append(0)  # 'a' token
    for _ in range(n):
        seq.append(1)  # 'b' token
    
    x_seq = torch.zeros(len(seq), 2, device=device)
    for i, tok in enumerate(seq):
        x_seq[i, tok] = 1.0
    
    label = torch.tensor([1], device=device)  # valid sequence
    return label, x_seq

# Test with simple MoE
model = RecurrentMoE(
    d_model=64,
    n_heads=2, 
    n_slots=8,
    n_experts=8,
    topk=2,
    d_in=2,
    d_out=2
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

losses = []
correct = []

print("\nTraining for 500 steps with BPTT...")
for step in range(500):
    tgt, x_seq = make_seq()
    h_t = model.init_state(1, device=device)
    
    # Forward through full sequence
    for t in range(x_seq.shape[0]):
        x_t = x_seq[t:t+1].unsqueeze(0)  # [B, 1, D]
        y, info, h_t = model(x_t, h_t)
    
    # Compute loss on final output
    loss = criterion(y, tgt)
    
    # Standard backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    pred = y.argmax(dim=-1).item()
    correct.append(1 if pred == tgt.item() else 0)
    
    if (step + 1) % 100 == 0:
        avg_loss = sum(losses[-100:]) / 100
        avg_acc = sum(correct[-100:]) / 100 * 100
        print(f"Step {step+1}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")

initial_loss = sum(losses[:100]) / 100
final_loss = sum(losses[-100:]) / 100
initial_acc = sum(correct[:100]) / 100 * 100
final_acc = sum(correct[-100:]) / 100 * 100

print(f"\nResults:")
print(f"  Initial: Loss = {initial_loss:.4f}, Acc = {initial_acc:.1f}%")
print(f"  Final:   Loss = {final_loss:.4f}, Acc = {final_acc:.1f}%")
print(f"  Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")

if final_acc > 80:
    print("✓ SUCCESS: MoE can learn the task with BPTT")
elif (initial_loss - final_loss) / initial_loss > 0.3:
    print("⚠ PARTIAL: MoE shows learning but slow convergence")
else:
    print("✗ FAILURE: MoE fails to learn even with BPTT")
