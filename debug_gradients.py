"""
Debug script to verify RTRL gradient computation is working correctly.
Compares RTRL gradients to BPTT gradients on a simple sequence.
"""

import torch
import torch.nn as nn
from moe import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cpu")
torch.manual_seed(42)

print("=" * 60)
print("GRADIENT VERIFICATION: RTRL vs BPTT")
print("=" * 60)

# Create a simple task: detect if sequence has more 'a' than 'b'
def make_simple_seq(n=4):
    """Generate simple a^n b^m sequence"""
    n_a = torch.randint(1, 4, (1,)).item()
    n_b = torch.randint(1, 4, (1,)).item()
    
    seq = []
    for _ in range(n_a):
        seq.append(0)  # 'a' token
    for _ in range(n_b):
        seq.append(1)  # 'b' token
    
    x_seq = torch.zeros(len(seq), 2, device=device)
    for i, tok in enumerate(seq):
        x_seq[i, tok] = 1.0
    
    label = 1 if n_a == n_b else 0
    return torch.tensor([label], device=device), x_seq

# Small model for testing
model = RecurrentMoE(
    d_model=32,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=2,
    d_out=2
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

# Test 1: Check if BPTT works at all
print("\n" + "=" * 60)
print("TEST 1: BPTT Baseline (verify task is learnable)")
print("=" * 60)

model_bptt = RecurrentMoE(
    d_model=32,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=2,
    d_out=2
).to(device)

optimizer_bptt = torch.optim.Adam(model_bptt.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

losses_bptt = []
for step in range(200):
    tgt, x_seq = make_simple_seq()
    h_t = model_bptt.init_state(1, device=device)
    
    # Forward through sequence with BPTT
    for t in range(x_seq.shape[0]):
        y, info, h_t = model_bptt(x_seq[t:t+1].unsqueeze(0), h_t)
    
    loss = criterion(y, tgt)
    optimizer_bptt.zero_grad()
    loss.backward()
    optimizer_bptt.step()
    
    losses_bptt.append(loss.item())
    
    if (step + 1) % 50 == 0:
        avg_loss = sum(losses_bptt[-50:]) / 50
        print(f"Step {step+1}: Avg Loss = {avg_loss:.4f}")

initial_bptt = sum(losses_bptt[:50]) / 50
final_bptt = sum(losses_bptt[-50:]) / 50
print(f"\nBPTT: Initial loss {initial_bptt:.4f} → Final loss {final_bptt:.4f}")
print(f"Improvement: {(initial_bptt - final_bptt) / initial_bptt * 100:.1f}%")

if final_bptt < 0.5:
    print("✓ BPTT works - task is learnable")
else:
    print("✗ WARNING: Even BPTT struggles - task might be too hard")

# Test 2: Check RTRL gradient magnitudes
print("\n" + "=" * 60)
print("TEST 2: RTRL Gradient Magnitude Check")
print("=" * 60)

model_rtrl = RecurrentMoE(
    d_model=32,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=2,
    d_out=2
).to(device)

rtrl = BlockRTRL(model=model_rtrl, len_buffer=128)
optimizer_rtrl = torch.optim.Adam(model_rtrl.parameters(), lr=1e-2)

print("\nTraining with RTRL for 200 steps...")
losses_rtrl = []
grad_norms = []

for step in range(200):
    tgt, x_seq = make_simple_seq()
    h_t = model_rtrl.init_state(1, device=device)
    rtrl.reset()
    
    # Forward through sequence with RTRL
    for t in range(x_seq.shape[0]):
        y, info, h_t = model_rtrl(x_seq[t:t+1].unsqueeze(0), h_t)
        
        # Get active parameters for this step
        active_params, write_idx = get_expert_latent_activated(model_rtrl, info)
        
        # RTRL step
        rtrl.step(h_t, active_params, write_idx)
    
    # Compute loss and get gradients
    loss = criterion(y, tgt)
    optimizer_rtrl.zero_grad()
    
    # Backprop through output layer
    loss.backward()
    
    # Add RTRL gradients
    dy = y.grad if y.grad is not None else torch.autograd.grad(loss, y, retain_graph=True)[0]
    rtrl.compute_grad(dy.squeeze(0))
    rtrl.accumulate_grad()
    
    # Check gradient magnitudes
    total_grad_norm = 0
    for p in model_rtrl.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    grad_norms.append(total_grad_norm)
    
    optimizer_rtrl.step()
    losses_rtrl.append(loss.item())
    
    if (step + 1) % 50 == 0:
        avg_loss = sum(losses_rtrl[-50:]) / 50
        avg_grad = sum(grad_norms[-50:]) / 50
        print(f"Step {step+1}: Avg Loss = {avg_loss:.4f}, Avg Grad Norm = {avg_grad:.4f}")

initial_rtrl = sum(losses_rtrl[:50]) / 50
final_rtrl = sum(losses_rtrl[-50:]) / 50
avg_grad_early = sum(grad_norms[:50]) / 50
avg_grad_late = sum(grad_norms[-50:]) / 50

print(f"\nRTRL: Initial loss {initial_rtrl:.4f} → Final loss {final_rtrl:.4f}")
print(f"Improvement: {(initial_rtrl - final_rtrl) / initial_rtrl * 100:.1f}%")
print(f"Gradient norm: Early {avg_grad_early:.4f} → Late {avg_grad_late:.4f}")

if avg_grad_early < 1e-6:
    print("✗ CRITICAL: Gradients near zero - RTRL not computing gradients!")
elif final_rtrl < 0.5:
    print("✓ RTRL works - model converges")
elif (initial_rtrl - final_rtrl) / initial_rtrl > 0.2:
    print("⚠ RTRL shows learning but slow convergence")
else:
    print("✗ RTRL fails to learn")

# Test 3: Direct comparison on same sequence
print("\n" + "=" * 60)
print("TEST 3: Gradient Direction Comparison (single step)")
print("=" * 60)

model_test = RecurrentMoE(
    d_model=32,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=2,
    d_out=2
).to(device)

tgt, x_seq = make_simple_seq()

# Get BPTT gradients
h_t = model_test.init_state(1, device=device)
for t in range(x_seq.shape[0]):
    y_bptt, info, h_t = model_test(x_seq[t:t+1].unsqueeze(0), h_t)

loss_bptt = criterion(y_bptt, tgt)
grads_bptt = torch.autograd.grad(loss_bptt, model_test.parameters(), retain_graph=True)

# Get RTRL gradients
rtrl_test = BlockRTRL(model=model_test, len_buffer=128)
rtrl_test.reset()

h_t = model_test.init_state(1, device=device)
for t in range(x_seq.shape[0]):
    y_rtrl, info, h_t = model_test(x_seq[t:t+1].unsqueeze(0), h_t)
    active_params, write_idx = get_expert_latent_activated(model_test, info)
    rtrl_test.step(h_t, active_params, write_idx)

loss_rtrl = criterion(y_rtrl, tgt)
for p in model_test.parameters():
    p.grad = None

dy = torch.autograd.grad(loss_rtrl, y_rtrl)[0]
rtrl_test.compute_grad(dy.squeeze(0))
rtrl_test.accumulate_grad()

grads_rtrl = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model_test.parameters()]

# Compare gradient directions
print("\nGradient comparison:")
for i, (g_bptt, g_rtrl, name) in enumerate(zip(grads_bptt, grads_rtrl, [n for n, _ in model_test.named_parameters()])):
    if g_bptt.numel() > 0 and g_rtrl.numel() > 0:
        norm_bptt = g_bptt.norm().item()
        norm_rtrl = g_rtrl.norm().item()
        
        if norm_bptt > 1e-8 and norm_rtrl > 1e-8:
            cos_sim = (g_bptt * g_rtrl).sum() / (norm_bptt * norm_rtrl)
            print(f"  {name[:40]:40s}: BPTT={norm_bptt:8.4f}, RTRL={norm_rtrl:8.4f}, cos={cos_sim:6.3f}")
        elif norm_bptt > 1e-8:
            print(f"  {name[:40]:40s}: BPTT={norm_bptt:8.4f}, RTRL=0.0000 ⚠")
        elif norm_rtrl > 1e-8:
            print(f"  {name[:40]:40s}: BPTT=0.0000, RTRL={norm_rtrl:8.4f} ⚠")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"BPTT converges: {final_bptt < 0.5}")
print(f"RTRL converges: {final_rtrl < 0.5}")
print(f"RTRL has gradients: {avg_grad_early > 1e-6}")
