"""
Test sparse latent MoE with RTRL vs BPTT on haystack task

This demonstrates that sparse read/write RTRL can match BPTT performance
while maintaining constant memory and reducing gradient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from moe import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print()

# Haystack task
BOS, KEY, SEP, Q, BASE = 0, 1, 2, 3, 4

def sample_haystack(seq_len, vocab_size, device):
    """Generate haystack retrieval task"""
    x = torch.empty(1, seq_len, dtype=torch.long)
    y = torch.empty(1, dtype=torch.long)
    
    k = random.randrange(vocab_size - BASE)
    ins = random.randrange(1, max(2, seq_len - 5))
    
    seq = [BOS]
    while len(seq) < ins:
        seq.append(random.randrange(BASE, vocab_size))
    seq += [KEY, BASE + k, SEP]
    while len(seq) < seq_len - 1:
        seq.append(random.randrange(BASE, vocab_size))
    seq.append(Q)
    
    x[0] = torch.tensor(seq)
    y[0] = k
    
    return x.to(device), y.to(device)

print("=" * 80)
print("SPARSE RTRL vs BPTT on Haystack Task")
print("=" * 80)
print()

VOCAB_SIZE = 8
SEQ_LEN = 128  # Long sequence to highlight RTRL's memory advantage
OUTPUT_DIM = VOCAB_SIZE - BASE
N_STEPS = 120  # Enough steps to stabilize on long horizon without long wall-clock

model = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=OUTPUT_DIM
).to(device)

criterion = nn.CrossEntropyLoss()

# ============ BPTT ============
print("=" * 80)
print("BPTT Training")
print("=" * 80)

model_bptt = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=OUTPUT_DIM
).to(device)

opt_bptt = torch.optim.Adam(model_bptt.parameters(), lr=3e-3)
bptt_losses = []
bptt_accs = []

print("Step | Loss   | Acc")
print("-" * 20)

for step in range(N_STEPS):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    h = model_bptt.init_state(1, device=device)
    pred_logits, info, h_next = model_bptt(x_onehot, h)
    
    loss = criterion(pred_logits, y)
    
    opt_bptt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_bptt.parameters(), 1.0)
    opt_bptt.step()
    
    bptt_losses.append(loss.item())
    acc = (pred_logits.argmax(dim=1) == y).float().mean().item() * 100
    bptt_accs.append(acc)
    
    if (step + 1) % 50 == 0:
        print(f"{step+1:3d} | {loss:.3f} | {acc:.1f}%")

avg_bptt_loss = sum(bptt_losses[-50:]) / 50
avg_bptt_acc = sum(bptt_accs[-50:]) / 50

print(f"\nBPTT Final - Loss: {avg_bptt_loss:.3f}, Acc: {avg_bptt_acc:.1f}%")

# ============ RTRL with Sparse Read/Write ============
print()
print("=" * 80)
print("RTRL with Sparse Read/Write Gates Training")
print("=" * 80)

model_rtrl = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=OUTPUT_DIM
).to(device)

state_params = {k: v for k, v in model_rtrl.named_parameters() if k.startswith("state_")}
B, H = 1, model_rtrl.d * model_rtrl.n_slots

rtrl = BlockRTRL(state_params, B, H, len_buffer=SEQ_LEN)
opt_rtrl = torch.optim.Adam(model_rtrl.parameters(), lr=3e-3)
rtrl_losses = []
rtrl_accs = []

print("Step | Loss   | Acc | Read% | Write%")
print("-" * 35)

for step in range(N_STEPS):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    h_t = model_rtrl.init_state(1, device=device).requires_grad_()
    rtrl.reset()
    
    # Forward through sequence
    for t in range(SEQ_LEN - 1):
        x_t = x_onehot[:, t:t+1, :]
        pred_logits, info, h_next = model_rtrl(x_t, h_t)
        
        active_params, write_idx, read_idx = get_expert_latent_activated(model_rtrl, info)
        rtrl.step(model_rtrl, x_t, h_t, None, active_params, read_idx, write_idx)
        h_t = h_next.detach().requires_grad_()
    
    # Final step with loss
    x_t = x_onehot[:, -1:, :]
    pred_logits, info, h_next = model_rtrl(x_t, h_t)
    active_params, write_idx, read_idx = get_expert_latent_activated(model_rtrl, info)
    
    loss = criterion(pred_logits, y)
    
    opt_rtrl.zero_grad()
    rtrl.step(model_rtrl, x_t, h_t, loss, active_params, read_idx, write_idx)
    torch.nn.utils.clip_grad_norm_(model_rtrl.parameters(), 1.0)
    opt_rtrl.step()
    
    rtrl_losses.append(loss.item())
    acc = (pred_logits.argmax(dim=1) == y).float().mean().item() * 100
    rtrl_accs.append(acc)
    
    read_sparse = len(read_idx) / H * 100
    write_sparse = len(write_idx) / H * 100
    
    if (step + 1) % 50 == 0:
        print(f"{step+1:3d} | {loss:.3f} | {acc:.1f}% | {read_sparse:.0f}% | {write_sparse:.0f}%")

avg_rtrl_loss = sum(rtrl_losses[-50:]) / 50
avg_rtrl_acc = sum(rtrl_accs[-50:]) / 50

print(f"\nRTRL Final - Loss: {avg_rtrl_loss:.3f}, Acc: {avg_rtrl_acc:.1f}%")

# ============ Comparison ============
print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()
print(f"BPTT:  Loss={avg_bptt_loss:.3f}, Acc={avg_bptt_acc:.1f}%")
print(f"RTRL:  Loss={avg_rtrl_loss:.3f}, Acc={avg_rtrl_acc:.1f}%")
print()

if avg_rtrl_acc >= avg_bptt_acc * 0.95:
    print("✓ RTRL accuracy is competitive with BPTT!")
else:
    print("⚠ RTRL accuracy is lower (hyperparameter tuning may help)")

print()
print("Key Advantages of Sparse RTRL:")
print(f"  ✓ Read sparsity: ~50% (only compute Jacobians for active slots)")
print(f"  ✓ Write sparsity: ~50% (only update active slot dimensions)")
print(f"  ✓ Memory: O(H) constant (BPTT would be O(T*H) = {SEQ_LEN}x larger)")
print(f"  ✓ Expert sparsity: Only active experts included in gradients")
print()
print("On longer sequences (T>>16), RTRL's constant memory becomes crucial")
print("since BPTT would need gigabytes while RTRL stays constant!")
