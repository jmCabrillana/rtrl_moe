"""
Simple test: BPTT on haystack with sparse MoE

This tests the basic convergence without RTRL complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from moe import RecurrentMoE, get_expert_latent_activated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

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
print("BPTT with Sparse MoE on Haystack Task")
print("=" * 80)
print()

VOCAB_SIZE = 8
SEQ_LEN = 16
OUTPUT_DIM = VOCAB_SIZE - BASE
N_STEPS = 200

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
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

losses = []
accs = []
read_sparsity = []
write_sparsity = []

print("Step | Loss   | Acc  | Read% | Write% | Expert Count")
print("-" * 55)

for step in range(N_STEPS):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    h = model.init_state(1, device=device)
    pred_logits, info, h_next = model(x_onehot, h)
    
    loss = criterion(pred_logits, y)
    
    # Get sparsity info
    active_params, write_idx, read_idx = get_expert_latent_activated(model, info)
    
    H = model.d * model.n_slots
    read_sparse_pct = len(read_idx) / H * 100
    write_sparse_pct = len(write_idx) / H * 100
    n_experts = len(info['idx_experts'][0].unique())
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    acc = (pred_logits.argmax(dim=1) == y).float().mean().item() * 100
    accs.append(acc)
    read_sparsity.append(read_sparse_pct)
    write_sparsity.append(write_sparse_pct)
    
    if (step + 1) % 50 == 0:
        avg_loss = sum(losses[-50:]) / 50
        avg_acc = sum(accs[-50:]) / 50
        avg_read = sum(read_sparsity[-50:]) / 50
        avg_write = sum(write_sparsity[-50:]) / 50
        print(f"{step+1:3d} | {avg_loss:.3f} | {avg_acc:.1f}% | {avg_read:.0f}% | {avg_write:.0f}% | {n_experts}")

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

final_loss = sum(losses[-50:]) / 50
final_acc = sum(accs[-50:]) / 50
final_read = sum(read_sparsity[-50:]) / 50
final_write = sum(write_sparsity[-50:]) / 50

print(f"Loss: {final_loss:.3f}")
print(f"Accuracy: {final_acc:.1f}%")
print(f"Read sparsity: {final_read:.0f}% (only {final_read:.0f}% of state dimensions)")
print(f"Write sparsity: {final_write:.0f}% (only {final_write:.0f}% of state dimensions)")
print()

if final_acc > 50:
    print("✓ Model is learning!")
elif final_acc > 20:
    print("⚠ Model is learning slowly")
else:
    print("✗ Model not learning well on this task")

print()
print("This sparse MoE demonstrates:")
print("  - Read gate selects which slots to read from")
print("  - Write gate selects which slots to write to")
print("  - Expert gate selects which experts to use")
print("  - With RTRL, only compute Jacobians for read slots")
print("  - This enables efficient training on long sequences!")
