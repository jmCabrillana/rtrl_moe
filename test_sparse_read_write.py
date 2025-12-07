"""
Test sparse read/write RTRL with proper read gating

This demonstrates the speedup from:
1. Sparse WRITE indices (only update selected slots)
2. Sparse READ indices (only compute Jacobians for selected slots)
3. Sparse EXPERT indices (only use active expert parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from moe import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cpu")

print("=" * 80)
print("SPARSE READ/WRITE RTRL SPEEDUP TEST")
print("=" * 80)
print()

# Configuration
VOCAB_SIZE = 8
SEQ_LEN = 32
D_MODEL = 32
N_SLOTS = 4
N_EXPERTS = 4

model = RecurrentMoE(
    d_model=D_MODEL,
    n_heads=2,
    n_slots=N_SLOTS,
    n_experts=N_EXPERTS,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=4
).to(device)

state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
B, H = 1, model.d * model.n_slots

# Test 1: With sparse read/write indices
print("Test 1: SPARSE read/write indices")
print("-" * 80)

rtrl_sparse = BlockRTRL(state_params, B, H, len_buffer=64)
x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
h_t = model.init_state(1, device=device).requires_grad_()

rtrl_sparse.reset()
start = time.time()

for t in range(SEQ_LEN):
    x_t = x_onehot[:, t:t+1, :]
    pred_logits, info, h_next = model(x_t, h_t)
    
    active_params, write_idx, read_idx = get_expert_latent_activated(model, info)
    
    # Pass sparse read and write indices
    rtrl_sparse.step(model, x_t, h_t, None, active_params, read_idx, write_idx)
    h_t = h_next.detach().requires_grad_()
    
    if t == 0:
        print(f"  Step 0: Read {len(read_idx)}/{H} dims ({len(read_idx)/H:.1%}), "
              f"Write {len(write_idx)}/{H} dims ({len(write_idx)/H:.1%})")

sparse_time = time.time() - start
print(f"  Total time: {sparse_time:.3f}s ({sparse_time/SEQ_LEN*1000:.1f}ms per step)")
print()

# Test 2: Without sparse indices (read/write all dims)
print("Test 2: FULL read/write indices (no sparsity)")
print("-" * 80)

rtrl_full = BlockRTRL(state_params, B, H, len_buffer=64)
h_t = model.init_state(1, device=device).requires_grad_()

rtrl_full.reset()
start = time.time()

for t in range(SEQ_LEN):
    x_t = x_onehot[:, t:t+1, :]
    pred_logits, info, h_next = model(x_t, h_t)
    
    active_params, _, _ = get_expert_latent_activated(model, info)
    
    # Pass None for read/write indices (uses all dimensions)
    rtrl_full.step(model, x_t, h_t, None, active_params, None, None)
    h_t = h_next.detach().requires_grad_()
    
    if t == 0:
        print(f"  Step 0: Read {H}/{H} dims (100%), Write {H}/{H} dims (100%)")

full_time = time.time() - start
print(f"  Total time: {full_time:.3f}s ({full_time/SEQ_LEN*1000:.1f}ms per step)")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Sparse read/write time: {sparse_time:.3f}s")
print(f"Full read/write time:   {full_time:.3f}s")
print(f"Speedup:                {full_time/sparse_time:.2f}x")
print()

if full_time/sparse_time > 1.2:
    print("✓ Sparse read/write provides significant speedup!")
else:
    print("⚠ Speedup is modest (may be overhead-limited on CPU)")

print()
print("Key insight:")
print("  - Sparse READ: Only compute Jacobians for active slots")
print("  - Sparse WRITE: Only update active slot dimensions")
print("  - This is the core speedup mechanism for efficient RTRL!")
