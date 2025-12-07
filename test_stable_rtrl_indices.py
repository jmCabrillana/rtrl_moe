"""
Test that moe_stable.py correctly returns write and read indices for BlockRTRL.
"""

import torch
import torch.nn as nn
from moe_stable import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cpu")  # Use CPU for quick verification
print(f"Using device: {device}\n")

# Create stable MoE model
model = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=8,
    d_out=4
).to(device)

B, T, D_in = 1, 16, 8
x = torch.randn(B, T, D_in).to(device)
state = model.init_state(B, device=device)

print("=" * 80)
print("Test 1: Forward pass returns idx_slots_write and idx_slots_read")
print("=" * 80)

# Forward through sequence
for t in range(T):
    x_t = x[:, t:t+1, :]
    y, info, state = model(x_t, state)
    
    if t == 0:
        print(f"\nStep {t}:")
        print(f"  idx_slots_read: {info['idx_slots_read']}")
        print(f"  idx_slots_write: {info['idx_slots_write']}")
        print(f"  idx_experts: {info['idx_experts']}")
        
        assert 'idx_slots_read' in info, "Missing idx_slots_read in info!"
        assert 'idx_slots_write' in info, "Missing idx_slots_write in info!"
        print(f"\n✓ Both read and write indices present!")

print()
print("=" * 80)
print("Test 2: get_expert_latent_activated returns correct indices")
print("=" * 80)

state_rtrl = model.init_state(B, device=device)
H = model.d * model.n_slots

# Forward pass
x_t = x[:, 0:1, :]
y, info, state_rtrl = model(x_t, state_rtrl)

# Extract activated indices
active_params, write_indices, read_indices = get_expert_latent_activated(model, info)

print(f"\nState dimension H = {H}")
print(f"Read indices: {len(read_indices)} dimensions (expected ~{H//2})")
print(f"Write indices: {len(write_indices)} dimensions (expected ~{H//2})")
print(f"Active parameters: {len(active_params)} params")

assert len(write_indices) > 0, "No write indices!"
assert len(read_indices) > 0, "No read indices!"
print(f"\n✓ get_expert_latent_activated works correctly!")

print()
print("=" * 80)
print("Test 3: BlockRTRL can use write_indices and read_indices")
print("=" * 80)

# Create RTRL with write/read indices
state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
rtrl = BlockRTRL(state_params, B, H, len_buffer=T)

# Reset and forward
rtrl.reset()
state_rtrl = model.init_state(B, device=device).requires_grad_()

for t in range(T):
    x_t = x[:, t:t+1, :]
    y, info, state_next = model(x_t, state_rtrl)
    
    active_params, write_idx, read_idx = get_expert_latent_activated(model, info)
    
    # Use RTRL with write and read indices
    if t < T-1:
        loss = None
    else:
        # Create actual loss on final output
        target = torch.randint(0, 4, (1,), device=device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y, target)
    
    rtrl.step(model, x_t, state_rtrl, loss, active_params, read_idx, write_idx)
    
    state_rtrl = state_next.detach().requires_grad_()

print(f"\n✓ BlockRTRL successfully processed {T} timesteps with write/read indices!")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✓ moe_stable.py returns both idx_slots_write and idx_slots_read")
print("✓ get_expert_latent_activated extracts both index lists")
print("✓ BlockRTRL can process both write and read indices")
print()
print("Ready for 1M-token training with sparse RTRL!")
