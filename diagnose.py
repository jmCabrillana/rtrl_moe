"""
Diagnostic script to understand why training isn't working.
"""
import torch
import torch.nn as nn
from rtrl_moe.model.moe import RecurrentMoE
from rtrl_moe.tasks import anbn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small test
D = 16
model = RecurrentMoE(
    d_model=D,
    n_heads=1,
    n_slots=2,
    n_experts=2,
    topk=1,
    topk_read=1,
    topk_write=1,
    d_in=4,
    d_out=2
).to(device)

# Get one sample
seq_len = 8
x_onehot, y = anbn.sample(seq_len, device, batch_size=1)

print(f"Input shape: {x_onehot.shape}")
print(f"Target: {y}")

# Process through model
h_t = model.init_state(1, device=device)
print(f"Initial state shape: {h_t.shape}")
print(f"Initial state norm: {h_t.norm().item():.6f}")

print("\n--- Processing sequence ---")
for t in range(seq_len - 1):
    x_t = x_onehot[:, t:t+1, :]
    pred_logits, info, h_next = model(x_t, h_t)
    
    print(f"t={t}: pred={pred_logits.argmax(1).item()}, "
          f"h_norm={h_next.norm().item():.6f}, "
          f"read_slots={info['idx_slots_read'].tolist()}, "
          f"write_slots={info['idx_slot_write'].tolist()}")
    
    h_t = h_next

# Final prediction
x_t = x_onehot[:, -1:, :]
pred_logits, info, h_next = model(x_t, h_t)

print(f"\nt={seq_len-1}: pred={pred_logits.argmax(1).item()}")
print(f"Final logits: {pred_logits.squeeze(0)}")
print(f"Target: {y.item()}")

# Check if gradients flow
criterion = nn.CrossEntropyLoss()
loss = criterion(pred_logits, y)
print(f"\nLoss: {loss.item():.6f}")

# Try backprop
loss.backward()
print("\nGradients after backward:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad norm = {param.grad.norm().item():.6e}")
    else:
        print(f"  {name}: NO GRADIENT")

print("\n--- Key insight ---")
print("Problem 1: Sparse read/write gating may break RTRL state tracking")
print("Problem 2: State updates only at written slots may cause gradient flow issues")
print("Problem 3: The final step needs gradient w.r.t state to propagate backward through RTRL")
