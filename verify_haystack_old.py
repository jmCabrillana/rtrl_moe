"""
Haystack Task Verification: Complete thesis test suite

Tests:
1. Sparse MoE with BPTT can solve haystack (convergence despite sparse gating)
2. Sparse MoE with RTRL can solve haystack (RTRL gradients work)
3. Segment tree lazy updates are faster than full updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from moe import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cpu")

# Haystack task constants
BOS, KEY, SEP, Q, BASE = 0, 1, 2, 3, 4

def sample_haystack(seq_len, vocab_size, device):
    """Generate haystack retrieval task"""
    x = torch.empty(1, seq_len, dtype=torch.long)
    y = torch.empty(1, dtype=torch.long)
    
    k = random.randrange(vocab_size - BASE)
    # Insert key somewhere with space before final Q token
    ins = random.randrange(1, seq_len - 17)
    
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

class DenseRNN(nn.Module):
    """Simple RNN baseline for haystack"""
    def __init__(self, vocab_size=20, hidden_dim=64, output_dim=16):
        super().__init__()
        self.h_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def init_state(self, B, device=None):
        return torch.zeros(B, self.h_dim, device=device)
    
    def forward(self, x, h):
        # x: [B, T]
        B, T = x.shape
        emb = self.emb(x)  # [B, T, D]
        
        for t in range(T):
            h = self.rnn_cell(emb[:, t], h)
        
        y = self.out(h)
        return y, {}, h

print("=" * 75)
print("HAYSTACK TASK VERIFICATION")
print("=" * 75)
print()

# Task parameters - small for fast testing
SEQ_LEN = 16  # Short sequences
VOCAB_SIZE = 12  # Small vocabulary
OUTPUT_DIM = VOCAB_SIZE - BASE

# Test 1: Can MoE solve haystack with BPTT?
print("=" * 75)
print("TEST 1: Sparse MoE Convergence with BPTT")
print("=" * 75)
print("Task: Retrieve key-value from sequence of length", SEQ_LEN)
print()

torch.manual_seed(42)
model_moe = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=4,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=OUTPUT_DIM
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model_moe.parameters())}")

optimizer_moe = torch.optim.Adam(model_moe.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()

print("\nTraining MoE with BPTT for 1000 steps...")
losses_moe = []
correct_moe = []

for step in range(1000):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    h_t = model_moe.init_state(1, device=device)
    
    # Convert to one-hot
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()  # [B, T, V]
    
    # Forward through sequence
    pred_logits, info, h_next = model_moe(x_onehot, h_t)
    
    loss = criterion(pred_logits, y)
    optimizer_moe.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_moe.parameters(), 1.0)
    optimizer_moe.step()
    
    losses_moe.append(loss.item())
    pred = pred_logits.argmax(dim=-1).item()
    correct_moe.append(1 if pred == y.item() else 0)
    
    if (step + 1) % 200 == 0:
        avg_loss = sum(losses_moe[-50:]) / 50
        avg_acc = sum(correct_moe[-50:]) / 50 * 100
        print(f"  Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")

initial_loss_moe = sum(losses_moe[:50]) / 50
final_loss_moe = sum(losses_moe[-50:]) / 50
initial_acc_moe = sum(correct_moe[:50]) / 50 * 100
final_acc_moe = sum(correct_moe[-50:]) / 50 * 100

print(f"\nMoE Results:")
print(f"  Initial: Loss = {initial_loss_moe:.4f}, Acc = {initial_acc_moe:.1f}%")
print(f"  Final:   Loss = {final_loss_moe:.4f}, Acc = {final_acc_moe:.1f}%")
print(f"  Improvement: {(initial_loss_moe - final_loss_moe) / initial_loss_moe * 100:.1f}%")

# Dense RNN baseline
print("\n--- Dense RNN Baseline ---")
torch.manual_seed(42)
model_rnn = DenseRNN(VOCAB_SIZE, 32, OUTPUT_DIM).to(device)
optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=5e-3)

print("Training Dense RNN with BPTT for 1000 steps...")
losses_rnn = []
correct_rnn = []

for step in range(1000):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    h_t = model_rnn.init_state(1, device=device)
    
    pred_logits, _, h_next = model_rnn(x, h_t)
    
    loss = criterion(pred_logits, y)
    optimizer_rnn.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_rnn.parameters(), 1.0)
    optimizer_rnn.step()
    
    losses_rnn.append(loss.item())
    pred = pred_logits.argmax(dim=-1).item()
    correct_rnn.append(1 if pred == y.item() else 0)
    
    if (step + 1) % 200 == 0:
        avg_loss = sum(losses_rnn[-50:]) / 50
        avg_acc = sum(correct_rnn[-50:]) / 50 * 100
        print(f"  Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")

final_loss_rnn = sum(losses_rnn[-50:]) / 50
final_acc_rnn = sum(correct_rnn[-50:]) / 50 * 100

print(f"\nDense RNN Results:")
print(f"  Final: Loss = {final_loss_rnn:.4f}, Acc = {final_acc_rnn:.1f}%")

print("\nComparison:")
print(f"  Sparse MoE: {final_acc_moe:.1f}% accuracy")
print(f"  Dense RNN:  {final_acc_rnn:.1f}% accuracy")

test1_pass = final_acc_moe > 50 or (final_acc_moe > 25 and final_loss_moe < initial_loss_moe * 0.7)
if test1_pass:
    print("✓ VERIFIED: Sparse MoE can learn haystack with BPTT")
elif final_loss_moe < initial_loss_moe:
    print("⚠ PARTIAL: Sparse MoE shows learning but slow convergence")
else:
    print("✗ FAILED: Sparse MoE struggles on haystack")

# Test 2: Can MoE solve haystack with RTRL?
print("\n" + "=" * 75)
print("TEST 2: Sparse MoE with RTRL (online learning)")
print("=" * 75)

torch.manual_seed(42)
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

rtrl = BlockRTRL(state_params, B, H, len_buffer=24)
optimizer_rtrl = torch.optim.Adam(model_rtrl.parameters(), lr=5e-3)

print(f"\nTraining MoE with RTRL for 500 steps...")
print("(Computing gradients online with BlockRTRL)")

losses_rtrl = []
correct_rtrl = []

for step in range(500):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    h_t = model_rtrl.init_state(1, device=device).requires_grad_()
    rtrl.reset()
    
    # Convert to one-hot
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    # Forward through sequence with RTRL
    for t in range(SEQ_LEN - 1):
        x_t = x_onehot[:, t:t+1, :]  # [B, 1, V]
        pred_logits, info, h_next = model_rtrl(x_t, h_t)
        
        # Get active parameters for RTRL
        active_params, write_idx = get_expert_latent_activated(model_rtrl, info)
        
        # RTRL step without loss
        rtrl.step(model_rtrl, x_t, h_t, None, active_params, None, write_idx)
        h_t = h_next.detach().requires_grad_()
    
    # Final step with loss
    x_t = x_onehot[:, -1:, :]
    pred_logits, info, h_next = model_rtrl(x_t, h_t)
    active_params, write_idx = get_expert_latent_activated(model_rtrl, info)
    
    loss = criterion(pred_logits, y)
    
    # RTRL step with loss
    optimizer_rtrl.zero_grad()
    rtrl.step(model_rtrl, x_t, h_t, loss, active_params, None, write_idx)
    torch.nn.utils.clip_grad_norm_(model_rtrl.parameters(), 1.0)
    optimizer_rtrl.step()
    
    losses_rtrl.append(loss.item())
    pred = pred_logits.argmax(dim=-1).item()
    correct_rtrl.append(1 if pred == y.item() else 0)
    
    if (step + 1) % 100 == 0:
        avg_loss = sum(losses_rtrl[-50:]) / min(50, len(losses_rtrl))
        avg_acc = sum(correct_rtrl[-50:]) / min(50, len(correct_rtrl)) * 100
        print(f"  Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}%")

initial_loss_rtrl = sum(losses_rtrl[:50]) / 50
final_loss_rtrl = sum(losses_rtrl[-50:]) / 50
final_acc_rtrl = sum(correct_rtrl[-50:]) / 50 * 100

print(f"\nRTRL Results:")
print(f"  Initial: Loss = {initial_loss_rtrl:.4f}")
print(f"  Final:   Loss = {final_loss_rtrl:.4f}, Acc = {final_acc_rtrl:.1f}%")
print(f"  Improvement: {(initial_loss_rtrl - final_loss_rtrl) / initial_loss_rtrl * 100:.1f}%")

test2_pass = final_acc_rtrl > 25 or final_loss_rtrl < initial_loss_rtrl * 0.7
if test2_pass:
    print("✓ VERIFIED: Sparse RTRL works on haystack")
elif final_loss_rtrl < initial_loss_rtrl:
    print("⚠ PARTIAL: RTRL shows learning but slow convergence")
else:
    print("⚠ WARNING: RTRL shows limited learning on haystack")

# Test 3: Segment Tree Speedup
print("\n" + "=" * 75)
print("TEST 3: Segment Tree Lazy Update Speedup")
print("=" * 75)

torch.manual_seed(42)
model_perf = RecurrentMoE(
    d_model=32,
    n_heads=2,
    n_slots=4,
    n_experts=8,
    topk=2,
    d_in=VOCAB_SIZE,
    d_out=OUTPUT_DIM
).to(device)

state_params_perf = {k: v for k, v in model_perf.named_parameters() if k.startswith("state_")}
B_perf, H_perf = 1, model_perf.d * model_perf.n_slots

# Test with segment tree (sparse + lazy updates)
print("\nRunning 15 steps with SPARSE + LAZY UPDATES (segment tree)...")
rtrl_sparse = BlockRTRL(state_params_perf, B_perf, H_perf, len_buffer=24)

start = time.time()
for step in range(15):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    h_t = model_perf.init_state(1, device=device).requires_grad_()
    rtrl_sparse.reset()
    
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    for t in range(SEQ_LEN):
        x_t = x_onehot[:, t:t+1, :]
        pred_logits, info, h_next = model_perf(x_t, h_t)
        
        active_params, write_idx = get_expert_latent_activated(model_perf, info)
        rtrl_sparse.step(model_perf, x_t, h_t, None, active_params, None, write_idx)
        h_t = h_next.detach().requires_grad_()

sparse_time = time.time() - start

# Test without segment tree (sparse but full updates each step)
print("Running 15 steps with SPARSE but NO LAZY UPDATES (full updates)...")
rtrl_full = BlockRTRL(state_params_perf, B_perf, H_perf, len_buffer=24)

start = time.time()
for step in range(15):
    x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
    h_t = model_perf.init_state(1, device=device).requires_grad_()
    rtrl_full.reset()
    
    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
    
    for t in range(SEQ_LEN):
        x_t = x_onehot[:, t:t+1, :]
        pred_logits, info, h_next = model_perf(x_t, h_t)
        
        # Use ALL state params instead of just active ones
        rtrl_full.step(model_perf, x_t, h_t, None, state_params_perf, None, None)
        h_t = h_next.detach().requires_grad_()

full_time = time.time() - start

print(f"\nResults:")
print(f"  Sparse + Lazy (w/ segment tree): {sparse_time:.3f}s")
print(f"  Sparse (no lazy, full updates):  {full_time:.3f}s")
print(f"  Speedup:                         {full_time / sparse_time:.2f}x")

test3_pass = sparse_time < full_time
if test3_pass:
    print("✓ VERIFIED: Segment tree lazy updates are faster!")
else:
    print("✗ WARNING: No speedup observed")

# Final Summary
print("\n" + "=" * 75)
print("HAYSTACK VERIFICATION SUMMARY")
print("=" * 75)
print()
print(f"1. Sparse MoE with BPTT:  {'✓ PASS' if test1_pass else '✗ FAIL'}")
print(f"   Final accuracy: {final_acc_moe:.1f}% (baseline: {final_acc_rnn:.1f}%)")
print()
print(f"2. Sparse MoE with RTRL:  {'✓ PASS' if test2_pass else '✗ FAIL'}")
print(f"   Final accuracy: {final_acc_rtrl:.1f}%")
print()
print(f"3. Segment tree speedup:  {'✓ PASS' if test3_pass else '✗ FAIL'}")
print(f"   Speedup: {full_time / sparse_time:.2f}x")
print()

if test1_pass and test2_pass and test3_pass:
    print("🎉 ALL TESTS PASSED - THESIS VERIFIED ON HAYSTACK TASK!")
elif test1_pass and test3_pass:
    print("✓ Core thesis verified (BPTT convergence + speedup)")
    print("⚠ RTRL needs tuning for this task")
else:
    print("⚠ Some tests need attention")
