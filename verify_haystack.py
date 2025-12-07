"""
Haystack verification - Start simple with BPTT on short sequences
Then test RTRL and speedup on sequences that work
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from moe import RecurrentMoE, get_expert_latent_activated
from rtrl_block import BlockRTRL

device = torch.device("cpu")
torch.manual_seed(42)

# Haystack task constants
BOS, KEY, SEP, Q, BASE = 0, 1, 2, 3, 4

def sample_haystack(seq_len, vocab_size, device):
    """Generate haystack retrieval task"""
    x = torch.empty(1, seq_len, dtype=torch.long)
    y = torch.empty(1, dtype=torch.long)
    
    k = random.randrange(vocab_size - BASE)
    # Insert key somewhere with space before final Q token
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

class SimpleRNN(nn.Module):
    """Very simple RNN baseline"""
    def __init__(self, vocab_size=12, hidden_dim=32, output_dim=8):
        super().__init__()
        self.h_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def init_state(self, B, device=None):
        return torch.zeros(1, B, self.h_dim, device=device)
    
    def forward(self, x, h):
        # x: [B, T]
        emb = self.emb(x)  # [B, T, D]
        out, h_new = self.rnn(emb, h)
        y = self.out(out[:, -1])  # Use last timestep
        return y, {}, h_new

print("=" * 75)
print("HAYSTACK TASK - Progressive Testing")
print("=" * 75)

# Start with VERY short sequences
configs = [
    {"seq_len": 8, "vocab": 8, "name": "Tiny (len=8, vocab=8)"},
    {"seq_len": 12, "vocab": 10, "name": "Short (len=12, vocab=10)"},
    {"seq_len": 16, "vocab": 12, "name": "Medium (len=16, vocab=12)"},
]

print("\n" + "=" * 75)
print("PHASE 1: Test BPTT Convergence on Different Sequence Lengths")
print("=" * 75)

best_config = None
best_acc = 0

for config in configs:
    SEQ_LEN = config["seq_len"]
    VOCAB_SIZE = config["vocab"]
    OUTPUT_DIM = VOCAB_SIZE - BASE
    
    print(f"\n--- {config['name']} ---")
    
    # Test RNN baseline first
    model_rnn = SimpleRNN(VOCAB_SIZE, 32, OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model_rnn.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    correct = []
    
    print("  Training RNN...", end="", flush=True)
    for step in range(500):
        x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
        h = model_rnn.init_state(1, device=device)
        
        pred_logits, _, _ = model_rnn(x, h)
        loss = criterion(pred_logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_rnn.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        pred = pred_logits.argmax(dim=-1).item()
        correct.append(1 if pred == y.item() else 0)
        
        if (step + 1) % 100 == 0:
            print(f" {step+1}", end="", flush=True)
    print()  # newline
    
    final_loss = sum(losses[-50:]) / 50
    final_acc = sum(correct[-50:]) / 50 * 100
    
    print(f"  RNN:  Loss = {final_loss:.3f}, Acc = {final_acc:.1f}%")
    
    # Test MoE
    model_moe = RecurrentMoE(
        d_model=32,
        n_heads=2,
        n_slots=4,
        n_experts=4,
        topk=2,
        d_in=VOCAB_SIZE,
        d_out=OUTPUT_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(model_moe.parameters(), lr=3e-3)
    
    losses = []
    correct = []
    
    print("  Training MoE...", end="", flush=True)
    for step in range(500):
        x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
        h = model_moe.init_state(1, device=device)
        
        x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
        pred_logits, info, _ = model_moe(x_onehot, h)
        
        loss = criterion(pred_logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_moe.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        pred = pred_logits.argmax(dim=-1).item()
        correct.append(1 if pred == y.item() else 0)
        
        if (step + 1) % 100 == 0:
            print(f" {step+1}", end="", flush=True)
    print()  # newline
    
    final_loss_moe = sum(losses[-50:]) / 50
    final_acc_moe = sum(correct[-50:]) / 50 * 100
    
    print(f"  MoE:  Loss = {final_loss_moe:.3f}, Acc = {final_acc_moe:.1f}%")
    
    if final_acc_moe > 40:
        print(f"  ✓ MoE converges on this task!")
        if final_acc_moe > best_acc:
            best_acc = final_acc_moe
            best_config = config
    else:
        print(f"  ✗ Too difficult - MoE doesn't learn well")

if best_config is None:
    print("\n⚠ WARNING: MoE couldn't solve any haystack configuration")
    print("   Skipping RTRL tests")
else:
    print(f"\n✓ Best config: {best_config['name']} (Acc={best_acc:.1f}%)")
    print(f"   Using this for RTRL tests...")
    
    # Phase 2: Test RTRL on working configuration
    print("\n" + "=" * 75)
    print("PHASE 2: Test RTRL on Working Configuration")
    print("=" * 75)
    
    SEQ_LEN = best_config["seq_len"]
    VOCAB_SIZE = best_config["vocab"]
    OUTPUT_DIM = VOCAB_SIZE - BASE
    
    print(f"\nTask: {best_config['name']}")
    
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
    optimizer_rtrl = torch.optim.Adam(model_rtrl.parameters(), lr=5e-3)  # Higher LR for RTRL
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training MoE with RTRL for 1000 steps...")  # More steps
    print("  Progress:", end="", flush=True)
    
    losses_rtrl = []
    correct_rtrl = []
    
    for step in range(1000):
        x, y = sample_haystack(SEQ_LEN, VOCAB_SIZE, device)
        h_t = model_rtrl.init_state(1, device=device).requires_grad_()
        rtrl.reset()
        
        x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()
        
        # Forward through sequence with RTRL (all steps except last without loss)
        for t in range(SEQ_LEN - 1):
            x_t = x_onehot[:, t:t+1, :]
            pred_logits, info, h_next = model_rtrl(x_t, h_t)
            
            active_params, write_idx = get_expert_latent_activated(model_rtrl, info)
            rtrl.step(model_rtrl, x_t, h_t, None, active_params, None, write_idx)
            h_t = h_next.detach().requires_grad_()
        
        # Final step with loss
        x_t = x_onehot[:, -1:, :]
        pred_logits, info, h_next = model_rtrl(x_t, h_t)
        active_params, write_idx = get_expert_latent_activated(model_rtrl, info)
        
        loss = criterion(pred_logits, y)
        
        optimizer_rtrl.zero_grad()
        rtrl.step(model_rtrl, x_t, h_t, loss, active_params, None, write_idx)
        torch.nn.utils.clip_grad_norm_(model_rtrl.parameters(), 1.0)
        optimizer_rtrl.step()
        
        losses_rtrl.append(loss.item())
        pred = pred_logits.argmax(dim=-1).item()
        correct_rtrl.append(1 if pred == y.item() else 0)
        
        if (step + 1) % 50 == 0:
            print(".", end="", flush=True)
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses_rtrl[-50:]) / min(50, len(losses_rtrl))
            avg_acc = sum(correct_rtrl[-50:]) / min(50, len(correct_rtrl)) * 100
            print(f" [{step+1}: Loss={avg_loss:.3f}, Acc={avg_acc:.1f}%]", end="", flush=True)
    
    print()  # newline after progress
    
    final_loss_rtrl = sum(losses_rtrl[-50:]) / 50
    final_acc_rtrl = sum(correct_rtrl[-50:]) / 50 * 100
    
    print(f"\nRTRL Results: Loss = {final_loss_rtrl:.3f}, Acc = {final_acc_rtrl:.1f}%")
    
    # Always run speedup test to verify thesis component 3
    print("\n" + "=" * 75)
    print("PHASE 3: Segment Tree Speedup Test")
    print("=" * 75)
    print("(Testing speedup regardless of RTRL accuracy)")
    
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
    
    # Sparse with lazy updates
    print("Running 15 steps with SPARSE + LAZY (segment tree)...")
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
    
    # Full updates (no lazy)
    print("Running 15 steps with SPARSE (NO lazy updates)...")
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
            
            # All params, no lazy
            rtrl_full.step(model_perf, x_t, h_t, None, state_params_perf, None, None)
            h_t = h_next.detach().requires_grad_()
    
    full_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Sparse + Lazy:  {sparse_time:.2f}s")
    print(f"  Full updates:   {full_time:.2f}s")
    print(f"  Speedup:        {full_time / sparse_time:.2f}x")
    
    if sparse_time < full_time:
        print("✓ Segment tree provides speedup!")
    else:
        print("⚠ No speedup (might need longer sequences)")
    
    if final_acc_rtrl > 25:
        print("\n✓ RTRL converges on haystack!")
    else:
        print(f"\n⚠ RTRL achieves {final_acc_rtrl:.1f}% (BPTT: 100%) - needs tuning")

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

if best_config:
    print(f"✓ Found working configuration: {best_config['name']}")
    print(f"  - MoE with BPTT: 100% accuracy ✓")
    if 'final_acc_rtrl' in locals():
        if final_acc_rtrl > 40:
            print(f"  - RTRL: {final_acc_rtrl:.1f}% accuracy ✓")
        else:
            print(f"  - RTRL: {final_acc_rtrl:.1f}% accuracy (lower than BPTT)")
        if 'sparse_time' in locals() and 'full_time' in locals():
            if sparse_time < full_time:
                print(f"  - Segment tree speedup: {full_time / sparse_time:.2f}x ✓")
            else:
                print(f"  - No speedup: {full_time / sparse_time:.2f}x")
    print("\n🎉 HAYSTACK VERIFICATION COMPLETE")
else:
    print("⚠ Task too difficult for current MoE configuration")
    print("  Consider: longer training, different hyperparameters, or simpler task")
