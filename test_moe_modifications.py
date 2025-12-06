"""
Test modifications to MoE architecture to achieve zero loss:
1. Reduce residual strength (lower beta)
2. Write to more slots
3. Add learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
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

# Create modified MoE with adjustable beta
class ModifiedRecurrentMoE(RecurrentMoE):
    def __init__(self, beta=0.6, n_write_slots=2, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.n_write_slots = n_write_slots
    
    def forward(self, x, state_flat):
        B, S, D = state_flat.shape[0], self.n_slots, self.d
        _, T, _ = x.shape
        device, dtype = state_flat.device, state_flat.dtype
        state_old = state_flat.reshape([B, S, D]).contiguous()
        latent = state_old.clone()
        
        from moe import fourier_pos_enc
        pos = torch.arange(T, device=device).unsqueeze(-1).float()
        pe = fourier_pos_enc(pos, D, base=S)

        # (1) Mixed attention
        latent_state_x = self.state_embedding(x) + pe
        q = self.state_ln_q(latent)
        kv = torch.cat([latent, latent_state_x], dim=1)
        kv = self.state_ln_kv(kv)
        attn_out, _ = self.state_mha(q, kv, kv, need_weights=False)
        latent = latent + attn_out
        latent = latent + self.state_ffn(self.state_ln_ffn(latent))

        # (2) MoE with top-k experts
        pooled = self.state_ln_moe_in(latent.mean(dim=1))
        w, idx_experts = self.state_gate(pooled)
        mixed = self.state_experts(latent, w, idx_experts)
        latent = latent + mixed

        # (3) Modified: Write to more slots with lower beta
        logits = self.state_slot_ctx(self.state_ln_slot(latent)).squeeze(-1)
        logits, tgt_idx = torch.topk(logits, self.n_write_slots, dim=-1)
        w = F.softmax(logits, dim=-1)
        alpha = torch.zeros(B, S, device=device, dtype=dtype)
        alpha = alpha.scatter_add_(1, tgt_idx, w)
        
        # Use configurable beta
        state = self.beta * state_old + (1 - self.beta) * alpha.unsqueeze(-1) * torch.tanh(latent)

        # (4) Output 
        latent_out = self.out_embedding(x) + pe
        q = self.out_ln_q(latent_out)
        kv = torch.cat([latent_out, state], dim=1)
        kv = self.out_ln_kv(kv)
        attn_out, _ = self.out_mha(q, kv, kv, need_weights=False, attn_mask=None)
        latent_out = latent_out + attn_out
        latent_out = latent_out + self.out_ffn(self.out_ln_ffn(latent_out))
        y = self.out_proj(latent_out[:, -1])

        info = {"idx_experts": idx_experts.detach(), "idx_slots": tgt_idx.detach()}
        return y, info, state.reshape([B, S*D]).contiguous()

def train_model(model, name, steps=2000, lr=1e-2, lr_schedule=False):
    """Train a model and return final performance"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
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
        
        if lr_schedule:
            scheduler.step()
        
        losses.append(loss.item())
        pred = y.argmax(dim=-1).item()
        correct.append(1 if pred == tgt.item() else 0)
        
        if (step + 1) % 400 == 0:
            avg_loss = sum(losses[-100:]) / 100
            avg_acc = sum(correct[-100:]) / 100 * 100
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  {name:35s} Step {step+1:4d}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.1f}% (lr={current_lr:.4f})")
    
    final_loss = sum(losses[-100:]) / 100
    final_acc = sum(correct[-100:]) / 100 * 100
    return final_loss, final_acc

print("=" * 75)
print("ARCHITECTURAL IMPROVEMENTS TO REACH ZERO LOSS")
print("=" * 75)

# Baseline
print("\nBaseline: Original MoE (beta=0.6, write 2 slots)")
print("-" * 75)
torch.manual_seed(42)
model_baseline = ModifiedRecurrentMoE(beta=0.6, n_write_slots=2, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss_baseline, acc_baseline = train_model(model_baseline, "Baseline", steps=2000)

# Test 1: Lower beta (less residual)
print("\nTest 1: Lower beta=0.3 (more new information)")
print("-" * 75)
torch.manual_seed(42)
model1 = ModifiedRecurrentMoE(beta=0.3, n_write_slots=2, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss1, acc1 = train_model(model1, "Beta=0.3", steps=2000)

# Test 2: Write to more slots
print("\nTest 2: Write to 4 slots (all slots)")
print("-" * 75)
torch.manual_seed(42)
model2 = ModifiedRecurrentMoE(beta=0.6, n_write_slots=4, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss2, acc2 = train_model(model2, "Write 4 slots", steps=2000)

# Test 3: Combine lower beta + more slots
print("\nTest 3: Beta=0.3 + Write 4 slots")
print("-" * 75)
torch.manual_seed(42)
model3 = ModifiedRecurrentMoE(beta=0.3, n_write_slots=4, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss3, acc3 = train_model(model3, "Beta=0.3 + Write 4", steps=2000)

# Test 4: LR scheduling
print("\nTest 4: Beta=0.3 + Write 4 + LR Schedule")
print("-" * 75)
torch.manual_seed(42)
model4 = ModifiedRecurrentMoE(beta=0.3, n_write_slots=4, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss4, acc4 = train_model(model4, "Beta=0.3 + Write 4 + LR Sched", steps=2000, lr_schedule=True)

# Test 5: Very low beta (almost no residual)
print("\nTest 5: Beta=0.1 (minimal residual)")
print("-" * 75)
torch.manual_seed(42)
model5 = ModifiedRecurrentMoE(beta=0.1, n_write_slots=4, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss5, acc5 = train_model(model5, "Beta=0.1 + Write 4", steps=2000)

# Test 6: Best config with longer training
print("\nTest 6: Best config + longer training (3000 steps)")
print("-" * 75)
torch.manual_seed(42)
model6 = ModifiedRecurrentMoE(beta=0.1, n_write_slots=4, d_model=32, n_heads=2, n_slots=4, n_experts=4, topk=2, d_in=2, d_out=2).to(device)
loss6, acc6 = train_model(model6, "Beta=0.1 + Write 4 (3000 steps)", steps=3000)

print("\n" + "=" * 75)
print("RESULTS SUMMARY")
print("=" * 75)
results = [
    ("Baseline (beta=0.6, write 2)", loss_baseline, acc_baseline),
    ("Beta=0.3", loss1, acc1),
    ("Write 4 slots", loss2, acc2),
    ("Beta=0.3 + Write 4", loss3, acc3),
    ("Beta=0.3 + Write 4 + LR Sched", loss4, acc4),
    ("Beta=0.1 + Write 4", loss5, acc5),
    ("Beta=0.1 + Write 4 (3000 steps)", loss6, acc6),
]

for name, loss, acc in results:
    marker = " ✓✓" if loss < 0.05 else " ✓" if loss < 0.2 else ""
    print(f"  {name:35s}: Loss = {loss:.4f}, Acc = {acc:5.1f}%{marker}")

best_loss = min(r[1] for r in results)
best_acc = max(r[2] for r in results)
best_name = [r[0] for r in results if r[1] == best_loss][0]

print(f"\n🏆 Best: {best_name}")
print(f"   Loss = {best_loss:.4f}, Acc = {best_acc:.1f}%")

if best_loss < 0.05:
    print("\n✓✓ MoE CAN reach near-zero loss!")
    print("    Architectural fix: Reduce residual strength (beta) and write to more slots")
elif best_loss < 0.2:
    print("\n✓ MoE reaches good performance with modifications")
else:
    print("\n⚠ MoE still struggles - may need more fundamental changes")
