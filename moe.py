import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

class TopKGate(nn.Module):
    def __init__(self, d, n_experts, k=2):
        super().__init__()
        assert 1 <= k <= n_experts
        self.k, self.n = k, n_experts
        self.proj = nn.Linear(d, n_experts, bias=False)
    def forward(self, h):                         # h: [B,D]
        logits = self.proj(h)                     # [B,E]
        val, idx = torch.topk(logits, self.k, dim=-1)  # [B,k]
        return val.softmax(-1), idx               # normalize over top-k

# ---------- Recurrent MoE ----------
class RecurrentMoE(nn.Module):
    """
    Step:
      1) Mixed-attn: Q=latent, K/V=concat(latent, x)
      2) Pool latent -> gate -> mixture of experts (sequence-level)
      3) Write expert update back to *one* latent slot chosen by a small gate
    """
    def __init__(self, d_model=512, n_heads=8, n_slots=4, n_experts=4, topk=2, dropout=0.0):
        super().__init__()
        d = d_model
        self.d, self.n_slots = d, n_slots

        # Attention
        self.state_mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.state_ln_q = nn.LayerNorm(d)
        self.state_ln_kv = nn.LayerNorm(d)
        self.state_ffn = MLP(d, dropout=dropout)
        self.state_ln_ffn = nn.LayerNorm(d)

        # Sequence level gating with topk expert
        self.state_gate = TopKGate(d, n_experts, k=topk)
        self.state_experts = nn.ModuleList([MLP(d, dropout=dropout) for _ in range(n_experts)])
        self.state_ln_moe_in = nn.LayerNorm(d)
        self.state_latent_proj = nn.Linear(d, d)

        # Latent projection
        self.state_ln_slot = nn.LayerNorm(d)
        self.state_slot_ctx = nn.Linear(d, 1, bias=False)


    @torch.no_grad()
    def init_state(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        latent = torch.randn(batch_size, self.n_slots, self.d, device=device, dtype=dtype) * 0.02
        return latent

    def forward(self, x, state):
        B, S, D = state.shape
        device = state.device
        latent = state

        # (1) Mixed attention
        q = self.state_ln_q(latent)
        kv = torch.cat([latent, x], dim=1)
        kv = self.state_ln_kv(kv)
        attn_out, _ = self.state_mha(q, kv, kv, need_weights=False)
        latent = latent + attn_out
        latent = latent + self.state_ffn(self.state_ln_ffn(latent))

        # (2) MoE with top-k experts (sequence-level) =1 here
        pooled = self.state_ln_moe_in(latent.mean(dim=1))       # [B,D]
        w, idx = self.state_gate(pooled)                        # [B,k]]
        mixed = torch.zeros(B, S, D, device=latent.device, dtype=latent.dtype)
        for i in range(self.state_gate.k):
            for e in range(len(self.state_experts)):
                mask = (idx[:, i] == e)                   # [B]
                if mask.any():
                    acc = torch.einsum('b, b S D-> b S D', w[mask, i], self.state_experts[e](latent[mask]))
                    mixed = mixed.index_put((mask,), acc, accumulate=True)
        latent = mixed

        # (3) Choose *one* target slot per sample and update state only there
        logits = self.state_slot_ctx(self.state_ln_slot(latent)).squeeze(-1)        # [B,S]
        w, tgt_idx = torch.topk(logits, 2, dim=-1)                          # [B,2]
        for i in range(2):
            for s in range(S):
                mask = (tgt_idx[:, i] == s)
                acc = torch.einsum('b, b D-> b D', w[mask, i], latent[mask, s])
                state = state.index_put((mask, torch.tensor(s, device=device)),
                                        acc, accumulate=True)

        # Output and info
        y = latent.mean(dim=1)
        info = {
            "idx_experts": idx.detach(),
            "idx_slots": tgt_idx.detach()
        }
        return y, info, latent

if __name__ == "__main__": 
    # ---------- Tiny usage ----------
    B, T, D = 1, 6, 128
    model = RecurrentMoE(d_model=D, n_heads=4, n_slots=4, n_experts=64, topk=2)
    x = torch.randn(B, T, D)
    state = model.init_state(B, device=x.device)
    y, info, state = model(x, state)
