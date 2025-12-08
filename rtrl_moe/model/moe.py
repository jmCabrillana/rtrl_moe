import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Expert (vmap compatible)
class ExpertBank(nn.Module):
    """
    E experts, each: y = act(W[e] @ x + b[e])
    Params are stored per-expert (ParameterList) but used via stacked banks in forward.
    """
    def __init__(self, E, d):
        super().__init__()
        self.E, self.d = E, d
        self.W = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(E)])
        self.b = nn.ParameterList([nn.Parameter(torch.empty(d))    for _ in range(E)])
        self.reset_parameters()

    def reset_parameters(self):
        for W, b in zip(self.W, self.b):
            nn.init.kaiming_uniform_(W, a=5**0.5)  # good for ReLU
            fan_in = W.size(1)
            bound = 1.0 / fan_in**0.5
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x, w, idx):
        B, T, D = x.shape
        _, k = idx.shape

        W_bank = torch.stack(list(self.W), dim=0).contiguous()  # [E, D, D]
        b_bank = torch.stack(list(self.b), dim=0).contiguous()  # [E, D]

        flat_idx = idx.reshape(-1)
        W_sel = W_bank.index_select(0, flat_idx).reshape(B, k, D, D).contiguous()   # [B,k,D,D]
        b_sel = b_bank.index_select(0, flat_idx).reshape(B, k, D).contiguous()      # [B,k,D]

        y_k = torch.einsum('b t i, b k i o -> b t k o', x, W_sel) + b_sel.unsqueeze(1)
        y_k = F.relu(y_k)
        y = (y_k * w.view(B,1,k,1)).sum(dim=2)  # [B,T,D]
        return y


class MLP(nn.Module):
    def __init__(self, d, mult=1, dropout=0.0):
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
    
def fourier_pos_enc(pos, d, base=10000): 
    sin = torch.sin(pos / base**(torch.arange(0,d//2).to(pos)/ (d//2))).to(pos)
    cos = torch.cos(pos / base**(torch.arange(0,d//2).to(pos)/ (d//2))).to(pos)
    return torch.cat([sin, cos], dim=-1)


# ---------- Recurrent MoE ----------
class RecurrentMoE(nn.Module):
    """
    Step:
      1) Mixed-attn: Q=latent, K/V=concat(latent, x)
      2) Pool latent -> gate -> mixture of experts (sequence-level)
      3) Write expert update back to latent slots chosen by a small gate
    """
    def __init__(self, d_model=512, n_heads=2, n_slots=4, n_experts=4, topk=2, topk_read=2, topk_write=2,
                  dropout=0.0, d_in=None, d_out=None, orthogonalize=False):
        super().__init__()
        d, E = d_model, n_experts
        self.d, self.n_slots = d, n_slots
        if not d_in: d_in = d; self.d_in = d_in
        if not d_out: d_out = d; self.d_out = d_out
        self.topk = topk
        self.topk_read, self.topk_write = topk_read, topk_write
        self.orthogonalize = orthogonalize

        # Attention
        self.state_embedding = nn.Linear(d_in, d)
        self.state_mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.state_ln_q = nn.LayerNorm(d)
        self.state_ln_kv = nn.LayerNorm(d)
        self.state_ffn = MLP(d, dropout=dropout)
        self.state_ln_ffn = nn.LayerNorm(d)

        # Sequence level gating with topk expert
        self.state_gate = TopKGate(d, n_experts, k=topk)
        # self.state_experts = nn.ModuleList([MLP(d, dropout=dropout) for _ in range(n_experts)])
        self.state_experts = ExpertBank(E, d)
        self.state_ln_moe_in = nn.LayerNorm(d)
        self.state_latent_proj = nn.Linear(d, d)

        # Read gate: select which slots to READ from
        self.state_read_gate = nn.Linear(d, 1, bias=False)
        
        # Write gate: select which slots to WRITE to
        self.state_ln_slot = nn.LayerNorm(d)
        self.state_slot_ctx = nn.Linear(d, 1, bias=False)

        # Output Attention
        self.out_embedding = nn.Linear(d_in, d)
        self.out_mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.out_ln_q = nn.LayerNorm(d)
        self.out_ln_kv = nn.LayerNorm(d)
        self.out_ffn = MLP(d, dropout=dropout)
        self.out_ln_ffn = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d_out)

        self.t=0

    @torch.no_grad()
    def init_state(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        latent = torch.randn(batch_size, self.n_slots*self.d, device=device, dtype=dtype) * 0.02
        return latent
    
    def orthogonalize_slots(self, y):
        # We want T orthonormal vectors in ℝ^D
        # Do QR on the transposed matrix so columns become orthonormal
        Q, R = torch.linalg.qr(y.transpose(1, 2))  # [B, D, T], [B, T, T]
        y_orth = Q.transpose(1, 2)                 # [B, T, D]
        return y_orth


    def forward(self, x, state_flat):
        B, S, D = state_flat.shape[0], self.n_slots, self.d
        _, T, _ = x.shape
        device, dtype = state_flat.device, state_flat.dtype
        state_old = state_flat.reshape([B, S, D]).contiguous()
        latent = state_old.clone()
        pos = torch.arange(T, device=device).unsqueeze(-1).float() #+ self.t
        pe = fourier_pos_enc(pos, D, base=S)
        # self.t += T

        # (0) Read gating: select which slots to READ from
        read_scores = self.state_read_gate(latent).squeeze(-1)  # [B, k]
        logits, read_idx = torch.topk(read_scores, self.topk_read, dim=-1)  # [B, k] - select top-k slots to read
        read_idx_expanded = read_idx.unsqueeze(-1).expand(B, self.topk_read, D)  # [B, k, D]
        latent_read = torch.gather(latent, 1, read_idx_expanded) * logits.unsqueeze(-1)  # [B, k, D] - selected slots
        
        # (1) Mixed attention
        latent_state_x = self.state_embedding(x) + pe
        q = self.state_ln_q(latent_read)  # Query from selected read slots
        kv = torch.cat([latent_read, latent_state_x], dim=1)
        kv = self.state_ln_kv(kv)
        attn_out, _ = self.state_mha(q, kv, kv, need_weights=False)
        latent_read = latent_read + attn_out
        latent_read = latent_read + self.state_ffn(self.state_ln_ffn(latent_read))

        # (2) MoE with top-k experts (sequence-level)
        pooled = self.state_ln_moe_in(latent_read.mean(dim=1))       # [B,D]
        w, idx_experts = self.state_gate(pooled)                # [B,k]
        mixed = self.state_experts(latent_read, w, idx_experts)
        latent_read = latent_read + mixed
        if self.orthogonalize:
            latent_read = self.orthogonalize_slots(latent_read)

        # (3) Choose *one* target slot per sample and update state only there
        logits = self.state_slot_ctx(self.state_ln_slot(latent)).squeeze(-1)        # [B,S]
        logits, write_idx = torch.topk(logits, self.topk_write, dim=-1)             # [B, k]
        # Aggregate latent_read into single update vector and weighted write
        latent_update = latent_read.mean(dim=1).unsqueeze(1).expand(B, self.topk_write, D)  # [B, k, D]
        write_weights = F.softmax(logits, dim=-1)  # [B, k]
        state_update = torch.zeros(B, S, D, device=device, dtype=dtype)
        write_idx_expanded = write_idx.unsqueeze(-1).expand(B, self.topk_write, D)  # [B, k, D]
        weighted_update = latent_update * write_weights.unsqueeze(-1)  # [B, k, D]
        state_update = state_update.scatter_add_(1, write_idx_expanded, weighted_update)  # [B, S, D]
        beta = 0.0
        state = beta * state_old + (1 - beta) * state_update  # residual update

        # (4) Output 
        latent_out = self.out_embedding(x) + pe
        q = self.out_ln_q(latent_out)
        kv = torch.cat([latent_out, state], dim=1)
        kv = self.out_ln_kv(kv)
        attn_out, _ = self.out_mha(q, kv, kv, need_weights=False, attn_mask=None)
        latent_out = latent_out + attn_out
        latent_out = latent_out + self.out_ffn(self.out_ln_ffn(latent_out))
        y = self.out_proj(latent_out[:, -1]) # autoregressive

        info = {
            "idx_experts": idx_experts.detach(), 
            "idx_slot_write": write_idx.detach(),  # write slots
            "idx_slots_read": read_idx.detach()  # read slots
        }
        return y, info, state.reshape([B, S*D]).contiguous()
    
def get_expert_latent_activated(model, info):
    """
    Extract active parameters and read/write indices based on MoE routing.
    
    Args:
        model: RecurrentMoE model
        info: dict with 'idx_slots' (latent slots) and 'idx_experts' (expert indices)
    
    Returns:
        active_params: dict of parameters that were active this step
        write_indices: list of state indices that were written to
        read_indices: list of state indices that were read from (for Jacobian computation)
    """
    D = model.d
    S = model.n_slots
    state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    
    # Extract activated slot indices for WRITE (these are updated)
    idx_slots_write = list(set(info['idx_slot_write'].flatten().tolist()))
    write_indices = sum((list(range(D*i, D*(i+1))) for i in idx_slots_write), start=[])
    
    # Extract activated slot indices for READ (from read gating)
    if 'idx_slots_read' in info:
        idx_slots_read = list(set(info['idx_slots_read'].flatten().tolist()))
        read_indices = sum((list(range(D*i, D*(i+1))) for i in idx_slots_read), start=[])
    else:
        # Fallback: read from write slots if read gating not available
        read_indices = sum((list(range(D*i, D*(i+1))) for i in range(S)), start=[])
    
    
    # Extract activated expert indices
    expert_ids = info.get('idx_experts', [])
    if not isinstance(expert_ids, list):
        expert_ids = list(set(expert_ids.flatten().tolist()))
    
    # Build pattern for expert parameters
    pattern = re.compile(r'^state_experts\.(W|b)\.\d+$')
    core_params = {name: p for name, p in state_params.items() if pattern.match(name) is None}
    
    if expert_ids:
        ids_pattern = "|".join(map(str, expert_ids))
        expert_pattern = re.compile(rf'^state_experts\.(W|b)\.({ids_pattern})$')
        active_experts_params = {name: p for name, p in state_params.items() if expert_pattern.match(name)}
        active_params = {**core_params, **active_experts_params}
    else:
        active_params = core_params
    
    return active_params, write_indices, read_indices

# ---------- Tiny usage ----------
if __name__ == "__main__": 
    B, T, D = 1, 8, 64
    model = RecurrentMoE(d_model=D, n_heads=4, n_slots=64, n_experts=64, topk=2).to(device)
    x = torch.randn(B, T, D).to(device)
    pattern = re.compile(r'^state_experts\.(W|b)\.\d+$')
    state_params = {name:p for name,p in model.named_parameters() if name.startswith("state_")}
    core_params = {name:p for name, p in state_params.items() if pattern.match(name) is None}
    expert_params = {name:p for name, p in state_params.items() if pattern.match(name)}
    state = model.init_state(B, device=x.device).requires_grad_()
    rtrl = BlockRTRL(state_params, B, state.shape[-1])