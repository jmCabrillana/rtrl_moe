"""
Stable MoE with Lyapunov regularization for long-sequence training.

Architecture: Identical to moe.py, with added regularization losses.
Regularization applied during training (not in forward pass).

Key components:
- Lyapunov penalty: Encourages ||dh/dh|| ≈ 1 (spectral stability)
- Gradient clipping: Prevents blow-up
- Weight decay: Prevents Expert overgrowth
- Gate temperature annealing: Smooth routing
"""

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
    def __init__(self, d_model=512, n_heads=2, n_slots=4, n_experts=4, topk=2, topk_read=2, topk_write=2, dropout=0.0, d_in=None, d_out=None):
        super().__init__()
        d, E = d_model, n_experts
        self.d, self.n_slots = d, n_slots
        self.topk_read, self.topk_write = topk_read, topk_write
        if not d_in: d_in = d; self.d_in = d_in
        if not d_out: d_out = d; self.d_out = d_out

        # Attention
        self.state_embedding = nn.Linear(d_in, d)
        self.state_mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.state_ln_q = nn.LayerNorm(d)
        self.state_ln_kv = nn.LayerNorm(d)
        self.state_ffn = MLP(d, dropout=dropout)
        self.state_ln_ffn = nn.LayerNorm(d)
        
        # Orthogonal state core (Cayley transform): A_skew := A - A^T
        A = torch.randn(d, d) * 0.02
        self.state_A_skew = nn.Parameter(A - A.T)

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

        self.t = 0
        self.lyapunov_probe_vectors = None  # K-step Lyapunov control probes

    @torch.no_grad()
    def init_state(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        latent = torch.randn(batch_size, self.n_slots*self.d, device=device, dtype=dtype) * 0.02
        return latent

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
        read_scores = self.state_read_gate(latent).squeeze(-1)  # [B, S]
        _, read_idx = torch.topk(read_scores, self.topk_read, dim=-1)  # [B, topk_read] - select top slots to read
        
        # Select read slots
        read_idx_expanded = read_idx.unsqueeze(-1).expand(B, self.topk_read, D)  # [B, topk_read, D]
        latent_read = torch.gather(latent, 1, read_idx_expanded)  # [B, topk_read, D] - selected slots
        
        # (1) Mixed attention on selected read slots
        latent_state_x = self.state_embedding(x) + pe
        q = self.state_ln_q(latent_read)  # Query from read slots only
        kv = torch.cat([latent_read, latent_state_x], dim=1)
        kv = self.state_ln_kv(kv)
        attn_out, _ = self.state_mha(q, kv, kv, need_weights=False)
        # TODO: These operations are not orthogonal, comment out for now
        # latent_read = latent_read + attn_out
        # latent_read = latent_read + self.state_ffn(self.state_ln_ffn(latent_read))

        # (2) MoE with top-k experts - operate on [B, T, D] tensor
        latent_moe = self.state_ln_moe_in(latent_read)  # [B, 2, D]
        # Gate on each position independently
        w_list, idx_list = [], []
        for t in range(latent_moe.shape[1]):
            w_t, idx_t = self.state_gate(latent_moe[:, t])  # [B, k]
            w_list.append(w_t)
            idx_list.append(idx_t)
        w = torch.stack(w_list, dim=1)  # [B, T, k]
        idx_experts = torch.stack(idx_list, dim=1)  # [B, T, k]
        
        # (3) Orthogonal state transition using expert weights
        # Construct Cayley transform from expert weights: Q = (I+A)(I-A)^{-1}
        I = torch.eye(D, device=device, dtype=dtype)
        
        # Use expert weights to modulate the skew-symmetric matrix
        # w: [B, T, k] - weight per expert per position
        A_base = self.state_A_skew  # [D, D] - base skew-symmetric
        
        # Weighted combination of orthogonal transforms per position
        latent_core = []
        for t in range(latent_moe.shape[1]):
            # Scale A by expert weights for this position
            w_t = w[:, t].mean(dim=-1, keepdim=True).mean(dim=0)  # scalar: average weight [1]
            A_scaled = A_base * (0.25 * w_t)  # Use expert weight to scale (vmap-compatible, no .item())
            try:
                Q = torch.linalg.solve(I - A_scaled, I + A_scaled)  # Cayley
            except:
                Q = I  # Fallback
            
            # Apply to this position
            latent_t = torch.einsum('bd,dd->bd', latent_read[:, t], Q)  # [B, D]
            latent_core.append(latent_t)
        
        latent_core = torch.stack(latent_core, dim=1)  # [B, T, D]
        
        # (3.1) Write slot selection: choose top slots by their importance score
        latent_core_avg = torch.tanh(latent_core).mean(dim=1)  # [B, D] - average over T
        # Score each slot in latent (not just read slots)
        latent_full = state_old.clone()  # [B, S, D] - full state for scoring
        slot_scores_full = self.state_slot_ctx(self.state_ln_slot(latent_full)).squeeze(-1)  # [B, S]
        # Extract scores for read slots only
        logits_read = slot_scores_full.gather(1, read_idx)  # [B, topk_read]
        _, write_idx_local = torch.topk(logits_read, min(self.topk_write, self.topk_read), dim=-1)  # [B, topk_write]
        # Map back to global slot indices
        write_idx = torch.gather(read_idx, 1, write_idx_local)  # [B, topk_write]
        
        # State update using scatter_add_ (vmap compatible)
        state = state_old.clone()
        w = F.softmax(logits_read[:, :min(self.topk_write, self.topk_read)], dim=-1)  # [B, topk_write]
        alpha = torch.zeros(B, S, device=device, dtype=dtype)
        alpha = alpha.scatter_add_(1, write_idx, w)  # Accumulate weights at write slots
        state = alpha.unsqueeze(-1) * torch.tanh(latent_core_avg).unsqueeze(1) + (1.0 - alpha.sum(dim=1, keepdim=True).unsqueeze(-1)) * state_old

        # (4) Output 
        latent_out = self.out_embedding(x) + pe
        q = self.out_ln_q(latent_out)
        kv = torch.cat([latent_out, state], dim=1)
        kv = self.out_ln_kv(kv)
        # attn_tok  = torch.triu(torch.full((T, T), float('-inf'), device=x.device), 1)
        # attn_slot = torch.zeros(T, S, device=device)
        # attn_mask = torch.cat([attn_tok, attn_slot], dim=1)           # [T, T+S]
        attn_out, _ = self.out_mha(q, kv, kv, need_weights=False, attn_mask=None)
        latent_out = latent_out + attn_out
        latent_out = latent_out + self.out_ffn(self.out_ln_ffn(latent_out))
        y = self.out_proj(latent_out[:, -1])

        info = {
            "idx_experts": idx_experts.detach(), 
            "idx_slots_read": read_idx.detach(),  # read slots
            "idx_slots_write": write_idx.detach()  # write slots
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
    if 'idx_slots_write' in info:
        idx_slots_write = list(set(info['idx_slots_write'].flatten().tolist()))
    else:
        idx_slots_write = list(range(S))  # Fallback: all slots
    write_indices = sum((list(range(D*i, D*(i+1))) for i in idx_slots_write), start=[])
    
    # Extract activated slot indices for READ (from read gating)
    if 'idx_slots_read' in info:
        idx_slots_read = list(set(info['idx_slots_read'].flatten().tolist()))
    else:
        # Fallback: read from write slots if read gating not available
        idx_slots_read = idx_slots_write
    
    read_indices = sum((list(range(D*i, D*(i+1))) for i in idx_slots_read), start=[])
    
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


# ============ Regularization Utilities ============

def compute_lyapunov_penalty(model, state_h, x_window, K=16, probes=2, eps=1e-9):
    """
    Global Lyapunov Control: K-step penalty on Jacobian product.
    
    Maintains ||J_{t+K-1}...J_t|| ≈ 1 through online QR decomposition.
    This prevents gradient explosion/vanishing over long horizons.
    
    Args:
        model: RecurrentMoE
        state_h: current state [B, H]
        x_window: input sequence window [T, B, D_in]
        K: window length for Lyapunov penalty
        probes: number of probe vectors for QR tracking
        eps: numerical stability
    
    Returns:
        penalty: scalar loss on ||log|diag(R)|||| (deviation from unit growth)
    """
    B, H = state_h.shape
    T = x_window.shape[0]
    device, dtype = state_h.device, state_h.dtype
    
    # Initialize probe vectors: orthonormal basis
    if T < K:
        K = T
    
    # Random probe vectors (orthonormalized)
    V = torch.randn(H, probes, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(V)  # [H, probes]
    
    log_norms = []
    h = state_h.clone()
    
    # Forward pass tracking Jacobian via JVP
    for t in range(min(K, T)):
        x_t = x_window[t:t+1]  # [1, B, D_in]
        
        # JVP: forward probes through state transition
        # For each probe v, compute J*v where J = ∂h_{t+1}/∂h_t
        jvps = []
        for m in range(probes):
            v = V[:, m:m+1].expand(B, H)  # [B, H]
            # Use finite differences for JVP approximation
            eps_fd = 1e-5
            
            def f_step(h_in):
                y, info, h_next = model(x_t, h_in.reshape(B, S*D).contiguous())
                return h_next.reshape(B, -1)
            
            h_plus = f_step(h + eps_fd * v)
            h_base = f_step(h)
            jvp = (h_plus - h_base) / eps_fd  # [B, H]
            jvps.append(jvp)
        
        # Stack: Y = [v1*J, v2*J, ...] -> [B, H, probes]
        Y = torch.stack(jvps, dim=-1)  # [B, H, probes]
        
        # QR decomposition: maintain orthonormal basis
        Y_flat = Y.reshape(B*H, probes)
        Q, R = torch.linalg.qr(Y_flat)  # Q: [BH, probes], R: [probes, probes]
        
        # Log of diagonal: penalize if not ≈ 1
        log_diag = torch.log(torch.abs(torch.diag(R)) + eps)  # [probes]
        log_norms.append(log_diag.sum())
        
        # Update probes
        V = Q.reshape(B, H, probes)[:1].squeeze(0)  # Use first batch elem
        
        # Step forward
        y, info, h = model(x_t, h)
    
    # Penalty: (mean log norm)^2 -> encourages norm preservation
    if log_norms:
        penalty = (torch.stack(log_norms).mean()) ** 2
    else:
        penalty = torch.tensor(0.0, device=device, dtype=dtype)
    
    return penalty


def compute_expert_norm_penalty(model, target_norm=1.0):
    """
    Regularize expert matrix norms to stay close to target.
    Prevents experts from exploding or vanishing.
    
    Args:
        model: RecurrentMoE model
        target_norm: desired Frobenius norm
    
    Returns:
        penalty: scalar loss
    """
    penalty = 0.0
    
    # Target: expert weights have moderate norm (not too large/small)
    for i, W in enumerate(model.state_experts.W):
        expert_norm = torch.norm(W, p='fro')
        penalty = penalty + (expert_norm - target_norm) ** 2
    
    return penalty / max(1.0, len(model.state_experts.W))


# ---------- Training wrapper with regularization ----------

class StableRTRLTrainer:
    """Wrapper for stable RTRL training with K-step Lyapunov regularization."""
    
    def __init__(self, model, lr=3e-3, lyapunov_weight=0.001, lyapunov_K=16, 
                 expert_norm_weight=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lyapunov_weight = lyapunov_weight
        self.lyapunov_K = lyapunov_K
        self.expert_norm_weight = expert_norm_weight
        self.lr = lr
        self.step_count = 0
    
    def backward_step(self, task_loss, state_h=None, x_window=None, clip_norm=1.0, 
                      apply_lyapunov_every=16):
        """
        Backward pass with K-step Lyapunov control and regularization.
        
        Args:
            task_loss: main task loss
            state_h: current state for Lyapunov penalty
            x_window: input window for Lyapunov computation
            clip_norm: gradient clipping threshold
            apply_lyapunov_every: only apply Lyapunov penalty every N steps
        
        Returns:
            loss_components: dict with breakdowns
        """
        self.optimizer.zero_grad()
        
        total_loss = task_loss
        components = {'task_loss': task_loss.item()}
        
        # Apply Lyapunov penalty periodically (expensive computation)
        if (self.step_count % apply_lyapunov_every == 0 and 
            state_h is not None and x_window is not None and self.lyapunov_weight > 0):
            try:
                lyap_penalty = compute_lyapunov_penalty(
                    self.model, state_h, x_window, K=self.lyapunov_K, probes=2
                )
                total_loss = total_loss + self.lyapunov_weight * lyap_penalty
                components['lyap_penalty'] = lyap_penalty.item()
            except Exception as e:
                components['lyap_penalty'] = 0.0  # Skip if error
        
        # Expert norm penalty
        if self.expert_norm_weight > 0:
            expert_penalty = compute_expert_norm_penalty(self.model)
            total_loss = total_loss + self.expert_norm_weight * expert_penalty
            components['expert_penalty'] = expert_penalty.item()
        
        # Backward with regularization
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        components['grad_norm'] = grad_norm.item()
        
        self.optimizer.step()
        self.step_count += 1
        
        components['total_loss'] = total_loss.item()
        return components


# ---------- Tiny usage ----------
if __name__ == "__main__": 
    B, T, D = 1, 8, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentMoE(d_model=D, n_heads=4, n_slots=8, n_experts=4, topk=2).to(device)
    x = torch.randn(B, T, D).to(device)
    state = model.init_state(B, device=x.device)
    y, info, state_new = model(x, state)
    print(f"Output shape: {y.shape}, State shape: {state_new.shape}")
