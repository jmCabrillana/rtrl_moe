"""
RTRL for a recurrent MoE transformer-like cell with **lazy deferral** via a factor queue.
- Top-1 experts; MoE active once every `moe_period` steps (e.g., 10).
- Exact online RTRL on the recurrent state S_t, but only **materialize** sensitivity
  for the **active** parameter block(s) each step.
- Inactive blocks accumulate a deferred left-operator (product of past J_S),
  applied in O(log m) using a segment-tree-backed factor queue when they wake.

This file replaces the previous RTRLMoE with a factor-queue implementation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -------------------- Utils --------------------

def seed_all(seed: int = 0):
    torch.manual_seed(seed)


def flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(params)


def unflatten_to_grads(flat: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    offset = 0
    for p in params:
        n = p.numel()
        grads.append(flat[offset : offset + n].view_as(p).clone())
        offset += n
    return grads


# -------------------- MoE components --------------------

class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Top1Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(h)  # (B, E)
        idx = torch.argmax(logits, dim=-1)  # (B,)
        return logits, idx


# -------------------- Recurrent Transformer-like cell --------------------

class MoERecurrentCell(nn.Module):
    def __init__(self, obs_dim: int, K: int, D: int, num_experts: int, d_hidden: int, moe_period: int = 10):
        super().__init__()
        self.K = K
        self.D = D
        self.num_experts = num_experts
        self.period = moe_period

        # Read projections
        self.W_q = nn.Linear(obs_dim, D)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)

        # Base write (used when MoE inactive)
        self.W_w = nn.Linear(obs_dim, D)
        self.W_g = nn.Linear(obs_dim + D, 1)

        # MoE stack
        self.router = Top1Router(D, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(D, d_hidden) for _ in range(num_experts)])

        # Policy / value heads (illustrative)
        self.policy_head = nn.Linear(D + obs_dim, 2)
        self.value_head = nn.Linear(D + obs_dim, 1)

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.K, self.D, requires_grad=True)

    def forward_step(self, S_prev: torch.Tensor, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict]:
        keys = self.W_k(S_prev)
        vals = self.W_v(S_prev)
        q = self.W_q(x)
        attn_logits = (keys @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.D)
        attn = F.softmax(attn_logits, dim=-1)
        context = (attn.unsqueeze(-1) * vals).sum(dim=1)
        features = torch.cat([context, x], dim=-1)

        use_moe = (t % self.period == 0)
        if use_moe:
            logits, idx = self.router(context)
            B = context.size(0)
            out = torch.zeros_like(context)
            for b in range(B):
                e = idx[b].item()
                out[b:b+1] = self.experts[e](context[b:b+1])
            write_vec = out
            gate = torch.sigmoid(self.W_g(torch.cat([x, context], dim=-1)))
            active_expert = idx
        else:
            write_vec = self.W_w(x)
            gate = torch.sigmoid(self.W_g(torch.cat([x, context], dim=-1)))
            active_expert = None

        pos = t % self.K
        S_new = S_prev.clone().requires_grad_(True)
        S_new[:, pos, :] = (1.0 - gate) * S_prev[:, pos, :] + gate * write_vec

        outs = {"context": context, "features": features, "use_moe": use_moe, "expert_idx": active_expert}
        return S_new, outs

    def policy_value(self, features: torch.Tensor, act_dim: int) -> Tuple[Normal, torch.Tensor]:
        out = self.policy_head(features)
        mean, log_std = out[:, :1], out[:, 1:2]
        mean = mean.repeat(1, act_dim)
        log_std = torch.clamp(log_std, -4, 1).repeat(1, act_dim)
        dist = Normal(mean, torch.exp(log_std))
        value = self.value_head(features)
        return dist, value.squeeze(-1)


# -------------------- Linear left-operators & Factor Queue --------------------

class LeftOp:
    def __init__(self, apply_fn: Callable[[torch.Tensor], torch.Tensor], name: str = ""):
        self.apply_fn = apply_fn
        self.name = name
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.apply_fn(X)


def identity_op() -> LeftOp:
    return LeftOp(lambda X: X, name="I")


def compose_ops(A: LeftOp, B: LeftOp) -> LeftOp:
    return LeftOp(lambda X: A(B(X)), name=f"({A.name}∘{B.name})")


def js_left_op(f_state: Callable[[torch.Tensor], torch.Tensor], S_prev: torch.Tensor) -> LeftOp:
    KD = S_prev.numel()
    def apply_on(X: torch.Tensor) -> torch.Tensor:
        B = X.shape[1]
        cols = []
        for j in range(B):
            v = X[:, j].view_as(S_prev)
            y_tan = torch.func.jvp(f_state, (S_prev,), (v,))[1]
            cols.append(y_tan.reshape(KD, 1))
        return torch.cat(cols, dim=1)
    return LeftOp(apply_on, name="J_S")


class FactorQueue:
    def __init__(self, window: int):
        self.m = window
        self.n = 1
        while self.n < self.m:
            self.n *= 2
        self.tree: List[LeftOp] = [identity_op() for _ in range(2 * self.n)]
        self.head = 0
        self.count = 0

    def _set_leaf(self, pos: int, op: LeftOp):
        i = self.n + pos
        self.tree[i] = op
        i //= 2
        while i >= 1:
            left = self.tree[2 * i]
            right = self.tree[2 * i + 1]
            self.tree[i] = compose_ops(right, left)
            i //= 2

    def push(self, op: LeftOp):
        self._set_leaf(self.head, op)
        self.head = (self.head + 1) % self.m
        self.count = min(self.count + 1, self.m)

    def _range_ops(self, L: int, R: int) -> LeftOp:
        l = self.n + L
        r = self.n + R
        left_acc = identity_op()
        right_acc = identity_op()
        while l < r:
            if l & 1:
                left_acc = compose_ops(self.tree[l], left_acc)
                l += 1
            if r & 1:
                r -= 1
                right_acc = compose_ops(right_acc, self.tree[r])
            l //= 2
            r //= 2
        return compose_ops(right_acc, left_acc)

    def apply_last(self, k: int, X: torch.Tensor) -> torch.Tensor:
        if k <= 0 or self.count == 0:
            return X
        k = min(k, self.count)
        end = (self.head - 0) % self.m
        start = (self.head - k) % self.m
        if start < end:
            op = self._range_ops(start, end)
            return op(X)
        else:
            op1 = self._range_ops(0, end)
            op2 = self._range_ops(start, self.m)
            op = compose_ops(op1, op2)
            return op(X)


# -------------------- Parameter partitioning --------------------

def params_of(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def split_params_core_and_experts(cell: MoERecurrentCell) -> Tuple[List[nn.Parameter], List[List[nn.Parameter]]]:
    core_modules = [cell.W_q, cell.W_k, cell.W_v, cell.W_w, cell.W_g, cell.policy_head, cell.value_head, cell.router]
    core_params: List[nn.Parameter] = []
    for m in core_modules:
        core_params += params_of(m)
    expert_params: List[List[nn.Parameter]] = [params_of(exp) for exp in cell.experts]
    return core_params, expert_params


# -------------------- RTRL with lazy deferral --------------------

class RTRLMoE:
    def __init__(self, cell: MoERecurrentCell, act_dim: int, lr: float = 3e-4, gamma: float = 0.99, window: int = 64):
        self.cell = cell
        self.act_dim = act_dim
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.cell.parameters(), lr=lr)

        self.params_core, self.params_experts = split_params_core_and_experts(self.cell)
        self.N_core = flatten_params(self.params_core).numel()
        self.N_experts = [flatten_params(pe).numel() for pe in self.params_experts]

        self.S = self.cell.init_state(1)
        KD = self.S.numel()
        self.P_core_mat = torch.zeros(KD, self.N_core)
        self.P_exp_mat = [torch.zeros(KD, n) for n in self.N_experts]
        self.t_last_core = -1
        self.t_last_exp = [-1 for _ in self.N_experts]

        self.fq = FactorQueue(window)
        self.t = 0
        self.baseline = 0.0
        self.b_decay = 0.99

    def reset(self):
        self.S = self.cell.init_state(1)
        KD = self.S.numel()
        self.P_core_mat.zero_()
        self.P_exp_mat = [torch.zeros(KD, n) for n in self.N_experts]
        self.t_last_core = -1
        self.t_last_exp = [-1 for _ in self.N_experts]
        self.fq = FactorQueue(self.fq.m)
        self.t = 0
        self.baseline = 0.0

    def _direct_grad(self, logp: torch.Tensor, params: List[nn.Parameter], size: int) -> torch.Tensor:
        grads = torch.autograd.grad(logp, params, retain_graph=True, allow_unused=True)
        g = torch.zeros(size)
        off = 0
        for p, gi in zip(params, grads):
            n = p.numel()
            if gi is not None:
                g[off: off+n] = gi.reshape(-1).detach()
            off += n
        return g

    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        S_prev = self.S
        def f_state(S_in: torch.Tensor) -> torch.Tensor:
            S_tmp, _ = self.cell.forward_step(S_in, obs, self.t)
            return S_tmp
        S_new, aux = self.cell.forward_step(S_prev, obs, self.t)
        features = torch.cat([aux["context"], obs], dim=-1)
        dist, value = self.cell.policy_value(features, self.act_dim)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)

        J_S_op = js_left_op(f_state, S_prev)

        active_blocks = ["core"]
        active_expert_idx: Optional[int] = None
        if aux["use_moe"]:
            active_expert_idx = aux["expert_idx"].item()
            active_blocks.append(("expert", active_expert_idx))

        if "core" in [b if isinstance(b, str) else b[0] for b in active_blocks]:
            delta = (self.t - 1) - self.t_last_core
            P_prev = self.fq.apply_last(max(0, delta), self.P_core_mat)
            J_theta_core_size = self.N_core
            J_theta_core = torch.zeros(P_prev.size(0), J_theta_core_size)
            KD = S_prev.numel()
            # Build J_theta_core by autograd per-state element (small KD recommended)
            S_flat = S_new.view(-1)
            for i in range(KD):
                grads = torch.autograd.grad(S_flat[i], self.params_core, retain_graph=True, allow_unused=True)
                off = 0
                for p, gi in zip(self.params_core, grads):
                    n = p.numel()
                    if gi is not None:
                        J_theta_core[i, off: off+n] = gi.reshape(-1).detach()
                    off += n
            P_core = J_theta_core + J_S_op(P_prev)
            self.P_core_mat = P_core
            self.t_last_core = self.t

        if active_expert_idx is not None:
            j = active_expert_idx
            delta = (self.t - 1) - self.t_last_exp[j]
            P_prev = self.fq.apply_last(max(0, delta), self.P_exp_mat[j])
            size_e = self.N_experts[j]
            J_theta_e = torch.zeros(P_prev.size(0), size_e)
            KD = S_prev.numel()
            S_flat = S_new.view(-1)
            for i in range(KD):
                grads = torch.autograd.grad(S_flat[i], self.params_experts[j], retain_graph=True, allow_unused=True)
                off = 0
                for p, gi in zip(self.params_experts[j], grads):
                    n = p.numel()
                    if gi is not None:
                        J_theta_e[i, off: off+n] = gi.reshape(-1).detach()
                    off += n
            P_e = J_theta_e + J_S_op(P_prev)
            self.P_exp_mat[j] = P_e
            self.t_last_exp[j] = self.t

        gS = torch.autograd.grad(logp, S_new, retain_graph=True, allow_unused=True)[0]
        gS_flat = torch.zeros(S_new.numel()) if gS is None else gS.view(-1).detach()

        grad_core = None
        if self.t_last_core == self.t:
            gtheta_core_dir = self._direct_grad(logp, self.params_core, self.N_core)
            grad_core = gtheta_core_dir + gS_flat @ self.P_core_mat

        grad_experts: Dict[int, torch.Tensor] = {}
        if active_expert_idx is not None and self.t_last_exp[active_expert_idx] == self.t:
            j = active_expert_idx
            gtheta_e_dir = self._direct_grad(logp, self.params_experts[j], self.N_experts[j])
            grad_experts[j] = gtheta_e_dir + gS_flat @ self.P_exp_mat[j]

        self.S = S_new

        self.fq.push(J_S_op)

        self.t += 1
        info = {"logp": logp, "value": value.detach(), "grad_core": grad_core, "grad_experts": grad_experts}
        return action, info

    def learn(self, traj: List[Dict], rewards: List[float]):
        G = 0.0
        returns: List[float] = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        for Gt in returns:
            self.baseline = self.b_decay * self.baseline + (1 - self.b_decay) * Gt

        g_core_total = torch.zeros(self.N_core)
        g_exp_total = [torch.zeros(n) for n in self.N_experts]

        for step, Gt in zip(traj, returns):
            adv = Gt - self.baseline
            if step["grad_core"] is not None:
                g_core_total += adv * step["grad_core"]
            for j, g in step["grad_experts"].items():
                g_exp_total[j] += adv * g

        self.opt.zero_grad(set_to_none=True)
        core_grads = unflatten_to_grads(g_core_total, self.params_core)
        for p, g in zip(self.params_core, core_grads):
            p.grad = (p.grad if p.grad is not None else torch.zeros_like(p)) + g
        for j, gflat in enumerate(g_exp_total):
            grads_j = unflatten_to_grads(gflat, self.params_experts[j])
            for p, g in zip(self.params_experts[j], grads_j):
                p.grad = (p.grad if p.grad is not None else torch.zeros_like(p)) + g
        self.opt.step()


# -------------------- Tiny smoke test --------------------
if __name__ == "__main__":
    seed_all(0)
    obs_dim = 8
    act_dim = 2
    K, D = 3, 16
    num_experts = 4
    d_hidden = 32
    period = 10

    cell = MoERecurrentCell(obs_dim, K, D, num_experts, d_hidden, moe_period=period)
    agent = RTRLMoE(cell, act_dim, window=32)

    T = 30
    traj = []
    rewards = []
    for t in range(T):
        x = torch.randn(1, obs_dim)
        a, info = agent.step(x)
        r = float(torch.randn(()).clamp(-1, 1))
        traj.append(info)
        rewards.append(r)

    agent.learn(traj, rewards)
    print("MoE RTRL with factor queue ran and updated params.")
