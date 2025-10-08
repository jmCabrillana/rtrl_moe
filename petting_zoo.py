"""
RTRL training of a tiny "recurrent transformer" policy on a PettingZoo multi-agent environment.

Notes
-----
- Environment: PettingZoo MPE `simple_spread_v3` (continuous actions, partial observability).
- Model: A small recurrent attention cell that maintains a compact memory state S (K slots x D).
  This is *transformer-flavored* (single-head content-addressed read + gated write), but kept tiny so
  exact RTRL is tractable.
- Optimization: Online REINFORCE with exact RTRL to propagate gradients through the recurrent state
  *without* backpropagation through time.
- This is a **didactic** reference implementation; keep K and D small (e.g., K<=4, D<=16), and the
  parameter count modest. RTRL has O(|θ|·|state| + |state|^2) per step costs here.

Requirements
------------
pip install pettingzoo==1.* supersuit==3.* numpy torch==2.*

Caveats
-------
- Exact RTRL on attention-like cells is heavy. We design the cell so Jacobians are cheap-ish.
- This code aims for clarity over speed. For research-scale runs, consider UORO/NoBackTrack approximations.
- Tested with small hyperparams on CPU.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --- PettingZoo env (Multi-Agent Particle Env: simple_spread) ---
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss
import imageio

# --------------- Utilities ---------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(params)


def unflatten_to_grads(flat: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    offset = 0
    for p in params:
        numel = p.numel()
        grads.append(flat[offset: offset + numel].view_as(p).clone())
        offset += numel
    return grads


# --------------- Recurrent Transformer Cell ---------------
class RTCell(nn.Module):
    """
    Tiny recurrent transformer-like cell with:
      - State S \in R^{K x D}
      - Read: attention over S using query from observation x
      - Write: gated update to a single slot (ring buffer) with a write vector from x

    State update S_t = F(S_{t-1}, x_t; θ) is differentiable.
    Output y_t = G(S_t, x_t; θ).

    We keep it tiny to make exact RTRL feasible.
    """
    def __init__(self, obs_dim: int, K: int = 4, D: int = 16):
        super().__init__()
        self.K = K
        self.D = D
        # Projections
        self.W_q = nn.Linear(obs_dim, D, bias=True)
        self.W_k = nn.Linear(D, D, bias=False)  # keys from slots
        self.W_v = nn.Linear(D, D, bias=False)  # values from slots
        self.W_w = nn.Linear(obs_dim, D, bias=True)  # write vector from obs
        self.W_g = nn.Linear(obs_dim + D, 1, bias=True)  # gate from obs+context
        # Policy & value heads (actor-critic flavor, but we only use baseline from value)
        self.policy_head = nn.Linear(D + obs_dim, 2)  # outputs mean, log_std for each action dim (we'll tile)
        self.value_head = nn.Linear(D + obs_dim, 1)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        # State shape: (batch, K, D)
        return torch.zeros(batch_size, self.K, self.D, requires_grad=True)

    def forward_step(self, S_prev: torch.Tensor, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        One time-step state transition and outputs.
        Inputs:
          S_prev: (1, K, D) requires_grad=True
          x: (1, obs_dim)
          t: int, time index (for write slot selection)
        Returns:
          S_new: (1, K, D) requires_grad=True
          aux: dict with 'context', 'features', etc.
        """
        Kslots = S_prev  # (1, K, D)
        # Read
        q = self.W_q(x)  # (1, D)
        keys = self.W_k(Kslots)  # (1, K, D)
        vals = self.W_v(Kslots)  # (1, K, D)
        attn_logits = (keys @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.D)  # (1, K)
        attn = F.softmax(attn_logits, dim=-1)  # (1, K)
        context = (attn.unsqueeze(-1) * vals).sum(dim=1, keepdim=False)  # (1, D)

        # Write
        write_vec = self.W_w(x)  # (1, D)
        gate = torch.sigmoid(self.W_g(torch.cat([x, context], dim=-1)))  # (1, 1)
        pos = t % self.K
        S_new = S_prev.clone()
        S_new = S_new.requires_grad_(True)
        # Update a single slot with gated interpolation
        S_new[:, pos, :] = (1.0 - gate) * S_prev[:, pos, :] + gate * write_vec

        features = torch.cat([context, x], dim=-1)  # (1, D + obs_dim)
        return S_new, {"context": context, "features": features}

    def policy_value(self, features: torch.Tensor, act_dim: int) -> Tuple[Normal, torch.Tensor]:
        # We share head for each action dim by broadcasting mean/log_std
        out = self.policy_head(features)  # (1, 2)
        mean, log_std = out[:, :1], out[:, 1:2]
        mean = mean.repeat(1, act_dim)
        log_std = torch.clamp(log_std, -4, 1).repeat(1, act_dim)
        dist = Normal(mean, torch.exp(log_std))
        value = self.value_head(features)  # (1, 1)
        return dist, value.squeeze(-1)


# --------------- Exact RTRL helper ---------------
@dataclass
class RTRLState:
    S: torch.Tensor            # (1, K, D), requires_grad=True
    P: torch.Tensor            # dS/dθ, shape (K*D, Np)


def jacobian_state_wrt_params(S_new: torch.Tensor, params: List[torch.nn.Parameter]) -> torch.Tensor:
    """J_theta: ∂vec(S_new)/∂θ -> (K*D, Np)"""
    flat_params = flatten_params(params)
    Np = flat_params.numel()
    KD = S_new.numel()
    J = torch.zeros(KD, Np)
    # Loop over each state element (small KD only!)
    S_new_flat = S_new.view(-1)
    for i in range(KD):
        grads = torch.autograd.grad(S_new_flat[i], params, retain_graph=True, allow_unused=True)
        if grads is None:
            continue
        gflat = torch.zeros(Np)
        offset = 0
        for p, g in zip(params, grads):
            n = p.numel()
            if g is not None:
                gflat[offset: offset + n] = g.reshape(-1).detach()
            offset += n
        J[i] = gflat
    return J


def jacobian_state_wrt_state(S_new: torch.Tensor, S_prev: torch.Tensor) -> torch.Tensor:
    """J_s: ∂vec(S_new)/∂vec(S_prev) -> (KD, KD)"""
    KD = S_new.numel()
    J = torch.zeros(KD, KD)
    S_new_flat = S_new.view(-1)
    eye = torch.eye(KD)
    for i in range(KD):
        grads = torch.autograd.grad(S_new_flat[i], S_prev, retain_graph=True, grad_outputs=torch.ones_like(S_new_flat[i]))
        # grads is a tuple with same shape as S_prev
        g = grads[0].view(-1).detach()
        J[i] = g
    return J


def total_grad_logprob(
    logp: torch.Tensor,
    S_t: torch.Tensor,
    params: List[torch.nn.Parameter],
    P_t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute d/dθ logπ(a_t|x_t, S_t; θ) = ∂logπ/∂θ + (∂logπ/∂S_t)·(dS_t/dθ)
    Returns flat gradient (Np,)
    """
    flat_params = flatten_params(params)
    Np = flat_params.numel()
    # Partial w.r.t params (treating S_t as constant):
    grads_theta = torch.autograd.grad(logp, params, retain_graph=True, allow_unused=True)
    gtheta = torch.zeros(Np)
    offset = 0
    for p, g in zip(params, grads_theta):
        n = p.numel()
        if g is not None:
            gtheta[offset: offset + n] = g.reshape(-1).detach()
        offset += n
    # Partial w.r.t state:
    glist = torch.autograd.grad(logp, S_t, retain_graph=True, allow_unused=True)
    glist = [g if g is not None else torch.zeros_like(S_t) for g in glist]
    gS = glist[0].view(-1).detach()  # (KD,)
    # Chain rule term:
    chain = gS @ P_t  # (Np,)
    return gtheta + chain


# --------------- Agent using RTRL ---------------
class RTRLAgent:
    def __init__(self, obs_dim: int, act_dim: int, K: int = 4, D: int = 16, lr: float = 3e-4, gamma: float = 0.99):
        self.core = RTCell(obs_dim, K, D)
        self.params = [p for p in self.core.parameters()]
        self.Np = flatten_params(self.params).numel()
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        # Baseline EMA
        self.baseline = 0.0
        self.b_decay = 0.99

        # Optimizer (we'll apply manual grads each update)
        self.optimizer = torch.optim.Adam(self.core.parameters(), lr=lr)

        # RTRL state
        self.rtrl = RTRLState(S=self.core.init_state(1), P=torch.zeros(self.core.K * self.core.D, self.Np))
        self.t = 0

    def reset(self):
        self.rtrl = RTRLState(S=self.core.init_state(1), P=torch.zeros(self.core.K * self.core.D, self.Np))
        self.t = 0

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict]:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        S_prev = self.rtrl.S
        S_new, aux = self.core.forward_step(S_prev, x, self.t)
        features = aux["features"]
        dist, value = self.core.policy_value(features, self.act_dim)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)  # (1,)

        # --- RTRL Update: propagate P_t = dS_t/dθ ---
        # Compute Jacobians for state transition
        J_theta = jacobian_state_wrt_params(S_new, self.params)  # (KD, Np)
        J_s = jacobian_state_wrt_state(S_new, S_prev)            # (KD, KD)
        P_new = J_theta + J_s @ self.rtrl.P

        # Save step data for learning
        step = {
            "x": x,
            "S": S_new,
            "dist": dist,
            "action": action.detach(),
            "logp": logp,
            "value": value.detach(),
            "P": P_new.clone(),
        }
        # Commit new state
        self.rtrl = RTRLState(S=S_new, P=P_new)
        self.t += 1
        return action.squeeze(0).numpy(), step

    def learn(self, trajectory: List[Dict], rewards: List[float]):
        # Compute discounted returns
        G = 0.0
        returns: List[float] = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()

        # Update baseline (EMA)
        for Gt in returns:
            self.baseline = self.b_decay * self.baseline + (1 - self.b_decay) * Gt

        # Accumulate policy gradients online
        g_total = torch.zeros(self.Np)
        for step, Gt in zip(trajectory, returns):
            adv = Gt - self.baseline
            # Recompute distribution & logp at this step (detach previous graph except what we need)
            x = step["x"]
            S = step["S"]
            features = torch.cat([self.core.W_v(self.rtrl.S).mean(dim=1), x], dim=-1)  # rough recompute; not used
            dist: Normal = step["dist"]  # reuse stored dist
            action = step["action"]
            logp = dist.log_prob(action).sum(dim=-1)

            # Total derivative via RTRL
            g_logp = total_grad_logprob(logp, S, self.params, step["P"])  # (Np,)
            g_total += (adv * g_logp)

        # Apply gradients manually
        grads = unflatten_to_grads(g_total, self.params)
        # Zero then set .grad
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.params, grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
        self.optimizer.step()


# --------------- Training loop (PettingZoo simple_spread) ---------------

def make_env(seed=0):
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=50, continuous_actions=True, render_mode="rgb_array")
    env = ss.black_death_v3(env)  # agents that are done return zeros
    env.reset(seed=seed)
    return env


def train(num_episodes: int = 50, seed: int = 0):
    seed_everything(seed)
    env = make_env(seed)
    agents = env.possible_agents
    # Infer obs/act dims from one reset
    obs, infos = env.reset(seed=seed)
    obs_dims = {a: obs[a].shape[0] for a in agents}
    act_dims = {a: env.action_space(a).shape[0] for a in agents}

    # One identical agent policy per entity (independent learners for simplicity)
    policies: Dict[str, RTRLAgent] = {
        a: RTRLAgent(obs_dims[a], act_dims[a], K=3, D=12, lr=3e-4, gamma=0.97) for a in agents
    }

    for ep in range(1, num_episodes + 1):
        obs, infos = env.reset(seed=seed + ep)
        for a in agents:
            policies[a].reset()
        done = {a: False for a in agents}
        ep_rewards = {a: 0.0 for a in agents}
        # Trajectories per agent
        traj: Dict[str, List[Dict]] = {a: [] for a in agents}
        rews: Dict[str, List[float]] = {a: [] for a in agents}

        t = 0
        frames = []
        while True:
            actions = {}
            step_data = {}
            for a in agents:
                if not done[a]:
                    action, data = policies[a].act(obs[a])
                    actions[a] = action
                    step_data[a] = data
                else:
                    actions[a] = np.zeros(act_dims[a], dtype=np.float32)
            frame = env.render()
            frames.append(frame)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for a in agents:
                if not done[a]:
                    traj[a].append(step_data[a])
                    rews[a].append(float(rewards[a]))
                    ep_rewards[a] += float(rewards[a])
                done[a] = terminations[a] or truncations[a]

            obs = next_obs
            t += 1
            if all(done.values()):
                break
        imageio.mimsave("episode.gif", frames, fps=30)

        # Learn per agent
        for a in agents:
            if len(traj[a]) > 0:
                policies[a].learn(traj[a], rews[a])

        print(f"Episode {ep:03d} | Rewards: " + ", ".join(f"{a}:{ep_rewards[a]:+.2f}" for a in agents))

    env.close()


if __name__ == "__main__":
    # Quick smoke run (short, toy settings)
    train(num_episodes=10, seed=42)
