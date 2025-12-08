Here’s your **clean Markdown README** — ready to drop into a repo (e.g., `README.md`) for your RTRL or recurrent model project:

---

# 🌐 Global Stability for Online Recurrent Learning (RTRL-Friendly)

A practical recipe for keeping **long-horizon Jacobian products** stable in streaming / online learning.
Designed for **RTRL**, **unrolled-free training**, and **robotics-style continuous updates**.

---

## 🧭 Goal

Maintain **global stability** — ensure that the **product of Jacobians**
[
J_{t+K-1}\cdots J_t
]
stays well-conditioned over time, avoiding both vanishing and exploding gradients.

---

## ⚙️ 1. Orthogonal Recurrent Core (Isometric Dynamics)

Parameterize the recurrent transformation to be **(nearly) orthogonal**:

[
O_\theta(h) = Q h, \quad Q \in \mathrm{Orthogonal}
]

### Implementation Options

* **Cayley transform:** ( Q = (I + W)(I - W)^{-1}, ; W = -W^\top )
* **Householder product:** ( Q = \prod_i (I - 2 v_i v_i^\top) )
* **Matrix exponential:** ( Q = \exp(W), ; W = -W^\top )

### Benefits

* Gradient norms preserved when the gate is open.
* No truncation or exploding Jacobian chains.
* Fully differentiable and RTRL-compatible.

---

## 🧩 2. Highway Residual Gating (Identity Carry Route)

Define the recurrent update as:

[
h_{t+1} = (1-\alpha_t)h_t + \alpha_t,\phi(O_\theta(h_t) + W_x x_t + b)
]
where the gate (\alpha_t \in [0,1]) can be scalar or per-unit.

### Jacobian

[
J_t = (1-\alpha_t)I + \alpha_t D_t Q, \quad D_t = \mathrm{diag}(\phi'(z_t))
]

### Why it helps

* Keeps the Jacobian spectrum clustered near **1**.
* Preserves gradient flow through identity path.
* Balances between identity (carry) and transform (update).

### Gate Hygiene

* **Init bias** ≈ −2 → (\alpha_t \approx 0.1)
* **Temperature** >1 early (e.g., 2.0 → 1.0 annealing)
* **Entropy / KL regularization** on gate activations to avoid saturation

---

## 📈 3. Global Lyapunov Control (K-Step Stability)

Penalize deviation of the **product of Jacobians** from unit growth.

### Objective

[
\mathcal{R}*{t:t+K} =
\Big(
\tfrac{1}{M}\sum*{m=1}^M
\sum_{k=t}^{t+K-1}
\log | J_k v^{(m)} |
\Big)^2
]

### Algorithm

1. Push a few **probe vectors** through Jacobian–Vector Products (JVPs).
2. **QR re-orthonormalize** at each step (“gradient flossing”).
3. Penalize deviation in log norms.

### Usage

Add ( \lambda \mathcal{R} ) to your task loss every (K) steps — no unrolling, no backward-through-time.

| Hyperparameter | Typical value |
| -------------- | ------------- |
| K              | 16–32         |
| probes         | 2             |
| λ              | 1e−6 – 5e−5   |

---

## 🧮 4. Minimal Gated Orthogonal Highway Cell

```python
import torch, torch.nn as nn, torch.nn.functional as F

def cayley(W):
    I = torch.eye(W.size(-1), device=W.device)
    return torch.linalg.solve(I - W, I + W)

class GatedOrthHighwayCell(nn.Module):
    def __init__(self, H, D, gate_temp=2.0):
        super().__init__()
        A = torch.randn(H, H) * 0.02
        self.A = nn.Parameter(A - A.T)  # skew-symmetric param
        self.Wx = nn.Linear(D, H)
        self.gate = nn.Linear(H + D, H)
        nn.init.constant_(self.gate.bias, -2.0)
        self.gate_temp = gate_temp

    def orth_core(self, h):
        Q = cayley(self.A * 0.25)
        return (Q @ h.T).T

    def step(self, x, h):
        z = self.orth_core(h) + self.Wx(x)
        u = F.silu(z)
        a = torch.sigmoid(self.gate(torch.cat([h, x], -1)) / self.gate_temp)
        return (1 - a) * h + a * u, a
```

**Key properties:**

* Orthogonal hidden dynamics
* Highway residual gating
* Smooth gradient flow over long horizons

---

## 🔁 5. Lyapunov Penalty (Streaming QR Implementation)

Approximate the global Jacobian growth using online JVPs and QR:

```python
from torch import func

@torch.enable_grad()
def lyapunov_penalty(f_step, params, h0, x_window, probes=2, eps=1e-9):
    B, H = h0.shape
    V = torch.linalg.qr(torch.randn(H, probes, device=h0.device)).Q
    logs, h = [], h0
    for x in x_window:
        def f_h(h_): return f_step(params, h_, x)
        Y = []
        for m in range(probes):
            _, Jv = func.jvp(f_h, (h,), (V[:, m].unsqueeze(0).expand(B, -1),))
            Y.append(Jv.mean(0))
        Y = torch.stack(Y, dim=1)
        Q, R = torch.linalg.qr(Y)
        logs.append(torch.log(torch.abs(torch.diag(R)) + eps).sum())
        h = f_step(params, h, x)
        V = Q.detach()
    return (torch.stack(logs).sum())**2
```

**Integration:**

```python
loss = task_loss + λ * lyapunov_penalty(...)
rtrl.step(model, x_t, h_t, loss, ...)
```

---

## ⏱️ 6. Optional: Multi-Timescale State

Split state into slow and fast paths:

[
h = [h^{\text{slow}}, h^{\text{fast}}]
]

Leaky-integrator slow path:
[
h^{\text{slow}}_{t+1} = h^{\text{slow}}_t + \beta_t \odot u^{\text{slow}}_t
]
This extends memory capacity while the fast core handles dynamics.

---

## 🧘 7. Stability Summary

| Mechanism        | Controls                | Implementation                 | Effect               |
| ---------------- | ----------------------- | ------------------------------ | -------------------- |
| Orthogonal core  | Local Jacobian norm     | Cayley / exp / Householder     | Norm-preserving      |
| Highway gate     | Spectrum radius         | (1−α)I + αDQ                   | Eigenvalues near 1   |
| Lyapunov penalty | Global Jacobian product | Online JVP + QR                | Long-term stability  |
| Gate hygiene     | Avoid saturation        | Bias, temperature, entropy reg | Smooth gradient path |
| Slow path        | Memory depth            | Leaky integrator               | Extended timescale   |

---

## ⚖️ 8. Recommended Hyperparameters

| Parameter            | Default         | Description                   |
| -------------------- | --------------- | ----------------------------- |
| **K**                | 16–32           | Lyapunov window length        |
| **probes**           | 2               | Number of probe vectors       |
| **λ**                | 1e−6–5e−5       | Penalty weight                |
| **gate bias**        | −2.0            | Initial carry fraction (~0.1) |
| **gate temperature** | 2.0 → 1.0       | Anneal early                  |
| **Cayley scale**     | 0.25–0.5        | Skew-step stability           |
| **Nonlinearity**     | `tanh` / `SiLU` | Use bounded slope ≤1 early    |

---

## 🧠 9. Why It Works

* **Orthogonality:** keeps local transformations norm-preserving.
* **Highway gating:** anchors the chain Jacobian near the identity.
* **Lyapunov control:** explicitly constrains the *product* of Jacobians.
* **Gate hygiene:** prevents dead or saturated gates.
* **Multi-timescale design:** extends memory without destabilization.

Together, these yield **globally stable**, **gradient-sane**, **online-trainable** recurrent networks.

---

### 💡 TL;DR

> Orthogonal core + Highway gate + K-step Lyapunov regularizer =
> **Stable long-horizon recurrence** without truncation — ready for RTRL and streaming tasks.

---

Would you like me to include a short **“Quick Start Example”** section (e.g., RTRL training loop snippet + sample config) at the end of the README?
