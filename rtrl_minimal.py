import torch
import torch.nn as nn
from torch.func import jacrev, vmap, functional_call
from einops import rearrange

# per-point function to batch jacobian later
def make_f_single(model, proj=slice(None)):
    def f(params, h, x, kw):
        x1, h1 = rearrange(x, '... -> 1 ...'), rearrange(h, '... -> 1 ...')
        *_, h_next_b = functional_call(model, params, (x1, h1), kw)
        return rearrange(h_next_b, '1 h -> h')[proj] # remove fake batch, slice
    return f

def add_grad_(p, g):
    if g is None: return
    g = g.detach()
    if p.grad is None:
        p.grad = g.clone()
    else:
        p.grad.add_(g)

class RTRL:
    @torch.no_grad()
    def __init__(self, state_parameters, B, H):
      self.state_parameters = state_parameters
      T = sum(p.numel() for p in self.state_parameters.values())
      self.P_t = torch.zeros([B, H, T])

    def step(self, model, x_t, h_t, loss, params, **kw):
        """
        x_t: [B,...], h_t: [B,H], P_t: [B,H,T], dL_dH_t: [B,H]
        """
        dL_dTheta = dict(zip(list(params.keys()), torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)))
        (dL_dH_t,) = torch.autograd.grad(loss, h_t, retain_graph=True)
        f1 = make_f_single(model)

        # flatten params -> vector (for splitting grads back)
        specs, flats = [], []
        for k, p in params.items():
            specs.append((k, p.shape, p.numel()))
            flats.append(p.flatten())

        # batched jacobian of per-sample f
        Jh = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(self.state_parameters, h_t, x_t, kw)  # [B,H,H]
        Jp_tree = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(self.state_parameters, h_t, x_t, kw) # [B,H,[...]]
        Jtheta = torch.cat([rearrange(v, 'b h ... -> b h (...)') for v in Jp_tree.values()], dim=2)  # [B,H,T]

        # RTRL recursion
        self.P_t = torch.bmm(Jh, self.P_t) + Jtheta  # [B,H,T]

        # per-sample param grads and batch-mean
        g_mean = torch.einsum('b h t, b h -> b t', self.P_t, dL_dH_t).mean(0) 

        # write gradients back into model params
        for k, g in dL_dTheta.items():
          add_grad_(params[k], g)
        i = 0
        for k, p in self.state_parameters.items():
            add_grad_(params[k], g_mean[i:i+p.numel()].view(p.shape))
            i += p.numel()

