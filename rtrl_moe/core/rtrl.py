# Block RTRL

import torch
import torch.nn as nn
from torch.func import jacrev, vmap, functional_call
from einops import rearrange


def make_f_single(model, write=slice(None)):
    def f(params, h, x, kw):
        x1, h1 = rearrange(x, '... -> 1 ...'), rearrange(h, '... -> 1 ...')
        *_, h_next_b = functional_call(model, params, (x1, h1), kw)
        return rearrange(h_next_b, '1 h -> h')[write] # remove fake batch, slice
    return f

def add_grad_(p, g):
    if g is None: return
    g = g.detach()
    if p.grad is None:
        p.grad = g.clone()
    else:
        p.grad.add_(g)

class BlockRTRL:
    @torch.no_grad()
    def __init__(self, state_params, B, H):
        self.state_params = state_params
        self.P_t = {k:torch.zeros([B, H, p.numel()]).to(p) for k,p in self.state_params.items()}

    @torch.no_grad()
    def reset(self):
        [P_t.zero_() for P_t in self.P_t.values()]

    @torch.no_grad()
    def step(self, model, x_t, h_t, loss, active_params, read=None, write=None, **kw):
        """
        x_t: [B,...], h_t: [B,H], P_t: [B,H,Tp], dL_dH_t: [B,H]
        """
        params = dict(model.named_parameters())
        B, H = h_t.shape[:2]
        read = list(range(H)) if read is None else read
        write = list(range(H)) if write is None else write
        f1 = make_f_single(model, write)

        if loss is not None:            
            # add direct term
            dL_dTheta = dict(zip(params.keys(), torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)))
            for k, g in dL_dTheta.items():
                # if k not in self.state_params.keys():
                add_grad_(params[k], g)

            # add RTRL term using *P_t* 
            (dL_dH_t,) = torch.autograd.grad(loss, h_t, retain_graph=False) 
            for k, p in self.state_params.items():
                g_mean = torch.einsum('b h, b h t -> b t', dL_dH_t, self.P_t[k]).mean(0) 
                add_grad_(params[k], g_mean.view(p.shape))

        # batched jacobian of per-sample f
        with torch.enable_grad():
            Jh_proj = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw)  # [B,W,H]
            # Jh = torch.eye(H).repeat(B, 1, 1).to(h_t), Jh[:, proj] = Jh_proj
            Jtheta_proj = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw) # [B,W,[...]]
            Jtheta_proj = {k:rearrange(v, 'b h ... -> b h (...)') for k, v in Jtheta_proj.items()}  # [B,W,[Tp]]
            # >>> Detach before storing <<<
            Jh_proj = Jh_proj.detach()
            Jtheta_proj = {k: v.detach() for k, v in Jtheta_proj.items()}
        
        # Track Jacobian norm for diagnostics
        self.last_jacobian_norm = torch.linalg.vector_norm(Jh_proj).item()

        # RTRL recursion on active or expiring sensitivities
        for k in self.state_params.keys():
            self.P_t[k][:, write] = Jh_proj[:,:,read] @ self.P_t[k][:, read]
            # Active parameters update have non-zero J_theta
            if k in active_params.keys():
                self.P_t[k][:, write] += Jtheta_proj[k]
            
            # Guard against NaNs
            self.P_t[k] = torch.nan_to_num(self.P_t[k], nan=0.0, posinf=0.0, neginf=0.0)        
