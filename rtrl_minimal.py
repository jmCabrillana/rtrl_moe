import torch
from torch.func import jacrev, vmap, functional_call
from einops import rearrange

# per-example function for jacobians (adds/squeezes a batch dim)
def make_f_single(model):
    def f(params, h, x, **kw):
        x1, h1 = rearrange(x, '... -> 1 ...'), rearrange(h, '... -> 1 ...')
        *_, h_next_b = functional_call(model, params, (x1, h1), kw)
        return rearrange(h_next_b[-1], '1 h -> h')   # remove fake batch
    return f

@torch.no_grad()
def rtrl_step(model, params: dict, x_t, h_prev, P_prev, L, **kw):
    """
    x_t: [B,...], h_prev: [B,H], P_prev: [B,H,T], dL_dh_t: [B,H]
    """
    (dL_dh_t,) = torch.autograd.grad(L.mean(), h_next, retain_graph=True)
    f1 = make_f_single(model)

    # flatten params -> vector (for splitting grads back)
    specs, flats = [], []
    for k, p in params.items():
        specs.append((k, p.shape, p.numel()))
        flats.append(p.flatten())
    T = torch.cat(flats).numel()

    # batched jacobian of per-sample f
    Jh = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(params, h_prev, x_t, kw)  # [B,H,H]
    Jp_tree = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(params, h_prev, x_t, kw) # [B,H,[...]]
    Jtheta = torch.cat([rearrange(v, 'b h ... -> b h (...)') for v in Jp_tree.values()], dim=2)  # [B,H,T]

    # RTRL recursion
    P_t = torch.bmm(Jh, P_prev) + Jtheta  # [B,H,T]

    # per-sample param grads and batch-mean
    g_b = torch.einsum('b h t, b h -> b t', P_t, dL_dh_t)
    g_mean = g_b.mean(0)                                                

    # write gradients back into model params
    i = 0
    for k, shape, n in specs:
        params[k].grad = g_mean[i:i+n].view(shape).detach().clone()
        i += n

    return P_t

#test
params = {k:v for k,v in model.named_parameters()}
with torch.no_grad():
    B, H = h0.shape
    T = sum(p.numel() for p in params.values())
    P = h0.new_zeros(B, H, T)
    h = h0

# training loop over time
for t in range(T_steps):
    x_t = x_seq[:, t]                 # [B, ...]
    # forward to get loss and dL/dh_t
    *_, h_next = f(x_t, h_prev)
    loss_t = criterion(h_next, y_seq[:, t])   # your per-step loss
    
    h, P, flat_g = rtrl_step(model, params, x_t, h, P, dL_dh_t)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
