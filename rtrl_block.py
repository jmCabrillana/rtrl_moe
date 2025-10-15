import torch
from structure import CircularTree, sparse_left_mul

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

class BlockRTRL:
    @torch.no_grad()
    def __init__(self, state_params, B, H, len_buffer=64):
        self.state_params = state_params
        self.P_t = {k:torch.zeros([B, H, p.numel()]).to(p) for k,p in self.state_params.items()}
        self.last_update = {k:0 for k in self.state_params.keys()}
        self.len_buffer = len_buffer
        self.buffer = CircularMatTree(len_buffer, torch.eye(H).expand(B, -1, -1).to(list(self.state_params.values())[0]))
        self.last_hk = [0]
        self.t = 0

    def reset(self):
        [P_t.zero_() for P_t in self.P_t.values()]

    def get_left_product(self, k, l=None):
        if l is None: l = len(self.buffer)
        return self.buffer.query(k, l)
      #   if l is None: l = len(self.buffer)
      #   L = torch.eye(H).expand(B, -1, -1)
      #   for i in range(k, l):
      #     L = torch.bmm(self.buffer[i], L)
      #   return L

    def step(self, model, x_t, h_t, loss, active_params, proj=slice(None), **kw):
        """
        x_t: [B,...], h_t: [B,H], P_t: [B,H,Tp], dL_dH_t: [B,H]
        """
        params = dict(model.named_parameters())
        B, H = h_t.shape[:2]
        f1 = make_f_single(model, proj)

        # batched jacobian of per-sample f
        Jh = torch.eye(H).repeat(B, 1, 1).to(h_t)
        Jh_proj = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw)  # [B,H,H]
        Jh[:, proj] = Jh_proj
        Jtheta_proj = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw) # [B,H,[...]]
        Jtheta_proj = {k:rearrange(v, 'b h ... -> b h (...)') for k, v in Jtheta_proj.items()}  # [B,H,[Tp]]

        # Update circular buffer
        self.buffer.update(Jh)
        # self.buffer.append(Jh); if len(self.buffer) > self.len_buffer: self.buffer.pop(0)

        # RTRL recursion on active or expiring sensitivities
        for k in self.state_params.keys():
            # Active parameters update
            if k in active_params.keys():
                # Jtheta = torch.zeros([B, H, self.state_params[k].numel()]); Jtheta[:, proj] = Jtheta_tree[k]
                # t <-> index [q-1]; product for time s < t starts at s+1, ie index [s - t + (q-1) + 1] = [s - t + q]
                if self.last_update[k] < self.t-1:
                  L_Jh = self.get_left_product(self.last_update[k] - self.t + self.len_buffer, self.len_buffer-1)
                  self.P_t[k] = torch.bmm(L_Jh, self.P_t[k])  # update to P_t-1 first
                self.P_t[k][:, proj] = Jtheta_proj[k] + torch.bmm(Jh_proj, self.P_t[k]) # Optimize to update only on projected lines
                self.last_update[k] = self.t
            # Expiring parameters update
            L_Jh = self.get_left_product(0)
            if k not in active_params.keys() and self.last_update[k] <= self.t - self.len_buffer: # == in practice
                self.P_t[k] = torch.bmm(L_Jh, self.P_t[k])
                self.last_update[k] = self.t


        # Backpropagate, Maintain last activated
        if proj != slice(None):
            I = list(set(self.last_hk) & set(proj))
            self.last_hk = [k for k in self.last_hk + proj if k not in I] + I
        if loss is not None:
            dL_dTheta = dict(zip(params.keys(), torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)))
            (dL_dH_t,) = torch.autograd.grad(loss, h_t, retain_graph=True) #[:,self.last_hk]
            for k, g in dL_dTheta.items():
                add_grad_(params[k], g)
            for k, p in self.state_params.items():
                g_mean = torch.einsum('b h, b h t -> b t', dL_dH_t, self.P_t[k]).mean(0)
                add_grad_(params[k], g_mean.view(p.shape))

        self.t += 1
