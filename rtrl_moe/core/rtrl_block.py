# Block RTRL
import torch
from torch.func import jacrev, vmap, functional_call
from einops import rearrange
from .circular_seg_tree import CircularTree, sparse_left_mm

# per-point function to batch jacobian later
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
    def __init__(self, state_params, B, H, len_buffer=64, len_buffer_hk=8):
        self.state_params = state_params
        self.P_t = {k:torch.zeros([B, H, p.numel()]).to(p) for k,p in self.state_params.items()}
        self.last_update = {k:0 for k in self.state_params.keys()}
        self.len_buffer = len_buffer
        self.len_buffer_hk = len_buffer_hk
        self.buffer = CircularTree(len_buffer, None, sparse_left_mm)
        self.last_hk = [0]
        self.t = 0
        self.last_jacobian_norm = 0.0

    def reset(self):
        [P_t.zero_() for P_t in self.P_t.values()]
        self.t = 0
        for k in self.last_update.keys():
            self.last_update[k] = 0

    @torch.no_grad()
    def get_left_product(self, k, l):
        return self.buffer.query(k, l)

    @torch.no_grad()
    def step(self, model, x_t, h_t, loss, active_params, read=None, write=None, **kw):
        """
        x_t: [B,...], h_t: [B,H], P_t: [B,H,Tp], dL_dH_t: [B,H]
        """
        params = dict(model.named_parameters())
        B, H = h_t.shape[:2]
        if B != 1:
            raise ValueError(f"BlockRTRL currently assumes batch size 1; got B={B}.")
        read = list(range(H)) if read is None else read
        write = list(range(H)) if write is None else write
        f1 = make_f_single(model, write)

        # Detect NaNs in inputs before any Jacobian work
        if torch.isnan(h_t).any():
            print(f"[RTRL] NaN in h_t at step {self.t}")
        if torch.isnan(x_t).any():
            print(f"[RTRL] NaN in x_t at step {self.t}")


        # batched jacobian of per-sample f
        with torch.enable_grad():
            Jh_proj = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw)  # [B,W,H]
            # Jh = torch.eye(H).repeat(B, 1, 1).to(h_t), Jh[:, proj] = Jh_proj
            Jtheta_proj = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw) # [B,W,[...]]
            Jtheta_proj = {k:rearrange(v, 'b h ... -> b h (...)') for k, v in Jtheta_proj.items()}  # [B,W,[Tp]]
            # >>> Detach before storing <<<
            Jh_proj = Jh_proj.detach()
            Jtheta_proj = {k: v.detach() for k, v in Jtheta_proj.items()}

        # Quick NaN diagnostics to narrow source of divergence
        if torch.isnan(Jh_proj).any():
            print(f"[RTRL] NaN in Jh_proj at step {self.t}")
        for k, v in Jtheta_proj.items():
            if torch.isnan(v).any():
                print(f"[RTRL] NaN in Jtheta_proj[{k}] at step {self.t}")

        # Update circular buffer with sparse Jacobian
        Jh_sub = Jh_proj[0][:, read]
        write_tensor = torch.tensor(write, device=Jh_proj.device, dtype=torch.long)
        read_tensor = torch.tensor(read, device=Jh_proj.device, dtype=torch.long)
        indices = torch.cartesian_prod(write_tensor, read_tensor).T
        values = Jh_sub.flatten()

        # Inject identity rows for hidden coordinates that were not written this step
        # so their sensitivities keep behaving like h_{t+1,i} = h_{t,i}
        if len(write) < H:
            inactive = torch.tensor(sorted(set(range(H)) - set(write)), device=Jh_proj.device, dtype=torch.long)
            if inactive.numel() > 0:
                diag_idx = torch.stack([inactive, inactive], dim=0)
                diag_vals = torch.ones(inactive.numel(), device=Jh_proj.device, dtype=values.dtype)
                indices = torch.cat([indices, diag_idx], dim=1)
                values = torch.cat([values, diag_vals], dim=0)

        sparse_jac_batch = torch.sparse_coo_tensor(indices, values, (H, H), device=Jh_proj.device).coalesce()
        self.buffer.update(sparse_jac_batch)

        # Track Frobenius norm (via sparse values) for logging diagnostics
        jac_values = sparse_jac_batch.values()
        self.last_jacobian_norm = torch.linalg.vector_norm(jac_values).item() if jac_values.numel() else 0.0


        # RTRL recursion on active or expiring sensitivities
        for k in self.state_params.keys():
            # Active parameters update
            if k in active_params.keys():
                # Use segment tree [last_update[k], t] for lazy update if parameter was inactive for a while
                if self.t - self.last_update[k] > 0: # always true unless first step
                    start_logical = self.len_buffer - self.t - self.last_update[k] - 2  
                    end_logical = self.len_buffer   # including current step
                    sparse_product = self.get_left_product(start_logical, end_logical)
                    sparse_coalesced = sparse_product.coalesce()
                    rows, cols = sparse_coalesced.indices()
                    vals = sparse_coalesced.values()
                    mask = (rows != cols) | (vals != 1)
                    if mask.any():
                        rows_unique = rows[mask].unique()
                        cols_unique = cols[mask].unique()
                        sub_dense = sparse_product.to_dense()[rows_unique][:, cols_unique]
                        self.P_t[k][:, rows_unique] = sub_dense @ self.P_t[k][:, cols_unique]
                
                # Current step update
                # self.P_t[k][:, write] = Jh_proj[:,:,read] @ self.P_t[k][:, read] # moved to segment_tree
                self.P_t[k][:, write] += Jtheta_proj[k]
                self.last_update[k] = self.t
            
            # Expiring parameters: haven't been updated for >= len_buffer steps
            elif self.t - self.last_update[k] >= self.len_buffer:
                # Apply full lazy update from entire buffer before it expires
                sparse_product = self.get_left_product(0, self.len_buffer)
                sparse_coalesced = sparse_product.coalesce()
                rows, cols = sparse_coalesced.indices()
                vals = sparse_coalesced.values()
                mask = (rows != cols) | (vals != 1)
                if mask.any():
                    rows_unique = rows[mask].unique()
                    cols_unique = cols[mask].unique()
                    sub_dense = sparse_product.to_dense()[rows_unique][:, cols_unique]
                    self.P_t[k][:, rows_unique] = sub_dense @ self.P_t[k][:, cols_unique]
                self.last_update[k] = self.t
            
            # Inactive parameters (not active, not expiring yet): skip this step
            # They will be lazily updated when they become active or when expiring


        # Backpropagate, Maintain last activated
        if loss is not None:
            dL_dTheta = dict(zip(params.keys(), torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)))
            (dL_dH_t,) = torch.autograd.grad(loss, h_t, retain_graph=False) #[:,self.last_hk]
            for k, g in dL_dTheta.items():
                # if k not in self.state_params.keys():
                add_grad_(params[k], g)
            for k, p in self.state_params.items():
                g_mean = torch.einsum('b h, b h t -> b t', dL_dH_t, self.P_t[k]).mean(0) #[self.last_hk]
                add_grad_(params[k], g_mean.view(p.shape))
        if len(write) != H:
            I = set(self.last_hk) & set(write)
            self.last_hk = ([k for k in self.last_hk + write if k not in I] + list(I))
            self.last_hk = self.last_hk[-self.len_buffer_hk:]

        self.t += 1
