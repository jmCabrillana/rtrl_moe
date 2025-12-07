# Block RTRL
import torch
import time
from torch.func import jacrev, vmap, functional_call
from einops import rearrange
from circular_seg_tree import CircularTree, sparse_left_mm

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
        # self.buffer = CircularMatTree(len_buffer, torch.eye(H).expand(B, -1, -1).to(list(self.state_params.values())[0]))
        self.buffer = CircularTree(len_buffer, None, sparse_left_mm)
        self.last_hk = [0]
        self.t = 0

    def reset(self):
        [P_t.zero_() for P_t in self.P_t.values()]
        self.t = 0
        for k in self.last_update.keys():
            self.last_update[k] = 0

    @torch.no_grad()
    def get_left_product(self, k, l):
        return self.buffer.query(k, l)
      #   if l is None: l = len(self.buffer)
      #   L = torch.eye(H).expand(B, -1, -1)
      #   for i in range(k, l):
      #     L = torch.bmm(self.buffer[i], L)
      #   return L

    @torch.no_grad()
    def step(self, model, x_t, h_t, loss, active_params, read=None, write=None, **kw):
        """
        x_t: [B,...], h_t: [B,H], P_t: [B,H,Tp], dL_dH_t: [B,H]
        """
        t0 = time.time()
        params = dict(model.named_parameters())
        B, H = h_t.shape[:2]
        read = list(range(H)) if read is None else read
        write = list(range(H)) if write is None else write
        f1 = make_f_single(model, write)

        print_time = False

        # batched jacobian of per-sample f
        with torch.enable_grad():
            if print_time: t1 = time.time(); print("0: ", t1-t0); t0=t1
            Jh_proj = vmap(jacrev(f1, argnums=1), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw)  # [B,W,H]
            # Jh = torch.eye(H).repeat(B, 1, 1).to(h_t), Jh[:, proj] = Jh_proj
            if print_time: t1 = time.time(); print("1: ", t1-t0)
            Jtheta_proj = vmap(jacrev(f1, argnums=0), in_dims=(None, 0, 0, None))(active_params, h_t, x_t, kw) # [B,W,[...]]
            Jtheta_proj = {k:rearrange(v, 'b h ... -> b h (...)') for k, v in Jtheta_proj.items()}  # [B,W,[Tp]]
            # >>> Detach before storing <<<
            Jh_proj = Jh_proj.detach()
            Jtheta_proj = {k: v.detach() for k, v in Jtheta_proj.items()}

        # Update circular buffer with sparse Jacobian
        if print_time: t1 = time.time(); print("2: ", t1-t0); t0=t1
        # Convert dense Jh_proj [B, len(write), len(read)] to sparse format
        # Create sparse COO tensor for each batch
        sparse_jac_list = []
        for b in range(B):
            # Build sparse COO format: rows from write_idx, cols from read_idx
            # Jh_proj[b] is [len(write), H], need to extract columns in read indices
            # Extract the sub-matrix [len(write), len(read)]
            Jh_sub = Jh_proj[b][:, read]  # [len(write), len(read)]
            
            num_write = len(write)
            num_read = len(read)
            
            # Create meshgrid of indices
            row_indices = torch.tensor(write, dtype=torch.long, device=Jh_proj.device).unsqueeze(1).repeat(1, num_read)
            col_indices = torch.tensor(read, dtype=torch.long, device=Jh_proj.device).unsqueeze(0).repeat(num_write, 1)
            
            # Flatten indices
            indices = torch.stack([row_indices.flatten(), col_indices.flatten()])
            
            # Flatten values
            values = Jh_sub.flatten()
            
            # Create sparse COO tensor [H, H]
            sparse_jac = torch.sparse_coo_tensor(
                indices, values, (H, H), device=Jh_proj.device
            ).coalesce()
            sparse_jac_list.append(sparse_jac)
        
        # Stack sparse tensors for batch (or just use first if B=1)
        sparse_jac_batch = sparse_jac_list[0] if B == 1 else torch.stack(sparse_jac_list)
        
        self.buffer.update(sparse_jac_batch)
        if print_time: t1 = time.time(); print("3: ", t1-t0); t0=t1

        # RTRL recursion on active or expiring sensitivities
        for k in self.state_params.keys():
            # Active parameters update
            if k in active_params.keys():
                # Use segment tree for lazy update if parameter was inactive for a while
                if self.t - self.last_update[k] > 1:
                    # Lazy update: apply product of Jacobians from last_update to now
                    start_idx = self.last_update[k] - self.t + self.len_buffer
                    end_idx = self.len_buffer - 1  # up to previous step
                    if start_idx >= 0 and end_idx > start_idx:
                        sparse_product = self.get_left_product(start_idx, end_idx)
                        if sparse_product is not None:
                            # sparse_product is a sparse torch tensor [H, H]
                            # Extract non-zero rows and columns
                            indices = sparse_product.coalesce().indices()
                            row_idx = indices[0].unique().tolist()
                            col_idx = indices[1].unique().tolist()
                            
                            # Apply sparse update: P[row_idx, :] = sparse_product[row_idx, col_idx] @ P[col_idx, :]
                            # Convert to dense for the relevant submatrix
                            sub_sparse = sparse_product.to_dense()[row_idx][:, col_idx]
                            self.P_t[k][:, row_idx] = sub_sparse @ self.P_t[k][:, col_idx]
                
                # Current step update
                self.P_t[k][:, write] = Jh_proj[:,:,read] @ self.P_t[k][:, read]
                self.P_t[k][:, write] += Jtheta_proj[k]
                self.last_update[k] = self.t
            
            # Expiring parameters: haven't been updated for >= len_buffer steps
            elif self.t - self.last_update[k] >= self.len_buffer:
                # Apply full lazy update from entire buffer before it expires
                sparse_product = self.get_left_product(0, self.len_buffer)
                if sparse_product is not None:
                    # sparse_product is a sparse torch tensor [H, H]
                    indices = sparse_product.coalesce().indices()
                    row_idx = indices[0].unique().tolist()
                    col_idx = indices[1].unique().tolist()
                    
                    # Apply sparse update
                    sub_sparse = sparse_product.to_dense()[row_idx][:, col_idx]
                    self.P_t[k][:, row_idx] = sub_sparse @ self.P_t[k][:, col_idx]
                self.last_update[k] = self.t
            
            # Inactive parameters (not active, not expiring yet): skip this step
            # They will be lazily updated when they become active or when expiring
        if print_time: t1 = time.time(); print("4.2: ", t1-t0); t0=t1


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
        if print_time: t1 = time.time(); print("5: ", t1-t0); t0=t1

        self.t += 1
