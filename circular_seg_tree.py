import torch

class CircularTree:
    def __init__(self, n, I, op):
        self.n, self.head = n, 0
        self.I = I
        self.op = op
        self.tree = [self.I.clone() if isinstance(I, torch.Tensor) else I for _ in range(2*n)]

    def __len__(self):
        return self.n

    def update(self, new):
        pos = self.head + self.n
        self.tree[pos] = new
        pos //= 2
        while pos > 0:
            self.tree[pos] = self.op(self.tree[2*pos], self.tree[2*pos+1])   # left-mult
            pos //= 2
        self.head = (self.head + 1) % self.n  # advance oldest

    def _query_phys(self, l, r):  # [l, r[ on physical index line, no wrap
        L, R = l + self.n, r + self.n
        left = right = self.I.clone() if isinstance(self.I, torch.Tensor) else self.I
        while L < R:
            if L % 2 == 1: left = self.op(left, self.tree[L]); L += 1
            if R % 2 == 1: R -= 1; right = self.op(self.tree[R], right)
            L //= 2; R //= 2
        return self.op(left, right)

    def query(self, l, r):  # logical [l, r[, 0 <= l < r <= n
        l = (self.head + l) % self.n
        r = (self.head + r) % self.n
        return self._query_phys(l, r) if r > l else self.op(self._query_phys(0, r), self._query_phys(l, self.n))

# Sparse matrix multiplication: sparse LEFT-multiply A @ B
def sparse_left_mm(a, b):
    if a is None: return b
    if b is None: return a
    return torch.sparse.mm(b, a)

# Binary operation: sparse LEFT-multiply B @ A
def sparse_left_mul(a, b):
    if a is None: return b
    elif b is None: return a
    else:
        (idx_a, A), (idx_b, B) = a, b
        d = a[1].shape[-1]
        idx_union = list(sorted(set(idx_a) | set(idx_b)))
        pos_a, pos_b = {i: p for p, i in enumerate(idx_a)}, {i: p for p, i in enumerate(idx_b)}
        A_delta = A.clone()
        A_delta[:, torch.arange(len(idx_a), device=A.device), idx_a] -= 1
        Cb = B + B[:, :, idx_a] @ A_delta
        rows = torch.stack([Cb[:, pos_b[i]] if i in pos_b else A[:, pos_a[i]] for i in idx_union], dim=1)
        return (idx_union, rows)
    
# def sparse_left_mul(a, b): # 1D
#     if a is None: return b
#     elif b is None: return a 
#     else:
#         (idx_a, A), (idx_b, B), d = a, b, a[1].shape[-1]
#         idx_union = list(sorted(set(idx_a) | set(idx_b)))
#         pos_a, pos_b = {i: p for p, i in enumerate(idx_a)}, {i: p for p, i in enumerate(idx_b)}
#         A_delta = A.clone()
#         A_delta[torch.arange(len(idx_a), device=A.device), idx_a] -= 1
#         Cb = B + B[:, idx_a] @ A_delta
#         rows = torch.stack([Cb[pos_b[i]] if i in pos_b else A[pos_a[i]] for i in idx_union], dim=0)
#         return (idx_union, rows)

def materialize(sparse, n, dtype=torch.int64):
    I = torch.eye(n, dtype=dtype)
    if sparse is None: 
        return I
    idx, rows = sparse
    M = I.clone()
    if len(idx):
        M[idx, :] = rows.to(dtype)
    return M

if __name__ == "__main__": 

    def fmt(t):
        if t is None: return "None"
        else: return f"{t}"

    def print_tree_state(step):
        print(f"\n=== After update step {step} (head={cmt.head}) ===")
        for i in range(2*n):  # 0 (unused)
            print(f"{i:2d}: {fmt(cmt.tree[i])}")

    print("TEST CIRCULAR MULT")
    def op(A, B):
        return B @ A
    I = torch.eye(2).unsqueeze(0)  # shape: (1, 2, 2)
    n = 5
    cmt = CircularTree(n=n, I=I, op=op)

    # Perform more than n inserts to exercise wrap-around
    for k in range(1, n + 3):
        new = I.clone()
        new[..., 0, 1] = float(k)  # set top-right entry to k
        cmt.update(new)
        print_tree_state(k)

    # Sum
    res = cmt.query(2, 5)  # adjust as you like
    print("\nQuery result (logical [2,5)):", fmt(res))

    print("TEST CIRCULAR SPARSE")
    I = None 
    n = 5
    cmt = CircularTree(n=n, I=I, op=sparse_left_mul)

    # Perform more than n inserts to exercise wrap-around
    for k in range(1, n + 3):
        cmt.update(([0], torch.tensor([[1, k]])))
        print_tree_state(k)

    # Sum
    res = cmt.query(2, 5)  # adjust as you like
    print("\nQuery result (logical [2,5)):", fmt(res))


    # ---- Minimal deterministic test ----
    n = 7
    dtype = torch.int64

    # Define A: identity except on rows 1, 4, 6
    idx_a = [1, 4, 6]
    A_rows = torch.tensor([
        [0, 1, 2, 0, 0, 0, 0],   # row 1
        [0, 0, 0, 0, 1, 3, 0],   # row 4
        [1, 0, 0, 0, 0, 0, 1],   # row 6
    ], dtype=dtype)
    a = (idx_a, A_rows)

    # Define B: identity except on rows 0, 4
    idx_b = [0, 4]
    B_rows = torch.tensor([
        [1, 0, 0, 0, 2, 0, 0],   # row 0
        [0, 0, 1, 0, 0, 0, 0],   # row 4
    ], dtype=dtype)
    b = (idx_b, B_rows)

    # Compute dense ground-truth
    A_dense = materialize(a, n, dtype)
    B_dense = materialize(b, n, dtype)
    C_dense = B_dense @ A_dense

    # Compute via sparse_left_mul
    print(a,b)
    c = sparse_left_mul(a, b)
    C_sparse_dense = materialize(c, n, dtype)

    print("A_dense =\n", A_dense)
    print("\nB_dense =\n", B_dense)
    print("\nC_dense = B @ A =\n", C_dense)
    print("\nC_from_sparse =\n", C_sparse_dense)
    print("\nExact match? ->", torch.equal(C_dense, C_sparse_dense))

    


