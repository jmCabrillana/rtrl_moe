class CircularTree:
    def __init__(self, n, I, op):
        self.n, self.head = n, 0
        self.I = I
        self.op = op
        self.tree = [self.I.clone() for _ in range(2*n)]

    def __len__(self):
        return self.n

    def update(self, new):
        pos = self.head + self.n
        self.tree[pos] = new
        pos //= 2
        while pos > 0:
            self.tree[pos] = self.op(self.tree[2*pos], self.tree[2*pos+1]) 
            pos //= 2
        self.head = (self.head + 1) % self.n  # advance oldest

    def _query_phys(self, l, r):  # [l, r[ on physical index line, no wrap
        L, R = l + self.n, r + self.n
        left = right = self.I.clone()
        while L < R:
            if L % 2 == 1: left = self.op(left, self.tree[L]); L += 1
            if R % 2 == 1: R -= 1; right = self.op(self.tree[R], right)
            L //= 2; R //= 2
        return self.op(left, right)

    def query(self, l, r):  # logical [l, r[, 0 <= l < r <= n
        l = (self.head + l) % self.n
        r = (self.head + r) % self.n
        return self._query_phys(l, r) if r > l else self.op(self._query_phys(0, r), self._query_phys(l, self.n))

if __name__ == "__main__": 

    # Binary operation: LEFT-multiply (note: B @ A)
    def op(A, B):
        return B @ A

    # Identity of shape (1, 2, 2)
    I = torch.eye(2).unsqueeze(0)  # shape: (1, 2, 2)

    # Build tree
    n = 5
    cmt = CircularTree(n=n, I=I, op=op)

    def fmt(t):
        # pretty format a (1,2,2) tensor
        m = t.squeeze(0)
        return f"[[{m[0,0].item():.1f}, {m[0,1].item():.1f}], [{m[1,0].item():.1f}, {m[1,1].item():.1f}]]"

    def print_tree_state(step):
        print(f"\n=== After update step {step} (head={cmt.head}) ===")
        for i in range(1, 2*n):  # skip index 0 (unused)
            print(f"{i:2d}: {fmt(cmt.tree[i])}")

    # Perform more than n inserts to exercise wrap-around
    for k in range(1, n + 3):
        new = I.clone()
        new[..., 0, 1] = float(k)  # set top-right entry to k
        cmt.update(new)
        print_tree_state(k)

    # Optional: demonstrate a logical query over the circular buffer
    # Example: product over the last 3 inserted elements (logical indices [n-3, n) => [2,5) for n=5)
    res = cmt.query(2, 5)  # adjust as you like
    print("\nQuery result (logical [2,5)):", fmt(res))


