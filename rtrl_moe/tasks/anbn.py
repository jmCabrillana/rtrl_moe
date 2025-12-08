"""a^n b^n counting task utilities."""
import random
from typing import Tuple

import torch
import torch.nn.functional as F

# Token definitions
PAD, A, B, Q = 0, 1, 2, 3
VOCAB_SIZE = 4
OUTPUT_DIM = 2


def sample(seq_len: int, device: torch.device, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate padded a^n b^n sequences ending with a query token.

    Sequences are fixed length (seq_len). Target is 1 if counts match, else 0.
    """
    max_a = max(1, (seq_len - 1) // 2)
    x = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    y = torch.empty(batch_size, dtype=torch.long)

    for b in range(batch_size):
        n_a = random.randint(1, max_a)

        if random.random() < 0.5:
            n_b = n_a
            y[b] = 1
        else:
            choices = [k for k in range(1, max_a + 1) if k != n_a]
            if choices:
                n_b = random.choice(choices)
            else:
                n_b = max(1, n_a - 1)
            y[b] = 0

        seq = [A] * n_a + [B] * n_b
        seq = seq[: seq_len - 1]  # leave room for query token

        for idx, token in enumerate(seq):
            x[b, idx] = token

        x[b, seq_len - 1] = Q

    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float().to(device)
    return x_onehot, y.to(device)
