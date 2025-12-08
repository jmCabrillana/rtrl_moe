"""Haystack retrieval task utilities."""
import random
from typing import Tuple

import torch
import torch.nn.functional as F

# Token definitions
BOS, KEY, SEP, Q, BASE = 0, 1, 2, 3, 4
VOCAB_SIZE = 8
OUTPUT_DIM = VOCAB_SIZE - BASE


def sample(seq_len: int, device: torch.device, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a haystack sequence and target.

    Returns one-hot encoded inputs (B, T, V) and targets (B,).
    """
    x = torch.empty(batch_size, seq_len, dtype=torch.long)
    y = torch.empty(batch_size, dtype=torch.long)

    for b in range(batch_size):
        k = random.randrange(VOCAB_SIZE - BASE)
        ins = random.randrange(1, max(2, seq_len - 5))

        seq = [BOS]
        while len(seq) < ins:
            seq.append(random.randrange(BASE, VOCAB_SIZE))
        seq += [KEY, BASE + k, SEP]
        while len(seq) < seq_len - 1:
            seq.append(random.randrange(BASE, VOCAB_SIZE))
        seq.append(Q)

        x[b] = torch.tensor(seq, dtype=torch.long)
        y[b] = k

    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float().to(device)
    return x_onehot, y.to(device)
