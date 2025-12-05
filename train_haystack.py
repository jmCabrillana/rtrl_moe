"""
Training script for Haystack retrieval task
Demonstrates RTRL's constant memory advantage on very long sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
import time

from rtrl_block import BlockRTRL
from moe import RecurrentMoE, get_expert_latent_activated


# Haystack task constants
BOS, KEY, SEP, Q, BASE = 0, 1, 2, 3, 4


class LSTMHaystack(nn.Module):
    """LSTM baseline for haystack"""
    def __init__(self, vocab_size, hidden_dim, num_layers=2):
        super().__init__()
        self.H = hidden_dim
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.head = nn.Linear(hidden_dim, vocab_size - BASE)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, h=None):
        out, h = self.rnn(self.emb(x), h)
        logits = self.head(self.norm(out[:, -1]))
        return logits, {}, h


def sample_haystack(seq_len, vocab_size, device):
    """Generate haystack retrieval task"""
    x = torch.empty(1, seq_len, dtype=torch.long)
    y = torch.empty(1, dtype=torch.long)
    
    k = random.randrange(vocab_size - BASE)
    # Insert key somewhere with space before final Q token
    ins = random.randrange(1, seq_len - 17)
    
    seq = [BOS]
    while len(seq) < ins:
        seq.append(random.randrange(BASE, vocab_size))
    seq += [KEY, BASE + k, SEP]
    while len(seq) < seq_len - 1:
        seq.append(random.randrange(BASE, vocab_size))
    seq.append(Q)
    
    x[0] = torch.tensor(seq)
    y[0] = k
    
    return x.to(device), y.to(device)


def train_bptt(model, optimizer, criterion, n_steps, seq_len, vocab_size, device, writer, 
               log_prefix="bptt", tbptt=None):
    """Train with BPTT (truncated if tbptt is set)"""
    model.train()
    correct = 0
    total_loss = 0.0
    
    for step in range(n_steps):
        x, y = sample_haystack(seq_len, vocab_size, device)
        
        if tbptt is None or isinstance(model, LSTMHaystack):
            # Full BPTT for LSTM
            logits, _, _ = model(x, None)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
        else:
            # Truncated BPTT
            B, H = 1, model.d if hasattr(model, 'd') else model.state_fc1.out_features
            h_t = torch.zeros(B, H).to(device)
            
            for s in range(0, seq_len, tbptt):
                chunk = x[:, s:min(s+tbptt, seq_len)]
                logits, _, h_next = model(chunk, h_t)
                h_t = h_next.detach()
            
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        pred = logits.argmax(dim=-1).item()
        correct += (pred == y.item())
        total_loss += loss.item()
        
        if (step + 1) % 100 == 0:
            acc = correct / 100
            avg_loss = total_loss / 100
            print(f"[{log_prefix}] Step {step+1}/{n_steps} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
            writer.add_scalar(f"{log_prefix}/loss", avg_loss, step)
            writer.add_scalar(f"{log_prefix}/accuracy", acc, step)
            correct = 0
            total_loss = 0.0


def train_rtrl(model, optimizer, criterion, rtrl, n_steps, seq_len, vocab_size, device, 
               writer, log_prefix="rtrl", chunk_size=1):
    """Train with RTRL (constant memory regardless of sequence length)"""
    model.train()
    correct = 0
    total_loss = 0.0
    
    state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    
    for step in range(n_steps):
        x, y = sample_haystack(seq_len, vocab_size, device)
        
        B, H = 1, model.d if hasattr(model, 'd') else model.d
        h_t = model.init_state(B, device=device).requires_grad_()
        rtrl.reset()
        
        # Process sequence in chunks
        for s in range(0, seq_len - chunk_size, chunk_size):
            x_chunk = F.one_hot(x[:, s:s+chunk_size], num_classes=vocab_size).float()
            logits, info, h_next = model(x_chunk, h_t)
            
            # Get sparse read/write indices from MoE gating
            active_params, write_idx = get_expert_latent_activated(model, info)
            read_idx = write_idx
            
            # Only update RTRL, no loss yet
            rtrl.step(model, x_chunk, h_t, None, active_params, read_idx, write_idx)
            h_t = h_next.detach().requires_grad_()
        
        # Final chunk with loss
        x_final = F.one_hot(x[:, -chunk_size:], num_classes=vocab_size).float()
        logits, info, h_next = model(x_final, h_t)
        active_params, write_idx = get_expert_latent_activated(model, info)
        read_idx = write_idx
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        rtrl.step(model, x_final, h_t, loss, active_params, read_idx, write_idx)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        pred = logits.argmax(dim=-1).item()
        correct += (pred == y.item())
        total_loss += loss.item()
        
        if (step + 1) % 100 == 0:
            acc = correct / 100
            avg_loss = total_loss / 100
            print(f"[{log_prefix}] Step {step+1}/{n_steps} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
            writer.add_scalar(f"{log_prefix}/loss", avg_loss, step)
            writer.add_scalar(f"{log_prefix}/accuracy", acc, step)
            P_t_norm = sum(p.abs().mean().item() for p in rtrl.P_t.values()) / len(rtrl.P_t)
            writer.add_scalar(f"{log_prefix}/P_t_norm", P_t_norm, step)
            correct = 0
            total_loss = 0.0


def measure_memory(model, seq_len, vocab_size, device, mode='bptt', tbptt=None):
    """Measure peak memory usage"""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    x, y = sample_haystack(seq_len, vocab_size, device)
    criterion = nn.CrossEntropyLoss()
    
    if mode == 'bptt':
        if tbptt is None:
            logits, _, _ = model(x, None)
            loss = criterion(logits, y)
            loss.backward()
        else:
            B, H = 1, model.d if hasattr(model, 'd') else 64
            h_t = torch.zeros(B, H).to(device)
            for s in range(0, seq_len, tbptt):
                chunk = x[:, s:min(s+tbptt, seq_len)]
                logits, _, h_next = model(chunk, h_t)
                h_t = h_next.detach()
            loss = criterion(logits, y)
            loss.backward()
    
    elif mode == 'rtrl':
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
        B, H = 1, model.d
        rtrl = BlockRTRL(state_params, B, H)
        h_t = model.init_state(B, device=device).requires_grad_()
        
        for s in range(0, seq_len - 1, 1):
            x_chunk = F.one_hot(x[:, s:s+1], num_classes=vocab_size).float()
            logits, info, h_next = model(x_chunk, h_t)
            rtrl.step(model, x_chunk, h_t, None, state_params)
            h_t = h_next.detach().requires_grad_()
        
        x_final = F.one_hot(x[:, -1:], num_classes=vocab_size).float()
        logits, info, h_next = model(x_final, h_t)
        loss = criterion(logits, y)
        rtrl.step(model, x_final, h_t, loss, state_params)
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    return peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both', choices=['bptt', 'rtrl', 'both', 'memory'])
    parser.add_argument('--model', type=str, default='moe', choices=['moe', 'lstm'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--vocab', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tbptt', type=int, default=None, help='Truncated BPTT window')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    # Memory profiling mode
    if args.mode == 'memory':
        print(f"\n{'='*60}")
        print(f"Memory Profiling (seq_len={args.seq_len})")
        print(f"{'='*60}\n")
        
        for seq_len in [64, 128, 256, 512, 1024, 2048]:
            print(f"\nSequence Length: {seq_len}")
            
            # BPTT memory
            if args.model == 'moe':
                model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden,
                                    n_experts=4, topk=2, d_in=args.vocab, d_out=args.vocab-BASE).to(device)
            else:
                model = LSTMHaystack(args.vocab, args.hidden).to(device)
            
            mem_bptt_full = measure_memory(model, seq_len, args.vocab, device, 'bptt', None)
            print(f"  BPTT (full):     {mem_bptt_full:.2f} MB")
            
            if args.tbptt:
                mem_bptt_trunc = measure_memory(model, seq_len, args.vocab, device, 'bptt', args.tbptt)
                print(f"  BPTT (tbptt={args.tbptt}): {mem_bptt_trunc:.2f} MB")
            
            # RTRL memory
            if args.model == 'moe':
                model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden,
                                    n_experts=4, topk=2, d_in=args.vocab, d_out=args.vocab-BASE).to(device)
                mem_rtrl = measure_memory(model, seq_len, args.vocab, device, 'rtrl')
                print(f"  RTRL:            {mem_rtrl:.2f} MB")
                print(f"  Ratio (BPTT/RTRL): {mem_bptt_full/mem_rtrl:.2f}x")
            
            torch.cuda.empty_cache()
        
        return
    
    # Train with BPTT
    if args.mode in ['bptt', 'both']:
        print(f"\n{'='*60}")
        print(f"Training with BPTT (seq_len={args.seq_len})")
        print(f"{'='*60}\n")
        
        if args.model == 'moe':
            model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden,
                                n_experts=4, topk=2, d_in=args.vocab, d_out=args.vocab-BASE).to(device)
        else:
            model = LSTMHaystack(args.vocab, args.hidden).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=f"runs/haystack_bptt_{args.model}_len{args.seq_len}")
        
        train_bptt(model, optimizer, criterion, args.steps, args.seq_len, args.vocab, 
                  device, writer, "bptt", tbptt=args.tbptt)
        writer.close()
    
    # Train with RTRL
    if args.mode in ['rtrl', 'both']:
        print(f"\n{'='*60}")
        print(f"Training with RTRL (seq_len={args.seq_len})")
        print(f"{'='*60}\n")
        
        if args.model == 'moe':
            model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden,
                                n_experts=4, topk=2, d_in=args.vocab, d_out=args.vocab-BASE).to(device)
        else:
            raise ValueError("RTRL only supported for MoE model currently")
        
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
        B, H = 1, model.d * model.n_slots
        rtrl = BlockRTRL(state_params, B, H)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=f"runs/haystack_rtrl_{args.model}_len{args.seq_len}")
        
        train_rtrl(model, optimizer, criterion, rtrl, args.steps, args.seq_len, args.vocab,
                  device, writer, "rtrl")
        writer.close()


if __name__ == "__main__":
    main()
