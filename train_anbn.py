"""
Training script for a^n b^n counting task
Compares RTRL vs BPTT on sequences of increasing length
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


class SimpleRNN(nn.Module):
    """Simple RNN baseline for a^n b^n"""
    def __init__(self, input_dim=2, hidden_dim=30, output_dim=2):
        super().__init__()
        self.state_fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.output_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        z = torch.cat([x, h], dim=-1)
        s = F.silu(self.state_fc1(z))
        s = s + F.silu(self.state_fc2(s))
        s = self.state_norm(s)
        h_next = torch.tanh(s)
        y = self.output_fc2(F.silu(self.output_fc1(h_next)))
        return y, {}, h_next


def make_anbn_seq(len_max_a=8):
    """Generate a^n b^n sequence with 50% valid, 50% invalid"""
    n_a = random.randint(1, len_max_a)
    if random.random() < 0.5:
        n_b = n_a
        tgt = torch.tensor([1])
    else:
        n_b = random.choice([k for k in range(1, len_max_a+1) if k != n_a])
        tgt = torch.tensor([0])
    # a = [1, 0], b = [0, 1]
    seq = torch.cat([
        torch.tensor([[1., 0.]]).repeat(n_a, 1),
        torch.tensor([[0., 1.]]).repeat(n_b, 1)
    ], dim=0)
    return tgt, seq


def train_bptt(model, optimizer, criterion, n_steps, len_max_a, device, writer, log_prefix="bptt"):
    """Train with standard BPTT"""
    model.train()
    correct = 0
    total_loss = 0.0
    
    for step in range(n_steps):
        tgt, x_seq = make_anbn_seq(len_max_a)
        tgt = tgt.to(device)
        x_seq = x_seq.to(device)
        
        B, H = 1, model.state_fc1.out_features if hasattr(model, 'state_fc1') else model.d
        h_t = torch.zeros(B, H).to(device)
        
        # Unroll entire sequence
        for k in range(x_seq.size(0)):
            y, _, h_next = model(x_seq[k].unsqueeze(0), h_t)
            h_t = h_next
        
        loss = criterion(y, tgt)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track accuracy
        pred = y.argmax(dim=-1).item()
        correct += (pred == tgt.item())
        total_loss += loss.item()
        
        if (step + 1) % 100 == 0:
            acc = correct / 100
            avg_loss = total_loss / 100
            print(f"[{log_prefix}] Step {step+1}/{n_steps} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
            writer.add_scalar(f"{log_prefix}/loss", avg_loss, step)
            writer.add_scalar(f"{log_prefix}/accuracy", acc, step)
            correct = 0
            total_loss = 0.0


def train_rtrl(model, optimizer, criterion, rtrl, n_steps, len_max_a, device, writer, log_prefix="rtrl", acc_steps=64):
    """Train with RTRL (constant memory)"""
    model.train()
    correct = 0
    total_loss = 0.0
    
    # Get state parameters
    if hasattr(model, 'state_fc1'):
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    else:
        # For RecurrentMoE
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    
    for step in range(n_steps):
        tgt, x_seq = make_anbn_seq(len_max_a)
        tgt = tgt.to(device)
        x_seq = x_seq.to(device)
        
        B, H = 1, model.state_fc1.out_features if hasattr(model, 'state_fc1') else model.d
        h_t = torch.zeros(B, H).requires_grad_().to(device)
        rtrl.reset()
        
        # Process sequence with RTRL
        for k in range(x_seq.size(0)):
            x_seq_k = x_seq[k].unsqueeze(0)
            y, info, h_next = model(x_seq_k, h_t)
            
            # Get sparse read/write indices from MoE gating
            if hasattr(model, 'd'):
                # RecurrentMoE: use sparse indices
                active_params, write_idx = get_expert_latent_activated(model, info)
                read_idx = write_idx  # For simplicity, read from same slots
            else:
                # SimpleRNN: all parameters active
                active_params = state_params
                write_idx = None
                read_idx = None
            
            if k < x_seq.size(0) - 1:
                # Intermediate steps: no loss, just update RTRL
                rtrl.step(model, x_seq_k, h_t, None, active_params, read_idx, write_idx)
                h_t = h_next.detach().requires_grad_()
        
        # Final step: compute loss and backprop through RTRL
        loss = criterion(y, tgt) / acc_steps
        rtrl.step(model, x_seq[k].unsqueeze(0), h_t, loss, active_params, read_idx, write_idx)
        
        if (step + 1) % acc_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track accuracy
        pred = y.argmax(dim=-1).item()
        correct += (pred == tgt.item())
        total_loss += loss.item() * acc_steps
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both', choices=['bptt', 'rtrl', 'both'])
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'moe'])
    parser.add_argument('--hidden', type=int, default=30)
    parser.add_argument('--len_max_a', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    # Train with BPTT
    if args.mode in ['bptt', 'both']:
        print(f"\n{'='*60}")
        print(f"Training with BPTT (len_max_a={args.len_max_a})")
        print(f"{'='*60}\n")
        
        if args.model == 'simple':
            model = SimpleRNN(input_dim=2, hidden_dim=args.hidden, output_dim=2).to(device)
        else:
            model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden, 
                                n_experts=4, topk=2, d_in=2, d_out=2).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=f"runs/anbn_bptt_{args.model}_len{args.len_max_a}")
        
        train_bptt(model, optimizer, criterion, args.steps, args.len_max_a, device, writer, "bptt")
        writer.close()
    
    # Train with RTRL
    if args.mode in ['rtrl', 'both']:
        print(f"\n{'='*60}")
        print(f"Training with RTRL (len_max_a={args.len_max_a})")
        print(f"{'='*60}\n")
        
        if args.model == 'simple':
            model = SimpleRNN(input_dim=2, hidden_dim=args.hidden, output_dim=2).to(device)
        else:
            model = RecurrentMoE(d_model=args.hidden, n_heads=2, n_slots=args.hidden,
                                n_experts=4, topk=2, d_in=2, d_out=2).to(device)
        
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
        B, H = 1, args.hidden
        rtrl = BlockRTRL(state_params, B, H)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=f"runs/anbn_rtrl_{args.model}_len{args.len_max_a}")
        
        train_rtrl(model, optimizer, criterion, rtrl, args.steps, args.len_max_a, device, writer, "rtrl")
        writer.close()


if __name__ == "__main__":
    main()
