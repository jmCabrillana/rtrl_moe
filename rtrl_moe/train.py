"""
RTRL/BPTT training for RNN and MoE models on sequence tasks.

Usage:
  python train.py --model moe --method rtrl --task anbn --extra-steps 100
  python train.py --model simple --method bptt --task anbn --extra-steps 100 --params "batch_size=32"

Key differences:
  RTRL: Constant memory O(H), batch_size=1, gradient accumulation via ACCUM_STEPS
  BPTT: Full backprop O(T*H), batch_size can be >1, no accumulation needed
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add package root to path for imports (supports execution from any directory)
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model.moe import RecurrentMoE, get_expert_latent_activated
from model.simple_rnn import SimpleRNN, get_active_params_simple
from core.rtrl_block import BlockRTRL
from core.utils import (
    parse_params, evaluate_accuracy, evaluate_accuracy_bptt,
    save_checkpoint, log_metrics_rtrl, log_metrics_bptt, log_gradient_norms,
    print_step_rtrl, print_step_bptt, print_experiment_header, print_final_summary,
    compute_sensitivity_norm, compute_grad_norm
)
from tasks import haystack, anbn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ============ Configuration & Arguments ============
SEQ_LEN = 128
BATCH_SIZE = 1
LR = 3e-3
D_MODEL = 32
N_SLOTS = 4
N_EXPERTS = 4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SAVE_EVERY = 100
EVAL_BATCH_SIZE = 10
ACCUM_STEPS = 32

parser = argparse.ArgumentParser(description="Train RNN/MoE with RTRL or BPTT")
parser.add_argument("--model", type=str, default="moe", choices=["moe", "simple"])
parser.add_argument("--method", type=str, default="rtrl", choices=["rtrl", "bptt"])
parser.add_argument("--task", type=str, default="anbn", choices=["anbn", "haystack"])
parser.add_argument("--exp-name", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--extra-steps", type=int, default=200)
parser.add_argument("--params", type=str, default="", help="key=val,key=val")
args = parser.parse_args()

user_params = parse_params(args.params)

# Apply parameter overrides
D_MODEL = user_params.get("d_model", D_MODEL)
N_SLOTS = user_params.get("n_slots", N_SLOTS)
N_EXPERTS = user_params.get("n_experts", N_EXPERTS)
LR = user_params.get("lr", LR)
WEIGHT_DECAY = user_params.get("weight_decay", WEIGHT_DECAY)
GRAD_CLIP = user_params.get("grad_clip", GRAD_CLIP)
SEQ_LEN = user_params.get("seq_len", SEQ_LEN)
EVAL_BATCH_SIZE = user_params.get("eval_batch_size", EVAL_BATCH_SIZE)
SAVE_EVERY = user_params.get("save_every", SAVE_EVERY)

# Method-specific batch size logic
if args.method == "bptt":
    BATCH_SIZE = user_params.get("batch_size", 32)
else:  # rtrl
    BATCH_SIZE = 1
    ACCUM_STEPS = user_params.get("accum_steps", ACCUM_STEPS)

# ============ Task Setup ============
task_module = haystack if args.task == "haystack" else anbn
VOCAB_SIZE = task_module.VOCAB_SIZE
OUTPUT_DIM = task_module.OUTPUT_DIM

# ============ Model Setup ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

model_config = {
    'd_model': D_MODEL, 'n_heads': 2, 'n_slots': N_SLOTS, 'n_experts': N_EXPERTS,
    'd_in': VOCAB_SIZE, 'd_out': OUTPUT_DIM,
}

# Merge user model-specific params
skip_training_params = {'lr', 'weight_decay', 'grad_clip', 'seq_len', 'batch_size', 'accum_steps', 'eval_batch_size', 'save_every'}
for k, v in user_params.items():
    if k not in skip_training_params:
        model_config[k] = v

# Instantiate model
if args.model == "moe":
    model = RecurrentMoE(**model_config).to(device)
    get_active_params = get_expert_latent_activated
    H = D_MODEL * N_SLOTS
elif args.model == "simple":
    simple_config = {k: model_config[k] for k in ['d_model', 'd_in', 'd_out']}
    model = SimpleRNN(**simple_config).to(device)
    get_active_params = get_active_params_simple
    H = D_MODEL
else:
    raise ValueError(f"Unknown model: {args.model}")

# ============ Training Setup ============
state_params = {k: v for k, v in model.named_parameters() if not k.startswith("output_")}
rtrl = BlockRTRL(state_params, BATCH_SIZE, H, len_buffer=min(SEQ_LEN, 8192))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# Setup experiment directory & logging
exp_name = args.exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = script_dir / "runs" / exp_name
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(log_dir))

# Load checkpoint if resuming
start_step = 0
if args.checkpoint:
    print(f"Resuming from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = ckpt.get("step", 0)

target_step = start_step + args.extra_steps
print(f"\nTraining {args.method.upper()} for {args.extra_steps} steps (target: {target_step})\n")

print_experiment_header(
    exp_name=exp_name, task_name=args.task, method=args.method, seq_len=SEQ_LEN,
    model_name=args.model, d_model=D_MODEL, n_slots=N_SLOTS, n_experts=N_EXPERTS,
    vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, user_params=user_params,
    accum_steps=ACCUM_STEPS, log_dir=log_dir
)

losses, accuracies, read_sparsities, write_sparsities = [], [], [], []

# ============ Training Loop ============
try:
    for step in range(start_step, target_step):
        t0 = time.time()
        x_onehot, y = task_module.sample(SEQ_LEN, device, batch_size=BATCH_SIZE)
        
        if args.method == "rtrl":
            # Forward pass: one timestep at a time (constant memory)
            h_t = model.init_state(BATCH_SIZE, device=device).requires_grad_()
            rtrl.reset()
            t_fwd_start = time.time()
            
            for t in range(SEQ_LEN - 1):
                x_t = x_onehot[:, t:t+1, :]
                _, info, h_next = model(x_t, h_t)
                active_params, write_idx, read_idx = get_active_params(model, info)
                rtrl.step(model, x_t, h_t, None, active_params, read_idx, write_idx)
                h_t = h_next.detach().requires_grad_()
            
            t_fwd = time.time() - t_fwd_start
            t_bwd_start = time.time()
            
            # Final step with loss
            x_t = x_onehot[:, -1:, :]
            pred_logits, info, _ = model(x_t, h_t)
            active_params, write_idx, read_idx = get_active_params(model, info)
            task_loss = criterion(pred_logits, y)
            rtrl.step(model, x_t, h_t, task_loss, active_params, read_idx, write_idx)
            
            # Accumulate gradients
            if (step + 1) % ACCUM_STEPS == 0 or step + 1 == target_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = compute_grad_norm(model)
            
            t_bwd = time.time() - t_bwd_start
            sensitivity_norm = compute_sensitivity_norm(rtrl)
            acc = evaluate_accuracy(model, task_module, SEQ_LEN, device, BATCH_SIZE, EVAL_BATCH_SIZE)
            read_sparse = len(read_idx) / H * 100
            write_sparse = len(write_idx) / H * 100
            
            loss_val = task_loss.item()
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            timing = {'forward': t_fwd, 'backward': t_bwd, 'total': time.time() - t0}
            
            print_step_rtrl(step+1, loss_val, acc, read_sparse, write_sparse, sensitivity_norm, timing)
            log_metrics_rtrl(writer, step, {'task_loss': loss_val, 'grad_norm': grad_val, 'total_loss': loss_val}, 
                           acc, read_sparse, write_sparse, sensitivity_norm, timing)
            losses.append(loss_val)
            read_sparsities.append(read_sparse)
            write_sparsities.append(write_sparse)
        
        else:  # BPTT
            # Forward pass: unroll entire sequence
            t_fwd_start = time.time()
            h_t = model.init_state(BATCH_SIZE, device=device)
            info_seq = []
            
            for t in range(SEQ_LEN):
                x_t = x_onehot[:, t:t+1, :]
                pred_logits, info, h_t = model(x_t, h_t)
                info_seq.append(info)
            
            t_fwd = time.time() - t_fwd_start
            t_bwd_start = time.time()
            
            # Backward pass
            task_loss = criterion(pred_logits, y)
            task_loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0 or step + 1 == target_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = compute_grad_norm(model)
            
            t_bwd = time.time() - t_bwd_start
            acc = evaluate_accuracy_bptt(model, task_module, SEQ_LEN, device, BATCH_SIZE, EVAL_BATCH_SIZE)
            
            # Sparsity from last step
            if args.model == "moe":
                read_idx = info_seq[-1].get('read_idx', [])
                write_idx = info_seq[-1].get('write_idx', [])
                read_sparse = len(read_idx) / H * 100 if H > 0 else 0
                write_sparse = len(write_idx) / H * 100 if H > 0 else 0
            else:
                read_sparse = write_sparse = 0
            
            loss_val = task_loss.item()
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            timing = {'forward': t_fwd, 'backward': t_bwd, 'total': time.time() - t0}
            
            print_step_bptt(step+1, loss_val, acc, timing, use_moe=(args.model == "moe"), 
                          read_sparsity=read_sparse, write_sparsity=write_sparse)
            log_metrics_bptt(writer, step, {'task_loss': loss_val, 'grad_norm': grad_val, 'total_loss': loss_val},
                           acc, read_sparse, write_sparse, timing, use_moe=(args.model == "moe"))
            losses.append(loss_val)
            read_sparsities.append(read_sparse)
            write_sparsities.append(write_sparse)
        
        accuracies.append(acc)
        
        # Logging & checkpointing
        if step % 10 == 0:
            log_gradient_norms(writer, model, step)
        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = save_checkpoint(log_dir, step+1, model, optimizer, loss_val, acc)
            print(f"  → Checkpoint: {ckpt_path.name}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

# ============ Final Summary ============
print_final_summary(losses, accuracies, read_sparsities, write_sparsities, log_dir, SEQ_LEN, H)

writer.close()
print("TensorBoard log closed. View results with:")
print(f"  tensorboard --logdir={log_dir.parent}")
