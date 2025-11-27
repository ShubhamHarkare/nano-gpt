import argparse
import torch
import os
import json
from datetime import datetime
from utils import encode, decode, text
from LLM.hyperparameter import block_size, batch_size, device, eval_iters
from LLM.languagemodel import BigramLanguageModel
from split_data import train_data, val_data


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# ========== ARGUMENT PARSER ==========
parser = argparse.ArgumentParser(description='Train nano-GPT with various configurations')

# Training args
parser.add_argument('--max_iters', type=int, default=5000, help='Maximum training iterations')
parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate every N iterations')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

# LR Schedule args
parser.add_argument('--lr_schedule', type=str, default='constant', 
                    choices=['constant', 'cosine', 'linear'],
                    help='Learning rate schedule type')
parser.add_argument('--warmup_iters', type=int, default=100, 
                    help='Number of warmup iterations for LR schedule')
parser.add_argument('--min_lr', type=float, default=1e-5,
                    help='Minimum learning rate for decay schedules')

# Model args
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

# Experiment tracking
parser.add_argument('--experiment_name', type=str, default='baseline',
                    help='Name of experiment for logging')
parser.add_argument('--save_model', action='store_true',
                    help='Save trained model checkpoint')

args = parser.parse_args()

# Override batch_size from args (since you import it from hyperparameter)
batch_size = args.batch_size

print(f'🚀 Starting Experiment: {args.experiment_name}')
print(f'Device: {device}')
print(f'Config: LR={args.learning_rate}, Schedule={args.lr_schedule}, '
      f'Max Iters={args.max_iters}, Dropout={args.dropout}')
print('-' * 80)

# ========== CREATE EXPERIMENT DIRECTORY ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = f'experiments/{args.experiment_name}_{timestamp}'
os.makedirs(exp_dir, exist_ok=True)

# Save config
config = vars(args)
with open(f'{exp_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

# ========== DATA LOADING ==========
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

torch.manual_seed(1337)

@torch.no_grad()
def eval_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ========== LEARNING RATE SCHEDULER ==========
def get_lr(iteration, warmup_iters, max_iters, learning_rate, min_lr, schedule_type):
    """Calculate learning rate for current iteration"""
    
    if schedule_type == 'constant':
        return float(learning_rate)
    
    # Warmup phase
    if iteration < warmup_iters:
        return float(learning_rate * (iteration + 1) / warmup_iters)
    
    # After warmup
    if schedule_type == 'cosine':
        # Cosine decay
        import math
        decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Use math.cos, not torch
        return float(min_lr + coeff * (learning_rate - min_lr))
    
    elif schedule_type == 'linear':
        # Linear decay
        decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
        return float(learning_rate - (learning_rate - min_lr) * decay_ratio)
    
    return float(learning_rate)
# ========== MODEL INITIALIZATION ==========
model = BigramLanguageModel()
m = model.to(device)
print(f'📊 Model: {sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters')

# ========== OPTIMIZER ==========
optimizer = torch.optim.AdamW(m.parameters(), lr=args.learning_rate)

# ========== TRAINING LOOP WITH TRACKING ==========
import matplotlib.pyplot as plt

# Tracking metrics
metrics = {
    'iterations': [],
    'train_loss': [],
    'val_loss': [],
    'perplexity': [],
    'learning_rates': []
}

print('\n🏃 Starting Training...\n')

for iteration in range(args.max_iters):
    
    # Update learning rate
    current_lr = get_lr(
        iteration, 
        args.warmup_iters, 
        args.max_iters, 
        args.learning_rate, 
        args.min_lr,
        args.lr_schedule
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Evaluation
    if iteration % args.eval_interval == 0 or iteration == args.max_iters - 1:
        losses = eval_loss()
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        perplexity = torch.exp(losses['val']).item()
        
        # Log metrics
        metrics['iterations'].append(iteration)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['perplexity'].append(perplexity)
        metrics['learning_rates'].append(float(current_lr))
        
        print(f'Step {iteration:5d} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f} | LR: {current_lr:.2e}')
    
    # Training step
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('\n✅ Training Complete!\n')

# ========== SAVE METRICS ==========
with open(f'{exp_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ========== GENERATE PLOTS ==========
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(metrics['iterations'], metrics['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(metrics['iterations'], metrics['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Perplexity
axes[0, 1].plot(metrics['iterations'], metrics['perplexity'], linewidth=2, color='green')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Perplexity')
axes[0, 1].set_title('Validation Perplexity')
axes[0, 1].grid(True, alpha=0.3)

# Learning rate schedule
axes[1, 0].plot(metrics['iterations'], metrics['learning_rates'], linewidth=2, color='red')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title(f'Learning Rate Schedule ({args.lr_schedule})')
axes[1, 0].grid(True, alpha=0.3)

# Loss histogram
axes[1, 1].hist(metrics['val_loss'], bins=20, alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Validation Loss')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Validation Loss Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{exp_dir}/training_curves.png', dpi=300, bbox_inches='tight')
print(f'📈 Training curves saved to {exp_dir}/training_curves.png')

# ========== SAVE RESULTS SUMMARY ==========
results = {
    'experiment_name': args.experiment_name,
    'timestamp': timestamp,
    'config': config,
    'final_train_loss': metrics['train_loss'][-1],
    'final_val_loss': metrics['val_loss'][-1],
    'final_perplexity': metrics['perplexity'][-1],
    'best_val_loss': min(metrics['val_loss']),
    'best_perplexity': min(metrics['perplexity']),
    'total_iterations': len(metrics['iterations'])
}

with open(f'{exp_dir}/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n📊 Final Results:')
print(f'  Best Validation Loss: {results["best_val_loss"]:.4f}')
print(f'  Best Perplexity: {results["best_perplexity"]:.2f}')
print(f'  Results saved to: {exp_dir}/')

# ========== SAVE MODEL (OPTIONAL) ==========
if args.save_model:
    torch.save({
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics,
    }, f'{exp_dir}/model_checkpoint.pt')
    print(f'💾 Model saved to {exp_dir}/model_checkpoint.pt')

# ========== GENERATE SAMPLE TEXT ==========
print('\n🎭 Sample Generation:')
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(idx, max_new_tokens=300)
sample_text = decode(generated[0].tolist())
print(sample_text[:500])  # Print first 500 chars

with open(f'{exp_dir}/sample_generation.txt', 'w') as f:
    f.write(sample_text)

print(f'\n✅ Experiment complete! All results in: {exp_dir}/')