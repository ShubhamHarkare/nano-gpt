import json
import glob
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("NANO-GPT EXPERIMENT RESULTS")
print("=" * 80)

# Find all experiments
exp_dirs = glob.glob('experiments/*/')
if not exp_dirs:
    print("\n❌ No experiment directories found in 'experiments/' folder!")
    print("Make sure you're running this from the nano-gpt directory.")
    exit(1)

results = {}

for exp_dir in exp_dirs:
    summary_file = os.path.join(exp_dir, 'results_summary.json')
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                exp_name = data.get('experiment_name', os.path.basename(exp_dir.rstrip('/')))
                results[exp_name] = data
        except Exception as e:
            print(f"⚠️  Warning: Could not load {summary_file}: {e}")

if not results:
    print("\n❌ No valid results found!")
    print("Make sure your experiments have 'results_summary.json' files.")
    exit(1)

print(f"\n✅ Found {len(results)} experiment(s)\n")

# Print comparison table
print("📊 RESULTS SUMMARY:\n")
print(f"{'Experiment':<40} {'Best Perplexity':<20} {'Best Val Loss':<20} {'Iterations':<15}")
print("-" * 95)

sorted_results = sorted(results.items(), key=lambda x: x[1].get('best_perplexity', float('inf')))

for name, data in sorted_results:
    ppl = data.get('best_perplexity', 'N/A')
    val_loss = data.get('best_val_loss', 'N/A')
    iters = data.get('total_iterations', 'N/A')
    
    ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
    val_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
    iter_str = str(iters)
    
    print(f"{name:<40} {ppl_str:<20} {val_str:<20} {iter_str:<15}")

# Calculate improvements (if we have multiple experiments)
if len(results) > 1:
    # Try to find baseline
    baseline_name = None
    baseline_candidates = [k for k in results.keys() if 'baseline' in k.lower()]
    
    if baseline_candidates:
        baseline_name = baseline_candidates[0]
    else:
        # Use the experiment with highest perplexity as baseline
        baseline_name = max(results.items(), key=lambda x: x[1].get('best_perplexity', 0))[0]
        print(f"\n⚠️  No 'baseline' experiment found. Using '{baseline_name}' as reference.")
    
    baseline_ppl = results[baseline_name].get('best_perplexity')
    
    if baseline_ppl and isinstance(baseline_ppl, (int, float)):
        print("\n" + "=" * 80)
        print(f"IMPROVEMENTS OVER {baseline_name.upper()}:")
        print("=" * 80)
        
        for name, data in results.items():
            if name != baseline_name:
                ppl = data.get('best_perplexity')
                if ppl and isinstance(ppl, (int, float)):
                    improvement = ((baseline_ppl - ppl) / baseline_ppl) * 100
                    symbol = "✅" if improvement > 0 else "❌"
                    print(f"{symbol} {name}: {improvement:+.1f}% ({baseline_ppl:.2f} → {ppl:.2f})")

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Perplexity comparison (bar chart)
if results:
    exp_names = [name.replace('_', '\n')[:30] for name in results.keys()]
    perplexities = [data.get('best_perplexity', 0) for data in results.values()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = axes[0, 0].bar(range(len(exp_names)), perplexities, 
                          color=colors[:len(exp_names)])
    axes[0, 0].set_xticks(range(len(exp_names)))
    axes[0, 0].set_xticklabels(exp_names, fontsize=9, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Best Perplexity', fontsize=12, weight='bold')
    axes[0, 0].set_title('Perplexity Comparison', fontsize=14, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, ppl in zip(bars, perplexities):
        if ppl > 0:
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{ppl:.2f}', ha='center', va='bottom', 
                           fontsize=10, weight='bold')

# Plot 2: Training curves overlay
curves_plotted = False
for name, data in results.items():
    # Find corresponding metrics file
    metrics_file = None
    for exp_dir in exp_dirs:
        if name in exp_dir or exp_dir.rstrip('/').endswith(name):
            metrics_file = os.path.join(exp_dir, 'metrics.json')
            break
    
    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            if 'iterations' in metrics and 'val_loss' in metrics:
                axes[0, 1].plot(metrics['iterations'], metrics['val_loss'], 
                               label=name.split('_')[0][:15], linewidth=2)
                curves_plotted = True
        except Exception as e:
            print(f"⚠️  Could not plot {metrics_file}: {e}")

if curves_plotted:
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12, weight='bold')
    axes[0, 1].set_title('Validation Loss Curves', fontsize=14, weight='bold')
    axes[0, 1].legend(fontsize=9, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'No training curves available', 
                    ha='center', va='center', fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Validation Loss Curves', fontsize=14, weight='bold')

# Plot 3: Learning rate schedules
schedules_plotted = False
for name, data in results.items():
    metrics_file = None
    for exp_dir in exp_dirs:
        if name in exp_dir or exp_dir.rstrip('/').endswith(name):
            metrics_file = os.path.join(exp_dir, 'metrics.json')
            break
    
    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            if 'iterations' in metrics and 'learning_rates' in metrics:
                axes[1, 0].plot(metrics['iterations'], metrics['learning_rates'], 
                               label=name.split('_')[0][:15], linewidth=2)
                schedules_plotted = True
        except Exception as e:
            print(f"⚠️  Could not plot LR schedule: {e}")

if schedules_plotted:
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, weight='bold')
    axes[1, 0].set_title('Learning Rate Schedules', fontsize=14, weight='bold')
    axes[1, 0].legend(fontsize=9, loc='best')
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No LR schedule data available', 
                    ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Learning Rate Schedules', fontsize=14, weight='bold')

# Plot 4: Improvement percentage
if len(results) > 1 and baseline_name and baseline_ppl:
    improvements = {}
    for name, data in results.items():
        if name != baseline_name:
            ppl = data.get('best_perplexity')
            if ppl and isinstance(ppl, (int, float)):
                improvement = ((baseline_ppl - ppl) / baseline_ppl) * 100
                improvements[name.split('_')[0][:15]] = improvement
    
    if improvements:
        names = list(improvements.keys())
        values = list(improvements.values())
        colors_imp = ['green' if v > 0 else 'red' for v in values]
        
        bars = axes[1, 1].barh(names, values, color=colors_imp, alpha=0.7)
        axes[1, 1].set_xlabel('Improvement over Baseline (%)', fontsize=12, weight='bold')
        axes[1, 1].set_title('Performance Improvements', fontsize=14, weight='bold')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, values):
            x_pos = val + (0.5 if val > 0 else -0.5)
            axes[1, 1].text(x_pos, bar.get_y() + bar.get_height()/2.,
                           f'{val:+.1f}%', ha='left' if val > 0 else 'right',
                           va='center', fontsize=10, weight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Only one experiment available', 
                        ha='center', va='center', fontsize=12, 
                        transform=axes[1, 1].transAxes)
else:
    axes[1, 1].text(0.5, 0.5, 'Need multiple experiments\nfor comparison', 
                    ha='center', va='center', fontsize=12, 
                    transform=axes[1, 1].transAxes)

axes[1, 1].set_title('Performance Improvements', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('experiments/master_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Master comparison plot saved to: experiments/master_comparison.png")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)