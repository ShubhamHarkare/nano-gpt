import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"experiments/{experiment_name}_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rates': [],
            'iterations': []
        }
        
    def log_metrics(self, iteration, train_loss, val_loss, lr):
        """Log metrics for each iteration"""
        self.metrics['iterations'].append(iteration)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['perplexity'].append(2.71828 ** val_loss)  # e^loss
        self.metrics['learning_rates'].append(lr)
        
    def save_metrics(self):
        """Save metrics to JSON"""
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def plot_training_curves(self):
        """Generate all training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['iterations'], self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['iterations'], self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Perplexity
        axes[0, 1].plot(self.metrics['iterations'], self.metrics['perplexity'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Validation Perplexity')
        axes[0, 1].grid(True)
        
        # Learning rate schedule
        axes[1, 0].plot(self.metrics['iterations'], self.metrics['learning_rates'])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        
        # Loss histogram
        axes[1, 1].hist(self.metrics['val_loss'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Validation Loss Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_curves.png", dpi=300)
        print(f"Plots saved to {self.log_dir}/training_curves.png")
        
    def save_final_results(self, model_params, training_config):
        """Save final experiment summary"""
        results = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_val_loss': self.metrics['val_loss'][-1],
            'final_perplexity': self.metrics['perplexity'][-1],
            'best_val_loss': min(self.metrics['val_loss']),
            'best_perplexity': min(self.metrics['perplexity']),
            'total_iterations': len(self.metrics['iterations']),
            'model_params': model_params,
            'training_config': training_config
        }
        
        with open(f"{self.log_dir}/results_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results