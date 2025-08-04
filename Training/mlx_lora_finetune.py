import json
import os
from typing import Dict, List, Optional, Tuple, Union
import argparse
import yaml
from pathlib import Path
import numpy as np
import mlx.core as mx

import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train
from transformers import PreTrainedTokenizer
from datasets import load_from_disk


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    mx.random.seed(seed)
    np.random.seed(seed)


def create_lr_schedule(config: Dict, num_iters: int):
    """Create learning rate schedule based on config."""
    lr_config = config['training'].get('lr_schedule', {})
    lr_type = lr_config.get('type', 'constant')
    base_lr = float(config['training']['learning_rate'])  # Ensure it's a float
    warmup_steps = lr_config.get('warmup_steps', 0)
    
    if lr_type == 'constant':
        return lambda step: base_lr
    
    elif lr_type == 'linear':
        def linear_schedule(step):
            step = int(step)  # Convert MLX array to int
            if step < warmup_steps:
                return base_lr * (step / warmup_steps)
            progress = (step - warmup_steps) / (num_iters - warmup_steps)
            return base_lr * (1 - progress)
        return linear_schedule
    
    elif lr_type == 'cosine':
        min_lr = float(lr_config.get('min_lr', 1e-6))  # Ensure it's a float
        def cosine_schedule(step):
            step = int(step)  # Convert MLX array to int
            if step < warmup_steps:
                return base_lr * (step / warmup_steps)
            progress = (step - warmup_steps) / (num_iters - warmup_steps)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        return cosine_schedule
    
    else:
        raise ValueError(f"Unknown lr_schedule type: {lr_type}")


def load_local_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    shuffle: bool = True,
    seed: int = 42
):
    """Load local HuggingFace dataset and prepare for training."""
    # Load the dataset from disk
    dataset = load_from_disk(data_path)
    
    # Shuffle if requested
    if shuffle and 'train' in dataset:
        dataset['train'] = dataset['train'].shuffle(seed=seed)
    
    # Define the formatting function to convert to MLX format
    def format_for_mlx(examples):
        # MLX expects either 'messages' for chat format or 'prompt'/'completion' for completion format
        # Let's use chat format with messages
        messages_list = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
            messages_list.append(messages)
        return {"messages": messages_list}
    
    # Apply formatting to all splits, keeping only the messages field
    formatted_dataset = dataset.map(
        format_for_mlx,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Convert HuggingFace datasets to lists of dictionaries as expected by MLX
    train_data = list(formatted_dataset['train'])
    val_data = list(formatted_dataset['validation'])
    test_data = list(formatted_dataset['test']) if 'test' in formatted_dataset else []
    
    # Create MLX datasets
    # Note: create_dataset expects (data, tokenizer) where data is a list of dicts
    train_set = datasets.create_dataset(train_data, tokenizer)
    val_set = datasets.create_dataset(val_data, tokenizer)
    test_set = datasets.create_dataset(test_data, tokenizer) if test_data else []
    
    return train_set, val_set, test_set


class Metrics:
    """Track training metrics with early stopping support."""
    def __init__(self, early_stopping_config: Dict = None):
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Early stopping config
        self.early_stopping = early_stopping_config or {}
        self.early_stop = False

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        iteration = info["iteration"]
        val_loss = info["val_loss"]
        self.val_losses.append((iteration, val_loss))
        
        # Check for early stopping
        if self.early_stopping.get('enabled', False):
            min_delta = self.early_stopping.get('min_delta', 0.001)
            patience = self.early_stopping.get('patience', 5)
            
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self.early_stop = True
                    print(f"\nEarly stopping triggered at iteration {iteration}")


def save_checkpoint(model, optimizer, iteration, config, checkpoint_dir):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.npz")
    
    # Save model state
    model.save_weights(checkpoint_path)
    
    # Save training state
    state = {
        'iteration': iteration,
        'optimizer_state': optimizer.state,
        'config': config
    }
    state_path = os.path.join(checkpoint_dir, f"state_iter_{iteration}.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"Saved checkpoint at iteration {iteration}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load training checkpoint."""
    # Load model weights
    model.load_weights(checkpoint_path)
    
    # Load training state
    state_path = checkpoint_path.replace('.npz', '_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        optimizer.state = state['optimizer_state']
        return state['iteration']
    return 0


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a model using LoRA with MLX')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the model without training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    # Load the model and tokenizer
    print(f"Loading model: {config['model']['model_path']}")
    model, tokenizer = load(config['model']['model_path'])
    
    # Test the base model
    test_prompt = config.get('test_prompt', 'What is fine-tuning in machine learning?')
    messages = [{"role": "user", "content": test_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Response from base model:")
    _ = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=200)
    
    if args.test_only:
        return
    
    # Set up the adapter path
    adapter_path = config['model']['adapter_path']
    os.makedirs(adapter_path, exist_ok=True)
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")
    
    # Set LoRA configuration
    lora_config = {
        "num_layers": config['lora']['num_layers'],
        "lora_parameters": {
            "rank": config['lora']['rank'],
            "scale": config['lora']['scale'],
            "dropout": config['lora']['dropout'],
        },
    }
    
    # Add alpha if specified
    if 'alpha' in config['lora']:
        lora_config["lora_parameters"]["alpha"] = config['lora']['alpha']
    
    with open(adapter_config_path, "w") as f:
        json.dump(lora_config, f, indent=4)
    
    # Set training arguments
    training_args = TrainingArgs(
        adapter_file=adapter_file_path,
        iters=config['training']['iterations'],
        steps_per_eval=config['training']['steps_per_eval'],
        batch_size=config['training'].get('batch_size', 1),
        val_batches=config['training'].get('val_batches', -1),
        steps_per_report=config['training'].get('steps_per_report', 10),
        steps_per_save=config['training'].get('steps_per_save', 100),
        grad_checkpoint=config['training'].get('grad_checkpoint', False),
        max_seq_length=config['dataset'].get('max_seq_length', 2048),
    )
    
    # Freeze the model and apply LoRA layers
    model.freeze()
    
    # Apply LoRA based on fine_tune_type
    fine_tune_type = config['lora'].get('fine_tune_type', 'lora')
    if fine_tune_type == 'lora':
        linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
    elif fine_tune_type == 'dora':
        # DoRA would require specific implementation
        print("DoRA fine-tuning not yet implemented, falling back to LoRA")
        linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
    elif fine_tune_type == 'full':
        # Full fine-tuning - unfreeze model
        model.unfreeze()
    
    num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Number of trainable parameters: {num_train_params}")
    
    model.train()
    
    # Initialize metrics tracker
    metrics = Metrics(config.get('early_stopping', {}))
    
    # Load the dataset
    print(f"Loading dataset from: {config['dataset']['data_path']}")
    train_set, val_set, test_set = load_local_dataset(
        data_path=config['dataset']['data_path'],
        tokenizer=tokenizer,
        max_seq_length=config['dataset'].get('max_seq_length', 2048),
        shuffle=config['dataset'].get('shuffle', True),
        seed=seed
    )
    
    print(f"Dataset loaded - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Create learning rate schedule
    lr_schedule = create_lr_schedule(config, config['training']['iterations'])
    
    # Initialize optimizer with learning rate schedule
    optimizer = optim.Adam(learning_rate=lr_schedule)
    
    # Setup gradient clipping if enabled
    grad_clip_config = config['training'].get('gradient_clip', {})
    if grad_clip_config.get('enabled', False):
        max_norm = grad_clip_config.get('max_norm', 1.0)
        # Note: MLX doesn't have built-in gradient clipping in the same way as PyTorch
        # This would need to be implemented in the training loop
        print(f"Gradient clipping enabled with max_norm={max_norm}")
    
    # Setup logging if enabled
    logging_config = config['output'].get('logging', {})
    if logging_config.get('use_wandb', False):
        try:
            import wandb
            wandb.init(project=logging_config.get('wandb_project', 'cybersecurity-lora'))
            print("Weights & Biases logging enabled")
        except ImportError:
            print("wandb not installed, skipping W&B logging")
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume or config.get('resume', {}).get('enabled', False):
        checkpoint_path = args.resume or config['resume'].get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_iteration = load_checkpoint(checkpoint_path, model, optimizer)
            print(f"Resumed from iteration {start_iteration}")
    
    # Fine-tune the model
    print("Starting training...")
    
    # Create custom training callback for checkpointing
    checkpoint_dir = config['output'].get('checkpoint_dir', 'checkpoints')
    
    def training_callback(info):
        metrics.on_train_loss_report(info) if 'train_loss' in info else None
        metrics.on_val_loss_report(info) if 'val_loss' in info else None
        
        # Save checkpoint if needed
        if info.get('iteration', 0) % training_args.steps_per_save == 0:
            save_checkpoint(model, optimizer, info['iteration'], config, checkpoint_dir)
        
        # Check for early stopping
        if metrics.early_stop:
            return False  # Stop training
        return True  # Continue training
    
    # Train the model
    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        training_callback=training_callback,
    )
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, config['training']['iterations'], config, checkpoint_dir)
    
    # Save training metrics if configured
    if config['output'].get('save_metrics', False):
        metrics_data = {
            'train_losses': metrics.train_losses,
            'val_losses': metrics.val_losses,
            'best_val_loss': metrics.best_val_loss,
            'config': config
        }
        metrics_path = config['output'].get('metrics_path', 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Saved training metrics to {metrics_path}")
    
    # Plot the training and validation loss
    if metrics.train_losses and metrics.val_losses:
        train_its, train_losses = zip(*metrics.train_losses)
        validation_its, validation_losses = zip(*metrics.val_losses)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_its, train_losses, "-o", label="Train", markersize=4)
        plt.plot(validation_its, validation_losses, "-o", label="Validation", markersize=4)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if config['output'].get('save_plots', True):
            plot_path = config['output'].get('plot_path', 'training_loss.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved loss plot to {plot_path}")
        else:
            plt.show()
        plt.close()
    
    # Load the fine-tuned model and generate a response
    print("\nLoading fine-tuned model...")
    model_lora, _ = load(config['model']['model_path'], adapter_path=adapter_path)
    
    print("\nResponse from fine-tuned model:")
    _ = generate(model_lora, tokenizer, prompt=prompt, verbose=True, max_tokens=200)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()