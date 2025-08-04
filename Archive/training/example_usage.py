#!/usr/bin/env python3
"""
Example usage of the cybersecurity LLM training framework.
Demonstrates common training scenarios and best practices.
"""

import os
import json
from pathlib import Path

def example_basic_training():
    """Basic training example with minimal configuration."""
    print("=== Basic Training Example ===")
    print("Train a small model on cybersecurity data:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --data-path ../dataset_creation/structured_data \\
    --model mlx-community/Qwen2.5-1B-Instruct-MLX-4bit \\
    --output-dir ./outputs/basic_model \\
    --num-epochs 1 \\
    --batch-size 2 \\
    --learning-rate 2e-4 \\
    --eval-steps 50 \\
    --save-steps 100
    """
    print(cmd)
    print("\nThis will train for 1 epoch with frequent evaluation.")


def example_advanced_lora():
    """Advanced LoRA training with custom parameters."""
    print("\n=== Advanced LoRA Training ===")
    print("Fine-tune with optimized LoRA settings:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --data-path ../dataset_creation/structured_data \\
    --model mlx-community/Qwen2.5-3B-Instruct-MLX-4bit \\
    --output-dir ./outputs/advanced_lora \\
    --use-lora \\
    --lora-rank 32 \\
    --lora-alpha 64 \\
    --lora-dropout 0.1 \\
    --num-epochs 3 \\
    --batch-size 4 \\
    --gradient-accumulation-steps 4 \\
    --learning-rate 1e-4 \\
    --warmup-steps 200 \\
    --lr-scheduler cosine \\
    --eval-steps 100 \\
    --use-tensorboard
    """
    print(cmd)
    print("\nHigher rank LoRA for better capacity, with warmup and cosine schedule.")


def example_filtered_training():
    """Training with filtered data by domain."""
    print("\n=== Domain-Filtered Training ===")
    print("Train only on specific cybersecurity domains:\n")
    
    # Create a temporary config
    config = {
        "model_name": "mlx-community/Llama-3.2-1B-Instruct-MLX-4bit",
        "use_lora": True,
        "lora_config": {
            "rank": 16,
            "alpha": 32
        },
        "dataset_config": {
            "paths": ["../dataset_creation/structured_data"],
            "format": "cybersec",
            "filter_by_type": ["vulnerability", "attack_pattern"],
            "min_response_length": 100,
            "system_prompt": "You are a vulnerability analysis expert."
        },
        "num_epochs": 2,
        "output_dir": "./outputs/vuln_specialist"
    }
    
    # Save config
    config_path = "configs/vuln_training.json"
    print(f"Creating config at {config_path}:")
    print(json.dumps(config, indent=2)[:500] + "...")
    
    print(f"\nThen run:")
    print(f"python train_cybersecurity_model.py --config {config_path}")


def example_multi_dataset():
    """Multi-dataset training example."""
    print("\n=== Multi-Dataset Training ===")
    print("Combine cybersecurity data with general instruction data:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --config configs/multi_dataset_config.yaml
    """
    print(cmd)
    print("\nThis mixes 60% cybersec, 30% general, 10% code data.")
    print("Edit configs/multi_dataset_config.yaml to adjust paths and ratios.")


def example_resume_training():
    """Resume interrupted training."""
    print("\n=== Resume Training ===")
    print("Continue from a checkpoint after interruption:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --config configs/cybersec_lora_config.yaml \\
    --resume-from-checkpoint ./outputs/my_model/checkpoint-500
    """
    print(cmd)
    print("\nTraining state, optimizer state, and metrics are restored.")


def example_evaluation():
    """Comprehensive model evaluation."""
    print("\n=== Model Evaluation ===")
    print("Evaluate a trained model:\n")
    
    cmd = """# Full evaluation
python evaluate_model.py \\
    --model ./outputs/my_model/checkpoint-best \\
    --data-path ../dataset_creation/structured_data \\
    --max-samples 500 \\
    --output detailed_eval.json

# Quick generation test
python evaluate_model.py \\
    --model ./outputs/my_model/checkpoint-best \\
    --data-path ../dataset_creation/structured_data \\
    --skip-perplexity \\
    --skip-domain \\
    --max-samples 20 \\
    --output quick_eval.json
    """
    print(cmd)


def example_model_merging():
    """Merge LoRA weights for deployment."""
    print("\n=== Model Merging ===")
    print("Merge LoRA adapters into base model:\n")
    
    cmd = """python merge_lora.py \\
    --base-model mlx-community/Qwen2.5-3B-Instruct-MLX-4bit \\
    --lora-checkpoint ./outputs/my_model/checkpoint-best \\
    --output-dir ./deployed_model
    """
    print(cmd)
    print("\nThe merged model can be loaded directly without LoRA setup.")


def example_preview_mode():
    """Preview data formatting before training."""
    print("\n=== Preview Mode ===")
    print("Preview how your data will be formatted and tokenized:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --data-path ../dataset_creation/structured_data \\
    --model mlx-community/Qwen2.5-3B-Instruct-MLX-4bit \\
    --output-dir ./outputs/test_model \\
    --preview \\
    --preview-samples 10
    """
    print(cmd)
    print("\nShows:")
    print("- How chat templates are applied")
    print("- Token counts and masking")
    print("- Data distribution statistics")
    print("- Memory and time estimates")


def example_custom_generation():
    """Custom generation during training."""
    print("\n=== Custom Generation Settings ===")
    print("Fine-tune generation behavior during training:\n")
    
    cmd = """python train_cybersecurity_model.py \\
    --config configs/cybersec_lora_config.yaml \\
    --generation-temperature 0.8 \\
    --generation-top-p 0.95 \\
    --generation-max-length 1024
    """
    print(cmd)
    print("\nAffects sample generation during training for monitoring.")


def show_monitoring():
    """Show how to monitor training."""
    print("\n=== Training Monitoring ===")
    print("Monitor training progress:\n")
    
    print("1. TensorBoard:")
    print("   tensorboard --logdir ./outputs/my_model/tensorboard")
    print("   # Then open http://localhost:6006")
    
    print("\n2. Weights & Biases:")
    print("   wandb login  # First time only")
    print("   python train_cybersecurity_model.py --config config.yaml --use-wandb")
    
    print("\n3. Training logs:")
    print("   tail -f ./outputs/my_model/training_*.log")
    
    print("\n4. Live metrics:")
    print("   watch -n 5 'tail -20 ./outputs/my_model/training_*.log | grep -E \"loss|eval\"'")


def show_tips():
    """Show optimization tips."""
    print("\n=== Optimization Tips ===")
    
    tips = [
        "Memory Management:",
        "  - Start with batch_size=1, increase gradually",
        "  - Use gradient_accumulation_steps for effective larger batches",
        "  - Reduce max_length if running out of memory",
        "",
        "Training Speed:",
        "  - Larger batch sizes are more efficient (if they fit)",
        "  - Reduce logging_steps for less overhead",
        "  - Use --num-workers 2 for data loading (if supported)",
        "",
        "Model Quality:",
        "  - Use warmup_steps = 5-10% of total steps",
        "  - Try learning rates: 5e-5 to 5e-4 for LoRA",
        "  - Increase LoRA rank for more capacity (8->16->32)",
        "  - Filter data by quality score if available",
        "",
        "Debugging:",
        "  - Use --max-samples 100 for quick tests",
        "  - Set --eval-steps 10 to catch issues early",
        "  - Check ./outputs/*/loss_curve.png for training dynamics"
    ]
    
    print("\n".join(tips))


def main():
    """Run all examples."""
    print("="*80)
    print("CYBERSECURITY LLM TRAINING EXAMPLES")
    print("="*80)
    
    example_basic_training()
    example_advanced_lora()
    example_filtered_training()
    example_multi_dataset()
    example_resume_training()
    example_evaluation()
    example_model_merging()
    example_preview_mode()
    example_custom_generation()
    show_monitoring()
    show_tips()
    
    print("\n" + "="*80)
    print("For production training, always:")
    print("1. Test on small data first")
    print("2. Monitor GPU memory usage") 
    print("3. Save configs for reproducibility")
    print("4. Keep evaluation data separate")
    print("5. Document model versions and datasets")
    print("="*80)


if __name__ == "__main__":
    main()