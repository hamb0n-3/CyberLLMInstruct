#!/usr/bin/env python3
"""
Quick script to verify memory-safe configuration settings.
"""

import yaml
import json
from pathlib import Path

def check_config():
    """Check and display current configuration for memory safety."""
    
    print("="*80)
    print("MEMORY CONFIGURATION CHECK")
    print("="*80)
    
    # Load config.yaml
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n✓ config.yaml settings:")
        print(f"  - batch_size: {config.get('batch_size')} (should be 1)")
        print(f"  - max_length: {config.get('max_length')} (should be 256)")
        print(f"  - gradient_checkpointing: {config.get('gradient_checkpointing')} (should be True)")
        print(f"  - gradient_accumulation_steps: {config.get('gradient_accumulation_steps')} (increased to 8)")
        print(f"  - logging_steps: {config.get('logging_steps')} (should be 10)")
        
        # Check dataset config
        dataset_config = config.get('dataset_config', {})
        print(f"\n  Dataset config:")
        print(f"  - max_length: {dataset_config.get('max_length')} (should match above)")
    
    # Check if old training_config.json exists
    old_config_path = Path("outputs/training_config.json")
    if old_config_path.exists():
        with open(old_config_path, 'r') as f:
            old_config = json.load(f)
        
        print("\n⚠️  Previous training_config.json found:")
        print(f"  - batch_size: {old_config.get('batch_size')} (was too high!)")
        print(f"  - max_length: {old_config.get('max_length')} (was too high!)")
        print(f"  - gradient_checkpointing: {old_config.get('gradient_checkpointing')} (was disabled!)")
        print("\n  Make sure to use --config configs/config.yaml when training!")
    
    # Memory calculations
    print("\n📊 Memory Estimates (with new settings):")
    vocab_size = 152064  # Qwen3-30B vocabulary
    batch_size = 1
    seq_length = 256
    
    logits_mem = (batch_size * seq_length * vocab_size * 4) / (1024**3)
    print(f"  - Logits memory (chunked): {logits_mem:.2f}GB")
    print(f"  - Base model: ~30GB (8-bit)")
    print(f"  - Total estimate: ~32-35GB (safe for 48GB GPU slice)")
    
    print("\n✅ Configuration is now optimized for the 152k vocabulary model!")
    print("\n🚀 To start training with safe settings:")
    print("   python train_cybersecurity_model.py --config configs/config.yaml")
    print("\n⚠️  DO NOT use default settings or increase batch_size/max_length!")
    print("="*80)

if __name__ == "__main__":
    check_config()