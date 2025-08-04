#!/usr/bin/env python3
"""
Utility to inspect model structure and identify targetable modules for LoRA.
"""

import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from typing import Dict, List, Tuple


def inspect_model_structure(model: nn.Module, max_depth: int = 5) -> Dict[str, List[str]]:
    """Inspect model structure and categorize modules."""
    linear_modules = []
    attention_modules = []
    mlp_modules = []
    other_modules = []
    all_modules = []
    
    def traverse(module: nn.Module, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
            
        # Check module type
        module_type = type(module).__name__
        full_name = f"{prefix} ({module_type})" if prefix else module_type
        all_modules.append(full_name)
        
        # Check if it's a Linear layer
        if isinstance(module, nn.Linear):
            linear_modules.append(prefix)
            
            # Categorize by name
            if any(key in prefix.lower() for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'query', 'key', 'value']):
                attention_modules.append(prefix)
            elif any(key in prefix.lower() for key in ['gate', 'up', 'down', 'mlp', 'fc']):
                mlp_modules.append(prefix)
        else:
            if prefix:  # Don't add root module
                other_modules.append(f"{prefix} ({module_type})")
        
        # Traverse children
        if hasattr(module, 'children'):
            for name, child in module.children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                traverse(child, child_prefix, depth + 1)
    
    traverse(model)
    
    return {
        'linear': linear_modules,
        'attention': attention_modules,
        'mlp': mlp_modules,
        'other': other_modules[:20],  # Limit other modules
        'all': all_modules[:50]  # Show first 50 modules
    }


def print_model_info(model_name: str):
    """Load model and print structure information."""
    print(f"\n{'='*60}")
    print(f"Inspecting Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load(model_name)
    
    # Get model info
    print(f"\nModel Type: {type(model).__name__}")
    
    # Count parameters
    try:
        total_params = sum(p.size for p in model.parameters().values() if hasattr(p, 'size'))
        print(f"Total Parameters: {total_params:,}")
    except Exception as e:
        print(f"Could not count parameters: {e}")
    
    # Inspect structure
    structure = inspect_model_structure(model)
    
    print(f"\n{'='*40}")
    print("LINEAR LAYERS (LoRA targetable):")
    print(f"{'='*40}")
    
    if structure['linear']:
        for i, module in enumerate(structure['linear'][:30]):  # Show first 30
            print(f"{i+1:3d}. {module}")
        if len(structure['linear']) > 30:
            print(f"... and {len(structure['linear']) - 30} more")
    else:
        print("No linear layers found!")
    
    print(f"\n{'='*40}")
    print("ATTENTION MODULES:")
    print(f"{'='*40}")
    if structure['attention']:
        for module in structure['attention'][:10]:
            print(f"  - {module}")
    else:
        print("No attention modules identified")
    
    print(f"\n{'='*40}")
    print("MLP/FFN MODULES:")
    print(f"{'='*40}")
    if structure['mlp']:
        for module in structure['mlp'][:10]:
            print(f"  - {module}")
    else:
        print("No MLP modules identified")
    
    # Suggest target modules
    print(f"\n{'='*40}")
    print("SUGGESTED TARGET MODULES FOR LORA:")
    print(f"{'='*40}")
    
    # Extract common patterns
    patterns = set()
    for module in structure['linear']:
        parts = module.split('.')
        if len(parts) > 1:
            # Get the last part (e.g., q_proj, gate_proj)
            last_part = parts[-1]
            if any(key in last_part for key in ['proj', 'gate', 'up', 'down', 'fc']):
                patterns.add(last_part)
    
    if patterns:
        print("target_modules:")
        for pattern in sorted(patterns):
            print(f'  - "{pattern}"')
    else:
        print("Could not determine target modules automatically")
        print("Try using the full module paths from the linear layers list")
    
    # Show all modules found
    print(f"\n{'='*40}")
    print("ALL MODULES FOUND (first 50):")
    print(f"{'='*40}")
    if structure.get('all'):
        for i, module in enumerate(structure['all']):
            print(f"{i+1:3d}. {module}")
    else:
        print("No modules found at all!")
    
    return structure


def test_lora_application(model_name: str, target_modules: List[str]):
    """Test if LoRA can be applied with given target modules."""
    print(f"\n{'='*60}")
    print(f"Testing LoRA Application")
    print(f"{'='*60}\n")
    
    print(f"Target modules: {target_modules}")
    
    # Load model
    model, _ = load(model_name)
    
    # Try to apply LoRA
    from mlx_lm.tuner.utils import linear_to_lora_layers
    
    lora_config = {
        "rank": 16,
        "scale": 1.0,
        "dropout": 0.0,
        "keys": target_modules
    }
    
    print("\nApplying LoRA...")
    linear_to_lora_layers(model, num_layers=-1, config=lora_config)
    
    # Count LoRA layers
    lora_count = 0
    lora_layers = []
    
    def count_lora(module: nn.Module, prefix: str = ""):
        nonlocal lora_count
        if hasattr(module, 'lora_a'):
            lora_count += 1
            lora_layers.append(prefix)
        
        if hasattr(module, 'children'):
            for name, child in module.children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                count_lora(child, child_prefix)
    
    count_lora(model)
    
    print(f"\nLoRA layers created: {lora_count}")
    if lora_layers:
        print("\nLoRA applied to:")
        for layer in lora_layers[:10]:
            print(f"  - {layer}")
        if len(lora_layers) > 10:
            print(f"  ... and {len(lora_layers) - 10} more")
    else:
        print("\n❌ No LoRA layers created! Check target module names.")
    
    return lora_count > 0


def main():
    parser = argparse.ArgumentParser(description="Inspect model structure for LoRA targeting")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-MLX-4bit",
        help="Model to inspect"
    )
    parser.add_argument(
        "--test-lora",
        action="store_true",
        help="Test LoRA application"
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Target modules for LoRA test"
    )
    
    args = parser.parse_args()
    
    # Inspect model
    structure = print_model_info(args.model)
    
    # Test LoRA if requested
    if args.test_lora:
        test_lora_application(args.model, args.target_modules)


if __name__ == "__main__":
    main()