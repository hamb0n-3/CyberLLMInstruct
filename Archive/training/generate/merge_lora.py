#!/usr/bin/env python3
"""
Script to merge LoRA adapters with base model for inference.
Supports exporting to various formats.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from transformers import AutoTokenizer

from train_cybersecurity_model import LoRAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model for MLX-LM LoRA layers."""
    
    def merge_recursive(module: nn.Module, prefix: str = ""):
        """Recursively find and merge LoRA layers."""
        for name, child in module.__dict__.items():
            if hasattr(child, 'lora_a') and hasattr(child, 'lora_b'):
                # This is a LoRA layer from MLX-LM
                # Merge weights: W = W + (B @ A) * scale
                if hasattr(child, 'scale'):
                    scale = child.scale
                else:
                    # Calculate scale from alpha/rank if available
                    scale = getattr(child, 'alpha', 16) / getattr(child, 'rank', 16)
                
                # Perform the merge
                merged_weight = child.weight + (child.lora_b @ child.lora_a) * scale
                
                # Create a new Linear layer with merged weights
                merged_layer = nn.Linear(
                    child.in_proj if hasattr(child, 'in_proj') else child.weight.shape[1],
                    child.out_proj if hasattr(child, 'out_proj') else child.weight.shape[0],
                    bias=hasattr(child, 'bias') and child.bias is not None
                )
                merged_layer.weight = merged_weight
                if hasattr(child, 'bias') and child.bias is not None:
                    merged_layer.bias = child.bias
                
                # Replace the LoRA layer with the merged layer
                setattr(module, name, merged_layer)
                logger.info(f"Merged LoRA weights for {prefix}.{name}")
                
            elif isinstance(child, nn.Module):
                # Recursively process child modules
                merge_recursive(child, f"{prefix}.{name}" if prefix else name)
    
    merge_recursive(model)
    return model


def export_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: str,
    export_format: str = "mlx",
    quantize: Optional[str] = None,
    adapter_info: Optional[Dict[str, Any]] = None
):
    """Export merged model in specified format."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if export_format == "mlx":
        # Save in MLX format
        model.save_weights(str(output_path / "model.safetensors"))
        
        # Save config
        config = {
            "model_type": "merged_lora",
            "quantization": quantize
        }
        
        # If we have adapter info, create a merged model metadata file
        if adapter_info:
            merged_info = {
                "model_type": "merged_adapter",
                "original_adapter": adapter_info,
                "merge_timestamp": datetime.now().isoformat(),
                "quantization": quantize,
                "notes": f"Model created by merging {adapter_info['adapter_name']} into base model"
            }
            with open(output_path / "merged_model_info.json", 'w') as f:
                json.dump(merged_info, f, indent=2)
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"Saved merged model in MLX format to {output_path}")
        
    elif export_format == "gguf":
        # Convert to GGUF format (requires additional tools)
        logger.warning("GGUF export not yet implemented. Use mlx_lm convert tool.")
        
    elif export_format == "pytorch":
        # Convert to PyTorch format
        logger.warning("PyTorch export not yet implemented.")
        
    else:
        raise ValueError(f"Unknown export format: {export_format}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="mlx",
        choices=["mlx", "gguf", "pytorch"],
        help="Export format"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        help="Quantization to apply"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading"
    )
    
    args = parser.parse_args()
    
    # Load base model
    logger.info(f"Loading base model: {args.base_model}")
    model, tokenizer = load(
        args.base_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code}
    )
    
    # Check if checkpoint has adapter config or training config
    checkpoint_path = Path(args.lora_checkpoint)
    adapter_config_path = checkpoint_path / "adapter_config.json"
    training_config_path = checkpoint_path / "training_config.json"
    
    # Load configuration
    adapter_info = None
    if adapter_config_path.exists():
        # New format with adapter config
        with open(adapter_config_path, 'r') as f:
            adapter_info = json.load(f)
        logger.info(f"Found adapter config: {adapter_info['adapter_name']}")
        logger.info(f"Adapter trained on: {adapter_info['trained_on']['base_model']}")
        
        # Check if base model matches
        if args.base_model != adapter_info['trained_on']['base_model']:
            logger.warning(
                f"Base model mismatch: Adapter trained on {adapter_info['trained_on']['base_model']}, "
                f"but merging with {args.base_model}. Results may vary!"
            )
        
        lora_config_dict = adapter_info['lora_config']
        
    elif training_config_path.exists():
        # Old format
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        logger.info(f"Found training config from checkpoint")
        lora_config_dict = training_config.get('lora_config', {})
    else:
        raise FileNotFoundError("No adapter_config.json or training_config.json found in checkpoint")
    
    # Apply LoRA configuration to model
    lora_config = LoRAConfig(**lora_config_dict)
    
    # Apply LoRA layers (same as training)
    from train_cybersecurity_model import apply_lora_to_model
    model = apply_lora_to_model(model, lora_config)
    
    # Load LoRA weights
    lora_weights_path = checkpoint_path / "lora_weights.safetensors"
    if not lora_weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}")
    
    # Load LoRA weights into model
    lora_state = mx.load(str(lora_weights_path))
    
    # Apply weights to LoRA layers (MLX-LM style)
    lora_layer_idx = 0
    def apply_lora_weights(module: nn.Module):
        nonlocal lora_layer_idx
        for name, child in module.__dict__.items():
            if hasattr(child, 'lora_a') and hasattr(child, 'lora_b'):
                # Found a LoRA layer
                if f'param_{lora_layer_idx}' in lora_state:
                    child.lora_a = lora_state[f'param_{lora_layer_idx}']
                    child.lora_b = lora_state[f'param_{lora_layer_idx + 1}']
                    lora_layer_idx += 2
            elif isinstance(child, nn.Module):
                apply_lora_weights(child)
    
    apply_lora_weights(model)
    logger.info(f"Applied {lora_layer_idx // 2} LoRA weight pairs")
    
    # Merge LoRA weights
    logger.info("Merging LoRA weights into base model...")
    merged_model = merge_lora_weights(model)
    
    # Export merged model
    export_merged_model(
        merged_model,
        tokenizer,
        args.output_dir,
        args.export_format,
        args.quantize,
        adapter_info=adapter_info
    )
    
    logger.info("Model merging completed successfully!")


if __name__ == "__main__":
    main()