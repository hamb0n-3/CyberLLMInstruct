#!/usr/bin/env python3
"""
Validation utilities for training configurations and model compatibility.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate training configurations."""
    
    @staticmethod
    def validate_training_config(config: 'TrainingConfig') -> List[str]:
        """Validate training configuration and return list of issues."""
        issues = []
        
        # Model validation
        if not config.model_name:
            issues.append("model_name is required")
        
        # Training parameters
        if config.learning_rate <= 0:
            issues.append(f"learning_rate must be positive, got {config.learning_rate}")
        if config.num_epochs <= 0:
            issues.append(f"num_epochs must be positive, got {config.num_epochs}")
        if config.batch_size <= 0:
            issues.append(f"batch_size must be positive, got {config.batch_size}")
        
        # LoRA validation
        if config.use_lora:
            if config.lora_config.rank <= 0:
                issues.append(f"LoRA rank must be positive, got {config.lora_config.rank}")
            if config.lora_config.rank > 256:
                issues.append(f"LoRA rank {config.lora_config.rank} is unusually high, consider 8-64")
            if config.lora_config.alpha <= 0:
                issues.append(f"LoRA alpha must be positive, got {config.lora_config.alpha}")
            if not config.lora_config.target_modules:
                issues.append("LoRA target_modules cannot be empty")
        
        # Dataset validation
        if not config.dataset_config.paths:
            issues.append("No dataset paths specified")
        else:
            for path in config.dataset_config.paths:
                if not Path(path).exists():
                    issues.append(f"Dataset path does not exist: {path}")
        
        if config.dataset_config.max_length <= 0:
            issues.append(f"max_length must be positive, got {config.dataset_config.max_length}")
        
        # Memory optimization warnings
        if config.batch_size * config.dataset_config.max_length > 8192:
            issues.append(
                f"Large batch_size * max_length ({config.batch_size * config.dataset_config.max_length}) "
                "may cause memory issues"
            )
        
        # Output directory
        output_path = Path(config.output_dir)
        if output_path.exists() and not output_path.is_dir():
            issues.append(f"output_dir exists but is not a directory: {config.output_dir}")
        
        return issues
    
    @staticmethod
    def validate_model_tokenizer_compatibility(
        model_name: str, 
        tokenizer_name: Optional[str] = None
    ) -> List[str]:
        """Check if model and tokenizer are compatible."""
        issues = []
        
        # If no custom tokenizer, assume it matches the model
        if not tokenizer_name:
            return issues
        
        # Extract base model names for comparison
        model_base = model_name.lower().split('/')[-1].split('-')[0]
        tokenizer_base = tokenizer_name.lower().split('/')[-1].split('-')[0]
        
        # Common incompatibilities
        incompatible_pairs = [
            ('qwen', 'llama'),
            ('llama', 'qwen'),
            ('glm', 'qwen'),
            ('glm', 'llama'),
            ('mistral', 'qwen'),
        ]
        
        for model_type, tokenizer_type in incompatible_pairs:
            if model_type in model_base and tokenizer_type in tokenizer_base:
                issues.append(
                    f"Model ({model_name}) and tokenizer ({tokenizer_name}) "
                    f"appear to be incompatible"
                )
                break
        
        return issues
    
    @staticmethod
    def validate_checkpoint(checkpoint_path: str) -> List[str]:
        """Validate a checkpoint directory."""
        issues = []
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            issues.append(f"Checkpoint directory does not exist: {checkpoint_path}")
            return issues
        
        # Check required files
        required_files = {
            'LoRA checkpoint': 'lora_weights.safetensors',
            'Training state': 'training_state.pkl',
            'Config': ['adapter_config.json', 'training_config.json']  # Either one
        }
        
        for desc, files in required_files.items():
            if isinstance(files, list):
                # At least one should exist
                if not any((checkpoint_dir / f).exists() for f in files):
                    issues.append(f"Missing {desc}: none of {files} found")
            else:
                if not (checkpoint_dir / files).exists():
                    issues.append(f"Missing {desc}: {files}")
        
        return issues


def validate_and_fix_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix common configuration issues."""
    fixed_config = config.copy()
    
    # Fix common path issues
    if 'dataset_config' in fixed_config:
        dataset_config = fixed_config['dataset_config']
        
        # Ensure paths is a list
        if 'paths' in dataset_config and isinstance(dataset_config['paths'], str):
            dataset_config['paths'] = [dataset_config['paths']]
        
        # Fix relative paths
        if 'paths' in dataset_config:
            fixed_paths = []
            for path in dataset_config['paths']:
                # Expand environment variables
                path = os.path.expandvars(path)
                # Make relative paths absolute from training directory
                if not os.path.isabs(path):
                    path = str(Path(__file__).parent / path)
                fixed_paths.append(path)
            dataset_config['paths'] = fixed_paths
    
    # Ensure LoRA config has required fields
    if 'lora_config' in fixed_config and isinstance(fixed_config['lora_config'], dict):
        lora_config = fixed_config['lora_config']
        
        # Add default target modules if missing
        if 'target_modules' not in lora_config or not lora_config['target_modules']:
            lora_config['target_modules'] = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    return fixed_config


def report_validation_issues(issues: List[str], raise_on_error: bool = True):
    """Report validation issues to user."""
    if not issues:
        logger.info("✅ Configuration validation passed")
        return
    
    logger.error("❌ Configuration validation failed:")
    for i, issue in enumerate(issues, 1):
        logger.error(f"  {i}. {issue}")
    
    if raise_on_error:
        raise ValueError(f"Configuration has {len(issues)} validation issues")


# Import guard for circular dependency
import os
if __name__ != "__main__":
    from train_cybersecurity_model import TrainingConfig