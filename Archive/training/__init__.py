"""
Cybersecurity LLM Training Suite

A comprehensive framework for fine-tuning Large Language Models on cybersecurity
instruction datasets using Apple's MLX framework with LoRA support.
"""

from .data_loader import (
    DatasetConfig,
    CyberSecDataset,
    DataCollator,
    MultiDatasetSampler,
    create_dataloaders,
    load_cybersec_datasets
)

from .train_cybersecurity_model import (
    LoRAConfig,
    TrainingConfig,
    LoRALinear,
    CyberSecTrainer,
    apply_lora_to_model,
    get_lora_parameters,
    load_config
)

from .evaluate_model import (
    EvaluationConfig,
    CyberSecEvaluator
)

__version__ = "1.0.0"
__author__ = "CyberLLMInstruct"

__all__ = [
    # Data loading
    "DatasetConfig",
    "CyberSecDataset", 
    "DataCollator",
    "MultiDatasetSampler",
    "create_dataloaders",
    "load_cybersec_datasets",
    
    # Training
    "LoRAConfig",
    "TrainingConfig",
    "LoRALinear",
    "CyberSecTrainer",
    "apply_lora_to_model",
    "get_lora_parameters",
    "load_config",
    
    # Evaluation
    "EvaluationConfig",
    "CyberSecEvaluator"
]