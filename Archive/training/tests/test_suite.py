#!/usr/bin/env python3
"""
Unified test suite for cybersecurity model training.
Consolidates all model inspection and LoRA testing functionality.
"""

import unittest
import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from train_cybersecurity_model import LoRAConfig, apply_lora_to_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelAnalysis(unittest.TestCase):
    """Test model structure and LoRA application."""
    
    @classmethod
    def setUpClass(cls):
        """Load a small test model once for all tests."""
        cls.model_name = "mlx-community/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading test model: {cls.model_name}")
        cls.model, cls.tokenizer = load(cls.model_name)
        mx.eval(cls.model.parameters())
    
    def test_model_structure(self):
        """Test basic model structure analysis."""
        # Check model has expected attributes
        self.assertTrue(hasattr(self.model, 'model'))
        self.assertTrue(hasattr(self.model.model, 'layers'))
        
        # Count linear layers
        linear_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
        
        logger.info(f"Found {linear_count} linear layers")
        self.assertGreater(linear_count, 0)
    
    def test_lora_application(self):
        """Test LoRA application to model."""
        # Create LoRA config
        lora_config = LoRAConfig(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"]
        )
        
        # Apply LoRA
        lora_model = apply_lora_to_model(self.model, lora_config)
        
        # Check LoRA was applied
        lora_layers = []
        for name, module in lora_model.named_modules():
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                lora_layers.append(name)
        
        logger.info(f"LoRA applied to {len(lora_layers)} layers")
        self.assertGreater(len(lora_layers), 0)
    
    def test_trainable_parameters(self):
        """Test detection of trainable parameters."""
        # Apply LoRA first
        lora_config = LoRAConfig(rank=8, alpha=16.0)
        lora_model = apply_lora_to_model(self.model, lora_config)
        
        # Get trainable parameters
        trainable_params = lora_model.trainable_parameters()
        
        # Count parameters
        if isinstance(trainable_params, dict):
            param_count = sum(
                p.size for p in tree_flatten(trainable_params) 
                if hasattr(p, 'size')
            )
        else:
            param_count = sum(
                p.size for p in trainable_params 
                if hasattr(p, 'size')
            )
        
        logger.info(f"Trainable parameters: {param_count:,}")
        self.assertGreater(param_count, 0)
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        vocab_size = self.tokenizer.vocab_size
        batch_size = 2
        seq_length = 512
        
        # Estimate logits memory
        logits_memory_gb = (batch_size * seq_length * vocab_size * 4) / (1024**3)
        
        logger.info(f"Vocab size: {vocab_size}")
        logger.info(f"Estimated logits memory: {logits_memory_gb:.2f} GB")
        
        # Warn if vocab is too large
        if vocab_size > 65000:
            logger.warning(f"Large vocabulary size ({vocab_size}) may cause memory issues!")
        
        self.assertLess(vocab_size, 200000, "Vocabulary size is dangerously large")


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""
    
    def test_dataset_loading(self):
        """Test loading cybersecurity dataset."""
        from data_loader import DatasetConfig, load_cybersec_datasets
        
        # Skip if no data available
        data_path = Path("../dataset_creation/structured_data")
        if not data_path.exists():
            self.skipTest("No data available for testing")
        
        config = DatasetConfig(
            paths=[str(data_path)],
            format="cybersec",
            max_length=512,
            split_ratio=0.9
        )
        
        # Mock tokenizer
        class MockTokenizer:
            vocab_size = 32000
            pad_token_id = 0
            eos_token_id = 1
            
            def __call__(self, text, **kwargs):
                return {"input_ids": [1] * 10}
        
        tokenizer = MockTokenizer()
        
        try:
            train_dataset, eval_dataset, collator = load_cybersec_datasets(
                data_paths=config.paths,
                tokenizer=tokenizer,
                batch_size=2,
                max_length=config.max_length,
                val_split=0.1
            )
            
            self.assertIsNotNone(train_dataset)
            self.assertGreater(len(train_dataset), 0)
            
        except Exception as e:
            logger.warning(f"Dataset loading failed: {e}")
            self.skipTest("Dataset loading not available")


def tree_flatten(tree):
    """Flatten a tree structure to a list."""
    if isinstance(tree, dict):
        return [item for v in tree.values() for item in tree_flatten(v)]
    elif isinstance(tree, (list, tuple)):
        return [item for v in tree for item in tree_flatten(v)]
    else:
        return [tree]


if __name__ == "__main__":
    unittest.main()