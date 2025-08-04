#!/usr/bin/env python3
"""
Data loader for cybersecurity instruction datasets with multi-format support.
Handles tokenization, batching, and data mixing for MLX training.
"""

import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from dataclasses import dataclass, field
import numpy as np
import hashlib

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    paths: List[str] = field(default_factory=list)
    format: str = "cybersec"  # cybersec, alpaca, sharegpt, openai
    split_ratio: float = 0.95  # train/val split
    max_length: int = 2048
    min_length: int = 10
    shuffle: bool = True
    seed: int = 42
    mix_ratio: Optional[Dict[str, float]] = None  # For multi-dataset mixing
    system_prompt: Optional[str] = None
    add_eos_token: bool = True
    padding_side: str = "left"
    truncation_side: str = "right"
    ignore_index: int = -100
    
    # Data filtering
    filter_by_domain: Optional[List[str]] = None
    filter_by_type: Optional[List[str]] = None
    min_response_length: int = 50
    
    # Augmentation
    use_paraphrasing: bool = False
    temperature_sampling: float = 1.0


class CyberSecDataset:
    """Dataset class for cybersecurity instruction data."""
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: AutoTokenizer,
        split: str = "train"
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load and process data
        self.examples = []
        self._load_datasets()
        
        # Apply train/val split
        self._split_data()
        
        # Use OrderedDict for LRU cache behavior
        from collections import OrderedDict
        self._token_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Limit cache size
        
        logger.info(f"Loaded {len(self.examples)} examples for {split} split")
    
    def _load_datasets(self):
        """Load datasets from configured paths."""
        for path in self.config.paths:
            path = Path(path)
            if path.is_file():
                self._load_file(path)
            elif path.is_dir():
                # Load all JSON files in directory
                for json_file in path.glob("*.json"):
                    self._load_file(json_file)
            else:
                logger.warning(f"Path not found: {path}")
    
    def _load_file(self, filepath: Path):
        """Load a single dataset file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if self.config.format == "cybersec":
                self._load_cybersec_format(data, filepath)
            elif self.config.format == "alpaca":
                self._load_alpaca_format(data, filepath)
            elif self.config.format == "sharegpt":
                self._load_sharegpt_format(data, filepath)
            elif self.config.format == "openai":
                self._load_openai_format(data, filepath)
            else:
                raise ValueError(f"Unknown format: {self.config.format}")
                
            logger.info(f"Loaded {filepath.name} with {len(data.get('data', data))} entries")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    def _load_cybersec_format(self, data: Dict, source_path: Path):
        """Load cybersecurity dataset format from structured_data output."""
        entries = data.get('data', data) if isinstance(data, dict) else data
        
        for entry in entries:
            # Skip if filtering is enabled and doesn't match
            if self.config.filter_by_type:
                if entry.get('type') not in self.config.filter_by_type:
                    continue
            
            if self.config.filter_by_domain:
                source_type = entry.get('source_data', {}).get('type', '')
                if source_type not in self.config.filter_by_domain:
                    continue
            
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            # Skip short responses
            if len(response.split()) < self.config.min_response_length:
                continue
            
            example = {
                'instruction': instruction,
                'response': response,
                'type': entry.get('type', 'unknown'),
                'source': source_path.name,
                'metadata': entry.get('source_data', {})
            }
            
            # Add system prompt if configured
            if self.config.system_prompt:
                example['system'] = self.config.system_prompt
            
            self.examples.append(example)
    
    def _load_alpaca_format(self, data: Union[Dict, List], source_path: Path):
        """Load Alpaca-style dataset."""
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            instruction = entry.get('instruction', '')
            input_text = entry.get('input', '')
            output = entry.get('output', '')
            
            # Combine instruction and input
            if input_text:
                full_instruction = f"{instruction}\n\nInput: {input_text}"
            else:
                full_instruction = instruction
            
            example = {
                'instruction': full_instruction,
                'response': output,
                'type': 'general',
                'source': source_path.name
            }
            
            if self.config.system_prompt:
                example['system'] = self.config.system_prompt
            
            self.examples.append(example)
    
    def _load_sharegpt_format(self, data: Union[Dict, List], source_path: Path):
        """Load ShareGPT conversation format."""
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for conversation in entries:
            messages = conversation.get('conversations', [])
            
            # Extract instruction-response pairs from conversation
            for i in range(0, len(messages) - 1, 2):
                if messages[i].get('from') == 'human' and messages[i + 1].get('from') == 'gpt':
                    example = {
                        'instruction': messages[i].get('value', ''),
                        'response': messages[i + 1].get('value', ''),
                        'type': 'conversation',
                        'source': source_path.name
                    }
                    
                    if self.config.system_prompt:
                        example['system'] = self.config.system_prompt
                    
                    self.examples.append(example)
    
    def _load_openai_format(self, data: Union[Dict, List], source_path: Path):
        """Load OpenAI chat format."""
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            messages = entry.get('messages', [])
            
            # Extract system, user, and assistant messages
            system_msg = None
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                role = msg.get('role')
                if role == 'system':
                    system_msg = msg.get('content', '')
                elif role == 'user' and user_msg is None:
                    user_msg = msg.get('content', '')
                elif role == 'assistant' and user_msg is not None:
                    assistant_msg = msg.get('content', '')
            
            if user_msg and assistant_msg:
                example = {
                    'instruction': user_msg,
                    'response': assistant_msg,
                    'type': 'chat',
                    'source': source_path.name
                }
                
                if system_msg:
                    example['system'] = system_msg
                elif self.config.system_prompt:
                    example['system'] = self.config.system_prompt
                
                self.examples.append(example)
    
    def _split_data(self):
        """Split data into train/validation sets."""
        if self.config.shuffle:
            random.seed(self.config.seed)
            random.shuffle(self.examples)
        
        split_idx = int(len(self.examples) * self.config.split_ratio)
        
        if self.split == "train":
            self.examples = self.examples[:split_idx]
        else:  # validation
            self.examples = self.examples[split_idx:]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for tokenized text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def tokenize_example(self, example: Dict) -> Dict[str, mx.array]:
        """Tokenize a single example with caching."""
        # Create full text with chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            
            # Add system message if present
            if 'system' in example:
                messages.append({"role": "system", "content": example['system']})
            
            # Add user and assistant messages
            messages.append({"role": "user", "content": example['instruction']})
            messages.append({"role": "assistant", "content": example['response']})
            
            # Apply chat template with thinking disabled for better LoRA fine-tuning
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            
            # Remove thinking tags if they were added despite enable_thinking=False
            # GLM sometimes adds them anyway, so we remove them for cleaner training
            text = text.replace("<think></think>\n", "").replace("<think></think>", "")
        else:
            # Fallback to simple format
            if 'system' in example:
                text = f"{example['system']}\n\nUser: {example['instruction']}\n\nAssistant: {example['response']}"
            else:
                text = f"User: {example['instruction']}\n\nAssistant: {example['response']}"
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._token_cache:
            self._cache_hits += 1
            # Move to end (most recently used) for LRU behavior
            self._token_cache.move_to_end(cache_key)
            return self._token_cache[cache_key]
        
        self._cache_misses += 1
        
        # Clear cache if it's too large (LRU eviction)
        if len(self._token_cache) >= self._max_cache_size:
            # Remove least recently used entries
            entries_to_remove = len(self._token_cache) // 4  # Remove 25% instead of 50%
            for _ in range(entries_to_remove):
                # popitem(last=False) removes oldest (least recently used) item
                self._token_cache.popitem(last=False)
            logger.debug(f"Evicted {entries_to_remove} LRU entries from token cache")
        
        # Tokenize
        # Handle TokenizerWrapper from mlx_lm
        if hasattr(self.tokenizer, '_tokenizer'):
            # It's a TokenizerWrapper, use the underlying tokenizer
            actual_tokenizer = self.tokenizer._tokenizer
        else:
            actual_tokenizer = self.tokenizer
            
        encoding = actual_tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = encoding['input_ids']
        
        # Create labels (mask instruction part for loss calculation)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Find where response starts - GLM uses <|assistant|> token
            assistant_token_id = actual_tokenizer.encode(
                "<|assistant|>", add_special_tokens=False
            )[0]  # Get the single token ID
            
            labels = input_ids.copy()
            
            # Find assistant token position
            assistant_pos = -1
            for i, token_id in enumerate(input_ids):
                if token_id == assistant_token_id:
                    assistant_pos = i
                    break
            
            if assistant_pos != -1:
                # Mask everything up to and including the assistant token
                for j in range(assistant_pos + 1):
                    labels[j] = self.config.ignore_index
            else:
                # Fallback: mask first half if assistant token not found
                mask_len = len(input_ids) // 2
                for i in range(mask_len):
                    labels[i] = self.config.ignore_index
        else:
            # Simple masking - mask first half
            labels = input_ids.copy()
            mask_len = len(input_ids) // 2
            for i in range(mask_len):
                labels[i] = self.config.ignore_index
        
        # Add EOS token if needed
        eos_token_id = actual_tokenizer.eos_token_id if hasattr(actual_tokenizer, 'eos_token_id') else self.tokenizer.eos_token_id
        if self.config.add_eos_token and input_ids[-1] != eos_token_id:
            input_ids.append(eos_token_id)
            labels.append(eos_token_id)
        
        # Convert to MLX arrays
        result = {
            'input_ids': mx.array(input_ids),
            'labels': mx.array(labels),
            'attention_mask': mx.ones_like(mx.array(input_ids))
        }
        
        # Cache result
        self._token_cache[cache_key] = result
        
        return result
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get tokenized example by index."""
        return self.tokenize_example(self.examples[idx])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._token_cache),
            'max_cache_size': self._max_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class DataCollator:
    """Collate batches of tokenized examples with padding."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        padding_side: str = "left",
        pad_to_multiple_of: int = 8
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Set padding token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def __call__(self, batch: List[Dict[str, mx.array]]) -> Dict[str, mx.array]:
        """Collate batch with padding."""
        # Find max length in batch
        max_len = max(len(ex['input_ids']) for ex in batch)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                      self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Ensure we don't exceed max_length
        max_len = min(max_len, self.max_length)
        
        # Prepare batch tensors
        batch_size = len(batch)
        input_ids = mx.full((batch_size, max_len), self.tokenizer.pad_token_id)
        labels = mx.full((batch_size, max_len), -100)
        attention_mask = mx.zeros((batch_size, max_len))
        
        for i, example in enumerate(batch):
            seq_len = len(example['input_ids'])
            
            if self.padding_side == "right":
                input_ids[i, :seq_len] = example['input_ids'][:max_len]
                labels[i, :seq_len] = example['labels'][:max_len]
                attention_mask[i, :seq_len] = example['attention_mask'][:max_len]
            else:  # left padding
                input_ids[i, -seq_len:] = example['input_ids'][-max_len:]
                labels[i, -seq_len:] = example['labels'][-max_len:]
                attention_mask[i, -seq_len:] = example['attention_mask'][-max_len:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class MultiDatasetSampler:
    """Sample from multiple datasets with configurable mixing ratios."""
    
    def __init__(
        self,
        datasets: Dict[str, CyberSecDataset],
        mix_ratios: Optional[Dict[str, float]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate sampling probabilities
        if mix_ratios:
            total = sum(mix_ratios.values())
            self.probabilities = {
                name: ratio / total 
                for name, ratio in mix_ratios.items()
            }
        else:
            # Equal probability for all datasets
            n_datasets = len(datasets)
            self.probabilities = {
                name: 1.0 / n_datasets 
                for name in datasets
            }
        
        # Initialize indices for each dataset
        self.indices = {
            name: list(range(len(dataset)))
            for name, dataset in datasets.items()
        }
        
        # Shuffle if needed
        self.rng = random.Random(seed)
        if shuffle:
            for indices in self.indices.values():
                self.rng.shuffle(indices)
        
        # Current positions
        self.positions = {name: 0 for name in datasets}
    
    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """Iterate over dataset names and indices."""
        # Calculate total number of batches
        total_examples = sum(len(dataset) for dataset in self.datasets.values())
        total_batches = total_examples // self.batch_size
        
        for _ in range(total_batches):
            batch = []
            
            for _ in range(self.batch_size):
                # Sample dataset based on probabilities
                dataset_name = self.rng.choices(
                    list(self.probabilities.keys()),
                    weights=list(self.probabilities.values())
                )[0]
                
                # Get next index for this dataset
                idx = self.indices[dataset_name][self.positions[dataset_name]]
                batch.append((dataset_name, idx))
                
                # Update position with wraparound
                self.positions[dataset_name] = (
                    self.positions[dataset_name] + 1
                ) % len(self.datasets[dataset_name])
                
                # Reshuffle if we've gone through all examples
                if self.positions[dataset_name] == 0 and self.shuffle:
                    self.rng.shuffle(self.indices[dataset_name])
            
            yield from batch


def create_dataloaders(
    config: DatasetConfig,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    num_workers: int = 0,
    multi_dataset_config: Optional[Dict[str, DatasetConfig]] = None
) -> Tuple[Any, Any]:
    """Create train and validation dataloaders."""
    
    if multi_dataset_config:
        # Create multiple datasets
        train_datasets = {}
        val_datasets = {}
        
        for name, dataset_config in multi_dataset_config.items():
            train_datasets[name] = CyberSecDataset(
                dataset_config, tokenizer, split="train"
            )
            val_datasets[name] = CyberSecDataset(
                dataset_config, tokenizer, split="validation"
            )
        
        # Create multi-dataset sampler
        train_sampler = MultiDatasetSampler(
            train_datasets,
            mix_ratios=config.mix_ratio,
            batch_size=batch_size,
            shuffle=config.shuffle,
            seed=config.seed
        )
        
        # For validation, use simple concatenation
        val_dataset = ConcatDataset(list(val_datasets.values()))
        
    else:
        # Single dataset
        train_dataset = CyberSecDataset(config, tokenizer, split="train")
        val_dataset = CyberSecDataset(config, tokenizer, split="validation")
    
    # Create collator
    collator = DataCollator(
        tokenizer,
        max_length=config.max_length,
        padding_side=config.padding_side
    )
    
    # Create dataloaders (simplified for MLX)
    # In practice, you might want to use a proper batching mechanism
    # For now, we'll return the datasets and collator
    return (train_dataset, val_dataset), collator


class ConcatDataset:
    """Simple concatenation of multiple datasets."""
    
    def __init__(self, datasets: List[CyberSecDataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]


def load_cybersec_datasets(
    data_paths: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    val_split: float = 0.05,
    **kwargs
) -> Tuple[Any, Any, DataCollator]:
    """Convenience function to load cybersecurity datasets."""
    
    # Remove 'format' from kwargs if present since we set it explicitly
    kwargs.pop('format', None)
    
    config = DatasetConfig(
        paths=data_paths,
        format="cybersec",
        split_ratio=1.0 - val_split,
        max_length=max_length,
        **kwargs
    )
    
    datasets, collator = create_dataloaders(
        config,
        tokenizer,
        batch_size=batch_size
    )
    
    return datasets[0], datasets[1], collator