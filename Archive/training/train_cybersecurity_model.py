#!/usr/bin/env python3
"""
Advanced training script for cybersecurity instruction-tuned models using MLX and LoRA.
Supports multiple models, datasets, and comprehensive monitoring.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import yaml
import numpy as np
from tqdm import tqdm
import pickle

# MLX imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

# Transformers for tokenizer
from transformers import AutoTokenizer

# Local imports
from data_loader import (
    DatasetConfig, CyberSecDataset, DataCollator, 
    load_cybersec_datasets
)

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))
from dataset_creation.utils import BenchmarkTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"  # none, all, lora_only
    modules_to_save: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    # Model
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-MLX-4bit"
    tokenizer_name: Optional[str] = None  # Use model_name if None
    trust_remote_code: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    min_learning_rate: float = 1e-5
    
    # Data
    max_length: int = 2048
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Evaluation
    eval_steps: int = 100
    eval_strategy: str = "steps"  # steps, epoch
    save_steps: int = 500
    logging_steps: int = 10
    
    # Checkpointing
    output_dir: str = "./outputs"
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Hardware
    use_mps: bool = True  # Metal Performance Shaders
    gradient_checkpointing: bool = False
    mixed_precision: str = "no"  # no, fp16, bf16
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "cybersec-llm"
    use_tensorboard: bool = True
    
    # Generation
    generation_max_length: int = 512
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001
    
    # Other
    seed: int = 42
    save_merged_model: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA adaptation to specified modules in the model using MLX-LM's implementation."""
    
    # First, let's inspect the model structure to understand what modules are available
    logger.info("Preparing model for LoRA...")
    
    # Freeze the model first (MLX-LM pattern)
    model.freeze()
    
    # Convert config to MLX-LM format
    lora_config = {
        "rank": config.rank,
        "scale": config.alpha / config.rank,  # MLX uses scale instead of alpha
        "dropout": config.dropout,
        "keys": config.target_modules if config.target_modules else None
    }
    
    logger.info(f"Applying LoRA with config: rank={config.rank}, alpha={config.alpha}, "
                f"target_modules={config.target_modules}")
    
    # Apply LoRA to model
    # The linear_to_lora_layers function will automatically handle model structure
    linear_to_lora_layers(
        model,
        num_layers=-1,  # Apply to all layers
        config=lora_config,
        use_dora=False
    )
    
    # Force evaluation to ensure LoRA layers are properly initialized
    mx.eval(model)
    
    logger.info("LoRA application complete.")
    
    # Print trainable parameters info
    try:
        from mlx_lm.tuner.utils import print_trainable_parameters
        print_trainable_parameters(model)
    except Exception as e:
        logger.warning(f"Could not print trainable parameters: {e}")
        # Manual count as fallback
        trainable_params = model.trainable_parameters()
        if isinstance(trainable_params, dict):
            num_params = sum(p.size for p in trainable_params.values() if hasattr(p, 'size'))
            logger.info(f"Trainable parameters: {num_params:,}")
        else:
            logger.info("Trainable parameters count unavailable")
    
    return model


def get_lora_parameters(model: nn.Module) -> List[mx.array]:
    """Extract trainable parameters for training."""
    # Force evaluation first to ensure LoRA parameters are initialized
    mx.eval(model)
    
    # Use MLX-LM's method to get trainable parameters
    trainable_params = model.trainable_parameters()
    
    # Handle different return types
    params = []
    if isinstance(trainable_params, dict):
        # Deep flatten nested dictionaries to find all array parameters
        def deep_flatten_dict(obj, parent_key=''):
            """Recursively flatten nested dicts/lists to find all array parameters."""
            items = []
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    items.extend(deep_flatten_dict(v, new_key))
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                    items.extend(deep_flatten_dict(item, new_key))
            elif hasattr(obj, '__dict__'):
                # Handle nn.Module objects by checking their attributes
                for attr_name in ['lora_a', 'lora_b', 'scale']:
                    if hasattr(obj, attr_name):
                        attr = getattr(obj, attr_name)
                        if hasattr(attr, 'size'):
                            items.append((f"{parent_key}.{attr_name}", attr))
            elif hasattr(obj, 'size'):
                # It's an array parameter
                items.append((parent_key, obj))
                
            return items
        
        param_list = deep_flatten_dict(trainable_params)
        params = [p for _, p in param_list]
        
        # Log parameter names for debugging
        if param_list:
            logger.debug("Trainable parameter names:")
            for name, _ in param_list[:5]:
                logger.debug(f"  - {name}")
            if len(param_list) > 5:
                logger.debug(f"  ... and {len(param_list) - 5} more")
    else:
        params = tree_flatten(trainable_params)
        params = [p for p in params if hasattr(p, 'size')]
    
    logger.info(f"Found {len(params)} trainable parameter arrays")
    
    # Calculate total trainable parameters
    if params:
        total_params = sum(p.size for p in params)
        logger.info(f"Total trainable parameters: {total_params:,}")
    else:
        logger.warning("No trainable parameters found! Check LoRA configuration.")
    
    return params


class CyberSecTrainer:
    """Main trainer class for cybersecurity model fine-tuning."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset: CyberSecDataset,
        eval_dataset: Optional[CyberSecDataset] = None,
        collator: Optional[DataCollator] = None,
        debug: bool = False
    ):
        self.debug = debug
        if self.debug:
            logger.info("Initializing CyberSecTrainer...")
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collator = collator or DataCollator(tokenizer)
        
        # Apply LoRA if configured
        if config.use_lora:
            logger.info("Applying LoRA to model...")
            lora_start = time.time()
            self.model = apply_lora_to_model(model, config.lora_config)
            self._debug_log(f"LoRA applied in {time.time() - lora_start:.2f}s")
            
            # Synchronize after LoRA application
            self._debug_log("Synchronizing model after LoRA...")
            mx.eval(self.model.parameters())
            self._debug_log("Model synchronized after LoRA")
            
            # Get LoRA parameters - need to use self.model not model
            self._debug_log("Getting LoRA parameters...")
            self.trainable_params = get_lora_parameters(self.model)
            logger.info(f"Found {len(self.trainable_params)} LoRA parameters to train")
            self._debug_log("LoRA parameters retrieved successfully")
        else:
            self.trainable_params = tree_flatten(model.parameters())
            logger.info(f"Training all {len(self.trainable_params)} model parameters")
        
        # Initialize optimizer
        self._debug_log("Creating optimizer...")
        try:
            self.optimizer = self._create_optimizer()
            self._debug_log("Optimizer created successfully")
        except Exception as e:
            logger.error(f"Error creating optimizer: {e}")
            raise
        
        # Training state
        self._debug_log("Initializing training state...")
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        
        # Setup output directory
        self._debug_log("Setting up output directory...")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._debug_log(f"Output directory: {self.output_dir}")
        
        # Initialize benchmark tracker
        self._debug_log("Initializing BenchmarkTracker...")
        try:
            self.benchmark = BenchmarkTracker(logger)
            self._debug_log("BenchmarkTracker initialized")
        except Exception as e:
            logger.error(f"Error initializing BenchmarkTracker: {e}")
            raise
        self.training_start_time = time.time()
        
        # Initialize memory monitor
        try:
            from .memory_monitor import MemoryMonitor
            self.memory_monitor = MemoryMonitor()
            logger.info("Memory monitoring enabled")
        except (ImportError, ValueError):
            # ValueError for when not running as a package
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(__file__))
                from memory_monitor import MemoryMonitor
                self.memory_monitor = MemoryMonitor()
                logger.info("Memory monitoring enabled")
            except ImportError:
                self.memory_monitor = None
                logger.info("Memory monitoring not available")
        
        # Setup logging
        self._debug_log("Setting up logging...")
        self._setup_logging()
        self._debug_log("Logging setup complete")
        
        # Load checkpoint if resuming
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
        
        logger.info("CyberSecTrainer initialization complete")
    
    def _debug_log(self, message: str):
        """Log message only in debug mode."""
        if self.debug:
            logger.info(f"[DEBUG] {message}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        self._debug_log(f"Creating {self.config.optimizer} optimizer...")
        self._debug_log(f"Learning rate: {self.config.learning_rate}")
        self._debug_log(f"Number of trainable params: {len(self.trainable_params)}")
        
        if self.config.optimizer == "adamw":
            self._debug_log("Initializing AdamW optimizer...")
            optimizer = optim.AdamW(
                learning_rate=self.config.learning_rate,
                betas=[self.config.adam_beta1, self.config.adam_beta2],
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
            self._debug_log("AdamW optimizer created")
            return optimizer
        elif self.config.optimizer == "adam":
            self._debug_log("Initializing Adam optimizer...")
            optimizer = optim.Adam(
                learning_rate=self.config.learning_rate,
                betas=[self.config.adam_beta1, self.config.adam_beta2],
                eps=self.config.adam_epsilon
            )
            self._debug_log("Adam optimizer created")
            return optimizer
        elif self.config.optimizer == "sgd":
            self._debug_log("Initializing SGD optimizer...")
            optimizer = optim.SGD(
                learning_rate=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
            self._debug_log("SGD optimizer created")
            return optimizer
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        self._debug_log("Entering _setup_logging()")
        
        # Create log file
        self._debug_log("Creating log file...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"training_{timestamp}.log"
        
        self._debug_log(f"Log file path: {log_file}")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        self._debug_log("File handler added to logger")
        
        # Initialize monitoring tools to None
        self.tb_writer = None
        self.wandb = None
        
        # Setup TensorBoard if enabled (deferred)
        if self.config.use_tensorboard:
            self._debug_log("TensorBoard is enabled, deferring initialization until first use")
            self.tb_writer = "pending"  # Placeholder to indicate deferred initialization
        else:
            self._debug_log("TensorBoard is disabled")
        
        # Setup Weights & Biases if enabled
        if self.config.use_wandb:
            self._debug_log("Setting up Weights & Biases...")
            try:
                import wandb
                self._debug_log("wandb imported successfully")
                wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                    name=f"cybersec_{timestamp}"
                )
                self.wandb = wandb
                self._debug_log("Weights & Biases initialized")
            except ImportError:
                logger.warning("Weights & Biases not available. Install with: pip install wandb")
                self.wandb = None
                self._debug_log("Weights & Biases import failed")
        else:
            self._debug_log("Weights & Biases is disabled")
        
        self._debug_log("Exiting _setup_logging()")
    
    def _initialize_tensorboard(self):
        """Lazily initialize TensorBoard when first needed."""
        if self.tb_writer == "pending":
            self._debug_log("Initializing TensorBoard (deferred)...")
            try:
                self._debug_log("Attempting to import torch.utils.tensorboard...")
                from torch.utils.tensorboard import SummaryWriter
                self._debug_log("TensorBoard import successful")
                self.tb_writer = SummaryWriter(
                    log_dir=str(self.output_dir / "tensorboard")
                )
                self._debug_log("TensorBoard SummaryWriter created")
            except ImportError as e:
                self._debug_log(f"TensorBoard import failed: {e}")
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.tb_writer = None
            except Exception as e:
                self._debug_log(f"TensorBoard initialization error: {e}")
                logger.error(f"Failed to initialize TensorBoard: {e}")
                self.tb_writer = None
    
    def get_learning_rate(self, step: int) -> float:
        """Calculate learning rate based on schedule."""
        warmup_steps = self.config.warmup_steps
        total_steps = len(self.train_dataset) * self.config.num_epochs // self.config.batch_size
        
        if step < warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / warmup_steps)
        
        if self.config.lr_scheduler == "cosine":
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.config.min_learning_rate + (
                self.config.learning_rate - self.config.min_learning_rate
            ) * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.lr_scheduler == "linear":
            # Linear decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.config.learning_rate * (1 - progress)
        else:
            # Constant
            return self.config.learning_rate
    
    def compute_loss(
        self, 
        input_ids: mx.array, 
        labels: mx.array, 
        attention_mask: Optional[mx.array] = None  # Currently unused, kept for API compatibility
    ) -> mx.array:
        """Compute cross-entropy loss for language modeling with memory-efficient chunking."""
        if self.global_step == 0 and self.debug:
            logger.info("[DEBUG] compute_loss called for first time")
            logger.info(f"[DEBUG] Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
        
        # Forward pass
        if self.global_step == 0 and self.debug:
            logger.info("[DEBUG] Running model forward pass...")
        logits = self.model(input_ids)
        if self.global_step == 0 and self.debug:
            logger.info(f"[DEBUG] Forward pass complete. Logits shape: {logits.shape}")
        
        _, seq_len, vocab_size = logits.shape  # batch_size not used
        
        # Memory-efficient loss computation with chunking
        # Process sequences in chunks to avoid materializing full logits tensor
        chunk_size = 256  # Process 256 tokens at a time
        total_loss = 0.0
        total_valid = 0
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            
            # Get chunk of logits and labels
            logits_chunk = logits[:, i:end_idx, :].reshape(-1, vocab_size)
            labels_chunk = labels[:, i:end_idx].reshape(-1)
            
            # Compute mask for this chunk
            mask_chunk = labels_chunk != -100
            
            if mask_chunk.sum() == 0:
                continue
            
            # Compute loss for this chunk
            losses_chunk = nn.losses.cross_entropy(
                logits_chunk,
                labels_chunk,
                reduction='none'
            )
            
            # Apply mask
            masked_losses = losses_chunk * mask_chunk
            chunk_valid = mask_chunk.sum()
            
            # Accumulate
            total_loss = total_loss + masked_losses.sum()
            total_valid = total_valid + chunk_valid
            
            # Force evaluation to free memory
            mx.eval(total_loss, total_valid)
        
        if total_valid == 0:
            return mx.zeros(1)
        
        # Compute average loss
        loss = total_loss / total_valid
        
        if self.global_step == 0 and self.debug:
            logger.info(f"[DEBUG] Loss computed successfully: {loss.item()}")
        
        return loss
    
    def training_step(self, batch: Dict[str, mx.array]) -> mx.array:
        """Perform a single training step."""
        # Forward pass and loss calculation
        loss = self.compute_loss(
            batch['input_ids'],
            batch['labels'],
            batch.get('attention_mask')
        )
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        return scaled_loss
    
    def evaluation_step(self, batch: Dict[str, mx.array]) -> mx.array:
        """Perform a single evaluation step."""
        with mx.no_grad():
            loss = self.compute_loss(
                batch['input_ids'],
                batch['labels'],
                batch.get('attention_mask')
            )
        return loss
    
    def evaluate(self) -> float:
        """Run evaluation on the validation dataset."""
        if self.eval_dataset is None:
            return 0.0
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        eval_losses = []
        eval_pbar = tqdm(range(len(self.eval_dataset)), desc="Evaluating")
        
        for i in eval_pbar:
            batch = self.collator([self.eval_dataset[i]])
            loss = self.evaluation_step(batch)
            eval_losses.append(loss.item())
            
            eval_pbar.set_postfix({'loss': np.mean(eval_losses)})
        
        avg_loss = np.mean(eval_losses)
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Log metrics
        if self.tb_writer:
            if self.tb_writer == "pending":
                self._initialize_tensorboard()
            if self.tb_writer:  # Check again after initialization
                self.tb_writer.add_scalar('eval/loss', avg_loss, self.global_step)
                self.tb_writer.add_scalar('eval/perplexity', perplexity, self.global_step)
        
        if self.wandb:
            self.wandb.log({
                'eval/loss': avg_loss,
                'eval/perplexity': perplexity,
                'global_step': self.global_step
            })
        
        self.model.train()
        return avg_loss
    
    def _create_adapter_config(self) -> Dict[str, Any]:
        """Create adapter configuration metadata."""
        # Calculate training duration
        training_hours = (time.time() - self.training_start_time) / 3600
        
        # Get latest metrics
        latest_loss = self.train_losses[-1] if self.train_losses else 0
        latest_perplexity = np.exp(latest_loss) if latest_loss > 0 else 0
        
        # Check if tokenizer supports chat template
        has_chat_template = hasattr(self.tokenizer, 'apply_chat_template')
        
        adapter_config = {
            "adapter_version": "1.0.0",
            "adapter_id": f"cybersec-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}-step{self.global_step}",
            "adapter_name": "Cybersecurity LoRA Adapter",
            "description": "LoRA adapter trained on cybersecurity instruction dataset for security Q&A, incident response, and threat analysis",
            
            "trained_on": {
                "base_model": self.config.model_name,
                "model_type": self._infer_model_type(self.config.model_name),
                "quantization": self._infer_quantization(self.config.model_name),
                "chat_template_used": has_chat_template,
                "tokenizer": self.config.tokenizer_name or self.config.model_name
            },
            
            "lora_config": asdict(self.config.lora_config),
            
            "training_info": {
                "dataset_paths": self.config.dataset_config.paths,
                "dataset_format": self.config.dataset_config.format,
                "dataset_size": len(self.train_dataset),
                "num_epochs": self.config.num_epochs,
                "current_epoch": self.epoch,
                "global_step": self.global_step,
                "latest_loss": float(latest_loss),
                "latest_perplexity": float(latest_perplexity),
                "best_eval_loss": float(self.best_eval_loss) if self.best_eval_loss != float('inf') else None,
                "training_duration_hours": training_hours,
                "timestamp": datetime.now().isoformat(),
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps
            },
            
            "architecture_hints": {
                "layer_patterns": "transformer",
                "expected_modules": self.config.lora_config.target_modules,
                "notes": f"Trained on {self._infer_model_type(self.config.model_name)} architecture. May work with similar transformer models with matching module names."
            },
            
            "files": {
                "weights": "lora_weights.safetensors",
                "tokenizer": "tokenizer/",
                "training_config": "training_config.json",
                "training_state": "training_state.pkl"
            },
            
            "usage_notes": "This adapter specializes the model for cybersecurity tasks including threat detection, vulnerability analysis, incident response, and security best practices.",
            "license": "apache-2.0"
        }
        
        return adapter_config
    
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name."""
        model_name_lower = model_name.lower()
        if 'qwen' in model_name_lower:
            return 'qwen'
        elif 'llama' in model_name_lower:
            return 'llama'
        elif 'mistral' in model_name_lower:
            return 'mistral'
        elif 'phi' in model_name_lower:
            return 'phi'
        else:
            return 'unknown'
    
    def _infer_quantization(self, model_name: str) -> Optional[str]:
        """Infer quantization from model name."""
        if '4bit' in model_name or '4-bit' in model_name:
            return '4bit'
        elif '8bit' in model_name or '8-bit' in model_name:
            return '8bit'
        elif '16bit' in model_name or '16-bit' in model_name or 'fp16' in model_name:
            return '16bit'
        else:
            return None
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None):
        """Save training checkpoint."""
        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state (LoRA weights only if using LoRA)
        if self.config.use_lora:
            lora_state = {}
            for i, param in enumerate(self.trainable_params):
                lora_state[f'param_{i}'] = param
            mx.save_safetensors(
                str(checkpoint_dir / "lora_weights.safetensors"),
                lora_state
            )
            
            # Create adapter config for LoRA checkpoints
            adapter_config = self._create_adapter_config()
            with open(checkpoint_dir / "adapter_config.json", 'w') as f:
                json.dump(adapter_config, f, indent=2)
        else:
            # Save full model
            self.model.save_weights(str(checkpoint_dir / "model_weights.safetensors"))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'optimizer_state': self.optimizer.state,
            'config': asdict(self.config)
        }
        
        with open(checkpoint_dir / "training_state.pkl", 'wb') as f:
            pickle.dump(training_state, f)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Manage checkpoint limit
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Validate checkpoint first
        from validation import ConfigValidator
        issues = ConfigValidator.validate_checkpoint(str(checkpoint_dir))
        if issues:
            raise ValueError(f"Invalid checkpoint: {', '.join(issues)}")
        
        # Load model weights
        if self.config.use_lora:
            lora_state = mx.load(str(checkpoint_dir / "lora_weights.safetensors"))
            for i, param in enumerate(self.trainable_params):
                param[:] = lora_state[f'param_{i}']
        else:
            self.model.load_weights(str(checkpoint_dir / "model_weights.safetensors"))
        
        # Load training state
        with open(checkpoint_dir / "training_state.pkl", 'rb') as f:
            training_state = pickle.load(f)
        
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.best_eval_loss = training_state['best_eval_loss']
        self.patience_counter = training_state['patience_counter']
        self.train_losses = training_state['train_losses']
        self.eval_losses = training_state['eval_losses']
        self.learning_rates = training_state['learning_rates']
        self.optimizer.state = training_state['optimizer_state']
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint)
    
    def generate_sample(self, prompt: str) -> str:
        """Generate a sample response for qualitative evaluation."""
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors=None,
            truncation=True,
            max_length=self.config.max_length
        )
        
        input_ids = mx.array(inputs['input_ids'])
        
        # Generate
        sampler = make_sampler(
            temp=self.config.generation_temperature,
            top_p=self.config.generation_top_p
        )
        
        # Simple generation loop
        generated = input_ids.tolist()
        for _ in range(self.config.generation_max_length):
            # Get model prediction
            logits = self.model(mx.array([generated]))
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            next_token = sampler(next_token_logits)
            generated.append(next_token.item())
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.model.train()
        
        return response
    
    def train_simple(self):
        """Simplified training loop for debugging."""
        logger.info("Starting simplified training...")
        
        # Check if we have trainable parameters
        logger.info(f"Number of trainable parameters: {len(self.trainable_params)}")
        if len(self.trainable_params) == 0:
            logger.error("No trainable parameters found! Make sure LoRA is applied correctly.")
            return
        
        # Get a single batch for testing
        batch_indices = list(range(min(4, len(self.train_dataset))))
        batch_examples = [self.train_dataset[i] for i in batch_indices]
        batch = self.collator(batch_examples)
        
        logger.info(f"Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        
        # Define loss function using MLX's approach
        # nn.value_and_grad passes parameters as first arg, not model
        def loss_fn(inputs, targets):
            # Forward pass using self.model
            logits = self.model(inputs)
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)
            
            # Compute loss with masking
            mask = targets != -100
            if mask.sum() == 0:
                return mx.zeros(1)
            
            all_losses = nn.losses.cross_entropy(logits, targets, reduction='none')
            masked_losses = all_losses * mask
            loss = masked_losses.sum() / mask.sum()
            
            return loss
        
        # Test forward pass
        logger.info("Testing forward pass...")
        test_loss = loss_fn(batch['input_ids'], batch['labels'])
        logger.info(f"Test loss: {test_loss.item()}")
        
        # Create value_and_grad function using standard MLX approach
        logger.info("Testing gradient computation...")
        
        def compute_loss_and_grad(inputs, targets):
            """Compute loss and gradients manually."""
            def _loss(params):
                # Update model parameters
                self.model.update({"model": params})
                return loss_fn(inputs, targets)
            
            # Get trainable parameters
            trainable_params = self.model.trainable_parameters()
            
            # Compute value and gradient
            loss, grads = mx.value_and_grad(_loss)(trainable_params)
            return loss, grads
        
        # Test gradient computation
        try:
            # Try the simple approach first
            logger.info("Trying direct gradient computation...")
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(batch['input_ids'], batch['labels'])
            logger.info(f"Direct gradient computation successful!")
        except Exception as e:
            logger.warning(f"Direct gradient computation failed: {e}")
            logger.info("Falling back to manual gradient computation...")
            loss, grads = compute_loss_and_grad(batch['input_ids'], batch['labels'])
        
        logger.info(f"Loss with gradients: {loss.item()}")
        logger.info(f"Gradient type: {type(grads)}")
        
        # Test optimizer update
        logger.info("Testing optimizer update...")
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        logger.info("Update completed successfully!")
        
        # Now run actual training
        logger.info("Starting actual training loop...")
        for _ in range(1):  # Single epoch for basic test
            for step in range(min(10, len(self.train_dataset) // self.config.batch_size)):
                # Get batch
                batch_indices = [
                    (step * self.config.batch_size + i) % len(self.train_dataset)
                    for i in range(self.config.batch_size)
                ]
                batch_examples = [self.train_dataset[i] for i in batch_indices]
                batch = self.collator(batch_examples)
                
                # Compute loss and gradients
                loss, grads = loss_and_grad_fn(batch['input_ids'], batch['labels'])
                
                # Update
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters())
                
                if step % 5 == 0:
                    logger.info(f"Step {step}, Loss: {loss.item():.4f}")
        
        logger.info("Simple training completed!")
    
    def train(self):
        """Main training loop."""
        self._debug_log("="*80)
        self._debug_log("ENTERING train() method")
        self._debug_log("="*80)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total training examples: {len(self.train_dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        # Memory safety check for large vocabulary models
        vocab_size = self.tokenizer.vocab_size
        if vocab_size > 100000:
            logger.warning(f"⚠️  Large vocabulary detected: {vocab_size:,} tokens")
            
            # Calculate expected memory usage
            logits_memory_gb = (self.config.batch_size * min(256, self.config.max_length) * vocab_size * 4) / (1024**3)
            logger.info(f"Expected logits memory (chunked): {logits_memory_gb:.2f}GB")
            
            # Check configuration safety
            if self.config.batch_size > 2:
                logger.warning(f"⚠️  WARNING: batch_size={self.config.batch_size} may be too large for {vocab_size:,} vocab model!")
                logger.warning("Recommended: batch_size=1 or 2 maximum for memory safety")
            
            if self.config.max_length > 512:
                logger.warning(f"WARNING: max_length={self.config.max_length} may cause OOM with {vocab_size:,} vocab")
                logger.warning("Recommended: max_length=256 or less")
            
            if not self.config.gradient_checkpointing and vocab_size > 150000:
                logger.warning("WARNING: gradient_checkpointing=False with very large vocabulary")
                logger.warning("Consider enabling gradient_checkpointing to save memory")
        
        # Calculate total steps
        steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.num_epochs
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
        
        # Check if we're using LoRA
        if self.config.use_lora and len(self.trainable_params) == 0:
            logger.error("No trainable LoRA parameters found!")
            return
        
        # Log initial memory status
        logger.info("Initial memory status:")
        try:
            # Get MLX memory if available
            mlx_memory_gb = mx.metal.get_active_memory() / (1024**3)
            mlx_peak_gb = mx.metal.get_peak_memory() / (1024**3)
            logger.info(f"MLX Memory - Active: {mlx_memory_gb:.2f}GB, Peak: {mlx_peak_gb:.2f}GB")
        except:
            pass
        
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage()
        
        # Log memory status
        if self.debug:
            logger.info("[DEBUG] Checking memory status...")
            import subprocess
            try:
                result = subprocess.run(['vm_stat'], capture_output=True, text=True)
                logger.info("[DEBUG] Memory stats:\n" + result.stdout[:500])
            except:
                logger.info("[DEBUG] Could not get memory stats")
        
        # Create loss function that works with current model
        self._debug_log("Creating loss function...")
        def compute_loss(input_ids, labels, attention_mask=None):
            """Compute loss using self.model."""
            return self.compute_loss(input_ids, labels, attention_mask)
        
        # Create the value_and_grad function using nn.value_and_grad
        self._debug_log("Creating value_and_grad function...")
        loss_value_and_grad = nn.value_and_grad(self.model, compute_loss)
        self._debug_log("value_and_grad function created successfully")
        
        # Training loop
        accumulated_loss = 0.0
        self.model.train()
        
        self._debug_log("About to start training loop...")
        self._debug_log(f"Current epoch: {self.epoch}, Target epochs: {self.config.num_epochs}")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self._debug_log(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            self.epoch = epoch
            epoch_pbar = tqdm(
                range(steps_per_epoch), 
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            accumulated_gradients = None
            
            for step in epoch_pbar:
                if step == 0:
                    self._debug_log(f"Processing first step of epoch {epoch + 1}")
                # Prepare batch
                batch_indices = [
                    (step * self.config.batch_size + i) % len(self.train_dataset)
                    for i in range(self.config.batch_size)
                ]
                batch_examples = [self.train_dataset[i] for i in batch_indices]
                batch = self.collator(batch_examples)
                
                if step == 0:
                    self._debug_log("Batch prepared, calling loss_value_and_grad...")
                
                # Forward and backward pass
                loss, grad = loss_value_and_grad(
                    batch['input_ids'],
                    batch['labels'],
                    batch.get('attention_mask')
                )
                
                if step == 0:
                    self._debug_log(f"First gradient computation complete! Loss: {loss.item()}")
                
                # Accumulate gradients
                if accumulated_gradients is None:
                    accumulated_gradients = grad
                else:
                    accumulated_gradients = tree_map(
                        lambda a, b: a + b, accumulated_gradients, grad
                    )
                
                accumulated_loss += loss.item()
                
                # Force evaluation to prevent memory buildup
                mx.eval(loss, accumulated_gradients)
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Average gradients
                    accumulated_gradients = tree_map(
                        lambda g: g / self.config.gradient_accumulation_steps,
                        accumulated_gradients
                    )
                    
                    # Update learning rate
                    lr = self.get_learning_rate(self.global_step)
                    self.optimizer.learning_rate = lr
                    self.learning_rates.append(lr)
                    
                    # Optimizer step
                    self.optimizer.update(self.model, accumulated_gradients)
                    
                    # Force evaluation of model parameters after update
                    mx.eval(self.model.parameters())
                    
                    # Reset accumulated gradients
                    accumulated_gradients = None
                    
                    # Memory cleanup - clear MLX cache periodically
                    if self.global_step % 50 == 0:
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Log memory usage periodically
                        if self.global_step % 100 == 0:
                            self.benchmark.record_memory_usage()
                            if self.memory_monitor:
                                self.memory_monitor.log_memory_usage(self.global_step)
                            logger.debug(f"Step {self.global_step}: Memory cleanup performed")
                    
                    # Enhanced memory monitoring every 10 steps
                    if self.global_step % 10 == 0:
                        try:
                            # Get MLX memory usage
                            mlx_memory_gb = mx.metal.get_active_memory() / (1024**3)
                            mlx_peak_gb = mx.metal.get_peak_memory() / (1024**3)
                            logger.info(f"Step {self.global_step}: MLX Memory - Active: {mlx_memory_gb:.2f}GB, Peak: {mlx_peak_gb:.2f}GB")
                            
                            # Check for memory leak pattern
                            if hasattr(self, '_prev_mlx_memory'):
                                memory_delta = mlx_memory_gb - self._prev_mlx_memory
                                if memory_delta > 0.5:  # Alert if memory increased by >500MB
                                    logger.warning(f"Memory increase detected: +{memory_delta:.2f}GB since step {self.global_step - 10}")
                            self._prev_mlx_memory = mlx_memory_gb
                        except Exception as e:
                            logger.debug(f"Could not get MLX memory stats: {e}")
                    
                    # Log metrics
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    self.train_losses.append(avg_loss)
                    
                    if self.global_step % self.config.logging_steps == 0:
                        epoch_pbar.set_postfix({
                            'loss': f"{avg_loss:.4f}",
                            'lr': f"{lr:.2e}"
                        })
                        
                        if self.tb_writer:
                            if self.tb_writer == "pending":
                                self._initialize_tensorboard()
                            if self.tb_writer:  # Check again after initialization
                                self.tb_writer.add_scalar('train/loss', avg_loss, self.global_step)
                                self.tb_writer.add_scalar('train/learning_rate', lr, self.global_step)
                        
                        if self.wandb:
                            self.wandb.log({
                                'train/loss': avg_loss,
                                'train/learning_rate': lr,
                                'global_step': self.global_step
                            })
                    
                    # Reset accumulated loss
                    accumulated_loss = 0.0
                    self.global_step += 1
                    
                    # Evaluation
                    if self.config.eval_strategy == "steps" and \
                       self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.eval_losses.append((self.global_step, eval_loss))
                        
                        # Early stopping check
                        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
                            self.best_eval_loss = eval_loss
                            self.patience_counter = 0
                            
                            # Save best model
                            self.save_checkpoint(
                                self.output_dir / "checkpoint-best"
                            )
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.config.early_stopping_patience:
                                logger.info("Early stopping triggered")
                                return
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                    
                    # Generate sample
                    if self.global_step % 1000 == 0:
                        sample_prompt = "What are the key indicators of a phishing attack?"
                        sample_response = self.generate_sample(sample_prompt)
                        logger.info(f"Sample generation:\nPrompt: {sample_prompt}\nResponse: {sample_response}")
            
            # Epoch evaluation
            if self.config.eval_strategy == "epoch":
                eval_loss = self.evaluate()
                self.eval_losses.append((self.global_step, eval_loss))
        
        # Save final model
        self.save_checkpoint(self.output_dir / "checkpoint-final")
        
        # Save training history
        self.save_training_history()
        
        logger.info("Training completed!")
    
    def preview_training_data(self, num_samples: int = 5):
        """Preview how data will be formatted and tokenized for training."""
        print("\n" + "="*80)
        print("TRAINING PREVIEW MODE")
        print("="*80)
        
        # Model info
        print(f"\nModel: {self.config.model_name}")
        print(f"Tokenizer: {self.config.tokenizer_name or self.config.model_name}")
        
        # Dataset info
        print(f"\nDataset: {', '.join(self.config.dataset_config.paths)}")
        print(f"Total Examples: {len(self.train_dataset) + (len(self.eval_dataset) if self.eval_dataset else 0)}")
        print(f"Training Examples: {len(self.train_dataset)}")
        print(f"Validation Examples: {len(self.eval_dataset) if self.eval_dataset else 0}")
        
        # Data formatting preview
        print("\n" + "="*80)
        print("DATA FORMATTING PREVIEW")
        print("="*80)
        
        # Get sample examples
        num_samples = min(num_samples, len(self.train_dataset))
        
        for i in range(num_samples):
            print(f"\n[Sample {i+1}/{num_samples}]")
            print("-" * 40)
            
            # Get raw example
            example = self.train_dataset.examples[i]
            
            print("Raw Data:")
            print(f"  Type: {example.get('type', 'unknown')}")
            print(f"  Instruction: {example['instruction']}")
            print(f"  Response: {example['response']}")
            
            # Show formatted text
            print("\nFormatted for Training:")
            
            # Create formatted text
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = []
                if 'system' in example:
                    messages.append({"role": "system", "content": example['system']})
                messages.append({"role": "user", "content": example['instruction']})
                messages.append({"role": "assistant", "content": example['response']})
                
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                if 'system' in example:
                    formatted_text = f"{example['system']}\n\nUser: {example['instruction']}\n\nAssistant: {example['response']}"
                else:
                    formatted_text = f"User: {example['instruction']}\n\nAssistant: {example['response']}"
            
            # Show full formatted text
            print(formatted_text)
            
            # Show tokenization info
            tokenized = self.train_dataset.tokenize_example(example)
            input_ids = tokenized['input_ids']
            labels = tokenized['labels']
            
            # Count masked vs unmasked tokens
            masked_tokens = mx.sum(labels == self.config.dataset_config.ignore_index).item()
            response_tokens = len(labels) - masked_tokens
            
            print("\nTokenization Info:")
            print(f"  Total tokens: {len(input_ids)}")
            print(f"  Instruction tokens: {masked_tokens} (masked for training)")
            print(f"  Response tokens: {response_tokens} (used for loss calculation)")
            print(f"  Truncated: {'Yes' if len(input_ids) >= self.config.max_length else 'No'}")
        
        # Training configuration
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        
        print(f"\nLoRA Configuration:")
        print(f"  Rank: {self.config.lora_config.rank}")
        print(f"  Alpha: {self.config.lora_config.alpha}")
        print(f"  Target Modules: {', '.join(self.config.lora_config.target_modules)}")
        
        print(f"\nTraining Parameters:")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Gradient Accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Max Sequence Length: {self.config.max_length}")
        
        # Calculate steps accounting for gradient accumulation
        gradient_steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        optimization_steps_per_epoch = gradient_steps_per_epoch // self.config.gradient_accumulation_steps
        total_optimization_steps = optimization_steps_per_epoch * self.config.num_epochs
        
        # More realistic time estimation for MLX on Apple Silicon
        # LoRA training typically takes 3-8 seconds per optimization step
        seconds_per_step = 5.0  # Conservative estimate for M4 Max
        estimated_time_hours = (total_optimization_steps * seconds_per_step) / 3600
        
        print(f"\nTraining Steps:")
        print(f"  Gradient Steps per Epoch: {gradient_steps_per_epoch}")
        print(f"  Optimization Steps per Epoch: {optimization_steps_per_epoch} (with gradient accumulation)")
        print(f"  Total Optimization Steps: {total_optimization_steps}")
        print(f"  Estimated Time: ~{estimated_time_hours:.1f} hours ({seconds_per_step:.1f}s per optimization step)")
        
        # Estimate memory based on model size and LoRA config
        # Detect model type and parameters
        model_name_lower = self.config.model_name.lower()
        vocab_size = self.tokenizer.vocab_size
        
        # Model-specific parameters
        if 'qwen3-30b' in model_name_lower:
            base_memory_gb = 30.0  # 30B model in 8-bit
            hidden_dim = 8192
            model_desc = "Qwen3-30B (8-bit)"
        elif 'qwen2.5-3b' in model_name_lower or 'qwen25-3b' in model_name_lower:
            base_memory_gb = 6.0  # 3B model in 4-bit
            hidden_dim = 2048
            model_desc = "Qwen2.5-3B (4-bit)"
        elif 'qwen2.5-1b' in model_name_lower or 'qwen25-1b' in model_name_lower:
            base_memory_gb = 2.0  # 1B model in 4-bit
            hidden_dim = 1024
            model_desc = "Qwen2.5-1B (4-bit)"
        elif 'llama' in model_name_lower and '3b' in model_name_lower:
            base_memory_gb = 6.0  # 3B model in 4-bit
            hidden_dim = 3072
            model_desc = "Llama-3B (4-bit)"
        elif 'glm' in model_name_lower:
            base_memory_gb = 18.0  # GLM model
            hidden_dim = 4096
            model_desc = "GLM (4-bit)"
        else:
            # Default/unknown model
            base_memory_gb = 10.0
            hidden_dim = 2048
            model_desc = "Unknown model"
        
        if self.config.use_lora:
            # LoRA memory estimation
            # Each LoRA module has rank * (input_dim + output_dim) parameters
            lora_params_per_module = self.config.lora_config.rank * hidden_dim * 2  # rank * (in + out)
            total_lora_params = lora_params_per_module * len(self.config.lora_config.target_modules)
            param_count_millions = total_lora_params / 1_000_000
            lora_memory_gb = param_count_millions * 4 / 1024  # LoRA weights in fp32
            optimizer_memory_gb = lora_memory_gb * 2  # Adam optimizer states
            
            # Activation memory depends on sequence length and hidden dim
            activation_memory_gb = (self.config.batch_size * self.config.max_length * hidden_dim * 4) / (1024**3)
            
            # Logits memory - critical for large vocab models
            logits_memory_gb = (self.config.batch_size * 256 * vocab_size * 4) / (1024**3)  # Chunked to 256 tokens
            
            total_memory_gb = base_memory_gb + lora_memory_gb + optimizer_memory_gb + activation_memory_gb + logits_memory_gb
        else:
            # Full fine-tuning would require much more memory
            total_memory_gb = base_memory_gb * 3  # Rough estimate
        
        print(f"\nMemory Estimate for {model_desc}:")
        print(f"  Vocabulary Size: {vocab_size:,} tokens")
        if vocab_size > 100000:
            print(f"  ⚠️  WARNING: Large vocabulary size may cause memory issues!")
        print(f"  Base Model: ~{base_memory_gb:.1f}GB")
        if self.config.use_lora:
            print(f"  LoRA Parameters: ~{lora_memory_gb:.1f}GB")
            print(f"  Optimizer States: ~{optimizer_memory_gb:.1f}GB")
            print(f"  Activations: ~{activation_memory_gb:.1f}GB")
            print(f"  Logits (chunked): ~{logits_memory_gb:.1f}GB")
        print(f"  Total Estimated: ~{total_memory_gb:.1f}GB")
        
        # Data distribution
        print("\n" + "="*80)
        print("DATA DISTRIBUTION")
        print("="*80)
        
        # Type distribution
        type_counts = defaultdict(int)
        for example in self.train_dataset.examples:
            type_counts[example.get('type', 'unknown')] += 1
        
        print("\nType Distribution:")
        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.train_dataset)) * 100
            print(f"  - {type_name}: {count} ({percentage:.1f}%)")
        
        # Response length distribution
        response_lengths = []
        # Handle TokenizerWrapper
        actual_tokenizer = self.tokenizer._tokenizer if hasattr(self.tokenizer, '_tokenizer') else self.tokenizer
        for example in self.train_dataset.examples[:min(1000, len(self.train_dataset.examples))]:  # Sample first 1000
            tokens = actual_tokenizer.encode(example['response'])
            response_lengths.append(len(tokens))
        
        print("\nResponse Length Distribution (tokens):")
        if response_lengths:
            print(f"  - Min: {min(response_lengths)}")
            print(f"  - Max: {max(response_lengths)}")
            print(f"  - Average: {np.mean(response_lengths):.0f}")
            print(f"  - Median: {np.median(response_lengths):.0f}")
        
        # Hardware info and warnings
        print("\n" + "="*80)
        print("HARDWARE & WARNINGS")
        print("="*80)
        
        try:
            import subprocess
            # Get total and available memory on macOS
            vm_stat = subprocess.check_output(['vm_stat'], text=True)
            page_size = 4096  # Default page size
            free_pages = 0
            inactive_pages = 0
            
            for line in vm_stat.split('\n'):
                if 'page size' in line:
                    page_size = int(line.split()[-2])
                elif 'Pages free:' in line:
                    free_pages = int(line.split()[-1].rstrip('.'))
                elif 'Pages inactive:' in line:
                    inactive_pages = int(line.split()[-1].rstrip('.'))
            
            # Available memory = free + inactive (can be reclaimed)
            available_gb = ((free_pages + inactive_pages) * page_size) / (1024**3)
            
            # Also get total memory
            total_mem = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
            total_gb = int(total_mem.strip()) / (1024**3)
            
            print(f"\nSystem Memory: {total_gb:.1f}GB total, ~{available_gb:.1f}GB available")
                    
            
            if total_memory_gb > available_gb * 0.8:  # Use 80% as threshold
                print(f"⚠️  WARNING: Estimated memory usage ({total_memory_gb:.1f}GB) may exceed available memory!")
                print("   Consider:")
                print("   - Reducing batch_size or max_length")
                print("   - Using a smaller model (e.g., Qwen2.5-3B)")
                print("   - Reducing LoRA rank")
                if vocab_size > 100000:
                    print(f"   - This model has a very large vocabulary ({vocab_size:,} tokens)!")
        except:
            pass
        
        # Additional useful information
        print("\n" + "="*80)
        print("ADDITIONAL INFORMATION")
        print("="*80)
        
        print(f"\nCheckpointing:")
        print(f"  Save frequency: Every {self.config.save_steps} steps")
        print(f"  Checkpoint directory: {self.config.output_dir}")
        
        print(f"\nEvaluation:")
        print(f"  Evaluation frequency: Every {self.config.eval_steps} steps")
        print(f"  Validation examples: {len(self.eval_dataset) if self.eval_dataset else 0}")
        
        print(f"\nData Loading:")
        print(f"  Shuffle training data: {self.config.dataset_config.shuffle}")
        print(f"  Random seed: {self.config.seed}")
        
        print(f"\nOptimizer:")
        print(f"  Type: AdamW")
        print(f"  Weight decay: {self.config.weight_decay}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  LR scheduler: {self.config.lr_scheduler}")
        
        print("\n" + "="*80)
        print("\nPress Enter to start training or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nTraining cancelled.")
            import sys
            sys.exit(0)
    
    def save_training_history(self):
        """Save training metrics and history."""
        history = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'config': asdict(self.config)
        }
        
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            # Loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Train Loss')
            if self.eval_losses:
                eval_steps, eval_values = zip(*self.eval_losses)
                plt.plot(eval_steps, eval_values, label='Eval Loss', marker='o')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig(self.output_dir / 'loss_curve.png')
            plt.close()
            
            # Learning rate curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.learning_rates)
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.savefig(self.output_dir / 'lr_curve.png')
            plt.close()
            
        except ImportError:
            logger.info("Matplotlib not available for plotting")


def load_config(config_path: Optional[str] = None, validate: bool = True) -> TrainingConfig:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Expand environment variables in model_name
        if 'model_name' in config_dict and isinstance(config_dict['model_name'], str):
            config_dict['model_name'] = os.path.expandvars(config_dict['model_name'])
        
        # Handle nested configs
        if 'lora_config' in config_dict:
            config_dict['lora_config'] = LoRAConfig(**config_dict['lora_config'])
        if 'dataset_config' in config_dict:
            config_dict['dataset_config'] = DatasetConfig(**config_dict['dataset_config'])
        
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Validate configuration if requested
    if validate:
        from validation import ConfigValidator, report_validation_issues, validate_and_fix_config
        
        # Fix common issues
        config_dict = asdict(config)
        fixed_dict = validate_and_fix_config(config_dict)
        
        # Recreate config from fixed dict if changes were made
        if fixed_dict != config_dict:
            if 'lora_config' in fixed_dict:
                fixed_dict['lora_config'] = LoRAConfig(**fixed_dict['lora_config'])
            if 'dataset_config' in fixed_dict:
                fixed_dict['dataset_config'] = DatasetConfig(**fixed_dict['dataset_config'])
            config = TrainingConfig(**fixed_dict)
            logger.info("Applied automatic configuration fixes")
        
        # Validate
        issues = ConfigValidator.validate_training_config(config)
        if issues:
            report_validation_issues(issues, raise_on_error=False)
            logger.warning("Proceeding with validation warnings...")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train cybersecurity instruction-tuned models with MLX and LoRA"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (YAML or JSON, default: configs/config.yaml)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (uses config file model if not specified)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer name or path (defaults to model)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=False,
        help="Path(s) to training data (uses config file path if not specified)"
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="cybersec",
        choices=["cybersec", "alpaca", "sharegpt", "openai"],
        help="Dataset format"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation split ratio"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for fine-tuning"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    
    # Resume training
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    
    # Hardware
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Monitoring
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="Use TensorBoard for logging"
    )
    
    # Preview mode
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview data formatting and training configuration without starting training"
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=5,
        help="Number of samples to preview (default: 5)"
    )
    
    # Debug mode
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simplified training loop for debugging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed diagnostics"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if args.config else TrainingConfig()
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.tokenizer:
        config.tokenizer_name = args.tokenizer
    if args.data_path:
        config.dataset_config.paths = args.data_path
    
    # Check if we have data paths from either command line or config
    if not args.data_path and not config.dataset_config.paths:
        parser.error("No data path specified. Provide --data-path or use a config file with dataset paths.")
    if args.data_format:
        config.dataset_config.format = args.data_format
    if args.max_length:
        config.max_length = args.max_length
        config.dataset_config.max_length = args.max_length
    if args.val_split:
        config.dataset_config.split_ratio = 1.0 - args.val_split
    
    # LoRA config
    config.use_lora = args.use_lora
    if args.lora_rank:
        config.lora_config.rank = args.lora_rank
    if args.lora_alpha:
        config.lora_config.alpha = args.lora_alpha
    if args.lora_dropout:
        config.lora_config.dropout = args.lora_dropout
    
    # Training config
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    
    # Output config
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.eval_steps:
        config.eval_steps = args.eval_steps
    if args.logging_steps:
        config.logging_steps = args.logging_steps
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Other config
    if args.seed:
        config.seed = args.seed
    if args.use_wandb:
        config.use_wandb = True
    if args.use_tensorboard:
        config.use_tensorboard = True
    
    # Set random seeds
    np.random.seed(config.seed)
    mx.random.seed(config.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    load_start = time.time()
    model, tokenizer = load(
        config.model_name,
        tokenizer_config={"trust_remote_code": config.trust_remote_code}
    )
    logger.info(f"Model loaded in {time.time() - load_start:.2f}s")
    
    # Force evaluation after model loading
    if args.debug:
        logger.info("[DEBUG] Synchronizing model after loading...")
    mx.eval(model.parameters())
    if args.debug:
        logger.info("[DEBUG] Model synchronized")
    
    # Use custom tokenizer if specified
    if config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=config.trust_remote_code
        )
    
    # Set padding token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load datasets
    logger.info("Loading datasets...")
    dataset_start = time.time()
    train_dataset, eval_dataset, collator = load_cybersec_datasets(
        data_paths=config.dataset_config.paths,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        val_split=1.0 - config.dataset_config.split_ratio,
        **{
            k: v for k, v in asdict(config.dataset_config).items()
            if k not in ['paths', 'split_ratio', 'max_length']
        }
    )
    logger.info(f"Datasets loaded in {time.time() - dataset_start:.2f}s")
    
    # Create trainer
    if args.debug:
        logger.info("[DEBUG] Creating trainer instance...")
    trainer_start = time.time()
    trainer = CyberSecTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        debug=args.debug
    )
    if args.debug:
        logger.info(f"[DEBUG] Trainer created in {time.time() - trainer_start:.2f}s")
    
    # Preview mode if requested
    if args.preview:
        logger.info("Running in preview mode...")
        trainer.preview_training_data(num_samples=args.preview_samples)
        # After preview, continue to training when user presses Enter
    
    # Start training - use simple version for debugging
    if args.debug:
        logger.info(f"[DEBUG] Starting training (simple={args.simple})...")
    if args.simple:
        trainer.train_simple()
    else:
        if args.debug:
            logger.info("[DEBUG] Calling trainer.train()...")
        trainer.train()
        
        # Save final configuration
        with open(Path(config.output_dir) / "training_config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        logger.info(f"Training completed! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()