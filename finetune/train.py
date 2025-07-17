import os
import logging
import torch
from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations (same as in data_prep.py)
MODEL_CONFIGS = {
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'mistral-7b': 'mistralai/Mistral-7B-v0.3',
    'llama-2-70b': 'meta-llama/Llama-2-70b',
    'qwen-coder-7b': 'Qwen/Qwen2.5-Coder-7B',
    'gemma-2-9b': 'google/gemma-2-9b',
    'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
    'phi-3-mini': 'microsoft/Phi-3.5-mini-instruct'
}

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name: str = field(
        metadata={"help": "Model identifier from huggingface.co/models", "choices": MODEL_CONFIGS.keys()}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter efficient fine-tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input."""
    dataset_path: str = field(
        default="processed_data",
        metadata={"help": "Path to the processed dataset directory"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """Custom training arguments."""
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use Weights & Biases logging"}
    )
    output_dir: str = field(
        default="finetuned_models",
        metadata={"help": "Directory to save the fine-tuned model"}
    )
    num_train_epochs: float = field(
        default=10.0,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Ratio of steps for warmup"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between evaluations"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )

def setup_wandb(model_name: str, training_args: TrainingArguments):
    """Initialize Weights & Biases logging."""
    if training_args.use_wandb:
        wandb.init(
            project="cyberllm-finetune",
            name=f"{model_name}-finetune",
            config={
                "model": model_name,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs
            }
        )

def load_tokenizer_and_model(model_args: ModelArguments, training_args: TrainingArguments):
    """Load the tokenizer and model with specified configurations."""
    model_name_or_path = MODEL_CONFIGS[model_args.model_name]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine quantization config
    quantization_config = None
    if model_args.use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if model_args.use_lora:
        logger.info("Applying LoRA to the model")
        # Prepare model for k-bit training if using quantization
        if model_args.use_4bit or model_args.use_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return tokenizer, model

def train():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup W&B logging
    if training_args.use_wandb:
        setup_wandb(model_args.model_name, training_args)
    
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(model_args, training_args)
    
    # Load datasets
    logger.info(f"Loading dataset from {data_args.dataset_path}")
    dataset = load_from_disk(data_args.dataset_path)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Close wandb run
    if training_args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train() 