# CyberLLMInstruct Fine-tuning Pipeline

This directory contains scripts for fine-tuning large language models on the CyberLLMInstruct dataset, as described in our paper ([arXiv:2503.09334](https://arxiv.org/abs/2503.09334)). The pipeline supports various models including Llama, Mistral, Qwen, Gemma, and Phi.

## Supported Models

- Llama 3.1 8B (`meta-llama/Llama-3.1-8B`)
- Mistral 7B (`mistralai/Mistral-7B-v0.3`)
- Llama 2 70B (`meta-llama/Llama-2-70b`)
- Qwen 2.5 Coder 7B (`Qwen/Qwen2.5-Coder-7B`)
- Gemma 2 9B (`google/gemma-2-9b`)
- Llama 3 8B (`meta-llama/Meta-Llama-3-8B`)
- Phi 3 Mini Instruct 3.8B (`microsoft/Phi-3.5-mini-instruct`)

## Pipeline Overview

The fine-tuning process consists of three main steps:

1. **Data Preparation** (`data_prep.py`): Preprocesses the CyberLLMInstruct dataset for training
2. **Training** (`train.py`): Fine-tunes the selected model using the preprocessed data
3. **Inference** (`inference.py`): Runs inference using the fine-tuned model

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the Hugging Face model hub and the required model weights.

## Usage

### 1. Data Preparation

The data preparation script processes the CyberLLMInstruct dataset that you can generate using the pipeline in the dataset_creation directory:

```bash
python data_prep.py \
    --model_name llama-3-8b \
    --input_path dataset/processed_data/CyberLLMInstruct_dataset.json \
    --output_dir processed_data
```

Key arguments:
- `--model_name`: Name of the model to prepare data for
- `--input_path`: Path to the input dataset (default: dataset/processed_data/CyberLLMInstruct_dataset.json)
- `--output_dir`: Directory to save processed datasets (default: processed_data)
- `--train_ratio`: Proportion of data for training (default: 0.8)
- `--val_ratio`: Proportion of data for validation (default: 0.1)
- `--test_ratio`: Proportion of data for testing (default: 0.1)

### 2. Training

The training script supports both full model fine-tuning and parameter-efficient fine-tuning using LoRA:

```bash
python train.py \
    --model_name llama-3-8b \
    --dataset_path processed_data/llama-3-8b \
    --output_dir finetuned_models/llama-3-8b \
    --use_lora True \
    --use_8bit True \
    --learning_rate 2e-4 \
    --num_train_epochs 10
```

Key arguments:
- `--model_name`: Name of the model to fine-tune
- `--dataset_path`: Path to the processed dataset
- `--output_dir`: Directory to save the fine-tuned model
- `--use_lora`: Whether to use LoRA (default: True)
- `--use_4bit`/`--use_8bit`: Enable quantisation for memory efficiency
- `--learning_rate`: Learning rate (default: 2e-4)
- `--num_train_epochs`: Number of training epochs (default: 10)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)

### 3. Inference

The inference script supports both interactive and batch processing modes:

```bash
# Interactive mode
python inference.py \
    --model_name llama-3-8b \
    --model_path finetuned_models/llama-3-8b \
    --interactive \
    --use_8bit True

# Batch processing
python inference.py \
    --model_name llama-3-8b \
    --model_path finetuned_models/llama-3-8b \
    --input_file prompts.json \
    --output_file responses.json
```

Key arguments:
- `--model_name`: Name of the model to use
- `--model_path`: Path to the fine-tuned model
- `--interactive`: Run in interactive mode
- `--input_file`: Path to input file (for batch processing)
- `--output_file`: Path to save outputs (for batch processing)
- `--use_4bit`/`--use_8bit`: Enable quantisation
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Nucleus sampling probability (default: 0.9)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)

### 4. Checkpoint Management

The checkpoint management script helps manage training checkpoints:

```bash
# List available checkpoints
python checkpoint_manager.py \
    --model_name llama-3-8b \
    --checkpoint_dir checkpoints \
    --action list

# Convert checkpoint for deployment
python checkpoint_manager.py \
    --model_name llama-3-8b \
    --checkpoint_dir checkpoints \
    --output_dir deployment \
    --action convert \
    --checkpoint_id checkpoint_20240220_123456_step1000
```

Key actions:
- `list`: List all available checkpoints
- `compress`: Compress a checkpoint for storage
- `convert`: Convert a checkpoint for deployment
- `clean`: Clean old checkpoints

## Directory Structure

```
finetune/
├── data_prep.py          # Data preparation script
├── train.py             # Training script
├── inference.py         # Inference script
├── checkpoint_manager.py # Checkpoint management script
└── README.md            # This file

dataset/
└── processed_data/      # Generated dataset from pipeline
    └── CyberLLMInstruct_dataset.json  # Generated dataset

processed_data/          # Processed datasets
└── <model_name>/
    ├── train/
    ├── validation/
    └── test/

finetuned_models/       # Fine-tuned models
└── <model_name>/
    └── ...
```

## Notes

- For large models (e.g., Llama 2 70B), it's recommended to use quantisation (4-bit or 8-bit) and LoRA to reduce memory requirements.
- The scripts include comprehensive error handling and logging.
- Training progress and metrics are logged using Weights & Biases (wandb).
- All scripts support both CPU and GPU execution, with automatic device selection.