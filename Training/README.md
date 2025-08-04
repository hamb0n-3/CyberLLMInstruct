# MLX LoRA Fine-tuning for Cybersecurity Dataset

This directory contains the training script for fine-tuning language models on the cybersecurity dataset using MLX and LoRA.

## Prerequisites

Ensure you have generated the dataset using the pipeline in `dataset_creation/`:
```bash
python3 dataset_creation/8_final_assembler.py --formats huggingface --split-ratios 0.8 0.1 0.1
```

## Configuration

Training is configured via `train_config.yaml` with the following sections:

### Model Configuration
- `model_path`: Base model to fine-tune
- `adapter_path`: Where to save LoRA adapters

### LoRA Parameters
- `fine_tune_type`: Type of fine-tuning (lora, dora, full)
- `num_layers`: Number of layers to apply LoRA
- `rank`: LoRA rank (decomposition dimension)
- `alpha`: LoRA alpha parameter (often 2*rank)
- `scale`: LoRA scaling factor
- `dropout`: Dropout rate for regularization

### Training Parameters
- `iterations`: Total training iterations
- `learning_rate`: Base learning rate
- `batch_size`: Training batch size
- `steps_per_eval`: Iterations between validation
- `steps_per_report`: Iterations between loss reports
- `val_batches`: Validation batches (-1 for all)
- `steps_per_save`: Checkpoint save frequency
- `grad_checkpoint`: Enable gradient checkpointing
- `seed`: Random seed for reproducibility

### Learning Rate Schedule
- `type`: Schedule type (constant, linear, cosine)
- `warmup_steps`: Number of warmup steps
- `min_lr`: Minimum LR for cosine schedule

### Gradient Clipping
- `enabled`: Enable/disable gradient clipping
- `max_norm`: Maximum gradient norm

### Dataset Configuration
- `data_path`: Path to HuggingFace dataset
- `max_seq_length`: Maximum sequence length
- `shuffle`: Whether to shuffle training data
- `num_workers`: Data loading workers

### Output Configuration
- `save_plots`: Save loss plots
- `plot_path`: Where to save plots
- `save_metrics`: Save training metrics
- `metrics_path`: Metrics JSON file path
- `checkpoint_dir`: Checkpoint directory

### Early Stopping
- `enabled`: Enable early stopping
- `patience`: Evaluations without improvement before stopping
- `min_delta`: Minimum improvement threshold

### Resume Training
- `enabled`: Enable resume from checkpoint
- `checkpoint_path`: Path to checkpoint file

## Usage

### Basic Training
```bash
cd Training
python3 mlx_lora_finetune.py
```

### With Custom Config
```bash
python3 mlx_lora_finetune.py --config my_config.yaml
```

### Test Model Only (No Training)
```bash
python3 mlx_lora_finetune.py --test-only
```

### Resume Training from Checkpoint
```bash
python3 mlx_lora_finetune.py --resume checkpoints/checkpoint_iter_100.npz
```

## Dataset Format

The script expects a HuggingFace dataset saved to disk with:
- Train, validation, and test splits
- Each sample containing 'instruction' and 'response' fields
- The script will automatically apply the model's chat template

## Output

After training:
- LoRA adapters saved to `adapters/` directory
- Training loss plot saved as `training_loss.png`
- Training metrics saved as `training_metrics.json`
- Checkpoints saved to `checkpoints/` directory
- Optional W&B or TensorBoard logs

## Advanced Features

### Learning Rate Schedules
The script supports three LR schedules:
- **constant**: Fixed learning rate
- **linear**: Linear decay after warmup
- **cosine**: Cosine annealing with warmup

### Early Stopping
Training automatically stops when validation loss stops improving based on patience and min_delta settings.

### Checkpoint Management
- Automatic checkpoint saving at specified intervals
- Resume training from any checkpoint
- Each checkpoint includes model weights and optimizer state

### Fine-tuning Types
- **lora**: Standard LoRA fine-tuning (most efficient)
- **dora**: Weight-Decomposed LoRA (not yet implemented)
- **full**: Full model fine-tuning (memory intensive)

## Memory Considerations

- Adjust `batch_size` in config based on available GPU memory
- Enable `grad_checkpoint` for memory-efficient training
- The default config uses 8-bit quantized models for efficiency
- Note: Gradient accumulation is not currently supported in MLX