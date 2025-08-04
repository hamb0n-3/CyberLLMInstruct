# Cybersecurity LLM Training Suite

This directory contains a comprehensive training framework for fine-tuning Large Language Models (LLMs) on cybersecurity instruction datasets using Apple's MLX framework with LoRA (Low-Rank Adaptation) support.

## 🚀 Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with configurable rank and target modules
- **Multi-Dataset Support**: Train on multiple datasets with configurable mixing ratios
- **MLX Optimization**: Optimized for Apple Silicon with Metal acceleration
- **Memory-Safe Training**: Chunked loss computation and automatic memory management
- **Comprehensive Evaluation**: Perplexity, generation quality, and domain-specific metrics
- **Flexible Configuration**: YAML/JSON config files for reproducible experiments
- **Advanced Monitoring**: TensorBoard, Weights & Biases, and memory tracking
- **Checkpoint Management**: Save/resume training with automatic cleanup
- **Model Merging**: Merge LoRA adapters back into base model for deployment

## 📋 Requirements

- Apple Silicon Mac (M1/M2/M3/M4) with 16GB+ RAM
- macOS 13.0+
- Python 3.8+
- Dependencies from `requirements.txt`

## 🛠️ Installation

```bash
# From the project root
pip install -r requirements.txt
```

## ⚡ Quick Start

### Recommended: Safe Training

For memory-safe training with automatic checks:

```bash
./train_safe.sh
```

This will:
1. Check available memory
2. Warn about potential issues
3. Start memory monitoring
4. Run training with config.yaml

### Alternative: Interactive Setup

```bash
./quickstart.sh
```

This interactive script will guide you through model selection and configuration.

## 📊 Training

### Using the Unified Configuration

The training setup is now consolidated in `config.yaml`:

```bash
# Default configuration (Qwen3-30B-A3B)
python train_cybersecurity_model.py

# Or explicitly specify config
python train_cybersecurity_model.py --config config.yaml
```

### Command-line Overrides

You can override any config setting via command line:

```bash
python train_cybersecurity_model.py \
    --batch-size 2 \
    --max-length 256 \
    --num-epochs 1
```

### Memory Safety Check

Before training, always check memory requirements:

```bash
python check_memory.py
```

This will analyze config.yaml and warn if memory usage might be too high.

### Multi-Dataset Training

Train on multiple datasets with custom mixing:

```bash
python train_cybersecurity_model.py \
    --config configs/multi_dataset_config.yaml
```

### Resume Training

Continue from a checkpoint:

```bash
python train_cybersecurity_model.py \
    --config configs/cybersec_lora_config.yaml \
    --resume-from-checkpoint ./outputs/my_model/checkpoint-1000
```

## 🧪 Evaluation

Evaluate a trained model:

```bash
python evaluate_model.py \
    --model ./outputs/my_model/checkpoint-best \
    --data-path ../dataset_creation/structured_data \
    --output evaluation_results.json
```

Evaluate with specific metrics only:

```bash
python evaluate_model.py \
    --model ./outputs/my_model/checkpoint-best \
    --data-path ../dataset_creation/structured_data \
    --skip-perplexity \
    --temperature 0.7 \
    --max-new-tokens 512
```

## 🔧 Model Merging

Merge LoRA adapters into the base model:

```bash
python merge_lora.py \
    --base-model mlx-community/Qwen2.5-3B-Instruct-MLX-4bit \
    --lora-checkpoint ./outputs/my_model/checkpoint-best \
    --output-dir ./merged_model
```

## 📁 Configuration Options

### Training Configuration

Key configuration parameters:

- **Model Settings**:
  - `model_name`: Base model to fine-tune
  - `trust_remote_code`: Allow custom model code
  
- **LoRA Parameters**:
  - `rank`: LoRA rank (typically 8-64)
  - `alpha`: LoRA scaling factor
  - `dropout`: Dropout for LoRA layers
  - `target_modules`: Which modules to apply LoRA to

- **Training Hyperparameters**:
  - `learning_rate`: Initial learning rate
  - `num_epochs`: Number of training epochs
  - `batch_size`: Training batch size
  - `gradient_accumulation_steps`: Steps before optimizer update
  
- **Dataset Configuration**:
  - `paths`: List of data directories/files
  - `format`: Dataset format (cybersec, alpaca, sharegpt, openai)
  - `max_length`: Maximum sequence length
  - `filter_by_type`: Filter data by type
  - `system_prompt`: System prompt to prepend

### Example Configurations

1. **Small Model Quick Training** (`configs/quick_train.yaml`):
   - 1B parameter model
   - LoRA rank 8
   - 1 epoch
   - Good for testing

2. **Production Training** (`configs/production.yaml`):
   - 7B+ parameter model
   - LoRA rank 32-64
   - 3-5 epochs
   - Extensive evaluation

3. **Multi-Dataset Training** (`configs/multi_dataset_config.yaml`):
   - Mix cybersecurity with general data
   - Custom mixing ratios
   - Enhanced generalization

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./outputs/my_model/tensorboard
```

### Weights & Biases

Set up W&B:

```bash
wandb login
python train_cybersecurity_model.py --config config.yaml --use-wandb
```

## 💾 Memory Optimizations

The training framework includes several memory optimizations to prevent crashes:

### Implemented Optimizations

1. **Chunked Loss Computation**: Processes sequences in 256-token chunks
2. **Token Cache Management**: Limited to 1000 entries with FIFO eviction
3. **Periodic Memory Cleanup**: Garbage collection every 50 steps
4. **Safe Default Configs**: Small batch sizes and sequence lengths

### Memory Usage Formula

```
Memory = Base_Model + Logits_Chunk + LoRA_Params + Optimizer_States + Activations

Example for Qwen2.5-3B with default settings:
- Base Model: ~6GB (4-bit quantized)
- Logits: 0.4GB (chunked)
- LoRA: 0.1GB
- Optimizer: 0.2GB
- Activations: 2GB
- Total: ~10-15GB (with overhead)
```

### Safe Model Recommendations

| Model | Vocab Size | Memory Usage | Speed |
|-------|------------|--------------|-------|
| Qwen2.5-1B | 151k | ~8GB | Fast |
| Qwen2.5-3B | 151k | ~15GB | Medium |
| Llama-3.2-3B | 128k | ~15GB | Medium |

**⚠️ Avoid**: Models with >100k vocabulary size on systems with <64GB RAM

## 🎯 Best Practices

1. **Start Small**: Use Qwen2.5-1B for initial testing
2. **Monitor Memory**: Run `memory_monitor.py` in parallel
3. **Use Safe Configs**: Start with configs in `configs/models/`
4. **Batch Size**: Start with 2, increase gradually
5. **Learning Rate**: 2e-4 for LoRA works well
6. **Checkpointing**: Save every 500 steps

## 🐛 Troubleshooting

See `docs/TROUBLESHOOTING.md` for detailed solutions. Quick fixes:

### Out of Memory
```bash
# Use minimal config
python train_cybersecurity_model.py \
    --model mlx-community/Qwen2.5-1B-Instruct-MLX-4bit \
    --batch-size 1 \
    --max-length 512
```

### Training Crashes
- Check model vocabulary size: avoid >100k
- Use configs in `configs/models/` (pre-tested)
- Run `./quickstart.sh` for guided setup

## 📈 Performance Tips

- **Batch Size**: Largest that fits in memory (typically 2-8)
- **Gradient Accumulation**: Use to simulate larger batches
- **Mixed Precision**: Not yet supported in MLX, coming soon
- **Data Loading**: Pre-tokenize data for faster training

## 📝 Output Structure

After training, the output directory contains:

```
outputs/my_model/
├── checkpoint-{step}/
│   ├── lora_weights.safetensors    # LoRA weights
│   ├── training_state.pkl          # Training state
│   └── tokenizer files             # Tokenizer config
├── checkpoint-best/                # Best model checkpoint
├── checkpoint-final/               # Final checkpoint
├── tensorboard/                    # TensorBoard logs
├── training_config.json            # Training configuration
├── training_history.json           # Loss and metrics history
├── loss_curve.png                  # Training curves
└── training_*.log                  # Training logs
```

## 🔗 Integration with Pipeline

The training module integrates with the data pipeline:

1. Run data collection and filtering (stages 1-7)
2. Train model on structured data
3. Evaluate on held-out cybersecurity benchmarks
4. Deploy merged model for inference

## 📄 License

Same as the parent project - see root LICENSE file.

## 🤝 Contributing

Contributions welcome! Please:
- Test changes on small datasets first
- Document new features
- Add configuration examples
- Update this README

## 📚 References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)