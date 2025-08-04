# Troubleshooting Guide

## Common Issues and Solutions

### Memory Issues

#### Problem: Training crashes with >400GB memory usage
**Cause**: Using models with very large vocabulary sizes (>100k tokens)

**Solutions**:
1. Use smaller models (Qwen2.5-1B or 3B)
2. Reduce batch size to 1-2
3. Reduce max sequence length to 512
4. Enable gradient checkpointing
5. Use memory-optimized configs in `configs/models/`

#### Memory Optimization Techniques
- **Chunked Loss Computation**: Processes sequences in 256-token chunks
- **Token Cache Management**: Limited to 1000 entries with FIFO eviction
- **Periodic Cleanup**: Garbage collection every 50 steps
- **Optimized Settings**: batch_size=2, max_length=1024, LoRA rank=8

### Model Loading Issues

#### Problem: Model fails to load
**Solutions**:
1. Check model name is correct
2. Ensure you have internet connection for first download
3. Verify model is MLX-compatible
4. Check available disk space

#### Problem: LoRA not applying correctly
**Solutions**:
1. Verify target modules match model architecture
2. Check model has been frozen before LoRA application
3. Ensure LoRA rank is reasonable (4-32)

### Training Issues

#### Problem: No trainable parameters found
**Cause**: LoRA not applied correctly or model not frozen

**Solution**:
```python
# Correct order:
model.freeze()  # First freeze
apply_lora_to_model(model, lora_config)  # Then apply LoRA
```

#### Problem: Loss not decreasing
**Solutions**:
1. Check learning rate (try 2e-4 to 5e-5)
2. Verify data is formatted correctly
3. Ensure labels are not all masked (-100)
4. Try smaller LoRA rank

### Data Issues

#### Problem: Dataset not loading
**Solutions**:
1. Check data path exists
2. Verify JSON format is correct
3. Ensure sufficient examples (>100)
4. Check tokenizer compatibility

### Hardware Issues

#### Problem: MPS (Metal) errors on Mac
**Solutions**:
1. Update macOS to latest version
2. Set `use_mps: false` in config
3. Reduce batch size
4. Clear MLX cache

### Quick Fixes

```bash
# Clear all outputs and start fresh
rm -rf outputs/* logs/*

# Test with minimal configuration
python train_cybersecurity_model.py \
    --model mlx-community/Qwen2.5-1B-Instruct-MLX-4bit \
    --batch-size 1 \
    --max-length 512 \
    --num-epochs 1 \
    --simple

# Use safe quickstart
./quickstart.sh  # Choose option 1 (smallest model)
```

## Error Messages

### "Vocabulary size is dangerously large"
- Switch to a model with smaller vocabulary (<65k tokens)
- Models to avoid: Qwen3-30B (151k vocab)

### "No module named 'mlx'"
```bash
pip install mlx mlx-lm
```

### "CUDA/GPU errors" (on non-Mac systems)
This codebase is designed for Apple Silicon. For NVIDIA GPUs, use standard PyTorch training.