# Automatic Batching and Speculative Decoding for CyberLLMInstruct

This document describes the automatic batching and speculative decoding features added to the CyberLLMInstruct pipeline to significantly improve performance.

## Overview

The pipeline now supports two major performance optimizations:

1. **Automatic Batching**: Groups multiple LLM requests together for parallel processing
2. **Speculative Decoding**: Uses a smaller "draft" model to predict tokens that are verified by the main model

## Performance Improvements

Expected performance gains:
- **Batching**: 3-5x throughput improvement for classification/filtering tasks
- **Speculative Decoding**: 1.5-2x speedup for longer generation tasks
- **Combined**: Up to 8x improvement for the overall pipeline

## Usage

### Basic Usage with Batching (Default)

All scripts now support batching by default:

```bash
# Data filtering with batching
python 2_data_filter.py --batch-size 32

# Data structuring with batching
python 3_data_structurer.py --batch-size 32

# Domain classification with batching
python 4_domain_classifier.py --batch-size 32

# Security alignment with batching
python 6_security_aligner.py --batch-size 16
```

### Speculative Decoding

Enable speculative decoding for faster generation:

```bash
# Use speculative decoding with default draft model
python 3_data_structurer.py --enable-speculative

# Use custom draft model
python 3_data_structurer.py --enable-speculative --draft-model mlx-community/gemma-3-1b-it-bf16
```

### Disable Batching

If you encounter issues or want to use the original sequential processing:

```bash
python 2_data_filter.py --no-batching
```

## Configuration Options

All scripts now support these additional arguments:

- `--batch-size`: Number of requests to process in parallel (default: 32)
- `--batch-timeout`: Time to wait for batch accumulation in seconds (default: 0.1)
- `--no-batching`: Disable batched processing
- `--enable-speculative`: Enable speculative decoding
- `--draft-model`: Model to use for speculative decoding (default: mlx-community/gemma-3-1b-it-bf16)

## Architecture

### BatchedMLXGenerator

The `BatchedMLXGenerator` class provides:
- Automatic request accumulation
- Dynamic batch sizing based on available memory
- Asynchronous processing with configurable timeout
- Graceful handling of different request sizes

### SpeculativeDecoder

The `SpeculativeDecoder` class implements:
- Draft token generation using a smaller model
- Parallel verification with the target model
- Adaptive speculation length based on acceptance rate
- Efficient caching for repeated contexts

## Testing

Run the test script to verify the batching utilities:

```bash
# Test all features
python test_batching.py

# Test only batching
python test_batching.py --test batch

# Test only speculative decoding
python test_batching.py --test speculative
```

## Memory Considerations

- Batching increases memory usage proportionally to batch size
- The system automatically adjusts batch size based on available memory
- Speculative decoding requires loading two models simultaneously

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Disable batching: `--no-batching`

### Slow Performance
- Increase batch size if memory allows: `--batch-size 64`
- Enable speculative decoding for long generations: `--enable-speculative`

### Model Compatibility
- Ensure draft and target models have compatible tokenizers for speculative decoding
- Some models may not support all batching features

## Performance Logging

The pipeline now includes comprehensive performance logging to track:
- Model inference metrics (tokens/second, memory usage)
- Pipeline stage performance (items/second, total time)
- Speculative decoding acceptance rates
- Error tracking and throughput analysis

### Using Performance Logging

```bash
# Run with automatic performance tracking
python example_with_performance.py

# Test with performance metrics
python test_batching.py
```

Performance logs are saved to `performance_logs/` directory with detailed metrics.

## Implementation Details

The batching system is implemented in four main files:
- `mlx_batch_utils.py`: Core batching utilities
- `mlx_speculative.py`: Speculative decoding implementation (fixed MLX compatibility)
- `performance_logger.py`: Performance tracking and reporting
- Updated pipeline scripts: Integration with existing workflow

Each pipeline stage maintains backward compatibility while offering significant performance improvements when batching is enabled.

## Known Issues Fixed

- **MLX Compatibility**: Removed `mx.no_grad()` usage as MLX handles gradients differently
- **Model Availability**: Updated default draft model to `mlx-community/gemma-3-1b-it-bf16`