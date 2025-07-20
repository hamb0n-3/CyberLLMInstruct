# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyberLLMInstruct is a data pipeline project for creating high-quality cybersecurity instruction datasets for training Large Language Models. The project uses Apple's MLX framework for local LLM processing and implements a 7-stage sequential pipeline.

## Environment Setup

This project requires:
- Apple Silicon Mac (M1/M2/M3/M4) for MLX framework
- Python 3.8+
- Environment variables in `.env` file:
  - `GITHUB_TOKEN` (required)
  - `NVD_API_KEY` (required)
  - `OPENCVE_EMAIL` and `OPENCVE_PASSWORD` (required)
  - Optional: Various threat intelligence API keys

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline

The pipeline must be run sequentially. Each stage depends on the output of the previous stage:

```bash
# Stage 1: Collect raw data from cybersecurity sources
python3 dataset_creation/1_data_collector.py --sources all

# Stage 2: Filter data using local MLX models
python3 dataset_creation/2_data_filter.py

# Stage 2 with batch processing and speculative decoding (faster)
python3 dataset_creation/2_data_filter.py --batch-size 32 --draft-model mlx-community/Qwen2.5-0.5B-Instruct --num-draft-tokens 4

# Stage 2 with verbose mode for debugging
python3 dataset_creation/2_data_filter.py --verbose

# Stage 3: Structure into instruction-response format
python3 dataset_creation/3_data_structurer.py

# Stage 4: Classify by cybersecurity domain
python3 dataset_creation/4_domain_classifier.py

# Stage 5: Manual review (interactive CLI)
python3 dataset_creation/5_manual_reviewer.py

# Stage 6: Add security-aligned synthetic examples
python3 dataset_creation/6_security_aligner.py --ratio 0.2

# Stage 7: Final assembly with deduplication
python3 dataset_creation/8_final_assembler.py
```

### Key Command-Line Options

- Most scripts support `--input-dir` and `--output-dir` for custom paths
- Scripts using LLMs support `--model` to specify the MLX model
- Scripts 3, 4, and 6 support `--disable-llm` for debugging without ML processing
- Script 3 supports `--workers` for parallel processing
- Script 2 (data filter) supports:
  - `--batch-size`: Number of entries to process in parallel (default: 16)
  - `--draft-model`: Path to a smaller draft model for speculative decoding
  - `--num-draft-tokens`: Number of tokens to generate speculatively (default: 4)
  - `--verbose`: Enable detailed logging for debugging

## Architecture Overview

### Pipeline Flow
```
raw_data/ → filtered/ → structured/ → domain_classified/ → manually_reviewed/ → security_aligned/ → final_dataset/
```

### Key Components

1. **Data Collection (`1_data_collector.py`)**: Fetches from NVD, MITRE ATT&CK, OpenCVE, CAPEC, ArXiv, and other security sources
2. **LLM Processing**: Uses local MLX models (default: Phi-3-mini) for filtering, structuring, and classification
3. **Human Review (`5_manual_reviewer.py`)**: Interactive CLI using questionary and rich for quality control
4. **Security Alignment (`6_security_aligner.py`)**: Generates synthetic examples focused on defensive security practices

### Data Format

Each stage produces JSONL files with entries containing:
- `instruction`: The question or task
- `response`: The answer or completion
- `metadata`: Source information, timestamps, and classification data

### MLX Model Usage

The project uses MLX models throughout the pipeline. Default model paths are configured for Phi-3-mini, but can be overridden with the `--model` parameter. Models are loaded locally without external API dependencies.

### Performance Optimizations

**Batch Processing**: Stage 2 now supports batch processing which processes multiple entries simultaneously, providing significant speedup.

**Speculative Decoding**: When using a draft model, the system uses speculative decoding to accelerate generation:
- The draft model quickly generates multiple token predictions
- The main model verifies these predictions in parallel
- Expected speedup: 1.5-3x over standard generation

**Model-Specific Prompt Formatting**: The system automatically detects model type (Qwen, Phi, Llama, etc.) and uses appropriate prompt formatting for better results.

**Performance Benchmarking**: Detailed metrics are tracked and logged:
- Tokens per second for each model
- Processing time per file
- Speculative decoding acceptance rate
- Batch processing efficiency

Recommended draft model combinations:
- For Qwen models: Use a smaller Qwen model as draft (e.g., Qwen3-4B with Qwen3-30B)
- For Phi models: `mlx-community/Qwen2.5-0.5B-Instruct` or smaller Phi variants

## Development Guidelines

### When modifying the pipeline:
1. Each stage should be independently runnable and resumable
2. Always save intermediate outputs for debugging
3. Use the existing logging infrastructure with rich console output
4. Maintain the JSONL format for data interchange between stages

### When adding new data sources:
1. Add the collection logic to `1_data_collector.py`
2. Update the `--sources` argument parser
3. Store API keys in environment variables
4. Implement proper error handling and rate limiting

### Code Style:
- The project uses descriptive variable names and comprehensive docstrings
- Error handling uses try-except blocks with detailed logging
- Progress bars (tqdm/rich) are used for long-running operations
- Interactive prompts use questionary with rich formatting

## Notes

- No test infrastructure currently exists
- The project is actively developed on branch `attempt-batch-decode-1-script-attempt3`
- Many documentation files have been removed in the current branch but core functionality remains
- The manual review stage (`5_manual_reviewer.py`) requires terminal interaction and cannot be run in automated environments