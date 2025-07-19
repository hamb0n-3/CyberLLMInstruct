# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyberLLMInstruct is a comprehensive cybersecurity dataset creation and LLM fine-tuning pipeline designed to build high-quality instruction-response datasets for training Large Language Models (LLMs) in cybersecurity tasks.

## Prerequisites & Environment

- **Platform**: Apple Silicon (M1/M2/M3/M4) optimized
- **Python**: 3.8+
- **Local LLM**: Ollama with models gemma:2b and mistral:7b
- **Environment Variables**: Create `.env` file with:
  - `GITHUB_TOKEN` (required)
  - `NVD_API_KEY` (required)
  - `OPENCVE_EMAIL` & `OPENCVE_PASSWORD` (required)
  - Optional: `VIRUSTOTAL_API_KEY`, `ALIENVAULT_API_KEY`, `SHODAN_API_KEY`

## Common Commands

### Dataset Creation Pipeline (Sequential)
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline stages
python dataset_creation/1_data_collector.py --sources all
python dataset_creation/2_data_filter.py
python dataset_creation/3_data_structurer.py
python dataset_creation/4_domain_classifier.py
python dataset_creation/5_manual_reviewer.py
python dataset_creation/6_security_aligner.py --ratio 0.2
python dataset_creation/8_final_assembler.py
```

### Fine-tuning Pipeline
```bash
# Data preparation
python finetune/data_prep.py \
    --model_name llama-3-8b \
    --input_path dataset/processed_data/CyberLLMInstruct_dataset.json \
    --output_dir processed_data

# Training
python finetune/train.py \
    --model_name llama-3-8b \
    --dataset_path processed_data/llama-3-8b \
    --output_dir finetuned_models/llama-3-8b \
    --use_lora True \
    --use_8bit True

# Inference
python finetune/inference.py \
    --model_name llama-3-8b \
    --model_path finetuned_models/llama-3-8b \
    --interactive
```

### Testing & Evaluation
```bash
# DeepEval testing
cd examples/deepeval && python test.py

# CyberLLM evaluation
cd examples/cybermetric && python3 CyberLLM-Eval.py
```

### Utility Scripts
```bash
# Categorize dataset
python scripts/categorise.py --input-dir dataset --output-dir categorised

# Export dataset
python scripts/dataset_export.py --input-dir dataset --output-dir release
```

## Architecture & Key Components

### Dataset Creation Pipeline (`/dataset_creation/`)
1. **Data Collection**: Gathers from NVD, MITRE ATT&CK, OpenCVE, GitHub, etc.
2. **Filtering**: Removes duplicates and low-quality entries
3. **Structuring**: Formats into instruction-response pairs using local LLMs
4. **Classification**: Categorizes by cybersecurity domains
5. **Human Review**: Manual quality control interface
6. **Security Alignment**: Ensures safe and helpful outputs
7. **Final Assembly**: Combines all processed data

### Fine-tuning Pipeline (`/finetune/`)
- Supports multiple architectures: Llama, Mistral, Qwen, Gemma, Phi
- Full fine-tuning and LoRA parameter-efficient methods
- Checkpoint management utilities
- Interactive inference mode

### Key Technologies
- **MLX Framework**: Apple's ML framework for on-device processing
- **Ollama**: Local LLM inference
- **Transformers**: Hugging Face ecosystem integration
- **Rich/Questionary**: Interactive CLI interfaces

## Performance Optimizations

The pipeline now supports automatic batching and speculative decoding:
- **Automatic Batching**: Groups LLM requests for 3-5x throughput improvement
- **Speculative Decoding**: Uses draft models for 1.5-2x generation speedup
- Enable with `--enable-speculative` or disable with `--no-batching`
- Default batch size: 32 (configurable with `--batch-size`)

## Important Notes

- No formal test suite or linting configuration exists
- Pipeline stages must be run sequentially
- Each stage saves intermediate results allowing resume capability
- Manual review stage (5) requires human interaction
- Security alignment stage ensures defensive security focus only
- Batching increases memory usage; reduce batch size if OOM occurs