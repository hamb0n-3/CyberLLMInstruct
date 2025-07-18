# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyberLLMInstruct is a dual-purpose project for cybersecurity-focused LLMs:
1. **Dataset Creation Pipeline**: 7-step pipeline for creating cybersecurity instruction-response datasets
2. **MLX ParaLLM**: Advanced inference engine with speculative decoding and continuous batching

## Key Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset Creation Pipeline
```bash
# Run pipeline steps sequentially:
python dataset_creation/1_data_collector.py
python dataset_creation/2_data_filter.py
python dataset_creation/3_data_structurer.py
python dataset_creation/4_domain_classifier.py
python dataset_creation/5_manual_reviewer.py
python dataset_creation/6_security_aligner.py
python dataset_creation/8_final_assembler.py
```

### MLX Inference Server
```bash
# Start the advanced MLX server
./dataset_creation/start-advanced-server.sh --model <model_name> --port 8080

# Test the server
python dataset_creation/test_advanced_client.py

# Common server options:
# --draft-model: Enable speculative decoding
# --max-batch-size: Set continuous batching size (default: 16)
# --cache-size: Set KV cache size
```

## Architecture

### Dataset Creation Pipeline (`dataset_creation/`)
- **1_data_collector.py**: Aggregates data from CVE, MITRE ATT&CK, CTF challenges, arXiv papers, security blogs
- **2_data_filter.py**: Quality filtering, removes irrelevant/low-quality entries
- **3_data_structurer.py**: Standardizes to instruction-response-source-date format
- **4_domain_classifier.py**: Categorizes into 15 cybersecurity domains (pen testing, malware analysis, etc.)
- **5_manual_reviewer.py**: Interactive UI for human validation
- **6_security_aligner.py**: Enhances security context and aligns responses
- **8_final_assembler.py**: Compiles final dataset with metadata

### MLX ParaLLM Inference (`dataset_creation/mlx_parallm/`)
- **advanced_server.py**: FastAPI server handling API requests
- **inference_engine.py**: Core MLX inference logic with model loading
- **speculative_decoding.py**: Implements draft model speculative decoding
- **continuous_batching.py**: Dynamic request batching for efficiency

### Data Flow
1. Raw data collected → `raw_data/` directory
2. Processed through pipeline → `processed_data/` directory  
3. Final dataset → `cybersec_instruct_dataset.json`
4. Inference server loads MLX models from HuggingFace or local paths

## Key Technologies
- **MLX Framework**: Apple's ML framework optimized for Apple Silicon
- **Python 3.13**: Requires Python 3.8+
- **Ollama**: Used for LLM inference during dataset processing (requires separate installation)
- **FastAPI**: REST API for inference server

## Important Notes
- The project is actively being refactored (many files recently deleted/reorganized)
- Ollama must be running locally for dataset creation steps 5 and 6
- MLX models must be in MLX format (use mlx-lm convert if needed)
- The inference server supports both standard and speculative decoding modes
- Dataset creation is computationally intensive and may take several hours

## Common Issues
- If Ollama connection fails, ensure Ollama is running (`ollama serve`)
- For MLX model loading errors, verify model is in MLX format
- Memory issues: Reduce batch size or use smaller models
- API rate limits: Adjust delay parameters in data collector