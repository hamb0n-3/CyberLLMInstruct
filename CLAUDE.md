# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyberLLMInstruct is a multi-stage pipeline for creating high-quality cybersecurity instruction datasets for LLM training. It runs on Apple Silicon M4 max with 128GB memory and 1TB SSD using the MLX framework for local LLM inference.

## Important
- Do not make fallback methods, workarounds, placeholders.
- Do not over engineer if possible.
- Attempt to research the best ways to implement a solution.

## Running the Pipeline

Execute scripts sequentially in order:

```bash
# Stage 1: Collect raw data from cybersecurity sources
python3 dataset_creation/1_data_collector.py --sources all

# Stage 2: Filter for relevance and enhance with LLM
python3 dataset_creation/2_data_filter.py --model mlx-community/Qwen3-8B-4bit-DWQ-053125

# Stage 3: Structure into instruction-response pairs
python3 dataset_creation/3_data_structurer.py --model mlx-community/Qwen3-8B-4bit

# Stage 4: Classify into cybersecurity domains
python3 dataset_creation/4_domain_classifier.py --model mlx-community/c4ai-command-r-v01-4bit

# Stage 5: Manual review (interactive CLI)
python3 dataset_creation/5_manual_reviewer.py

# Stage 6: Add security alignment examples
python3 dataset_creation/6_security_aligner.py --ratio 0.2

# Stage 7: Final assembly and deduplication
python3 dataset_creation/8_final_assembler.py
```

### Key Parameters

All LLM-based scripts (2-4, 6) support these parameters:
- `--temperature` (default: 0.6-0.7)
- `--top-p` (default: 0.95)
- `--top-k` (default: 20)
- `--repetition-penalty` (default: 1.0)
- `--disable-llm` (skip LLM processing for testing)

### Performance Options

For faster processing with `2_data_filter_batched.py`:
```bash
# Use parallel LLM instances (2-4x speedup)
python3 dataset_creation/2_data_filter_batched.py --num-llm-instances 2 --batch-size 8

# Process specific sources only
python3 dataset_creation/2_data_filter.py --sources opencve mitre_attack
```

## Architecture & Data Flow

### Pipeline Flow
```
raw_data/ → filtered_data/ → structured_data/ → domain_classified/ 
    → reviewed_data/ → security_aligned/ → final_dataset/
```

### Key Components

**utils.py**: Shared utilities
- `BenchmarkTracker`: Performance monitoring (memory, time, LLM metrics)
- `extract_first_json_object`: Robust JSON extraction from LLM responses

**MLX Model Pattern**: All LLM scripts follow this pattern:
1. Load model with fallback error handling
2. Apply chat template if supported (Qwen models)
3. Use retry logic with progressive token limits [512, 1024, 1500]
4. Track performance with BenchmarkTracker
5. Use 8-bit KV cache quantization for memory efficiency

**Data Sources** (1_data_collector.py):
- CVE: NVD API, OpenCVE
- Frameworks: MITRE ATT&CK, CAPEC
- Advisories: Ubuntu, Microsoft, Red Hat
- Research: arXiv papers
- CTF events

### Environment Setup

Required `.env` file:
```bash
GITHUB_TOKEN="your_token"
NVD_API_KEY="your_key"
OPENCVE_EMAIL="your_email"
OPENCVE_PASSWORD="your_password"
# Optional: VIRUSTOTAL_API_KEY, ALIENVAULT_API_KEY, SHODAN_API_KEY
```

### Adding New Data Sources

1. Add `fetch_*` method in `1_data_collector.py`
2. Register in `all_sources` dictionary in main()
3. Add handler in `3_data_structurer.py` for instruction generation

## Development Patterns

### Chat Template Usage
When using Qwen models, apply chat template:
```python
messages = [{"role": "user", "content": prompt}]
if hasattr(tokenizer, 'apply_chat_template'):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
```

### Performance Benchmarking
Use BenchmarkTracker for monitoring:
```python
benchmark = BenchmarkTracker(logger)
benchmark.record_llm_performance(tokens_per_sec, input_tokens, output_tokens, gen_time)
benchmark.log_benchmark_stats(force=True)
```

### Parallel Processing
- Rule-based filtering uses ProcessPoolExecutor
- LLM batching uses ThreadPoolExecutor
- Batched filter supports multiple model instances

## Troubleshooting

Common issues:
- **ModuleNotFoundError**: Run `pip install -r requirements.txt`
- **MLX errors**: Ensure running on Apple Silicon with latest macOS
- **Memory issues**: Reduce batch size or number of LLM instances