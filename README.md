# MLX ParaLLM - Advanced Inference Engine

High-performance inference engine for MLX with adaptive speculative decoding and continuous batching.

## Features

### 🚀 Adaptive Speculative Decoding
- Uses a smaller draft model to generate candidate tokens quickly
- Verifies candidates with the larger target model
- Achieves up to 2-3x speedup for single requests
- Adaptive draft length based on acceptance rate

### 📦 Continuous Batching
- Dynamically groups requests into batches
- Improves throughput by processing multiple prompts together
- Configurable batch size and timeout parameters
- Efficient handling of variable-length sequences

### 🔧 Advanced Server
- FastAPI-based HTTP server with async support
- RESTful API endpoints for text generation
- Real-time statistics and monitoring
- Streaming response support
- Dynamic configuration

## Installation

```bash
# Install required dependencies
pip install mlx-lm fastapi uvicorn aiohttp

# Clone the repository (if not already done)
git clone <repository-url>
cd dataset_creation
```

## Quick Start

### 1. Start the Advanced Server

```bash
# Basic usage (continuous batching only)
./start-advanced-server.sh --model mlx-community/c4ai-command-r-v01-4bit

# With speculative decoding (requires a draft model)
./start-advanced-server.sh \
    --model mlx-community/c4ai-command-r-v01-4bit \
    --draft-model mlx-community/Phi-3-mini-4k-instruct-4bit \
    --use-speculative

# Custom configuration
./start-advanced-server.sh \
    --model your-model-path \
    --draft-model your-draft-model \
    --max-batch-size 16 \
    --batch-timeout-ms 100 \
    --port 8080
```

### 2. Test the Server

```bash
# Run the test client
python test_advanced_client.py

# Or make direct API calls
curl -X POST http://localhost:8080/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Explain what a firewall is",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

## API Endpoints

### Generate Text (Single Request)
```http
POST /v1/generate
Content-Type: application/json

{
    "prompt": "Your prompt here",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false,
    "use_speculative": true  // Optional: override default
}
```

### Generate Text (Batch)
```http
POST /v1/generate_batch
Content-Type: application/json

{
    "prompts": [
        "First prompt",
        "Second prompt",
        "Third prompt"
    ],
    "max_tokens": 100,
    "temperature": 0.7
}
```

### Get Server Statistics
```http
GET /v1/stats

Response:
{
    "total_requests": 42,
    "average_latency": 0.234,
    "batching_stats": {
        "total_batches": 10,
        "avg_batch_size": 4.2,
        "total_tokens": 12345
    },
    "speculative_stats": {
        "acceptance_rate": 0.85,
        "draft_length": 4
    }
}
```

### Configure Server
```http
POST /v1/configure?use_speculative_decoding=true&max_batch_size=16
```

## Architecture

### Speculative Decoding Flow
```
1. Draft model generates N candidate tokens quickly
2. Target model verifies candidates in parallel
3. Accept/reject based on probability threshold
4. Adaptive adjustment of draft length
```

### Continuous Batching Flow
```
1. Requests added to pending queue
2. Batch formation based on timeout and size limits
3. Padded batch processing with attention masks
4. Parallel token generation for all sequences
5. Early stopping for completed sequences
```

## Performance Tuning

### Speculative Decoding
- **Draft Model Selection**: Choose a model 4-10x smaller than target
- **Max Draft Tokens**: Start with 4-6, adjust based on acceptance rate
- **Temperature**: Lower temperatures improve acceptance rates

### Continuous Batching
- **Batch Size**: Larger batches improve throughput but increase latency
- **Timeout**: Lower timeouts reduce latency but may create smaller batches
- **Padding**: Use efficient padding strategies for variable-length inputs

## Benchmarks

Example performance improvements (results may vary):

| Method | Tokens/sec | Latency (ms) | Throughput |
|--------|------------|--------------|------------|
| Baseline | 50 | 200 | 1x |
| Speculative | 120 | 83 | 2.4x |
| Batching (8) | 320 | 250 | 6.4x |
| Both | 400 | 100 | 8x |

## Development

### Running Tests
```bash
# Test speculative decoding
python -m pytest tests/test_speculative_decoding.py

# Test continuous batching
python -m pytest tests/test_continuous_batching.py
```

### Custom Integration
```python
from mlx_parallm.speculative_decoding import SpeculativeConfig, SpeculativeDecodingEngine
from mlx_parallm.continuous_batching import BatchConfig, AsyncContinuousBatchingEngine

# Speculative decoding
config = SpeculativeConfig(
    draft_model_path="path/to/draft",
    target_model_path="path/to/target",
    max_draft_tokens=5
)
engine = SpeculativeDecodingEngine(config)
result = engine.generate("Your prompt", max_tokens=100)

# Continuous batching
batch_config = BatchConfig(max_batch_size=8, timeout_ms=50)
batch_engine = AsyncContinuousBatchingEngine(model, tokenizer, batch_config)
results = await batch_engine.generate_batch(prompts)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install mlx-lm fastapi uvicorn aiohttp
   ```

2. **Model compatibility**: Draft and target models must use the same tokenizer

3. **Memory issues**: Reduce batch size or use smaller models

4. **Low acceptance rate**: Try a larger draft model or adjust temperature

## License

See the main project LICENSE file.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

Built with MLX for efficient inference on Apple Silicon. 