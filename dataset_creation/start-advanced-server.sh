#!/bin/bash

# Advanced MLX Inference Server Startup Script
# This script starts the server with speculative decoding and continuous batching

# Default values
MODEL="mlx-community/c4ai-command-r-v01-4bit"
DRAFT_MODEL="mlx-community/Phi-3-mini-4k-instruct-4bit"
HOST="0.0.0.0"
PORT="8080"
MAX_BATCH_SIZE="8"
BATCH_TIMEOUT_MS="50"
USE_SPECULATIVE=""
USE_COMBINED="--use-combined"  # Default to combined mode
KV_BITS="8"
KV_GROUP_SIZE="32"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-timeout-ms)
            BATCH_TIMEOUT_MS="$2"
            shift 2
            ;;
        --use-speculative)
            USE_SPECULATIVE="--use-speculative"
            shift
            ;;
        --use-combined)
            USE_COMBINED="--use-combined"
            shift
            ;;
        --kv-bits)
            KV_BITS="$2"
            shift 2
            ;;
        --kv-group-size)
            KV_GROUP_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL            Main model to use (default: $MODEL)"
            echo "  --draft-model MODEL      Draft model for speculative decoding (default: $DRAFT_MODEL)"
            echo "  --host HOST              Server host (default: $HOST)"
            echo "  --port PORT              Server port (default: $PORT)"
            echo "  --max-batch-size SIZE    Maximum batch size (default: $MAX_BATCH_SIZE)"
            echo "  --batch-timeout-ms MS    Batch timeout in milliseconds (default: $BATCH_TIMEOUT_MS)"
            echo "  --use-speculative        Use speculative decoding by default"
            echo "  --use-combined           Use combined speculative + continuous batching"
            echo "  --kv-bits BITS           KV cache quantization bits (default: $KV_BITS)"
            echo "  --kv-group-size SIZE     KV cache group size (default: $KV_GROUP_SIZE)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if mlx_lm is installed
if ! python -c "import mlx_lm" 2>/dev/null; then
    echo "Error: mlx-lm is not installed. Please install it with:"
    echo "  pip install mlx-lm"
    exit 1
fi

# Check if FastAPI and uvicorn are installed
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Error: FastAPI and uvicorn are not installed. Please install them with:"
    echo "  pip install fastapi uvicorn"
    exit 1
fi

echo "Starting Advanced MLX Inference Server..."
echo "Main Model: $MODEL"
echo "Draft Model: $DRAFT_MODEL"
echo "Host: $HOST:$PORT"
echo "Max Batch Size: $MAX_BATCH_SIZE"
echo "Batch Timeout: ${BATCH_TIMEOUT_MS}ms"
echo "Use Speculative: ${USE_SPECULATIVE:-No}"
echo "Use Combined: ${USE_COMBINED:-No}"
echo "KV Cache: ${KV_BITS}-bit quantization, group size ${KV_GROUP_SIZE}"
echo ""

# Start the server
python -m mlx_parallm.advanced_server \
    --model "$MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --batch-timeout-ms "$BATCH_TIMEOUT_MS" \
    $USE_SPECULATIVE \
    $USE_COMBINED 