#!/bin/bash
# Simple MLX Server Startup Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Default values
MODEL_PATH="mlx-community/Phi-3-mini-4k-instruct-4bit"
PORT=8080
HOST="0.0.0.0"
BATCH_SIZE=16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model PATH       Path to MLX model (default: $MODEL_PATH)"
            echo "  --port PORT        Server port (default: $PORT)"
            echo "  --host HOST        Server host (default: $HOST)"
            echo "  --batch-size SIZE  Max batch size (default: $BATCH_SIZE)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Simple MLX Server..."
echo "Model: $MODEL_PATH"
echo "Host: $HOST:$PORT"
echo "Batch Size: $BATCH_SIZE"

# Change to project root and run the server
cd "$PROJECT_ROOT"
python dataset_creation/mlx_parallm/simple_server.py \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --batch-size "$BATCH_SIZE"