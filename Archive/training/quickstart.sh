#!/bin/bash
# Quick start script for cybersecurity LLM training
# This demonstrates a complete training workflow

set -e  # Exit on error

echo "==================================="
echo "Cybersecurity LLM Training Quickstart"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "train_cybersecurity_model.py" ]; then
    echo "Error: Please run this script from the training directory"
    exit 1
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p outputs
mkdir -p configs
mkdir -p logs

# Check for data
DATA_PATH="../dataset_creation/structured_data"
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: No structured data found at $DATA_PATH"
    echo "Please run the data pipeline first (stages 1-3 minimum)"
    exit 1
fi

# Count available data
if [ -f "$DATA_PATH/consolidated_cybersecurity_dataset_*.json" ]; then
    echo "Found consolidated dataset"
else
    echo "Warning: No consolidated dataset found. Looking for individual files..."
fi

# Model selection
echo ""
echo "Select a model to train:"
echo "1) Qwen2.5-1B-Instruct (Smallest, fastest)"
echo "2) Qwen2.5-3B-Instruct (Recommended)"
echo "3) Llama-3.2-3B-Instruct"
echo "4) Custom model path"
read -p "Enter choice (1-4): " model_choice

case $model_choice in
    1)
        MODEL="mlx-community/Qwen2.5-1B-Instruct-MLX-4bit"
        ;;
    2)
        MODEL="mlx-community/Qwen2.5-3B-Instruct-MLX-4bit"
        ;;
    3)
        MODEL="mlx-community/Llama-3.2-3B-Instruct-MLX-4bit"
        ;;
    4)
        read -p "Enter model path/name: " MODEL
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "Using model: $MODEL"

# Training configuration
echo ""
echo "Select training configuration:"
echo "1) Quick test (1 epoch, small batch)"
echo "2) Standard training (3 epochs, LoRA)"
echo "3) Advanced training (5 epochs, larger LoRA)"
echo "4) Custom configuration"
read -p "Enter choice (1-4): " config_choice

# Set parameters based on choice
case $config_choice in
    1)
        EPOCHS=1
        BATCH_SIZE=2
        LORA_RANK=8
        LR=2e-4
        EVAL_STEPS=50
        SAVE_STEPS=100
        OUTPUT_NAME="quicktest"
        ;;
    2)
        EPOCHS=3
        BATCH_SIZE=4
        LORA_RANK=16
        LR=2e-4
        EVAL_STEPS=100
        SAVE_STEPS=500
        OUTPUT_NAME="standard"
        ;;
    3)
        EPOCHS=5
        BATCH_SIZE=4
        LORA_RANK=32
        LR=1e-4
        EVAL_STEPS=200
        SAVE_STEPS=1000
        OUTPUT_NAME="advanced"
        ;;
    4)
        read -p "Number of epochs: " EPOCHS
        read -p "Batch size: " BATCH_SIZE
        read -p "LoRA rank: " LORA_RANK
        read -p "Learning rate (e.g., 2e-4): " LR
        read -p "Eval steps: " EVAL_STEPS
        read -p "Save steps: " SAVE_STEPS
        read -p "Output name: " OUTPUT_NAME
        ;;
esac

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/${OUTPUT_NAME}_${TIMESTAMP}"

# Create training command
echo ""
echo "Starting training with the following configuration:"
echo "- Model: $MODEL"
echo "- Epochs: $EPOCHS"
echo "- Batch Size: $BATCH_SIZE"
echo "- LoRA Rank: $LORA_RANK"
echo "- Learning Rate: $LR"
echo "- Output: $OUTPUT_DIR"
echo ""
read -p "Continue? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Training cancelled"
    exit 0
fi

# Ask if user wants to preview
echo ""
read -p "Would you like to preview the data formatting before training? (y/n): " preview_confirm

if [ "$preview_confirm" = "y" ]; then
    echo "Running preview mode..."
    python train_cybersecurity_model.py \
        --data-path "$DATA_PATH" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --num-epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LR \
        --use-lora \
        --lora-rank $LORA_RANK \
        --lora-alpha $(($LORA_RANK * 2)) \
        --eval-steps $EVAL_STEPS \
        --save-steps $SAVE_STEPS \
        --logging-steps 10 \
        --warmup-steps 100 \
        --gradient-accumulation-steps 4 \
        --use-tensorboard \
        --seed 42 \
        --preview \
        --preview-samples 5
    
    echo ""
    read -p "Continue with training? (y/n): " continue_confirm
    if [ "$continue_confirm" != "y" ]; then
        echo "Training cancelled"
        exit 0
    fi
fi

# Run training
echo "Starting training..."
python train_cybersecurity_model.py \
    --data-path "$DATA_PATH" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LR \
    --use-lora \
    --lora-rank $LORA_RANK \
    --lora-alpha $(($LORA_RANK * 2)) \
    --eval-steps $EVAL_STEPS \
    --save-steps $SAVE_STEPS \
    --logging-steps 10 \
    --warmup-steps 100 \
    --gradient-accumulation-steps 4 \
    --use-tensorboard \
    --seed 42

echo ""
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"

# Offer to evaluate
echo ""
read -p "Run evaluation on the best checkpoint? (y/n): " eval_confirm

if [ "$eval_confirm" = "y" ]; then
    echo "Running evaluation..."
    python evaluate_model.py \
        --model "$OUTPUT_DIR/checkpoint-best" \
        --data-path "$DATA_PATH" \
        --max-samples 100 \
        --output "$OUTPUT_DIR/evaluation_results.json"
    
    echo "Evaluation complete! Results saved to: $OUTPUT_DIR/evaluation_results.json"
fi

# Offer to merge model
echo ""
read -p "Merge LoRA weights for deployment? (y/n): " merge_confirm

if [ "$merge_confirm" = "y" ]; then
    echo "Merging model..."
    python merge_lora.py \
        --base-model "$MODEL" \
        --lora-checkpoint "$OUTPUT_DIR/checkpoint-best" \
        --output-dir "$OUTPUT_DIR/merged_model"
    
    echo "Model merged! Ready for deployment at: $OUTPUT_DIR/merged_model"
fi

# Show summary
echo ""
echo "==================================="
echo "Training Summary"
echo "==================================="
echo "Model: $MODEL"
echo "Training time: Check $OUTPUT_DIR/training_*.log"
echo "Best checkpoint: $OUTPUT_DIR/checkpoint-best"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR/tensorboard"
echo ""
echo "Next steps:"
echo "1. Review training curves: open $OUTPUT_DIR/loss_curve.png"
echo "2. Check evaluation results: cat $OUTPUT_DIR/evaluation_results.json | jq"
echo "3. Test the model: python generate.py --model $OUTPUT_DIR/checkpoint-best"
echo "==================================="