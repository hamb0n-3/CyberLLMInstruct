#!/usr/bin/env python3
"""Test MLX in parallel processes."""

from concurrent.futures import ProcessPoolExecutor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model
_model = None
_tokenizer = None

def init_worker(model_path):
    """Initialize worker with MLX model."""
    global _model, _tokenizer
    
    logger.info(f"Worker {os.getpid()} loading model: {model_path}")
    
    try:
        from mlx_lm import load
        _model, _tokenizer = load(model_path)
        logger.info(f"Worker {os.getpid()} ready")
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to load model: {e}")
        raise

def process_text(args):
    """Process text with model."""
    text, max_tokens = args
    
    logger.info(f"Worker {os.getpid()} processing text of length {len(text)}")
    
    try:
        from mlx_lm import generate
        
        # Simple generation without chat template
        prompt = f"Question: {text}\nAnswer:"
        
        response = generate(
            _model,
            _tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        
        logger.info(f"Worker {os.getpid()} generated response")
        return response
    except Exception as e:
        logger.error(f"Worker {os.getpid()} error: {e}")
        return f"ERROR: {e}"

def main():
    model_path = "mlx-community/Qwen3-8B-4bit-DWQ-053125"
    
    logger.info("Creating process pool...")
    
    # Test with just 1 worker first
    with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(model_path,)) as pool:
        logger.info("Process pool created")
        
        # Submit one simple task
        future = pool.submit(process_text, ("What is cybersecurity?", 10))
        logger.info("Task submitted")
        
        # Wait for result
        logger.info("Waiting for result...")
        try:
            result = future.result(timeout=60)
            logger.info(f"Got result: {result[:100]}...")
        except Exception as e:
            logger.error(f"Failed: {e}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()