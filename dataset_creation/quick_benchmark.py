#!/usr/bin/env python3
"""
Quick benchmark to test batch processing performance
"""

import time
import importlib.util
from mlx_lm import load
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the batch utils
spec = importlib.util.spec_from_file_location("batch_utils", "mlx_batch_utils.py")
batch_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_utils)

def benchmark_batch_processing():
    """Benchmark batch vs sequential processing"""
    logger.info("Loading model...")
    model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
    
    # Test prompts
    test_prompts = [
        f'[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "Vulnerability {i} in authentication system"[/INST]'
        for i in range(16)
    ]
    
    # Test 1: Sequential processing
    logger.info("\n=== Sequential Processing ===")
    start_time = time.time()
    sequential_results = []
    for prompt in test_prompts:
        from mlx_lm import generate
        result = generate(model, tokenizer, prompt, max_tokens=5, verbose=False)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    logger.info(f"Sequential time: {sequential_time:.2f}s ({len(test_prompts)/sequential_time:.2f} prompts/sec)")
    
    # Test 2: Batch processing
    logger.info("\n=== Batch Processing ===")
    batch_gen = batch_utils.BatchedMLXGenerator(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        batch_timeout=0.05
    )
    
    start_time = time.time()
    batch_results = batch_gen.generate_batch(test_prompts, max_tokens=5)
    batch_time = time.time() - start_time
    logger.info(f"Batch time: {batch_time:.2f}s ({len(test_prompts)/batch_time:.2f} prompts/sec)")
    
    batch_gen.shutdown()
    
    # Calculate speedup
    speedup = sequential_time / batch_time
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    
    # Verify results are similar
    logger.info("\nSample results comparison:")
    for i in range(min(3, len(test_prompts))):
        logger.info(f"Prompt {i+1}:")
        logger.info(f"  Sequential: '{sequential_results[i][:20]}...'")
        logger.info(f"  Batch: '{batch_results[i][:20]}...'")

if __name__ == "__main__":
    benchmark_batch_processing()