#!/usr/bin/env python3
"""
Test script for MLX batching and speculative decoding utilities
"""

import time
import argparse
import logging
from mlx_lm import load
import mlx.core as mx

# Import batching utilities
try:
    from mlx_batch_utils import BatchedMLXGenerator, DynamicBatcher
    from mlx_speculative import SpeculativeDecoder, SpeculativeConfig, create_speculative_pipeline
    from performance_logger import get_performance_logger
except ImportError as e:
    print(f"Error importing batching utilities: {e}")
    print("Make sure mlx_batch_utils.py, mlx_speculative.py, and performance_logger.py are in the same directory")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_batched_generation(model_path: str = "mlx-community/gemma-3-1b-it-bf16"):
    """Test batched generation performance"""
    logger.info("Testing Batched Generation")
    perf_logger = get_performance_logger()
    
    # Start pipeline stage logging
    perf_logger.start_pipeline_stage("batched_generation_test")
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    
    # Create batched generator
    batch_gen = BatchedMLXGenerator(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=16,
        batch_timeout=0.1
    )
    
    # Test prompts
    test_prompts = [
        "What is a buffer overflow vulnerability?",
        "How does SQL injection work?",
        "Explain cross-site scripting (XSS)",
        "What is a DDoS attack?",
        "How to prevent phishing attacks?",
        "What is zero-day vulnerability?",
        "Explain ransomware attacks",
        "What is social engineering?"
    ]
    
    logger.info(f"Processing {len(test_prompts)} prompts with batching")
    
    # Start model inference logging
    perf_logger.start_model_inference(model_path)
    
    # Time batched generation
    start_time = time.time()
    batched_results = []
    for i, prompt in enumerate(test_prompts):
        result = batch_gen.generate(prompt, max_tokens=100)
        batched_results.append(result)
        # Update metrics
        perf_logger.update_model_tokens(model_path, len(tokenizer.encode(result)), batch_size=1)
        perf_logger.update_stage_progress("batched_generation_test", 1)
    batched_time = time.time() - start_time
    
    # End model inference logging
    perf_logger.end_model_inference(model_path)
    
    # Cleanup
    batch_gen.shutdown()
    
    # End pipeline stage logging
    perf_logger.end_pipeline_stage("batched_generation_test", model_name=model_path)
    
    logger.info(f"Batched generation completed in {batched_time:.2f}s")
    logger.info(f"Average time per prompt: {batched_time/len(test_prompts):.2f}s")
    
    # Show sample results
    logger.info("\nSample results:")
    for i in range(min(3, len(test_prompts))):
        logger.info(f"\nPrompt: {test_prompts[i]}")
        logger.info(f"Response: {batched_results[i][:100]}...")
    
    return batched_time


def test_speculative_decoding(
    target_model_path: str = "mlx-community/c4ai-command-r-v01-4bit",
    draft_model_path: str = "mlx-community/gemma-3-1b-it-bf16"
):
    """Test speculative decoding performance"""
    logger.info("Testing Speculative Decoding")
    perf_logger = get_performance_logger()
    
    # Start pipeline stage logging
    perf_logger.start_pipeline_stage("speculative_decoding_test")
    
    # Create speculative decoder
    spec_decoder = create_speculative_pipeline(
        target_model_path=target_model_path,
        draft_model_path=draft_model_path,
        speculation_length=5
    )
    
    # Test prompt
    test_prompt = """Explain the OWASP Top 10 security vulnerabilities in detail."""
    
    logger.info(f"Generating with speculative decoding...")
    
    # Start model inference logging
    perf_logger.start_model_inference("speculative_decoder")
    
    # Time speculative generation
    start_time = time.time()
    spec_result = spec_decoder.generate(
        prompt=test_prompt,
        max_tokens=500,
        temperature=0.0,
        verbose=True
    )
    spec_time = time.time() - start_time
    
    # Update metrics
    tokens_generated = len(spec_decoder.target_tokenizer.encode(spec_result)) - len(spec_decoder.target_tokenizer.encode(test_prompt))
    perf_logger.update_model_tokens("speculative_decoder", tokens_generated, batch_size=1)
    perf_logger.update_stage_progress("speculative_decoding_test", 1)
    
    # End model inference logging with acceptance rate if available
    acceptance_rate = None
    if hasattr(spec_decoder, 'acceptance_history') and spec_decoder.acceptance_history:
        acceptance_rate = sum(spec_decoder.acceptance_history) / len(spec_decoder.acceptance_history)
    perf_logger.end_model_inference("speculative_decoder", acceptance_rate=acceptance_rate)
    
    # End pipeline stage logging
    perf_logger.end_pipeline_stage("speculative_decoding_test", model_name="speculative_decoder")
    
    logger.info(f"\nSpeculative generation completed in {spec_time:.2f}s")
    logger.info(f"Generated text length: {len(spec_result)} characters")
    logger.info(f"\nGenerated text (first 500 chars):\n{spec_result[:500]}...")
    
    return spec_time


def test_dynamic_batcher():
    """Test the DynamicBatcher utility"""
    logger.info("Testing DynamicBatcher")
    
    # Create a mock batch processing function
    def process_batch(items):
        logger.info(f"Processing batch of {len(items)} items")
        return [f"Processed: {item}" for item in items]
    
    # Create batcher
    batcher = DynamicBatcher(process_batch, batch_size=3)
    
    # Add items
    test_items = ["item1", "item2", "item3", "item4", "item5"]
    results = []
    
    for i, item in enumerate(test_items):
        batch_result = batcher.add(item, index=i)
        if batch_result:
            results.extend(batch_result)
    
    # Flush remaining items
    final_batch = batcher.flush()
    if final_batch:
        results.extend(final_batch)
    
    logger.info(f"Processed {len(results)} total items")
    for idx, result in results:
        logger.info(f"  Index {idx}: {result}")


def main():
    parser = argparse.ArgumentParser(description="Test MLX batching and speculative decoding")
    parser.add_argument("--test", choices=["batch", "speculative", "dynamic", "all"], 
                        default="all", help="Which test to run")
    parser.add_argument("--model", type=str, default="mlx-community/Phi-3-mini-4k-instruct-4bit",
                        help="Model path for testing")
    parser.add_argument("--draft-model", type=str, default="mlx-community/gemma-3-1b-it-bf16",
                        help="Draft model for speculative decoding")
    args = parser.parse_args()
    
    logger.info("Starting MLX Batching Tests")
    logger.info("=" * 50)
    
    # Get performance logger
    perf_logger = get_performance_logger()
    
    if args.test in ["batch", "all"]:
        test_batched_generation(args.model)
        logger.info("=" * 50)
    
    if args.test in ["speculative", "all"]:
        # Note: Speculative decoding requires specific model compatibility
        try:
            test_speculative_decoding(args.model, args.draft_model)
        except Exception as e:
            logger.error(f"Speculative decoding test failed: {e}")
            logger.info("This may be due to model compatibility issues")
        logger.info("=" * 50)
    
    if args.test in ["dynamic", "all"]:
        test_dynamic_batcher()
        logger.info("=" * 50)
    
    logger.info("All tests completed!")
    
    # Print and save performance summary
    perf_logger.print_summary()
    log_file = perf_logger.save_session_log()
    logger.info(f"Performance log saved to: {log_file}")


if __name__ == "__main__":
    main()