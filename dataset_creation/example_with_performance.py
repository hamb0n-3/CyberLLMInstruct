#!/usr/bin/env python3
"""
Example script demonstrating batching, speculative decoding, and performance logging
in the CyberLLMInstruct pipeline
"""

import json
import time
from pathlib import Path
from mlx_lm import load
import mlx.core as mx

# Import our custom utilities
from mlx_batch_utils import BatchedMLXGenerator
from mlx_speculative import create_speculative_pipeline
from performance_logger import get_performance_logger

def example_batched_pipeline():
    """Example of using batched generation with performance logging"""
    print("=== Batched Generation Example ===")
    
    # Initialize performance logger
    perf_logger = get_performance_logger()
    perf_logger.start_pipeline_stage("example_batched_pipeline")
    
    # Load a small model for demonstration
    model_path = "mlx-community/gemma-3-1b-it-bf16"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    
    # Create batched generator
    batch_gen = BatchedMLXGenerator(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=16,
        batch_timeout=0.1
    )
    
    # Example cybersecurity prompts
    prompts = [
        "What is a SQL injection attack?",
        "Explain cross-site scripting (XSS)",
        "How to prevent buffer overflow?",
        "What is a DDoS attack?",
        "Explain zero-day vulnerability",
        "What is ransomware?",
        "How does phishing work?",
        "What is social engineering?"
    ]
    
    # Start model inference logging
    perf_logger.start_model_inference(model_path)
    
    results = []
    print(f"\nProcessing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        # Generate response
        response = batch_gen.generate(prompt, max_tokens=100)
        results.append({
            "prompt": prompt,
            "response": response
        })
        
        # Update performance metrics
        tokens_generated = len(tokenizer.encode(response))
        perf_logger.update_model_tokens(model_path, tokens_generated, batch_size=1)
        perf_logger.update_stage_progress("example_batched_pipeline", 1)
        
        print(f"  [{i+1}/{len(prompts)}] Processed: {prompt[:30]}...")
    
    # End logging
    perf_logger.end_model_inference(model_path)
    batch_gen.shutdown()
    perf_logger.end_pipeline_stage("example_batched_pipeline", model_name=model_path)
    
    # Save results
    output_file = Path("example_batched_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results


def example_speculative_pipeline():
    """Example of using speculative decoding with performance logging"""
    print("\n=== Speculative Decoding Example ===")
    
    # Initialize performance logger
    perf_logger = get_performance_logger()
    perf_logger.start_pipeline_stage("example_speculative_pipeline")
    
    # Create speculative decoder with a larger target model
    target_model = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    draft_model = "mlx-community/gemma-3-1b-it-bf16"
    
    print(f"Creating speculative decoder:")
    print(f"  Target model: {target_model}")
    print(f"  Draft model: {draft_model}")
    
    spec_decoder = create_speculative_pipeline(
        target_model_path=target_model,
        draft_model_path=draft_model,
        speculation_length=5
    )
    
    # Complex prompt that benefits from speculative decoding
    prompt = """Write a comprehensive cybersecurity incident response plan that includes:
1. Initial detection and assessment procedures
2. Containment strategies
3. Eradication steps
4. Recovery processes
5. Post-incident review

Focus on a ransomware attack scenario."""
    
    print("\nGenerating response with speculative decoding...")
    
    # Start model inference logging
    perf_logger.start_model_inference("speculative_decoder")
    
    # Generate response
    start_time = time.time()
    response = spec_decoder.generate(
        prompt=prompt,
        max_tokens=500,
        temperature=0.0,
        verbose=True
    )
    generation_time = time.time() - start_time
    
    # Update metrics
    tokens_generated = len(spec_decoder.target_tokenizer.encode(response)) - len(spec_decoder.target_tokenizer.encode(prompt))
    perf_logger.update_model_tokens("speculative_decoder", tokens_generated, batch_size=1)
    perf_logger.update_stage_progress("example_speculative_pipeline", 1)
    
    # Get acceptance rate if available
    acceptance_rate = None
    if hasattr(spec_decoder, 'acceptance_history') and spec_decoder.acceptance_history:
        acceptance_rate = sum(spec_decoder.acceptance_history) / len(spec_decoder.acceptance_history)
    
    perf_logger.end_model_inference("speculative_decoder", acceptance_rate=acceptance_rate)
    perf_logger.end_pipeline_stage("example_speculative_pipeline", model_name="speculative_decoder")
    
    # Display results
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Tokens per second: {tokens_generated/generation_time:.2f}")
    if acceptance_rate:
        print(f"Acceptance rate: {acceptance_rate:.2%}")
    
    print(f"\nGenerated response (first 500 chars):")
    print("-" * 50)
    print(response[:500] + "...")
    
    # Save result
    output_file = Path("example_speculative_result.json")
    with open(output_file, 'w') as f:
        json.dump({
            "prompt": prompt,
            "response": response,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "acceptance_rate": acceptance_rate
        }, f, indent=2)
    
    print(f"\nResult saved to: {output_file}")
    return response


def main():
    """Run examples and generate performance report"""
    print("CyberLLMInstruct Pipeline Examples")
    print("=" * 60)
    
    # Run batched generation example
    try:
        example_batched_pipeline()
    except Exception as e:
        print(f"Batched generation example failed: {e}")
    
    # Run speculative decoding example
    try:
        example_speculative_pipeline()
    except Exception as e:
        print(f"Speculative decoding example failed: {e}")
    
    # Generate and display performance report
    perf_logger = get_performance_logger()
    print("\n" + "=" * 60)
    perf_logger.print_summary()
    
    # Save performance log
    log_file = perf_logger.save_session_log()
    print(f"Performance log saved to: {log_file}")
    
    print("\nExamples completed successfully!")


if __name__ == "__main__":
    main()