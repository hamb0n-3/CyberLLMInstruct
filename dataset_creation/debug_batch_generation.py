#!/usr/bin/env python3
"""
Debug script to test batch generation output
"""

import importlib.util
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the modules
spec = importlib.util.spec_from_file_location("data_filter", "2_data_filter.py") 
data_filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_filter_module)

spec2 = importlib.util.spec_from_file_location("batch_utils", "mlx_batch_utils.py")
batch_utils_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(batch_utils_module)

from mlx_lm import load, generate
import mlx.core as mx

def test_batch_generator():
    """Test the batch generator directly"""
    logger.info("Loading model...")
    model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
    
    # Create batch generator
    batch_gen = batch_utils_module.BatchedMLXGenerator(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        batch_timeout=0.1
    )
    
    # Test prompts
    prompts = [
        '[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "A critical vulnerability in authentication allows attackers to bypass security"[/INST]',
        '[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "SQL injection attack"[/INST]',
        '[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "Regular software update"[/INST]'
    ]
    
    logger.info("Testing individual generation...")
    for i, prompt in enumerate(prompts):
        response = batch_gen.generate(prompt, max_tokens=5, temperature=0.0)
        logger.info(f"Prompt {i+1}: '{prompt[:50]}...' -> Response: '{response}'")
    
    batch_gen.shutdown()
    
def test_direct_generation():
    """Test direct generation without batching"""
    logger.info("\nTesting direct generation...")
    model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
    
    prompt = '[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "A critical vulnerability"[/INST]'
    
    response = generate(model, tokenizer, prompt, max_tokens=5, verbose=False)
    logger.info(f"Direct generation response: '{response}'")
    
    # Check what the response looks like after the prompt
    response_only = response[len(prompt):].strip()
    logger.info(f"Response after prompt removal: '{response_only}'")
    logger.info(f"Does response contain YES? {'YES' in response_only.upper()}")

if __name__ == "__main__":
    test_direct_generation()
    test_batch_generator()