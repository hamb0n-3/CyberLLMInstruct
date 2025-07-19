#!/usr/bin/env python3
"""
Debug script to understand response format
"""

from mlx_lm import load, generate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_response_format():
    """Test the exact response format"""
    logger.info("Loading model...")
    model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
    
    prompt = '[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "A critical vulnerability"[/INST]'
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Prompt length: {len(prompt)}")
    
    response = generate(model, tokenizer, prompt, max_tokens=10, verbose=False)
    
    logger.info(f"Full response: '{response}'")
    logger.info(f"Response length: {len(response)}")
    
    # Check if response starts with prompt
    if response.startswith(prompt):
        logger.info("Response DOES start with the prompt")
        actual_response = response[len(prompt):].strip()
        logger.info(f"Actual response after prompt: '{actual_response}'")
    else:
        logger.info("Response does NOT start with the prompt")
        logger.info(f"First 100 chars of response: '{response[:100]}'")
        
    # Also check with different approach
    logger.info("\nTrying a different prompt format...")
    prompt2 = "Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: \"A critical vulnerability\"\nAnswer:"
    
    response2 = generate(model, tokenizer, prompt2, max_tokens=10, verbose=False)
    logger.info(f"Response 2: '{response2}'")
    
    if response2.startswith(prompt2):
        actual_response2 = response2[len(prompt2):].strip()
        logger.info(f"Actual response 2: '{actual_response2}'")

if __name__ == "__main__":
    test_response_format()