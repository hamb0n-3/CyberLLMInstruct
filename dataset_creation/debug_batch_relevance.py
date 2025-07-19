#!/usr/bin/env python3
"""
Debug script to test batch relevance check
"""

import importlib.util
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the filter class
spec = importlib.util.spec_from_file_location("data_filter", "2_data_filter.py")
data_filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_filter_module)
CyberDataFilter = data_filter_module.CyberDataFilter

def test_single_relevance():
    """Test single relevance check"""
    logger.info("Testing single relevance check...")
    
    # Create filter with model
    filter_instance = CyberDataFilter(
        input_dir="test_raw_data",
        output_dir="test_filtered",
        model_path="mlx-community/Phi-3-mini-4k-instruct-4bit",
        use_batching=False  # Disable batching for this test
    )
    
    # Test text
    test_text = "A critical vulnerability in the authentication mechanism allows remote attackers to bypass security controls and gain unauthorized access to sensitive data."
    
    logger.info(f"Test text: {test_text}")
    
    # Check relevance
    is_relevant = filter_instance.check_relevance(test_text)
    logger.info(f"Relevance check result: {is_relevant}")
    
    # Also test with batching
    logger.info("\nTesting batch relevance check...")
    filter_batch = CyberDataFilter(
        input_dir="test_raw_data", 
        output_dir="test_filtered",
        model_path="mlx-community/Phi-3-mini-4k-instruct-4bit",
        use_batching=True,
        batch_size=4
    )
    
    test_texts = [
        test_text,
        "SQL injection attack allows database manipulation",
        "This is a general software update with no security implications"
    ]
    
    results = filter_batch.batch_check_relevance(test_texts)
    for i, (text, result) in enumerate(zip(test_texts, results)):
        logger.info(f"Text {i+1}: {text[:50]}... -> Relevant: {result}")
    
    # Cleanup
    filter_instance.cleanup()
    filter_batch.cleanup()

if __name__ == "__main__":
    test_single_relevance()