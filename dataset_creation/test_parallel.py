#!/usr/bin/env python3
"""Test script to debug the hanging issue."""

from concurrent.futures import ProcessPoolExecutor
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
_test_data = None

def init_worker(data):
    """Initialize worker."""
    global _test_data
    _test_data = data
    logger.info(f"Worker {os.getpid()} initialized with data: {data}")

def process_batch(args):
    """Process a batch."""
    batch_id, items = args
    logger.info(f"Worker {os.getpid()} processing batch {batch_id} with {len(items)} items")
    
    results = []
    for item in items:
        # Simulate work
        time.sleep(0.1)
        results.append(f"Processed: {item}")
    
    logger.info(f"Worker {os.getpid()} completed batch {batch_id}")
    return results

def main():
    logger.info("Starting test...")
    
    # Create test data
    batches = [
        (0, ["item1", "item2"]),
        (1, ["item3", "item4"]),
        (2, ["item5", "item6"])
    ]
    
    # Create process pool
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker, initargs=("test_data",)) as pool:
        logger.info("Process pool created")
        
        # Submit all tasks
        futures = []
        for batch in batches:
            future = pool.submit(process_batch, batch)
            futures.append(future)
            logger.info(f"Submitted batch {batch[0]}")
        
        # Collect results
        logger.info("Waiting for results...")
        for i, future in enumerate(futures):
            logger.info(f"Waiting for future {i}...")
            result = future.result(timeout=10)
            logger.info(f"Got result for future {i}: {len(result)} items")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()