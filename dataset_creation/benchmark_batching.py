#!/usr/bin/env python3
"""
Benchmark script to compare batching performance
"""

import time
import json
import argparse
from pathlib import Path
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(num_entries: int = 100):
    """Create test dataset for benchmarking"""
    test_data = []
    
    # Mix of relevant and non-relevant entries
    for i in range(num_entries):
        if i % 3 == 0:
            # Relevant cybersecurity content
            entry = {
                "id": f"test_{i}",
                "title": f"Critical vulnerability CVE-2024-{i:04d} in network protocol",
                "description": f"A critical security vulnerability has been discovered in the authentication mechanism of protocol XYZ. This exploit allows remote attackers to gain unauthorized access to systems running affected versions. The vulnerability affects encryption and could lead to data breach.",
                "source": "test_benchmark"
            }
        elif i % 3 == 1:
            # Semi-relevant content
            entry = {
                "id": f"test_{i}",
                "title": f"Software update {i} released",
                "description": f"New software update includes improvements to network performance and system stability. Minor bug fixes included.",
                "source": "test_benchmark"
            }
        else:
            # Non-relevant content
            entry = {
                "id": f"test_{i}",
                "title": f"Lorem ipsum test {i}",
                "description": f"This is a test sample with placeholder text. Lorem ipsum dolor sit amet.",
                "source": "test_benchmark"
            }
        
        test_data.append(entry)
    
    # Save test data
    test_file = Path("raw_data") / "benchmark_test_data.json"
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, 'w') as f:
        json.dump({"entries": test_data}, f, indent=2)
    
    return test_file

def run_benchmark(script_name: str, test_file: Path, batch_size: int = 32) -> float:
    """Run a benchmark test and return execution time"""
    logger.info(f"Running benchmark with {script_name} (batch_size={batch_size})...")
    
    start_time = time.time()
    
    # Run the script
    cmd = [
        sys.executable,
        script_name,
        "--input-dir", str(test_file.parent),
        "--output-dir", "benchmark_output",
        "--batch-size", str(batch_size),
        "--limit", "1"  # Only process our test file
    ]
    
    if "optimized" not in script_name:
        # Add batching flag for original script
        cmd.extend(["--model", "mlx-community/Phi-3-mini-4k-instruct-4bit"])
    else:
        cmd.extend(["--model", "mlx-community/Phi-3-mini-4k-instruct-4bit"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Script failed: {result.stderr}")
            return -1
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return -1
    
    elapsed = time.time() - start_time
    
    # Count results
    output_dir = Path("benchmark_output")
    filtered_files = list(output_dir.glob("*_filtered_*.json"))
    
    if filtered_files:
        with open(filtered_files[0], 'r') as f:
            results = json.load(f)
            logger.info(f"  Processed {len(results)} relevant entries")
    
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Benchmark batching performance")
    parser.add_argument("--num-entries", type=int, default=100, help="Number of test entries to create")
    parser.add_argument("--batch-sizes", type=int, nargs='+', default=[1, 8, 16, 32], help="Batch sizes to test")
    args = parser.parse_args()
    
    # Create test data
    test_file = create_test_data(args.num_entries)
    logger.info(f"Created test data with {args.num_entries} entries")
    
    # Clean up old benchmark output
    output_dir = Path("benchmark_output")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    results = {}
    
    # Test different configurations
    configs = [
        ("2_data_filter.py", "Original (fake batching)"),
        ("2_data_filter_optimized.py", "Optimized (true batching)")
    ]
    
    for script, label in configs:
        if not Path(script).exists():
            logger.warning(f"Skipping {script} - file not found")
            continue
        
        for batch_size in args.batch_sizes:
            key = f"{label} (batch={batch_size})"
            elapsed = run_benchmark(script, test_file, batch_size)
            
            if elapsed > 0:
                results[key] = {
                    "time": elapsed,
                    "items_per_second": args.num_entries / elapsed
                }
                logger.info(f"{key}: {elapsed:.2f}s ({results[key]['items_per_second']:.2f} items/sec)")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Test dataset: {args.num_entries} entries")
    print("-"*60)
    
    for config, metrics in sorted(results.items()):
        print(f"{config:40s} {metrics['time']:8.2f}s  {metrics['items_per_second']:8.2f} items/sec")
    
    # Calculate speedup
    if len(results) >= 2:
        original_time = next((v['time'] for k, v in results.items() if "Original" in k), None)
        optimized_time = next((v['time'] for k, v in results.items() if "Optimized" in k), None)
        
        if original_time and optimized_time:
            speedup = original_time / optimized_time
            print("-"*60)
            print(f"Speedup: {speedup:.2f}x faster with optimized batching")

if __name__ == "__main__":
    main()