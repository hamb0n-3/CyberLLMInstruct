#!/usr/bin/env python3
"""
Test client for the combined speculative + continuous batching mode
"""

import asyncio
import aiohttp
import time
import json
import argparse
from typing import List, Dict


async def test_single_request(session: aiohttp.ClientSession, base_url: str, prompt: str) -> Dict:
    """Test a single generation request"""
    url = f"{base_url}/v1/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        result = await response.json()
        result['client_latency'] = time.time() - start_time
        return result


async def test_batch_request(session: aiohttp.ClientSession, base_url: str, prompts: List[str]) -> Dict:
    """Test batch generation request"""
    url = f"{base_url}/v1/generate_batch"
    payload = {
        "prompts": prompts,
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        result = await response.json()
        result['client_latency'] = time.time() - start_time
        return result


async def test_concurrent_requests(session: aiohttp.ClientSession, base_url: str, prompts: List[str]) -> List[Dict]:
    """Test multiple concurrent single requests"""
    tasks = [test_single_request(session, base_url, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def get_server_stats(session: aiohttp.ClientSession, base_url: str) -> Dict:
    """Get server statistics"""
    url = f"{base_url}/v1/stats"
    async with session.get(url) as response:
        return await response.json()


async def configure_server(session: aiohttp.ClientSession, base_url: str, use_combined: bool) -> Dict:
    """Configure server mode"""
    url = f"{base_url}/v1/configure"
    payload = {
        "use_combined_mode": use_combined,
        "use_speculative_decoding": not use_combined
    }
    async with session.post(url, json=payload) as response:
        return await response.json()


async def main():
    parser = argparse.ArgumentParser(description="Test combined inference mode")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of test prompts")
    args = parser.parse_args()
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Describe the water cycle.",
        "What is artificial intelligence?",
        "How do vaccines work?",
        "Explain the theory of relativity.",
        "What causes climate change?",
        "How does the internet work?"
    ][:args.num_prompts]
    
    async with aiohttp.ClientSession() as session:
        print("MLX Combined Inference Mode Test Client")
        print("=" * 50)
        
        # Get initial stats
        stats = await get_server_stats(session, args.base_url)
        print(f"\nServer Mode: {stats['mode']}")
        print(f"Total Requests: {stats['total_requests']}")
        
        # Test 1: Single requests with combined mode
        print("\n\nTest 1: Single Requests (Combined Mode)")
        print("-" * 40)
        await configure_server(session, args.base_url, use_combined=True)
        
        for i, prompt in enumerate(test_prompts[:3]):
            print(f"\nPrompt {i+1}: {prompt[:50]}...")
            result = await test_single_request(session, args.base_url, prompt)
            print(f"Method: {result['method']}")
            print(f"Server Time: {result['generation_time']:.3f}s")
            print(f"Client Latency: {result['client_latency']:.3f}s")
            print(f"Generated: {result['text'][:100]}...")
        
        # Test 2: Batch request with combined mode
        print("\n\nTest 2: Batch Request (Combined Mode)")
        print("-" * 40)
        batch_result = await test_batch_request(session, args.base_url, test_prompts)
        print(f"Batch Size: {batch_result['batch_size']}")
        print(f"Total Time: {batch_result['total_time']:.3f}s")
        print(f"Client Latency: {batch_result['client_latency']:.3f}s")
        print(f"Avg Time per Request: {batch_result['total_time'] / batch_result['batch_size']:.3f}s")
        
        # Test 3: Concurrent requests with combined mode
        print("\n\nTest 3: Concurrent Requests (Combined Mode)")
        print("-" * 40)
        start_time = time.time()
        concurrent_results = await test_concurrent_requests(session, args.base_url, test_prompts)
        total_time = time.time() - start_time
        
        print(f"Total Concurrent Time: {total_time:.3f}s")
        avg_latency = sum(r['client_latency'] for r in concurrent_results) / len(concurrent_results)
        print(f"Average Latency: {avg_latency:.3f}s")
        
        # Get final statistics
        print("\n\nFinal Server Statistics")
        print("-" * 40)
        final_stats = await get_server_stats(session, args.base_url)
        print(json.dumps(final_stats, indent=2))
        
        # Compare modes if server supports both
        if final_stats.get('combined_stats'):
            print("\n\nCombined Mode Performance Summary")
            print("-" * 40)
            combined_stats = final_stats['combined_stats']
            print(f"Total Requests: {combined_stats.get('total_requests', 0)}")
            print(f"Average Latency: {combined_stats.get('avg_latency', 0):.3f}s")
            print(f"Average Batch Size: {combined_stats.get('avg_batch_size', 0):.2f}")
            
            if combined_stats.get('avg_acceptance_rate'):
                print(f"Speculative Acceptance Rate: {combined_stats['avg_acceptance_rate']:.2%}")
                if combined_stats.get('speculative_speedup'):
                    print(f"Speculative Speedup: {combined_stats['speculative_speedup']:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())