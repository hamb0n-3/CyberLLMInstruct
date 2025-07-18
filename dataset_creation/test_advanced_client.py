#!/usr/bin/env python3
"""
Test client for the Advanced MLX Inference Server

This script demonstrates:
- Single request generation with continuous batching
- Single request generation with speculative decoding
- Batch generation for multiple prompts
- Streaming responses
- Server statistics monitoring
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union


class AdvancedMLXClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async with self.session.get(f"{self.base_url}/health") as resp:
            return await resp.json()
    
    async def generate(self, 
                      prompt: str, 
                      max_tokens: int = 100,
                      temperature: float = 0.7,
                      use_speculative: Optional[bool] = None,
                      stream: bool = False) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate text for a single prompt"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if use_speculative is not None:
            payload["use_speculative"] = use_speculative
        
        async with self.session.post(f"{self.base_url}/v1/generate", json=payload) as resp:
            if stream:
                # Handle streaming response
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield json.loads(data).get("text", "")
            else:
                result = await resp.json()
                return result
    
    async def generate_batch(self, 
                           prompts: List[str],
                           max_tokens: int = 100,
                           temperature: float = 0.7) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(f"{self.base_url}/v1/generate_batch", json=payload) as resp:
            return await resp.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        async with self.session.get(f"{self.base_url}/v1/stats") as resp:
            return await resp.json()
    
    async def configure(self, 
                       use_speculative_decoding: Optional[bool] = None,
                       max_batch_size: Optional[int] = None,
                       batch_timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        """Configure server parameters"""
        params = {}
        if use_speculative_decoding is not None:
            params["use_speculative_decoding"] = use_speculative_decoding
        if max_batch_size is not None:
            params["max_batch_size"] = max_batch_size
        if batch_timeout_ms is not None:
            params["batch_timeout_ms"] = batch_timeout_ms
        
        async with self.session.post(f"{self.base_url}/v1/configure", params=params) as resp:
            return await resp.json()


async def test_single_generation(client: AdvancedMLXClient):
    """Test single prompt generation"""
    print("\n=== Testing Single Generation ===")
    
    # Test with continuous batching
    print("\n1. Continuous Batching:")
    start = time.time()
    result = await client.generate(
        "Explain what SQL injection is and how to prevent it",
        max_tokens=150,
        use_speculative=False
    )
    print(f"Response: {result['text'][:200]}...")
    print(f"Method: {result['method']}")
    print(f"Time: {result['generation_time']:.2f}s")
    
    # Test with speculative decoding (if available)
    print("\n2. Speculative Decoding:")
    start = time.time()
    try:
        result = await client.generate(
            "What are the best practices for password security?",
            max_tokens=150,
            use_speculative=True
        )
        print(f"Response: {result['text'][:200]}...")
        print(f"Method: {result['method']}")
        print(f"Time: {result['generation_time']:.2f}s")
    except Exception as e:
        print(f"Speculative decoding not available: {e}")


async def test_batch_generation(client: AdvancedMLXClient):
    """Test batch generation"""
    print("\n=== Testing Batch Generation ===")
    
    prompts = [
        "What is a buffer overflow attack?",
        "Explain the concept of zero-day vulnerabilities",
        "How does encryption protect data?",
        "What is the difference between authentication and authorization?",
        "Describe the principle of least privilege"
    ]
    
    start = time.time()
    result = await client.generate_batch(prompts, max_tokens=100)
    total_time = time.time() - start
    
    print(f"\nBatch size: {result['batch_size']}")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Average time per prompt: {result['total_time']/len(prompts):.2f}s")
    print(f"\nSample responses:")
    for i, res in enumerate(result['results'][:2]):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {res['text'][:150]}...")


async def test_streaming(client: AdvancedMLXClient):
    """Test streaming generation"""
    print("\n=== Testing Streaming Generation ===")
    
    print("Streaming response for: 'What are the steps in a penetration test?'")
    print("Response: ", end="", flush=True)
    
    async for chunk in client.generate(
        "What are the steps in a penetration test?",
        max_tokens=200,
        stream=True
    ):
        print(chunk, end="", flush=True)
    print("\n")


async def test_concurrent_requests(client: AdvancedMLXClient):
    """Test concurrent request handling"""
    print("\n=== Testing Concurrent Requests ===")
    
    prompts = [
        f"Explain cybersecurity concept #{i}" 
        for i in range(10)
    ]
    
    # Send all requests concurrently
    start = time.time()
    tasks = [
        client.generate(prompt, max_tokens=50)
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start
    
    print(f"Processed {len(prompts)} concurrent requests in {total_time:.2f}s")
    print(f"Average time per request: {total_time/len(prompts):.2f}s")
    
    # Check methods used
    methods = [r['method'] for r in results]
    print(f"Methods used: {set(methods)}")


async def monitor_stats(client: AdvancedMLXClient):
    """Monitor server statistics"""
    print("\n=== Server Statistics ===")
    
    stats = await client.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Average latency: {stats['average_latency']:.3f}s")
    
    if stats.get('batching_stats'):
        bs = stats['batching_stats']
        print(f"\nBatching Statistics:")
        print(f"  Total batches: {bs.get('total_batches', 0)}")
        print(f"  Average batch size: {bs.get('avg_batch_size', 0):.1f}")
        print(f"  Total tokens generated: {bs.get('total_tokens', 0)}")
    
    if stats.get('speculative_stats'):
        ss = stats['speculative_stats']
        print(f"\nSpeculative Decoding Statistics:")
        print(f"  Acceptance rate: {ss.get('acceptance_rate', 0):.2%}")
        print(f"  Draft length: {ss.get('draft_length', 0)}")


async def main():
    """Run all tests"""
    print("Advanced MLX Inference Server Test Client")
    print("========================================")
    
    async with AdvancedMLXClient() as client:
        # Check server health
        try:
            health = await client.health_check()
            print(f"Server status: {health['status']}")
        except Exception as e:
            print(f"Error: Could not connect to server at {client.base_url}")
            print(f"Make sure the server is running with: ./start-advanced-server.sh")
            return
        
        # Configure server for testing
        print("\nConfiguring server...")
        config = await client.configure(max_batch_size=8, batch_timeout_ms=50)
        print(f"Server configuration: {config}")
        
        # Run tests
        await test_single_generation(client)
        await test_batch_generation(client)
        await test_streaming(client)
        await test_concurrent_requests(client)
        
        # Show final statistics
        await monitor_stats(client)


if __name__ == "__main__":
    asyncio.run(main()) 