#!/usr/bin/env python3

from mlx_client import MLXClient
import time

# Test batch generation
client = MLXClient(
    server_url="http://localhost:8080",
    batch_size=8,  # Smaller batch
    use_server=True
)

print("Client initialized")

# Test single request
prompts = ["Is this about cybersecurity? Answer yes or no: Buffer overflow attack"]
print(f"Testing with {len(prompts)} prompts...")

start = time.time()
try:
    responses = client.generate_batch(
        prompts, 
        max_tokens=5,
        temperature=0.1,
        timeout=10
    )
    print(f"Success! Response: {responses}")
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
    print(f"Time taken: {time.time() - start:.2f}s")

# Test larger batch
prompts = [
    "Is this about cybersecurity? Answer yes or no: Buffer overflow attack",
    "Is this about cybersecurity? Answer yes or no: SQL injection vulnerability",
    "Is this about cybersecurity? Answer yes or no: Cross-site scripting exploit",
    "Is this about cybersecurity? Answer yes or no: The weather is nice today",
    "Is this about cybersecurity? Answer yes or no: Remote code execution flaw",
    "Is this about cybersecurity? Answer yes or no: Pizza recipe ingredients",
    "Is this about cybersecurity? Answer yes or no: Zero-day vulnerability found",
    "Is this about cybersecurity? Answer yes or no: Authentication bypass discovered",
]

print(f"\nTesting with {len(prompts)} prompts...")
start = time.time()
try:
    responses = client.generate_batch(
        prompts, 
        max_tokens=5,
        temperature=0.1,
        timeout=30
    )
    print(f"Success! Got {len(responses)} responses")
    for i, resp in enumerate(responses):
        print(f"  {i}: {resp}")
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
    print(f"Time taken: {time.time() - start:.2f}s")