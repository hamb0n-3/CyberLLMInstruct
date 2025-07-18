#!/usr/bin/env python3
"""
MLX Client Library for Dataset Creation Pipeline

This module provides a unified interface for using MLX models either through
the advanced server (with speculative decoding and continuous batching) or
directly via mlx-lm. It includes automatic fallback and batch processing support.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import aiohttp
import requests
from dataclasses import dataclass, field
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future

# Try to import MLX for fallback mode
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Represents a generation request"""
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    request_id: Optional[str] = None
    callback: Optional[callable] = None


@dataclass
class BatchRequest:
    """Represents a batch of generation requests"""
    requests: List[GenerationRequest]
    futures: List[Future] = field(default_factory=list)


class MLXClient:
    """
    Unified client for MLX inference with server/direct fallback support.
    
    Features:
    - Automatic server detection and fallback to direct MLX
    - Request batching for improved throughput
    - Async and sync interfaces
    - Connection pooling and retry logic
    """
    
    def __init__(self, 
                 server_url: Optional[str] = "http://localhost:8080",
                 model_path: Optional[str] = "mlx-community/Phi-3-mini-4k-instruct-4bit",
                 batch_size: int = 16,
                 batch_timeout_ms: int = 100,
                 max_retries: int = 3,
                 use_server: bool = True):
        """
        Initialize the MLX client.
        
        Args:
            server_url: URL of the MLX server
            model_path: Path to MLX model (for fallback mode)
            batch_size: Maximum batch size for requests
            batch_timeout_ms: Timeout for batch formation
            max_retries: Maximum retry attempts
            use_server: Whether to try server first
        """
        self.server_url = server_url
        self.model_path = model_path
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_retries = max_retries
        self.use_server = use_server
        
        # Server availability
        self.server_available = False
        self.mode = "unknown"
        
        # Direct MLX model (lazy loaded)
        self.model = None
        self.tokenizer = None
        
        # Batch processing
        self.batch_queue = Queue()
        self.batch_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Initialize based on configuration
        self._initialize()
    
    def _initialize(self):
        """Initialize the client based on available resources"""
        if self.use_server:
            self.server_available = self._check_server()
            if self.server_available:
                self.mode = "server"
                logger.info(f"MLX client initialized in server mode: {self.server_url}")
            elif MLX_AVAILABLE:
                self._load_model()
                self.mode = "direct"
                logger.info(f"MLX client initialized in direct mode: {self.model_path}")
            else:
                raise RuntimeError("Neither MLX server nor mlx-lm is available")
        else:
            if MLX_AVAILABLE:
                self._load_model()
                self.mode = "direct"
                logger.info(f"MLX client initialized in direct mode: {self.model_path}")
            else:
                raise RuntimeError("mlx-lm is not available and server mode is disabled")
        
        # Start batch processing thread
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor)
        self.batch_thread.start()
    
    def _check_server(self) -> bool:
        """Check if the MLX server is available"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _load_model(self):
        """Load the MLX model for direct inference"""
        if not MLX_AVAILABLE:
            raise RuntimeError("mlx-lm is not available")
        
        logger.info(f"Loading MLX model: {self.model_path}")
        self.model, self.tokenizer = load(self.model_path)
        mx.eval(self.model.parameters())
        
        # Compile for performance
        def _gen(prompt: str, **kwargs):
            return generate(self.model, self.tokenizer, prompt, **kwargs)
        self.fast_generate = mx.compile(_gen, shapeless=True)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text synchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if self.mode == "server":
            return self._generate_server(prompt, **kwargs)
        else:
            return self._generate_direct(prompt, **kwargs)
    
    def _generate_server(self, prompt: str, **kwargs) -> str:
        """Generate using the MLX server"""
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95)
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/v1/generate",
                    json=payload,
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()["text"]
                else:
                    logger.warning(f"Server returned {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"Server request failed (attempt {attempt + 1}): {e}")
                
                # Try fallback on last attempt
                if attempt == self.max_retries - 1 and MLX_AVAILABLE:
                    logger.info("Falling back to direct MLX inference")
                    if self.model is None:
                        self._load_model()
                    return self._generate_direct(prompt, **kwargs)
        
        raise RuntimeError("Failed to generate text after all retries")
    
    def _generate_direct(self, prompt: str, **kwargs) -> str:
        """Generate using direct MLX inference"""
        if self.model is None:
            self._load_model()
        
        # Use fast compiled generation
        return self.fast_generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            verbose=False,
            **kwargs
        )
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        if self.mode == "server":
            return self._generate_batch_server(prompts, **kwargs)
        else:
            # Process sequentially in direct mode
            return [self._generate_direct(p, **kwargs) for p in prompts]
    
    def _generate_batch_server(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate batch using the MLX server"""
        payload = {
            "prompts": prompts,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95)
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/v1/generate_batch",
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    results = response.json()["results"]
                    return [r["text"] for r in results]
            except Exception as e:
                logger.warning(f"Batch server request failed (attempt {attempt + 1}): {e}")
                
                # Fallback on last attempt
                if attempt == self.max_retries - 1:
                    logger.info("Falling back to sequential generation")
                    return [self.generate(p, **kwargs) for p in prompts]
        
        raise RuntimeError("Failed to generate batch after all retries")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            kwargs
        )
    
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate batch asynchronously"""
        if self.mode == "server":
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompts": prompts,
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.95)
                }
                
                try:
                    async with session.post(
                        f"{self.server_url}/v1/generate_batch",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return [r["text"] for r in data["results"]]
                except Exception as e:
                    logger.warning(f"Async batch request failed: {e}")
        
        # Fallback to concurrent single requests
        tasks = [self.generate_async(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)
    
    def submit_request(self, request: GenerationRequest) -> Future:
        """Submit a request for batch processing"""
        future = Future()
        self.batch_queue.put((request, future))
        return future
    
    def _batch_processor(self):
        """Background thread for batch processing"""
        while self.running:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.001)
    
    def _collect_batch(self) -> Optional[BatchRequest]:
        """Collect requests into a batch"""
        requests = []
        futures = []
        deadline = time.time() + (self.batch_timeout_ms / 1000.0)
        
        while len(requests) < self.batch_size and time.time() < deadline:
            try:
                timeout = max(0, deadline - time.time())
                request, future = self.batch_queue.get(timeout=timeout)
                requests.append(request)
                futures.append(future)
            except:
                break
        
        if requests:
            return BatchRequest(requests=requests, futures=futures)
        return None
    
    def _process_batch(self, batch: BatchRequest):
        """Process a batch of requests"""
        try:
            prompts = [r.prompt for r in batch.requests]
            
            # Get common generation parameters from first request
            first_req = batch.requests[0]
            kwargs = {
                "max_tokens": first_req.max_tokens,
                "temperature": first_req.temperature,
                "top_p": first_req.top_p
            }
            
            # Generate results
            results = self.generate_batch(prompts, **kwargs)
            
            # Set futures
            for future, result in zip(batch.futures, results):
                future.set_result(result)
                
        except Exception as e:
            # Set exception for all futures
            for future in batch.futures:
                future.set_exception(e)
    
    def close(self):
        """Close the client and cleanup resources"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join()
        self.executor.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            "mode": self.mode,
            "server_url": self.server_url if self.mode == "server" else None,
            "model_path": self.model_path if self.mode == "direct" else None,
            "batch_size": self.batch_size,
            "server_available": self.server_available
        }
        
        # Try to get server stats if available
        if self.mode == "server":
            try:
                response = requests.get(f"{self.server_url}/v1/stats", timeout=2)
                if response.status_code == 200:
                    stats["server_stats"] = response.json()
            except:
                pass
        
        return stats


# Convenience functions
_default_client = None

def get_client(**kwargs) -> MLXClient:
    """Get or create the default MLX client"""
    global _default_client
    if _default_client is None:
        _default_client = MLXClient(**kwargs)
    return _default_client


def generate(prompt: str, **kwargs) -> str:
    """Generate text using the default client"""
    return get_client().generate(prompt, **kwargs)


def generate_batch(prompts: List[str], **kwargs) -> List[str]:
    """Generate batch using the default client"""
    return get_client().generate_batch(prompts, **kwargs)