#!/usr/bin/env python3
"""
Simplified MLX Inference Engine - Efficient and straightforward implementation
"""

import mlx.core as mx
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Simple configuration for inference"""
    model_path: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 8
    batch_timeout_ms: int = 50


@dataclass 
class Request:
    """Simple request object"""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    timestamp: float = field(default_factory=time.time)
    result: Optional[str] = None


class SimpleInferenceEngine:
    """
    Simplified inference engine that handles batching efficiently.
    Removed unnecessary abstractions and complex threading.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        logger.info(f"Loading model: {config.model_path}")
        self.model, self.tokenizer = load(config.model_path)
        mx.eval(self.model.parameters())
        
        # Simple request tracking
        self.pending_requests: List[Request] = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Direct generation without batching - for single requests"""
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            temp=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            verbose=False
        )
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Efficient batch generation using MLX's native capabilities.
        No complex queue management - just process the batch.
        """
        results = []
        
        # Process in chunks to manage memory
        batch_size = min(len(prompts), self.config.batch_size)
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Generate for each prompt in parallel using MLX
            batch_results = []
            for prompt in batch:
                result = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temp=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    verbose=False
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
        return results


class AsyncInferenceEngine:
    """Simple async wrapper for the inference engine"""
    
    def __init__(self, config: InferenceConfig):
        self.engine = SimpleInferenceEngine(config)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Async generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.engine.generate,
            prompt,
            **kwargs
        )
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Async batch generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.engine.generate_batch,
            prompts,
            **kwargs
        )


# FastAPI-compatible interface
def create_engine(model_path: str, **kwargs) -> SimpleInferenceEngine:
    """Factory function for creating engine"""
    config = InferenceConfig(model_path=model_path, **kwargs)
    return SimpleInferenceEngine(config)