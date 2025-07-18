#!/usr/bin/env python3
"""
Unified Inference Engine for MLX

This module provides a simple interface that can use either:
- Direct MLX inference (original method)
- Advanced inference with speculative decoding and continuous batching
"""

import logging
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import asyncio

try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .speculative_decoding import SpeculativeConfig, SpeculativeDecodingEngine
from .continuous_batching import BatchConfig, AsyncContinuousBatchingEngine

logger = logging.getLogger(__name__)


class InferenceMode:
    """Inference mode options"""
    DIRECT = "direct"
    SPECULATIVE = "speculative"
    CONTINUOUS_BATCHING = "continuous_batching"


class UnifiedInferenceEngine:
    """
    Unified interface for MLX inference that supports multiple modes:
    - Direct: Original single-request inference
    - Speculative: Low-latency inference with draft model
    - Continuous Batching: High-throughput batch processing
    """
    
    def __init__(self, 
                 model_path: str,
                 mode: str = InferenceMode.DIRECT,
                 draft_model_path: Optional[str] = None,
                 batch_size: int = 8,
                 batch_timeout_ms: int = 50):
        """
        Initialize the unified inference engine.
        
        Args:
            model_path: Path to the main model
            mode: Inference mode (direct, speculative, continuous_batching)
            draft_model_path: Path to draft model (for speculative mode)
            batch_size: Maximum batch size (for batching mode)
            batch_timeout_ms: Batch formation timeout in ms
        """
        self.mode = mode
        self.model_path = model_path
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available. Please install mlx-lm")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        mx.eval(self.model.parameters())
        
        # Initialize mode-specific engines
        self.speculative_engine = None
        self.batch_engine = None
        
        if mode == InferenceMode.SPECULATIVE:
            if not draft_model_path:
                raise ValueError("Draft model path required for speculative mode")
            
            logger.info(f"Initializing speculative decoding with draft model: {draft_model_path}")
            spec_config = SpeculativeConfig(
                draft_model_path=draft_model_path,
                target_model_path=model_path,
                max_draft_tokens=5
            )
            self.speculative_engine = SpeculativeDecodingEngine(spec_config)
            
        elif mode == InferenceMode.CONTINUOUS_BATCHING:
            logger.info("Initializing continuous batching engine")
            batch_config = BatchConfig(
                max_batch_size=batch_size,
                timeout_ms=batch_timeout_ms,
                padding_token_id=self.tokenizer.pad_token_id or 0
            )
            self.batch_engine = AsyncContinuousBatchingEngine(
                self.model, 
                self.tokenizer, 
                batch_config
            )
        
        logger.info(f"Inference engine initialized in {mode} mode")
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 100,
                 temperature: float = 0.7,
                 **kwargs) -> str:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.mode == InferenceMode.SPECULATIVE:
            if not self.speculative_engine:
                raise RuntimeError("Speculative engine not initialized")
            result = self.speculative_engine.generate(
                prompt, 
                max_tokens=max_tokens,
                stream=False
            )
            # Handle generator return type
            return result if isinstance(result, str) else str(result)
        
        elif self.mode == InferenceMode.CONTINUOUS_BATCHING:
            # Run async method in sync context
            if not self.batch_engine:
                raise RuntimeError("Batch engine not initialized")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.batch_engine.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                )
            finally:
                loop.close()
        
        else:  # DIRECT mode
            # Compile for better performance
            if not hasattr(self, '_compiled_generate'):
                def _gen(prompt: str, **kw):
                    return generate(self.model, self.tokenizer, prompt, **kw)
                self._compiled_generate = mx.compile(_gen, shapeless=True)
            
            return self._compiled_generate(
                prompt,
                max_tokens=max_tokens,
                verbose=False,
                **kwargs
            )
    
    def generate_batch(self, 
                      prompts: List[str],
                      max_tokens: int = 100,
                      temperature: float = 0.7,
                      **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if self.mode == InferenceMode.CONTINUOUS_BATCHING:
            # Use native batch processing
            if not self.batch_engine:
                raise RuntimeError("Batch engine not initialized")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.batch_engine.generate_batch(
                        prompts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                )
            finally:
                loop.close()
        
        else:
            # Fall back to sequential processing
            results = []
            for prompt in prompts:
                result = self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                results.append(result)
            return results
    
    def shutdown(self):
        """Clean up resources"""
        if self.batch_engine:
            self.batch_engine.stop()


def create_inference_engine(
    model_path: str,
    use_advanced: bool = False,
    draft_model_path: Optional[str] = None,
    batch_size: int = 8
) -> UnifiedInferenceEngine:
    """
    Factory function to create an inference engine.
    
    Args:
        model_path: Path to the main model
        use_advanced: Whether to use advanced features
        draft_model_path: Path to draft model (enables speculative decoding)
        batch_size: Batch size for continuous batching
        
    Returns:
        Configured inference engine
    """
    if not use_advanced:
        mode = InferenceMode.DIRECT
    elif draft_model_path:
        mode = InferenceMode.SPECULATIVE
    else:
        mode = InferenceMode.CONTINUOUS_BATCHING
    
    return UnifiedInferenceEngine(
        model_path=model_path,
        mode=mode,
        draft_model_path=draft_model_path,
        batch_size=batch_size
    ) 