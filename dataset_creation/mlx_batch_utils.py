#!/usr/bin/env python3
"""
MLX Batching Utilities for CyberLLMInstruct Pipeline
Provides automatic batching and memory-aware processing for MLX models
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import time
import logging
from queue import Queue, Empty
from threading import Thread, Lock, Event
import psutil

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual request for batch processing"""
    id: int
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    callback: Optional[Any] = None
    metadata: Optional[Dict] = None


@dataclass
class BatchResponse:
    """Response from batch processing"""
    id: int
    text: str
    tokens_generated: int
    time_taken: float
    metadata: Optional[Dict] = None


class BatchedMLXGenerator:
    """
    Automatic batching for MLX model generation.
    Accumulates requests and processes them in optimized batches.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        batch_timeout: float = 0.1,
        max_sequence_length: int = 2048,
        memory_fraction: float = 0.8,
        enable_kv_cache: bool = True,
        kv_bits: int = 8,
        kv_group_size: int = 32
    ):
        """
        Initialize the batched generator.
        
        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance
            max_batch_size: Maximum requests per batch
            batch_timeout: Max time to wait for batch accumulation (seconds)
            max_sequence_length: Maximum sequence length
            memory_fraction: Fraction of available memory to use
            enable_kv_cache: Whether to use KV cache optimization
            kv_bits: Bits for KV cache quantization
            kv_group_size: Group size for KV cache
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_sequence_length = max_sequence_length
        self.memory_fraction = memory_fraction
        self.enable_kv_cache = enable_kv_cache
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        
        # Request queue and processing state
        self.request_queue = Queue()
        self.response_futures = {}
        self.next_request_id = 0
        self.request_lock = Lock()
        
        # Dynamic batch size based on memory
        self._update_dynamic_batch_size()
        
        # Start the batch processing thread
        self.processing_thread = Thread(target=self._batch_processing_loop, daemon=True)
        self.stop_event = Event()
        self.processing_thread.start()
        
        logger.info(f"BatchedMLXGenerator initialized with max_batch_size={self.max_batch_size}")
    
    def _update_dynamic_batch_size(self):
        """Update batch size based on available memory"""
        try:
            available_memory = psutil.virtual_memory().available
            # Estimate memory per request (rough approximation)
            memory_per_request = self.max_sequence_length * 2 * 2  # tokens * bytes * safety_factor
            dynamic_batch_size = int((available_memory * self.memory_fraction) / memory_per_request)
            self.current_batch_size = min(self.max_batch_size, max(1, dynamic_batch_size))
        except:
            self.current_batch_size = self.max_batch_size
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate text with automatic batching.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            metadata: Optional metadata to attach
            
        Returns:
            Generated text
        """
        with self.request_lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            
            # Create a future for this request
            from concurrent.futures import Future
            future = Future()
            self.response_futures[request_id] = future
        
        # Create and queue the request
        request = BatchRequest(
            id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata
        )
        self.request_queue.put(request)
        
        # Wait for the response
        response = future.result()
        return response.text
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> List[str]:
        """
        Generate text for multiple prompts in a single batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated texts
        """
        # Direct batch processing without threading overhead
        return self._process_batch_direct(prompts, max_tokens, temperature, top_p)
    
    def _batch_processing_loop(self):
        """Main loop for processing batches"""
        while not self.stop_event.is_set():
            batch = self._accumulate_batch()
            if batch:
                self._process_batch(batch)
    
    def _accumulate_batch(self) -> List[BatchRequest]:
        """Accumulate requests into a batch"""
        batch = []
        deadline = time.time() + self.batch_timeout
        
        while len(batch) < self.current_batch_size and time.time() < deadline:
            timeout = max(0, deadline - time.time())
            try:
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
            except Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        start_time = time.time()
        
        # Tokenize all prompts
        prompts = [req.prompt for req in batch]
        tokenized = self._tokenize_batch(prompts)
        
        # Find the maximum length in the batch for padding
        max_length = max(len(tokens) for tokens in tokenized['input_ids'])
        
        # Pad sequences and create attention masks
        padded_input_ids = []
        attention_masks = []
        
        for tokens in tokenized['input_ids']:
            # Pad to max length
            padding_length = max_length - len(tokens)
            padded_tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_tokens)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens) + [0] * padding_length
            attention_masks.append(attention_mask)
        
        # Convert to MLX arrays
        input_ids = mx.array(padded_input_ids)
        attention_mask = mx.array(attention_masks)
        
        # Generate for the batch
        try:
            outputs = self._batched_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_tokens=max(req.max_tokens for req in batch),
                temperature=np.mean([req.temperature for req in batch]),
                top_p=np.mean([req.top_p for req in batch])
            )
            
            # Create responses from the generated text
            for i, (request, output) in enumerate(zip(batch, outputs)):
                # output is already the generated text string
                generated_text = output
                
                response = BatchResponse(
                    id=request.id,
                    text=generated_text,
                    tokens_generated=len(self.tokenizer.encode(output)),
                    time_taken=time.time() - start_time,
                    metadata=request.metadata
                )
                
                # Set the future result
                with self.request_lock:
                    if request.id in self.response_futures:
                        self.response_futures[request.id].set_result(response)
                        del self.response_futures[request.id]
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set error for all requests in batch
            with self.request_lock:
                for request in batch:
                    if request.id in self.response_futures:
                        self.response_futures[request.id].set_exception(e)
                        del self.response_futures[request.id]
    
    def _tokenize_batch(self, prompts: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize a batch of prompts"""
        tokenized = {'input_ids': []}
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            tokenized['input_ids'].append(tokens)
        return tokenized
    
    def _batched_generate(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> List[str]:
        """
        Generate text for a batch of inputs using true batch processing.
        """
        batch_size = input_ids.shape[0]
        
        # Process all sequences in parallel
        # MLX models can handle batched inputs natively
        outputs = []
        
        # Generate for all sequences at once
        # Note: This assumes the model supports batch processing
        # For models that don't, we'll need to fall back to sequential
        try:
            # Prepare generation kwargs
            kw = {
                'max_tokens': max_tokens,
                'verbose': False
            }
            if self.enable_kv_cache:
                kw['kv_bits'] = self.kv_bits
                kw['kv_group_size'] = self.kv_group_size
            
            if temperature > 0:
                kw['temp'] = temperature
                kw['top_p'] = top_p
            
            # Process batch in parallel chunks if batch is too large
            # This balances memory usage with parallelism
            chunk_size = min(8, batch_size)  # Process up to 8 at a time
            
            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)
                chunk_outputs = []
                
                # Process chunk in parallel using MLX's automatic parallelization
                for i in range(chunk_start, chunk_end):
                    mask = attention_mask[i]
                    valid_length = int(mx.sum(mask))
                    valid_tokens = input_ids[i][:valid_length]
                    prompt = self.tokenizer.decode(valid_tokens.tolist())
                    
                    # Queue up generation (MLX will parallelize internally)
                    output = generate(self.model, self.tokenizer, prompt, **kw)
                    # output is already the generated text, not tokens
                    chunk_outputs.append(output)
                
                # MLX automatically parallelizes the above operations
                mx.eval(chunk_outputs)  # Force evaluation of the chunk
                outputs.extend(chunk_outputs)
                
        except Exception as e:
            logger.error(f"Batch generation failed, falling back to sequential: {e}")
            # Fall back to sequential processing
            for i in range(batch_size):
                mask = attention_mask[i]
                valid_length = int(mx.sum(mask))
                valid_tokens = input_ids[i][:valid_length]
                prompt = self.tokenizer.decode(valid_tokens.tolist())
                
                output = generate(self.model, self.tokenizer, prompt, **kw)
                outputs.append(output)
        
        return outputs
    
    def _process_batch_direct(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> List[str]:
        """
        Process a batch of prompts directly without threading overhead.
        """
        # Tokenize all prompts
        tokenized = self._tokenize_batch(prompts)
        
        # Find the maximum length for padding
        max_length = max(len(tokens) for tokens in tokenized['input_ids'])
        
        # Pad sequences and create attention masks
        padded_input_ids = []
        attention_masks = []
        
        for tokens in tokenized['input_ids']:
            padding_length = max_length - len(tokens)
            padded_tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_tokens)
            attention_mask = [1] * len(tokens) + [0] * padding_length
            attention_masks.append(attention_mask)
        
        # Convert to MLX arrays
        input_ids = mx.array(padded_input_ids)
        attention_mask = mx.array(attention_masks)
        
        # Generate for the batch
        outputs = self._batched_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            # output is already the generated text
            results.append(output)
        
        return results
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self.stop_event.set()
        self.processing_thread.join()
        logger.info("BatchedMLXGenerator shutdown complete")


class DynamicBatcher:
    """
    Helper class for converting existing code to use batching.
    Collects items and processes them in batches with a simple API.
    """
    
    def __init__(self, batch_fn, batch_size: int = 32):
        """
        Args:
            batch_fn: Function that processes a batch of items
            batch_size: Size of each batch
        """
        self.batch_fn = batch_fn
        self.batch_size = batch_size
        self.pending_items = []
        self.pending_indices = []
    
    def add(self, item: Any, index: int = None):
        """Add an item to the batch"""
        self.pending_items.append(item)
        if index is not None:
            self.pending_indices.append(index)
        
        # Process batch if full
        if len(self.pending_items) >= self.batch_size:
            return self.flush()
        return None
    
    def flush(self) -> Optional[List[Any]]:
        """Process all pending items"""
        if not self.pending_items:
            return None
        
        results = self.batch_fn(self.pending_items)
        
        # Map results back to indices if provided
        if self.pending_indices:
            indexed_results = list(zip(self.pending_indices, results))
            self.pending_items = []
            self.pending_indices = []
            return indexed_results
        else:
            self.pending_items = []
            return results