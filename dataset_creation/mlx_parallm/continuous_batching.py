#!/usr/bin/env python3
"""
Continuous Batching for MLX

This module implements continuous batching (also known as dynamic batching)
to improve throughput by processing multiple requests together.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Set
import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    """Represents a single inference request"""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    timestamp: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[str] = None
    tokens_generated: int = 0
    completion_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class BatchConfig:
    """Configuration for continuous batching"""
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    timeout_ms: int = 50  # Max time to wait for batch formation
    min_batch_size: int = 1
    padding_token_id: int = 0


class ContinuousBatchingEngine:
    """
    Implements continuous batching for MLX models.
    Dynamically groups requests into batches for efficient processing.
    """
    
    def __init__(self, model, tokenizer, config: BatchConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Request management
        self.pending_queue: queue.Queue = queue.Queue()
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.batch_thread = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_tokens': 0,
            'avg_batch_size': 0.0,
            'avg_latency': 0.0
        }
        
        mx.eval(self.model.parameters())

    def start(self):
        """Start the batching engine"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.start()
        logger.info("Continuous batching engine started")

    def stop(self):
        """Stop the batching engine"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("Continuous batching engine stopped")

    def submit_request(self, request: InferenceRequest) -> str:
        """Submit a request for processing"""
        self.active_requests[request.request_id] = request
        self.pending_queue.put(request)
        self.stats['total_requests'] += 1
        return request.request_id

    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[str]:
        """Get the result of a request (blocking)"""
        request = self.active_requests.get(request_id)
        if not request:
            request = self.completed_requests.get(request_id)
            if request:
                return request.result
            return None
        
        # Wait for completion
        if request.completion_event.wait(timeout):
            return request.result
        return None

    def _batch_processing_loop(self):
        """Main loop for processing batches"""
        while self.running:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch"""
        batch = []
        deadline = time.time() + (self.config.timeout_ms / 1000.0)
        
        while len(batch) < self.config.max_batch_size and time.time() < deadline:
            try:
                timeout = max(0, deadline - time.time())
                request = self.pending_queue.get(timeout=timeout)
                batch.append(request)
                
                # Early exit if we have enough for efficient processing
                if len(batch) >= self.config.min_batch_size and self.pending_queue.empty():
                    break
                    
            except queue.Empty:
                break
        
        return batch

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests"""
        if not batch:
            return
        
        start_time = time.time()
        self.stats['total_batches'] += 1
        
        # Update request statuses
        for req in batch:
            req.status = RequestStatus.PROCESSING
        
        try:
            # Tokenize all prompts
            input_ids_list = []
            attention_masks = []
            max_length = 0
            
            for req in batch:
                tokens = self.tokenizer.encode(req.prompt)
                input_ids_list.append(tokens)
                max_length = max(max_length, len(tokens))
            
            # Pad sequences to same length
            padded_input_ids = []
            for tokens in input_ids_list:
                padding_length = max_length - len(tokens)
                padded_tokens = tokens + [self.config.padding_token_id] * padding_length
                padded_input_ids.append(padded_tokens)
                
                # Create attention mask
                mask = [1] * len(tokens) + [0] * padding_length
                attention_masks.append(mask)
            
            # Convert to MLX arrays
            input_ids = mx.array(padded_input_ids)
            attention_mask = mx.array(attention_masks)
            
            # Generate tokens iteratively
            generated_sequences = self._generate_batch(
                input_ids, 
                attention_mask,
                batch
            )
            
            # Decode and update results
            for i, (req, generated_ids) in enumerate(zip(batch, generated_sequences)):
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                # Remove the prompt from generated text
                prompt_text = req.prompt
                if generated_text.startswith(prompt_text):
                    generated_text = generated_text[len(prompt_text):].strip()
                
                req.result = generated_text
                req.status = RequestStatus.COMPLETED
                req.tokens_generated = len(generated_ids)
                req.completion_event.set()
                
                # Move to completed
                self.completed_requests[req.request_id] = req
                del self.active_requests[req.request_id]
                
                self.stats['total_tokens'] += req.tokens_generated
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for req in batch:
                req.status = RequestStatus.FAILED
                req.result = None
                req.completion_event.set()
        
        # Update statistics
        batch_time = time.time() - start_time
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * 0.9 + len(batch) * 0.1
        )
        self.stats['avg_latency'] = (
            self.stats['avg_latency'] * 0.9 + batch_time * 0.1
        )
        
        logger.debug(f"Processed batch of {len(batch)} requests in {batch_time:.2f}s")

    def _generate_batch(self, 
                       input_ids: mx.array, 
                       attention_mask: mx.array,
                       requests: List[InferenceRequest]) -> List[List[int]]:
        """Generate tokens for a batch of sequences"""
        batch_size = input_ids.shape[0]
        max_new_tokens = max(req.max_tokens for req in requests)
        
        # Track which sequences are still generating
        active_sequences = set(range(batch_size))
        generated_tokens = [[] for _ in range(batch_size)]
        
        current_ids = input_ids
        current_mask = attention_mask
        
        for step in range(max_new_tokens):
            if not active_sequences:
                break
            
            # Get logits from model
            # Pass None for mask to let model handle it internally
            outputs = self.model(current_ids)
            logits = outputs[:, -1, :]  # Get last token logits
            
            # Sample next tokens
            next_tokens = []
            for i in range(batch_size):
                if i not in active_sequences:
                    # Pad with padding token
                    next_tokens.append(self.config.padding_token_id)
                    continue
                
                # Apply temperature and sampling
                token_logits = logits[i]
                temperature = requests[i].temperature
                
                if temperature > 0:
                    token_logits = token_logits / temperature
                    probs = mx.softmax(token_logits)
                    next_token = mx.random.categorical(mx.log(probs)).item()
                else:
                    next_token = mx.argmax(token_logits).item()
                
                next_tokens.append(next_token)
                generated_tokens[i].append(next_token)
                
                # Check stopping conditions
                if (next_token == self.tokenizer.eos_token_id or 
                    len(generated_tokens[i]) >= requests[i].max_tokens):
                    active_sequences.remove(i)
            
            # Update sequences
            next_tokens_array = mx.array([next_tokens])
            current_ids = mx.concatenate([current_ids, next_tokens_array.T], axis=1)
            
            # Don't update attention mask since we're not using it
        
        return generated_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return self.stats.copy()


class AsyncContinuousBatchingEngine:
    """
    Async wrapper for the continuous batching engine
    """
    
    def __init__(self, model, tokenizer, config: BatchConfig):
        self.engine = ContinuousBatchingEngine(model, tokenizer, config)
        self.engine.start()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Async interface for text generation"""
        request = InferenceRequest(
            request_id=f"req_{time.time()}_{id(prompt)}",
            prompt=prompt,
            **kwargs
        )
        
        # Submit request
        request_id = self.engine.submit_request(request)
        
        # Wait for result in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.engine.get_result, 
            request_id,
            120.0  # 120 second timeout for large models
        )
        
        if result is None:
            raise TimeoutError("Request timed out")
        
        return result
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts concurrently"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def stop(self):
        """Stop the engine"""
        self.engine.stop() 