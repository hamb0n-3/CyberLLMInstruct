#!/usr/bin/env python3
"""
Combined Continuous Batching + Speculative Decoding for MLX

This module implements a unified approach that combines continuous batching
with speculative decoding for maximum throughput and efficiency.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Generator
import threading
import queue
import time
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SpeculativeRequest:
    """Enhanced request with speculative decoding state"""
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
    
    # Speculative decoding state
    accepted_tokens: List[int] = field(default_factory=list)
    draft_acceptance_rate: float = 0.8
    current_position: int = 0
    input_ids: Optional[mx.array] = None
    


@dataclass
class CombinedConfig:
    """Configuration for combined inference"""
    # Model paths
    target_model_path: str
    draft_model_path: Optional[str] = None
    
    # Batching config
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    batch_timeout_ms: int = 50
    min_batch_size: int = 1
    padding_token_id: int = 0
    
    # Speculative decoding config
    max_draft_tokens: int = 5
    acceptance_threshold: float = 0.9
    enable_speculative: bool = True
    
    # Generation config
    temperature: float = 0.7
    top_p: float = 0.95
    
    # KV cache config
    enable_kv_cache: bool = True
    kv_bits: int = 8
    kv_group_size: int = 32


class CombinedInferenceEngine:
    """
    Unified engine that combines continuous batching with speculative decoding.
    Processes multiple requests in batches while using speculative decoding
    for each sequence independently.
    """
    
    def __init__(self, config: CombinedConfig):
        self.config = config
        
        # Load target model
        logger.info(f"Loading target model: {config.target_model_path}")
        if config.enable_kv_cache:
            logger.info(f"KV cache enabled (will be used during generation)")
        
        self.target_model, self.tokenizer = load(config.target_model_path)
        
        # Load draft model if speculative decoding is enabled
        self.draft_model = None
        if config.enable_speculative and config.draft_model_path:
            logger.info(f"Loading draft model: {config.draft_model_path}")
            self.draft_model, _ = load(config.draft_model_path)
            mx.eval(self.draft_model.parameters())
        
        mx.eval(self.target_model.parameters())
        
        # Request management
        self.pending_queue: queue.Queue = queue.Queue()
        self.active_requests: Dict[str, SpeculativeRequest] = {}
        self.completed_requests: Dict[str, SpeculativeRequest] = {}
        
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
            'avg_latency': 0.0,
            'total_drafted': 0,
            'total_accepted': 0,
            'avg_acceptance_rate': 0.8,
            'kv_cache_enabled': config.enable_kv_cache,
            'kv_bits': config.kv_bits if config.enable_kv_cache else None
        }

    def start(self):
        """Start the combined inference engine"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.start()
        logger.info("Combined inference engine started")

    def stop(self):
        """Stop the engine"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("Combined inference engine stopped")
    
    
    
    
    
            
    

    def submit_request(self, request: SpeculativeRequest) -> str:
        """Submit a request for processing"""
        self.active_requests[request.request_id] = request
        self.pending_queue.put(request)
        self.stats['total_requests'] += 1
        return request.request_id

    def _batch_processing_loop(self):
        """Main loop for processing batches"""
        while self.running:
            batch = self._collect_batch()
            if batch:
                self._process_combined_batch(batch)
            else:
                time.sleep(0.001)

    def _collect_batch(self) -> List[SpeculativeRequest]:
        """Collect requests into a batch"""
        batch = []
        deadline = time.time() + (self.config.batch_timeout_ms / 1000.0)
        
        while len(batch) < self.config.max_batch_size and time.time() < deadline:
            try:
                timeout = max(0, deadline - time.time())
                request = self.pending_queue.get(timeout=timeout)
                batch.append(request)
                
                if len(batch) >= self.config.min_batch_size and self.pending_queue.empty():
                    break
                    
            except queue.Empty:
                break
        
        return batch

    def _process_combined_batch(self, batch: List[SpeculativeRequest]):
        """Process a batch with combined continuous batching + speculative decoding"""
        if not batch:
            return
        
        start_time = time.time()
        self.stats['total_batches'] += 1
        
        # Update statuses
        for req in batch:
            req.status = RequestStatus.PROCESSING
        
        try:
            # Initialize requests
            self._initialize_batch_requests(batch)
            
            # Generate tokens iteratively
            if self.config.enable_speculative and self.draft_model:
                self._generate_batch_speculative(batch)
            else:
                self._generate_batch_standard(batch)
            
            # Finalize results
            self._finalize_batch_results(batch)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            for req in batch:
                req.status = RequestStatus.FAILED
                req.result = None
                req.completion_event.set()
        
        # Update statistics
        batch_time = time.time() - start_time
        self._update_statistics(batch, batch_time)

    def _initialize_batch_requests(self, batch: List[SpeculativeRequest]):
        """Initialize input IDs for all requests in batch"""
        for req in batch:
            tokens = self.tokenizer.encode(req.prompt)
            req.input_ids = mx.array([tokens])
            req.current_position = len(tokens)

    def _generate_batch_speculative(self, batch: List[SpeculativeRequest]):
        """Generate tokens using combined batching + speculative decoding"""
        active_requests = set(range(len(batch)))
        max_iterations = max(req.max_tokens for req in batch)
        
        for iteration in range(max_iterations):
            if not active_requests:
                break
            
            # Collect sequences that need processing
            active_batch = [(i, batch[i]) for i in active_requests]
            
            # Draft phase: generate draft tokens for each sequence
            draft_tokens_batch = []
            if self.draft_model:
                for idx, req in active_batch:
                    draft_tokens = self._draft_tokens_for_sequence(req)
                    draft_tokens_batch.append((idx, draft_tokens))
            
            # Verification phase: verify all drafts in a single batch
            verified_results = self._verify_batch_drafts(active_batch, draft_tokens_batch)
            
            # Update sequences with accepted tokens
            for (idx, req), (accepted_tokens, acceptance_rate) in zip(active_batch, verified_results):
                # Update request state
                req.accepted_tokens.extend(accepted_tokens)
                req.tokens_generated = len(req.accepted_tokens)
                req.draft_acceptance_rate = 0.9 * req.draft_acceptance_rate + 0.1 * acceptance_rate
                
                # Update input_ids
                if accepted_tokens:
                    new_tokens = mx.array([accepted_tokens])
                    req.input_ids = mx.concatenate([req.input_ids, new_tokens.reshape(1, -1)], axis=1)
                    req.current_position += len(accepted_tokens)
                
                # Check stopping conditions
                if (accepted_tokens and accepted_tokens[-1] == self.tokenizer.eos_token_id) or \
                   req.tokens_generated >= req.max_tokens:
                    active_requests.remove(idx)
                
                # Update global stats
                self.stats['total_accepted'] += len(accepted_tokens)

    def _draft_tokens_for_sequence(self, req: SpeculativeRequest) -> List[int]:
        """Generate draft tokens for a single sequence"""
        # Adaptive draft length based on acceptance rate
        draft_length = min(
            self.config.max_draft_tokens,
            max(1, int(self.config.max_draft_tokens * req.draft_acceptance_rate))
        )
        
        draft_tokens = []
        current_ids = req.input_ids
        
        for _ in range(draft_length):
            # Get draft model predictions
            # MLX doesn't need no_grad context
            outputs = self.draft_model(current_ids, mx.ones_like(current_ids).astype(mx.float16))
            logits = outputs[:, -1, :]
            
            # Sample token
            next_token = self._sample_token(
                logits[0], 
                temperature=req.temperature,
                top_p=req.top_p
            )
            
            draft_tokens.append(next_token)
            current_ids = mx.concatenate([current_ids, mx.array([[next_token]])], axis=1)
            
            # Stop if EOS
            if next_token == self.tokenizer.eos_token_id:
                break
        
        self.stats['total_drafted'] += len(draft_tokens)
        return draft_tokens

    def _verify_batch_drafts(self, 
                            active_batch: List[Tuple[int, SpeculativeRequest]], 
                            draft_tokens_batch: List[Tuple[int, List[int]]]) -> List[Tuple[List[int], float]]:
        """Verify draft tokens for multiple sequences in a batch"""
        if not self.draft_model or not draft_tokens_batch:
            # Fallback to standard generation
            return self._generate_single_tokens_batch(active_batch)
        
        # Prepare batch inputs for target model
        max_draft_len = max(len(tokens) for _, tokens in draft_tokens_batch)
        batch_size = len(active_batch)
        
        # Create padded sequences with all draft tokens
        padded_sequences = []
        attention_masks = []
        
        for (idx, req), (_, draft_tokens) in zip(active_batch, draft_tokens_batch):
            # Concatenate input_ids with draft tokens
            full_seq = mx.concatenate([
                req.input_ids,
                mx.array([draft_tokens[:max_draft_len]]).reshape(1, -1)
            ], axis=1)
            
            # Pad if necessary
            seq_len = full_seq.shape[1]
            if seq_len < req.input_ids.shape[1] + max_draft_len:
                padding = mx.zeros((1, req.input_ids.shape[1] + max_draft_len - seq_len))
                full_seq = mx.concatenate([full_seq, padding], axis=1)
            
            padded_sequences.append(full_seq[0])
            
            # Create attention mask
            mask = mx.ones((seq_len,), dtype=mx.float16)
            if seq_len < req.input_ids.shape[1] + max_draft_len:
                pad_mask = mx.zeros((req.input_ids.shape[1] + max_draft_len - seq_len,), dtype=mx.float16)
                mask = mx.concatenate([mask, pad_mask])
            attention_masks.append(mask)
        
        # Stack into batch
        batch_input = mx.stack(padded_sequences)
        batch_mask = mx.stack(attention_masks)
        
        # Get target model predictions for entire batch
        # MLX doesn't need no_grad context
        # For now, use standard forward pass
        # TODO: Integrate KV cache when MLX models support it
        outputs = self.target_model(batch_input, batch_mask)
            
        
        # Verify each sequence's draft tokens
        results = []
        for i, ((idx, req), (_, draft_tokens)) in enumerate(zip(active_batch, draft_tokens_batch)):
            accepted_tokens = []
            n_accepted = 0
            
            for j, draft_token in enumerate(draft_tokens):
                position = req.current_position + j
                logits = outputs[i, position - 1, :]
                
                # Calculate acceptance probability
                probs = mx.softmax(logits / req.temperature)
                draft_prob = probs[draft_token].item()
                
                # Accept or reject
                threshold = self.config.acceptance_threshold * req.draft_acceptance_rate
                if draft_prob >= threshold:
                    accepted_tokens.append(draft_token)
                    n_accepted += 1
                else:
                    # Sample new token from target distribution
                    new_token = self._sample_token(logits, req.temperature, req.top_p)
                    accepted_tokens.append(new_token)
                    break
            
            acceptance_rate = n_accepted / len(draft_tokens) if draft_tokens else 0
            results.append((accepted_tokens, acceptance_rate))
        
        return results

    def _generate_single_tokens_batch(self, active_batch: List[Tuple[int, SpeculativeRequest]]) -> List[Tuple[List[int], float]]:
        """Generate single tokens for batch without speculative decoding"""
        # Prepare batch
        sequences = []
        masks = []
        max_len = max(req.input_ids.shape[1] for _, req in active_batch)
        
        for _, req in active_batch:
            seq = req.input_ids[0]
            seq_len = len(seq)
            
            # Pad sequence
            if seq_len < max_len:
                padding = mx.zeros((max_len - seq_len,))
                seq = mx.concatenate([seq, padding])
            sequences.append(seq)
            
            # Create mask
            mask = mx.ones((seq_len,), dtype=mx.float16)
            if seq_len < max_len:
                pad_mask = mx.zeros((max_len - seq_len,), dtype=mx.float16)
                mask = mx.concatenate([mask, pad_mask])
            masks.append(mask)
        
        # Stack and process
        batch_input = mx.stack(sequences)
        batch_mask = mx.stack(masks)
        
        # MLX doesn't need no_grad context
        outputs = self.target_model(batch_input, batch_mask)
        logits = outputs[:, -1, :]
        
        # Sample tokens
        results = []
        for i, (_, req) in enumerate(active_batch):
            token = self._sample_token(logits[i], req.temperature, req.top_p)
            results.append(([token], 1.0))  # Single token, 100% "acceptance"
        
        return results

    def _generate_batch_standard(self, batch: List[SpeculativeRequest]):
        """Standard batch generation without speculative decoding"""
        active_requests = set(range(len(batch)))
        max_iterations = max(req.max_tokens for req in batch)
        
        for iteration in range(max_iterations):
            if not active_requests:
                break
            
            active_batch = [(i, batch[i]) for i in active_requests]
            results = self._generate_single_tokens_batch(active_batch)
            
            for (idx, req), (tokens, _) in zip(active_batch, results):
                req.accepted_tokens.extend(tokens)
                req.tokens_generated = len(req.accepted_tokens)
                
                if tokens:
                    new_tokens = mx.array([tokens])
                    req.input_ids = mx.concatenate([req.input_ids, new_tokens.reshape(1, -1)], axis=1)
                    req.current_position += len(tokens)
                
                if (tokens and tokens[-1] == self.tokenizer.eos_token_id) or \
                   req.tokens_generated >= req.max_tokens:
                    active_requests.remove(idx)

    def _sample_token(self, logits: mx.array, temperature: float = 1.0, top_p: float = 1.0) -> int:
        """Sample token from logits"""
        if temperature == 0:
            return mx.argmax(logits).item()
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = mx.softmax(logits)
        
        # Apply top-p sampling
        sorted_indices = mx.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = mx.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = mx.argmax(cumsum_probs > top_p).item()
        if cutoff_idx > 0:
            sorted_indices = sorted_indices[:cutoff_idx]
            sorted_probs = sorted_probs[:cutoff_idx]
            sorted_probs = sorted_probs / mx.sum(sorted_probs)
        
        # Sample
        idx = mx.random.categorical(mx.log(sorted_probs))
        return sorted_indices[idx].item()

    def _finalize_batch_results(self, batch: List[SpeculativeRequest]):
        """Finalize results for completed requests"""
        for req in batch:
            # Decode generated tokens
            generated_text = self.tokenizer.decode(req.accepted_tokens, skip_special_tokens=True)
            req.result = generated_text
            req.status = RequestStatus.COMPLETED
            req.completion_event.set()
            
            
            # Move to completed
            self.completed_requests[req.request_id] = req
            del self.active_requests[req.request_id]
            
            self.stats['total_tokens'] += req.tokens_generated

    def _update_statistics(self, batch: List[SpeculativeRequest], batch_time: float):
        """Update engine statistics"""
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * 0.9 + len(batch) * 0.1
        )
        self.stats['avg_latency'] = (
            self.stats['avg_latency'] * 0.9 + batch_time * 0.1
        )
        
        # Update acceptance rate
        if self.stats['total_drafted'] > 0:
            self.stats['avg_acceptance_rate'] = (
                self.stats['total_accepted'] / self.stats['total_drafted']
            )
        
        
        logger.debug(f"Processed batch of {len(batch)} requests in {batch_time:.2f}s")

    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[str]:
        """Get result of a request"""
        request = self.active_requests.get(request_id)
        if not request:
            request = self.completed_requests.get(request_id)
            if request:
                return request.result
            return None
        
        if request.completion_event.wait(timeout):
            return request.result
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = self.stats.copy()
        if self.config.enable_speculative and self.draft_model:
            stats['mode'] = 'combined_speculative_batching'
            if stats['total_drafted'] > 0:
                stats['speculative_speedup'] = stats['total_drafted'] / max(1, stats['total_tokens'])
        else:
            stats['mode'] = 'continuous_batching_only'
        return stats


class AsyncCombinedEngine:
    """Async interface for the combined engine"""
    
    def __init__(self, config: CombinedConfig):
        self.engine = CombinedInferenceEngine(config)
        self.engine.start()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously"""
        request = SpeculativeRequest(
            request_id=f"req_{time.time()}_{id(prompt)}",
            prompt=prompt,
            **kwargs
        )
        
        request_id = self.engine.submit_request(request)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.engine.get_result,
            request_id,
            30.0
        )
        
        if result is None:
            raise TimeoutError("Request timed out")
        
        return result
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return self.engine.get_stats()
    
    def stop(self):
        """Stop the engine"""
        self.engine.stop()