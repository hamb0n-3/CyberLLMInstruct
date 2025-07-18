#!/usr/bin/env python3
"""
Adaptive Speculative Decoding for MLX

This module implements adaptive speculative decoding to accelerate LLM inference
by using a smaller draft model to generate candidate tokens that are verified
by the larger target model.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import Optional, Tuple, List, Dict, Any, Generator, Union
import numpy as np
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    draft_model_path: str
    target_model_path: str
    max_draft_tokens: int = 5  # Max tokens to draft at once
    acceptance_threshold: float = 0.9  # Adaptive threshold
    temperature: float = 0.7
    top_p: float = 0.95
    max_retries: int = 3


class AdaptiveSpeculativeDecoder:
    """
    Implements adaptive speculative decoding with MLX.
    Uses a smaller draft model to generate candidate tokens quickly,
    then verifies them with the larger target model.
    """
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        
        # Load models
        logger.info(f"Loading draft model: {config.draft_model_path}")
        self.draft_model, self.draft_tokenizer = load(config.draft_model_path)
        
        logger.info(f"Loading target model: {config.target_model_path}")
        self.target_model, self.target_tokenizer = load(config.target_model_path)
        
        # Check vocabulary compatibility (following mlx_lm standard)
        if self.draft_tokenizer.vocab_size != self.target_tokenizer.vocab_size:
            logger.warning(
                f"Draft and target models have different vocabulary sizes "
                f"({self.draft_tokenizer.vocab_size} vs {self.target_tokenizer.vocab_size}). "
                f"Speculative decoding may be less effective."
            )
        
        # Store both EOS tokens
        self.draft_eos_token_id = self.draft_tokenizer.eos_token_id
        self.target_eos_token_id = self.target_tokenizer.eos_token_id
        
        if self.draft_eos_token_id != self.target_eos_token_id:
            logger.info(
                f"Draft and target models use different EOS tokens "
                f"({self.draft_eos_token_id} vs {self.target_eos_token_id})"
            )
        
        # Adaptive parameters
        self.acceptance_rate = 0.8  # Running average of acceptance rate
        self.draft_length = config.max_draft_tokens
        
        mx.eval(self.draft_model.parameters())
        mx.eval(self.target_model.parameters())

    def _get_logits(self, model: nn.Module, input_ids: mx.array) -> mx.array:
        """Get logits from a model given input token IDs"""
        # Create attention mask
        mask = mx.ones(input_ids.shape, dtype=mx.int32)
        
        # Get model output
        outputs = model(input_ids, mask)
        return outputs

    def _sample_token(self, logits: mx.array, temperature: float = 1.0, top_p: float = 1.0) -> int:
        """Sample a token from logits using temperature and top-p sampling"""
        if temperature == 0:
            return mx.argmax(logits, axis=-1).item()
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Apply top-p (nucleus) sampling
        sorted_indices = mx.argsort(probs, axis=-1)[::-1]
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

    def _draft_tokens(self, input_ids: mx.array, n_tokens: int) -> Tuple[List[int], List[mx.array]]:
        """Generate draft tokens using the smaller model"""
        draft_tokens = []
        draft_logits = []
        current_ids = input_ids
        
        for _ in range(n_tokens):
            logits = self._get_logits(self.draft_model, current_ids)
            next_token_logits = logits[:, -1, :]
            
            # Sample next token
            next_token = self._sample_token(
                next_token_logits[0], 
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            draft_tokens.append(next_token)
            draft_logits.append(next_token_logits)
            
            # Update input
            current_ids = mx.concatenate([current_ids, mx.array([[next_token]])], axis=1)
            
            # Stop if EOS
            if next_token == self.draft_eos_token_id:
                break
        
        return draft_tokens, draft_logits

    def _verify_tokens(self, input_ids: mx.array, draft_tokens: List[int]) -> Tuple[List[int], int]:
        """Verify draft tokens using the target model"""
        accepted_tokens = []
        
        # Get target model logits for all positions at once
        extended_ids = mx.concatenate([
            input_ids,
            mx.array([draft_tokens]).reshape(1, -1)
        ], axis=1)
        
        target_logits = self._get_logits(self.target_model, extended_ids)
        
        # Verify each draft token
        for i, draft_token in enumerate(draft_tokens):
            position = input_ids.shape[1] + i
            token_logits = target_logits[:, position - 1, :]
            
            # Calculate acceptance probability
            probs = mx.softmax(token_logits[0] / self.config.temperature)
            draft_prob = probs[draft_token].item()
            
            # Adaptive acceptance threshold
            if draft_prob >= self.config.acceptance_threshold * self.acceptance_rate:
                accepted_tokens.append(draft_token)
            else:
                # Rejection - sample from target distribution
                new_token = self._sample_token(
                    token_logits[0],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                accepted_tokens.append(new_token)
                break
        
        return accepted_tokens, len(accepted_tokens)

    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 100,
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using adaptive speculative decoding"""
        # Tokenize prompt
        input_ids = mx.array([self.target_tokenizer.encode(prompt)])
        generated_tokens = []
        
        total_drafted = 0
        total_accepted = 0
        
        while len(generated_tokens) < max_tokens:
            # Adaptively adjust draft length based on acceptance rate
            adaptive_draft_length = min(
                self.config.max_draft_tokens,
                max(1, int(self.draft_length * self.acceptance_rate))
            )
            
            # Draft tokens
            draft_tokens, draft_logits = self._draft_tokens(
                input_ids, 
                adaptive_draft_length
            )
            
            if not draft_tokens:
                break
            
            # Verify tokens
            accepted_tokens, n_accepted = self._verify_tokens(
                input_ids,
                draft_tokens
            )
            
            # Update statistics
            total_drafted += len(draft_tokens)
            total_accepted += n_accepted
            
            # Update acceptance rate (exponential moving average)
            current_rate = n_accepted / len(draft_tokens) if draft_tokens else 0
            self.acceptance_rate = 0.9 * self.acceptance_rate + 0.1 * current_rate
            
            # Add accepted tokens
            generated_tokens.extend(accepted_tokens)
            
            # Update input_ids
            input_ids = mx.concatenate([
                input_ids,
                mx.array([accepted_tokens]).reshape(1, -1)
            ], axis=1)
            
            # Check for EOS
            if accepted_tokens and accepted_tokens[-1] == self.target_eos_token_id:
                break
            
            # Stream output if requested
            if stream:
                yield self.target_tokenizer.decode(accepted_tokens)
        
        # Log statistics
        speedup = total_drafted / max(1, len(generated_tokens))
        logger.info(f"Speculative decoding stats: drafted={total_drafted}, "
                   f"accepted={total_accepted}, speedup={speedup:.2f}x, "
                   f"acceptance_rate={self.acceptance_rate:.2f}")
        
        # Decode final output
        full_text = self.target_tokenizer.decode(generated_tokens)
        
        if stream:
            yield ""  # Signal end of stream
        else:
            return full_text


class SpeculativeDecodingEngine:
    """
    High-level engine that manages multiple speculative decoders
    and provides a simple interface for text generation.
    """
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self.decoder = AdaptiveSpeculativeDecoder(config)
        
    def generate(self, prompt: str, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using speculative decoding"""
        return self.decoder.generate(prompt, **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        # For now, process sequentially
        # Continuous batching will be implemented separately
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results 