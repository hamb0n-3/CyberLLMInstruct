#!/usr/bin/env python3
"""
MLX Speculative Decoding for CyberLLMInstruct Pipeline
Implements speculative decoding with draft and target models for faster generation
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    draft_model_path: str = "mlx-community/gemma-3-1b-it-bf16"  # Smaller, faster model
    target_model_path: str = "mlx-community/c4ai-command-r-v01-4bit"  # Main model
    speculation_length: int = 5  # Initial tokens to speculate
    max_speculation_length: int = 10
    min_speculation_length: int = 2
    acceptance_threshold: float = 0.8  # Min acceptance rate to increase speculation
    adaptation_window: int = 100  # Tokens to consider for adaptation
    temperature: float = 0.0  # Temperature for draft model
    enable_kv_cache: bool = True
    kv_bits: int = 4  # Lower bits for draft model
    kv_group_size: int = 64


class SpeculativeDecoder:
    """
    Implements speculative decoding using a smaller draft model
    to generate candidate tokens that are verified by the target model.
    """
    
    def __init__(
        self,
        target_model=None,
        target_tokenizer=None,
        config: Optional[SpeculativeConfig] = None
    ):
        """
        Initialize speculative decoder.
        
        Args:
            target_model: Target (main) model instance
            target_tokenizer: Target tokenizer instance
            config: Speculative decoding configuration
        """
        self.config = config or SpeculativeConfig()
        
        # Load models if not provided
        if target_model is None:
            logger.info(f"Loading target model: {self.config.target_model_path}")
            self.target_model, self.target_tokenizer = load(self.config.target_model_path)
        else:
            self.target_model = target_model
            self.target_tokenizer = target_tokenizer
        
        # Load draft model
        logger.info(f"Loading draft model: {self.config.draft_model_path}")
        self.draft_model, self.draft_tokenizer = load(self.config.draft_model_path)
        
        # Ensure tokenizers are compatible
        if self.draft_tokenizer.vocab_size != self.target_tokenizer.vocab_size:
            logger.warning("Draft and target tokenizers have different vocab sizes. This may affect performance.")
        
        # Adaptive speculation length
        self.current_speculation_length = self.config.speculation_length
        self.acceptance_history = []
        
        # Cache for model outputs
        self.draft_cache = {}
        self.target_cache = {}
        
        # Compile models for better performance
        mx.eval(self.draft_model.parameters())
        mx.eval(self.target_model.parameters())
        
        logger.info("SpeculativeDecoder initialized successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False
    ) -> str:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for target model
            top_p: Top-p sampling parameter
            verbose: Whether to print debug information
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        # Tokenize prompt
        input_ids = self.target_tokenizer.encode(prompt)
        generated_ids = input_ids.copy()
        
        tokens_generated = 0
        draft_tokens_generated = 0
        accepted_tokens = 0
        
        while tokens_generated < max_tokens:
            # Current context
            context = generated_ids[-self.config.adaptation_window:]
            
            # Step 1: Generate draft tokens
            draft_tokens = self._generate_draft_tokens(
                context,
                self.current_speculation_length
            )
            draft_tokens_generated += len(draft_tokens)
            
            # Step 2: Verify draft tokens with target model
            accepted, rejected_at = self._verify_tokens(
                context,
                draft_tokens,
                temperature,
                top_p
            )
            
            # Step 3: Accept verified tokens
            if accepted:
                generated_ids.extend(draft_tokens[:rejected_at])
                accepted_tokens += rejected_at
                tokens_generated += rejected_at
            else:
                # Generate one token with target model as fallback
                next_token = self._generate_target_token(
                    context,
                    temperature,
                    top_p
                )
                generated_ids.append(next_token)
                tokens_generated += 1
            
            # Step 4: Update speculation length based on acceptance rate
            self._update_speculation_length(accepted, rejected_at)
            
            # Check for EOS token
            if generated_ids[-1] == self.target_tokenizer.eos_token_id:
                break
            
            if verbose and tokens_generated % 10 == 0:
                acceptance_rate = accepted_tokens / draft_tokens_generated if draft_tokens_generated > 0 else 0
                logger.info(f"Tokens: {tokens_generated}/{max_tokens}, "
                          f"Acceptance rate: {acceptance_rate:.2%}, "
                          f"Speculation length: {self.current_speculation_length}")
        
        # Decode final output
        output_text = self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            total_time = time.time() - start_time
            acceptance_rate = accepted_tokens / draft_tokens_generated if draft_tokens_generated > 0 else 0
            logger.info(f"Generation complete. Time: {total_time:.2f}s, "
                      f"Tokens/sec: {tokens_generated/total_time:.2f}, "
                      f"Overall acceptance rate: {acceptance_rate:.2%}")
        
        return output_text
    
    def _generate_draft_tokens(
        self,
        context: List[int],
        num_tokens: int
    ) -> List[int]:
        """Generate draft tokens using the smaller model"""
        draft_ids = []
        current_context = context.copy()
        
        for _ in range(num_tokens):
            # Get logits from draft model
            logits = self._get_model_logits(
                self.draft_model,
                current_context,
                cache_key="draft"
            )
            
            # Sample next token (greedy for draft model)
            if self.config.temperature == 0:
                next_token = int(mx.argmax(logits, axis=-1).item())
            else:
                # Apply temperature sampling
                probs = mx.softmax(logits / self.config.temperature, axis=-1)
                next_token = int(mx.random.categorical(probs).item())
            
            draft_ids.append(next_token)
            current_context.append(next_token)
            
            # Stop if EOS is generated
            if next_token == self.draft_tokenizer.eos_token_id:
                break
        
        return draft_ids
    
    def _verify_tokens(
        self,
        context: List[int],
        draft_tokens: List[int],
        temperature: float,
        top_p: float
    ) -> Tuple[bool, int]:
        """
        Verify draft tokens using the target model.
        
        Returns:
            (accepted, rejected_at): Whether any tokens were accepted and 
                                   the index of first rejection
        """
        current_context = context.copy()
        
        for i, draft_token in enumerate(draft_tokens):
            # Get target model's prediction
            logits = self._get_model_logits(
                self.target_model,
                current_context,
                cache_key="target"
            )
            
            # Get probability of draft token
            if temperature == 0:
                # Greedy: check if draft token is the argmax
                predicted_token = int(mx.argmax(logits, axis=-1).item())
                if predicted_token != draft_token:
                    return (i > 0, i)
            else:
                # Sampling: check if draft token has sufficient probability
                probs = mx.softmax(logits / temperature, axis=-1)
                
                # Apply top-p if needed
                if top_p < 1.0:
                    probs = self._apply_top_p(probs, top_p)
                
                draft_prob = float(probs[draft_token].item())
                
                # Accept based on probability
                acceptance_prob = min(1.0, draft_prob / 0.1)  # Threshold
                if np.random.random() > acceptance_prob:
                    return (i > 0, i)
            
            current_context.append(draft_token)
        
        # All tokens accepted
        return (True, len(draft_tokens))
    
    def _generate_target_token(
        self,
        context: List[int],
        temperature: float,
        top_p: float
    ) -> int:
        """Generate a single token using the target model"""
        logits = self._get_model_logits(
            self.target_model,
            context,
            cache_key="target"
        )
        
        if temperature == 0:
            return int(mx.argmax(logits, axis=-1).item())
        else:
            probs = mx.softmax(logits / temperature, axis=-1)
            if top_p < 1.0:
                probs = self._apply_top_p(probs, top_p)
            return int(mx.random.categorical(probs).item())
    
    def _get_model_logits(
        self,
        model,
        context: List[int],
        cache_key: str
    ) -> mx.array:
        """Get logits from model with caching"""
        context_key = tuple(context[-50:])  # Use last 50 tokens as key
        
        cache = self.draft_cache if cache_key == "draft" else self.target_cache
        
        if context_key in cache:
            return cache[context_key]
        
        # Convert to tensor and get logits
        input_tensor = mx.array([context])
        
        # Forward pass through model
        # MLX doesn't need no_grad context - gradients are computed explicitly
        outputs = model(input_tensor)
        logits = outputs[0, -1, :]  # Get logits for last position
        
        # Cache result
        cache[context_key] = logits
        
        # Limit cache size
        if len(cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(cache.keys())[:100]
            for key in keys_to_remove:
                del cache[key]
        
        return logits
    
    def _apply_top_p(self, probs: mx.array, top_p: float) -> mx.array:
        """Apply top-p (nucleus) sampling"""
        sorted_indices = mx.argsort(probs, axis=-1, descending=True)
        sorted_probs = probs[sorted_indices]
        cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
        
        # Find cutoff
        cutoff_idx = int(mx.argmax(cumsum_probs > top_p).item())
        cutoff_idx = max(0, cutoff_idx - 1)
        
        # Zero out probabilities after cutoff
        indices_to_zero = sorted_indices[cutoff_idx + 1:]
        probs = probs.at[indices_to_zero].set(0)
        
        # Renormalize
        return probs / mx.sum(probs)
    
    def _update_speculation_length(self, accepted: bool, num_accepted: int):
        """Adaptively update speculation length based on acceptance rate"""
        # Record acceptance
        if accepted:
            self.acceptance_history.append(num_accepted / self.current_speculation_length)
        else:
            self.acceptance_history.append(0.0)
        
        # Keep window size
        if len(self.acceptance_history) > self.config.adaptation_window:
            self.acceptance_history.pop(0)
        
        # Calculate recent acceptance rate
        if len(self.acceptance_history) >= 10:
            recent_rate = np.mean(self.acceptance_history[-10:])
            
            if recent_rate > self.config.acceptance_threshold:
                # Increase speculation length
                self.current_speculation_length = min(
                    self.current_speculation_length + 1,
                    self.config.max_speculation_length
                )
            elif recent_rate < 0.5:
                # Decrease speculation length
                self.current_speculation_length = max(
                    self.current_speculation_length - 1,
                    self.config.min_speculation_length
                )
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False
    ) -> List[str]:
        """
        Generate text for multiple prompts using speculative decoding.
        Processes each prompt sequentially but could be extended for parallel processing.
        """
        results = []
        for i, prompt in enumerate(prompts):
            if verbose:
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, max_tokens, temperature, top_p, verbose=False)
            results.append(result)
        return results


def create_speculative_pipeline(
    target_model_path: str,
    draft_model_path: Optional[str] = None,
    **kwargs
) -> SpeculativeDecoder:
    """
    Factory function to create a speculative decoder with custom configuration.
    
    Args:
        target_model_path: Path to the main model
        draft_model_path: Path to the draft model (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SpeculativeDecoder instance
    """
    config = SpeculativeConfig(
        target_model_path=target_model_path,
        draft_model_path=draft_model_path or "mlx-community/gemma-3-1b-it-bf16",
        **kwargs
    )
    return SpeculativeDecoder(config=config)