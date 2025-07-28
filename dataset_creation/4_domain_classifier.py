#!/usr/bin/env python3

import os
# Disable tokenizers parallelism to avoid multiprocessing conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
from tqdm import tqdm
# Import shared utilities
from utils import extract_first_json_object, BenchmarkTracker

# Import MLX libraries
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    def __init__(self, input_dir: str, output_dir: str, llm_model: str, disable_llm: bool,
                 temperature: float = 0.7, top_p: float = 1.0, top_k: int = 0,
                 min_p: float = 0.0, repetition_penalty: float = 1.0):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer, self.llm_available = None, None, False
        self.sampler = None
        
        # Store sampling parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        
        # Initialize benchmark tracker
        self.benchmark = BenchmarkTracker(logger=logger)

        if not disable_llm and MLX_AVAILABLE:
            try:
                logger.info(f"Loading MLX model '{llm_model}'...")
                self.model, self.tokenizer = load(llm_model)
                self.llm_available = True
                logger.info("MLX model loaded successfully.")
                
                # Create sampler with configured parameters
                self.sampler = make_sampler(
                    temp=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    min_tokens_to_keep=1
                )
                
                # Log sampler configuration
                logger.info(f"Sampler configured: temp={self.temperature}, top_p={self.top_p}, "
                           f"top_k={self.top_k}, min_p={self.min_p}, rep_penalty={self.repetition_penalty}")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                self.llm_available = False
        else:
            logger.warning("LLM disabled or mlx-lm not found.")

        self.domains = ['malware', 'phishing', 'zero_day', 'iot_security', 'web_security', 'network_security', 'vulnerability_management', 'cloud_security', 'cryptography', 'identity_access_management', 'incident_response', 'threat_intelligence', 'compliance_and_frameworks', 'social_engineering']

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the local MLX model with retry logic."""
        if not self.llm_available: 
            return None
            
        max_attempts = 3
        token_limits = [256, 512, 1024]
        
        for attempt in range(max_attempts):
            try:
                # Count input tokens
                input_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else len(prompt.split())
                
                start_time = time.time()
                
                # Create logits processors for repetition penalty if needed
                logits_processors = []
                if self.repetition_penalty != 1.0:
                    rep_penalty_processor = make_repetition_penalty(
                        penalty=self.repetition_penalty,
                        context_size=20
                    )
                    logits_processors.append(rep_penalty_processor)
                
                response = generate(
                    self.model, 
                    self.tokenizer, 
                    prompt=prompt, 
                    verbose=False,
                    max_tokens=token_limits[attempt],
                    sampler=self.sampler,
                    logits_processors=logits_processors if logits_processors else None,
                    kv_bits=8,
                    kv_group_size=64
                )
                
                generation_time = time.time() - start_time
                
                # Count output tokens
                output_tokens = len(self.tokenizer.encode(response)) if self.tokenizer else len(response.split())
                tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
                
                # Record in benchmark
                self.benchmark.record_llm_performance(tokens_per_second, input_tokens, output_tokens, generation_time)
                
                # Check if response was likely cut off
                if output_tokens >= token_limits[attempt] - 10 and attempt < max_attempts - 1:
                    # Response likely truncated, retry with higher limit
                    logger.warning(f"Response likely truncated at {output_tokens} tokens (limit: {token_limits[attempt]}), retrying with higher limit...")
                    continue
                
                if attempt > 0:
                    logger.info(f"Successfully generated response on attempt {attempt + 1} with {token_limits[attempt]} max tokens")
                
                return response
                
            except Exception as e:
                logger.error(f"Error during MLX generation (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    return None
                    
        return None

    def classify_entry_with_llm(self, entry: Dict) -> Dict:
        prompt = f"""[INST]
You are a cybersecurity domain classification expert. Analyze the following content and classify it into one or more of these domains: {', '.join(self.domains)}.
Return your classification as a JSON array of objects with 'domain' and 'confidence' (from 0.0 to 1.0) fields, sorted by confidence. Only include domains with confidence > 0.2.

Content:
Instruction: {entry.get('instruction', '')}
Response: {entry.get('response', '')}

Respond with ONLY the JSON array.
[/INST]"""
        
        try:
            response = self._call_llm(prompt)
            if response:
                # Try to extract JSON array from response
                json_obj = extract_first_json_object('[' + response if not response.strip().startswith('[') else response)
                if json_obj and isinstance(json_obj, list):
                    entry['domains'] = json_obj
                    entry['primary_domain'] = json_obj[0]['domain'] if json_obj else 'uncategorized'
                else:
                    # Fallback to regex
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        classifications = json.loads(json_match.group(0))
                        entry['domains'] = classifications
                        entry['primary_domain'] = classifications[0]['domain'] if classifications else 'uncategorized'
                    else:
                        entry['domains'] = []
                        entry['primary_domain'] = 'uncategorized'
            else:
                entry['domains'] = []
                entry['primary_domain'] = 'uncategorized'
        except Exception as e:
            logger.error(f"Error during LLM classification: {e}")
            entry['domains'] = []
            entry['primary_domain'] = 'uncategorized'
            
        return entry

    def process_directory(self):
        files_to_process = list(self.input_dir.glob('*.json'))
        for file_path in files_to_process:
            logger.info(f"Processing {file_path.name}")
            # Reset file timing for benchmark tracking
            self.benchmark.reset_file_timing()
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f: 
                    data_obj = json.load(f)
                entries_to_process = data_obj.get('data', [])
                if not isinstance(entries_to_process, list): 
                    continue
                if not self.llm_available:
                    logger.warning("LLM not available. Cannot classify entries.")
                    break

                # Using a sequential loop with tqdm
                classified_entries = []
                for entry in tqdm(entries_to_process, desc=f"Classifying {file_path.name}"):
                    classified_entry = self.classify_entry_with_llm(entry)
                    classified_entries.append(classified_entry)
                    
                    # Log benchmark stats periodically
                    self.benchmark.log_benchmark_stats()

                data_obj['data'] = classified_entries
                data_obj['metadata']['classification_timestamp'] = datetime.now().isoformat()
                
                output_file = self.output_dir / f"{file_path.stem}_classified.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_obj, f, indent=2)
                logger.info(f"Saved classified data to {output_file}")
                
                # Increment files completed
                self.benchmark.metrics['files_completed'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
        
        # Log final benchmark stats
        self.benchmark.log_benchmark_stats(force=True)

def main():
    parser = argparse.ArgumentParser(description="Classify cybersecurity data into domains using an LLM.")
    parser.add_argument("--input-dir", default="structured_data", help="Directory containing structured data.")
    parser.add_argument("--output-dir", default="domain_classified", help="Directory to save classified data.")
    parser.add_argument("--model", type=str, default="mlx-community/c4ai-command-r-v01-4bit", help="The MLX-compatible model to use.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM usage entirely.")
    # Add sampling parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter (default: 0.0)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling parameter (default: 0.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    args = parser.parse_args()

    classifier = CyberDomainClassifier(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        llm_model=args.model, 
        disable_llm=args.disable_llm,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty
    )
    classifier.process_directory()

if __name__ == "__main__":
    main()