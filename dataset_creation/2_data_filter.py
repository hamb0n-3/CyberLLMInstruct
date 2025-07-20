#!/usr/bin/env python3

import json
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
import argparse
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
from dataclasses import dataclass, field
import sys
import platform
import signal

# Import MLX components for batch processing and speculative decoding
from mlx_lm import load, generate
from mlx_lm.generate import speculative_generate_step
import mlx.core as mx

# Setup logging with both console and file handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
benchmarks_dir = logs_dir / "benchmarks"
benchmarks_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"filter_{timestamp}.log"

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

@dataclass
class FileMetrics:
    """Metrics for a single file."""
    items_total: int = 0
    items_passed_rules: int = 0
    items_passed_relevance: int = 0
    items_enhanced: int = 0
    processing_time: float = 0.0
    tokens_per_second: float = 0.0
    avg_batch_size: float = 0.0
    speculative_acceptance_rate: float = 0.0
    rule_filter_time: float = 0.0
    relevance_check_time: float = 0.0
    enhancement_time: float = 0.0

@dataclass
class BenchmarkMetrics:
    """Track performance metrics for the filtering process."""
    # Model information
    model_name: str = ""
    draft_model_name: str = ""
    model_type: str = ""
    
    # System information
    timestamp_start: str = ""
    timestamp_end: str = ""
    platform_info: str = platform.platform()
    python_version: str = sys.version.split()[0]
    
    # Processing parameters
    batch_size: int = 0
    num_draft_tokens: int = 0
    use_speculative_decoding: bool = False
    min_content_length: int = 0
    min_keyword_matches: int = 0
    
    # Performance tracking
    total_tokens_generated: int = 0
    total_generation_time: float = 0.0
    relevance_check_time: float = 0.0
    enhancement_time: float = 0.0
    file_processing_times: Dict[str, float] = field(default_factory=dict)
    speculative_acceptance_rate: float = 0.0
    speculative_tokens_accepted: int = 0
    speculative_tokens_total: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    
    # Per-file metrics
    file_metrics: Dict[str, FileMetrics] = field(default_factory=dict)
    
    # Aggregate metrics
    total_items_processed: int = 0
    total_items_retained: int = 0
    
    def add_generation_metrics(self, tokens: int, time_taken: float):
        self.total_tokens_generated += tokens
        self.total_generation_time += time_taken
    
    def get_tokens_per_second(self) -> float:
        if self.total_generation_time > 0:
            return self.total_tokens_generated / self.total_generation_time
        return 0.0
    
    def get_speculative_acceptance_rate(self) -> float:
        if self.speculative_tokens_total > 0:
            return self.speculative_tokens_accepted / self.speculative_tokens_total
        return 0.0
    
    def get_overall_retention_rate(self) -> float:
        if self.total_items_processed > 0:
            return self.total_items_retained / self.total_items_processed
        return 0.0
    
    def save_to_file(self, filename: str, file_specific: Optional[str] = None):
        """Save benchmark metrics to JSON file."""
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "main": self.model_name,
                "draft": self.draft_model_name,
                "type": self.model_type
            },
            "system": {
                "platform": self.platform_info,
                "python_version": self.python_version,
                "start_time": self.timestamp_start,
                "end_time": self.timestamp_end
            },
            "parameters": {
                "batch_size": self.batch_size,
                "num_draft_tokens": self.num_draft_tokens,
                "use_speculative_decoding": self.use_speculative_decoding,
                "min_content_length": self.min_content_length,
                "min_keyword_matches": self.min_keyword_matches
            },
            "overall_performance": {
                "total_tokens_generated": self.total_tokens_generated,
                "tokens_per_second": self.get_tokens_per_second(),
                "total_generation_time": self.total_generation_time,
                "speculative_acceptance_rate": self.get_speculative_acceptance_rate() if self.use_speculative_decoding else None
            },
            "aggregate_results": {
                "total_items_processed": self.total_items_processed,
                "total_items_retained": self.total_items_retained,
                "overall_retention_rate": self.get_overall_retention_rate()
            }
        }
        
        # Add file-specific metrics if provided
        if file_specific and file_specific in self.file_metrics:
            fm = self.file_metrics[file_specific]
            benchmark_data["file_metrics"] = {
                "filename": file_specific,
                "items_total": fm.items_total,
                "items_passed_rules": fm.items_passed_rules,
                "items_passed_relevance": fm.items_passed_relevance,
                "items_enhanced": fm.items_enhanced,
                "retention_rate": fm.items_enhanced / fm.items_total if fm.items_total > 0 else 0,
                "processing_time": fm.processing_time,
                "tokens_per_second": fm.tokens_per_second,
                "stage_breakdown": {
                    "rule_filtering": {
                        "passed": fm.items_passed_rules,
                        "time": fm.rule_filter_time,
                        "pass_rate": fm.items_passed_rules / fm.items_total if fm.items_total > 0 else 0
                    },
                    "relevance_check": {
                        "passed": fm.items_passed_relevance,
                        "time": fm.relevance_check_time,
                        "pass_rate": fm.items_passed_relevance / fm.items_passed_rules if fm.items_passed_rules > 0 else 0
                    },
                    "enhancement": {
                        "passed": fm.items_enhanced,
                        "time": fm.enhancement_time,
                        "pass_rate": fm.items_enhanced / fm.items_passed_relevance if fm.items_passed_relevance > 0 else 0
                    }
                }
            }
        else:
            # Add all file metrics
            benchmark_data["all_files"] = {}
            for fname, fm in self.file_metrics.items():
                benchmark_data["all_files"][fname] = {
                    "items_total": fm.items_total,
                    "items_retained": fm.items_enhanced,
                    "retention_rate": fm.items_enhanced / fm.items_total if fm.items_total > 0 else 0,
                    "processing_time": fm.processing_time,
                    "tokens_per_second": fm.tokens_per_second
                }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
    
    def log_file_benchmark(self, filename: str):
        """Log detailed benchmark for a specific file."""
        if filename not in self.file_metrics:
            return
            
        fm = self.file_metrics[filename]
        logger.info(f"\n=== Benchmark: {filename} ===")
        logger.info(f"Model: {self.model_name} ({self.model_type})")
        if self.draft_model_name:
            logger.info(f"Draft Model: {self.draft_model_name}")
        logger.info(f"Batch Size: {self.batch_size}, Draft Tokens: {self.num_draft_tokens}")
        
        logger.info("\nProcessing Stats:")
        logger.info(f"- Total items: {fm.items_total}")
        logger.info(f"- Passed rules: {fm.items_passed_rules} ({fm.items_passed_rules/fm.items_total*100:.1f}%)")
        logger.info(f"- Passed relevance: {fm.items_passed_relevance} ({fm.items_passed_relevance/fm.items_passed_rules*100:.1f}% of rules-passed)")
        logger.info(f"- Successfully enhanced: {fm.items_enhanced} ({fm.items_enhanced/fm.items_passed_relevance*100:.1f}% of relevant)")
        logger.info(f"- Final retained: {fm.items_enhanced} ({fm.items_enhanced/fm.items_total*100:.1f}% of total)")
        
        logger.info("\nPerformance:")
        logger.info(f"- Processing time: {fm.processing_time:.1f}s ({fm.items_total/fm.processing_time:.1f} items/s)")
        logger.info(f"- Tokens/second: {fm.tokens_per_second:.1f}")
        if self.use_speculative_decoding:
            logger.info(f"- Speculative acceptance: {fm.speculative_acceptance_rate:.1%}")
        logger.info(f"- Avg batch utilization: {fm.avg_batch_size:.1f}/{self.batch_size}")
        
        # Save benchmark file
        benchmark_file = benchmarks_dir / f"{Path(filename).stem}_{timestamp}.json"
        self.save_to_file(benchmark_file, filename)
        logger.info(f"\nSaved to: {benchmark_file}")
        logger.info("="*40)
    
    def log_summary(self):
        logger.info("\n=== Overall Benchmark Summary ===")
        logger.info(f"Model: {self.model_name} ({self.model_type})")
        if self.draft_model_name:
            logger.info(f"Draft Model: {self.draft_model_name}")
        logger.info(f"Total files processed: {len(self.file_metrics)}")
        logger.info(f"Total items processed: {self.total_items_processed}")
        logger.info(f"Total items retained: {self.total_items_retained} ({self.get_overall_retention_rate():.1%})")
        logger.info(f"Total tokens generated: {self.total_tokens_generated}")
        logger.info(f"Overall tokens/second: {self.get_tokens_per_second():.2f}")
        logger.info(f"Total generation time: {self.total_generation_time:.2f}s")
        if self.draft_model_name:
            logger.info(f"Overall speculative acceptance rate: {self.get_speculative_acceptance_rate():.2%}")
        
        # Save final summary
        summary_file = benchmarks_dir / f"summary_{timestamp}.json"
        self.timestamp_end = datetime.now().isoformat()
        self.save_to_file(summary_file)
        logger.info(f"\nFinal summary saved to: {summary_file}")
        logger.info("="*40)

class CyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, model_path: str, 
                 draft_model_path: Optional[str] = None, batch_size: int = 16,
                 num_draft_tokens: int = 4, verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_draft_tokens = num_draft_tokens
        self.verbose = verbose
        
        # Initialize metrics with full configuration
        self.metrics = BenchmarkMetrics(
            model_name=model_path,
            draft_model_name=draft_model_path or "",
            timestamp_start=datetime.now().isoformat(),
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            use_speculative_decoding=bool(draft_model_path)
        )

        # Load the main model
        logger.info(f"Loading main model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        logger.info("Main model loaded successfully.")
        mx.eval(self.model.parameters())
        
        # Detect model type for prompt formatting
        self.model_type = self._detect_model_type(model_path)
        self.metrics.model_type = self.model_type
        logger.info(f"Detected model type: {self.model_type}")
        
        # Load draft model for speculative decoding
        if draft_model_path:
            logger.info(f"Loading draft model: {draft_model_path}...")
            self.draft_model, self.draft_tokenizer = load(draft_model_path)
            logger.info("Draft model loaded successfully.")
            mx.eval(self.draft_model.parameters())
        else:
            self.draft_model = None
            self.draft_tokenizer = None
            logger.info("No draft model specified, using standard generation.")

        self.trusted_sources = ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft']
        self.cybersecurity_keywords = {'high_relevance': {'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security', 'attack', 'threat', 'breach', 'cve-', 'patch', 'authentication', 'authorization', 'encryption', 'cryptography', 'backdoor', 'botnet', 'phishing', 'injection', 'zero-day', '0day', 'penetration', 'pentest', 'firewall', 'malicious'}, 'medium_relevance': {'network', 'system', 'software', 'hardware', 'protocol', 'server', 'client', 'database', 'web', 'application', 'code', 'programming', 'access', 'control', 'monitoring', 'detection', 'response', 'incident'}}
        self.exclusion_patterns = {'generic_terms': r'\b(test|sample|example|dummy|todo)\b', 'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b', 'empty_content': r'^\s*$'}
        self.min_content_length = 20
        self.min_keyword_matches = 1
        
        # Update metrics with filtering parameters
        self.metrics.min_content_length = self.min_content_length
        self.metrics.min_keyword_matches = self.min_keyword_matches

    def _detect_model_type(self, model_path: str) -> str:
        """Detect the model type from the path for proper prompt formatting."""
        model_path_lower = model_path.lower()
        if 'qwen' in model_path_lower:
            return 'qwen'
        elif 'phi' in model_path_lower:
            return 'phi'
        elif 'llama' in model_path_lower:
            return 'llama'
        elif 'mistral' in model_path_lower:
            return 'mistral'
        else:
            return 'generic'

    def format_prompt(self, text: str, task: str = "relevance") -> str:
        """Format prompt based on model type."""
        if task == "relevance":
            base_prompt = f'Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "{text[:500]}"'
        else:  # enhancement
            base_prompt = f'Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".\nText to analyze:\n"{text}"'
        
        # Apply model-specific formatting
        if self.model_type in ['phi', 'llama', 'mistral']:
            if task == "relevance":
                return f"[INST]{base_prompt}[/INST]"
            else:
                return f"[INST]\n{base_prompt}\n[/INST]```json\n"
        elif self.model_type == 'qwen':
            # Qwen models work better with simple prompts
            return base_prompt
        else:
            # Generic format
            return base_prompt

    def batch_generate_speculative(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """Generate responses for a batch of prompts using speculative decoding."""
        if not self.draft_model:
            # Fallback to standard generation if no draft model
            return self.batch_generate_standard(prompts, max_tokens)
        
        responses = []
        
        # Create progress bar for individual prompts if in verbose mode
        prompt_iter = tqdm(prompts, desc="Speculative generation", leave=False) if self.verbose else prompts
        
        for prompt in prompt_iter:
            prompt_start_time = time.time()
            
            # Tokenize the prompt
            prompt_tokens = mx.array(self.tokenizer.encode(prompt))
            
            # Generate using speculative decoding
            generated_tokens = []
            tokens_accepted = 0
            tokens_total = 0
            
            for token, _, is_draft in speculative_generate_step(
                prompt_tokens,
                self.model,
                self.draft_model,
                num_draft_tokens=self.num_draft_tokens,
                max_tokens=max_tokens,
                kv_bits=8,
                kv_group_size=32
            ):
                generated_tokens.append(token)
                if is_draft:
                    tokens_total += 1
                else:
                    tokens_accepted += 1
                    tokens_total += 1
                    
                if token == self.tokenizer.eos_token_id:
                    break
            
            # Update metrics
            self.metrics.speculative_tokens_accepted += tokens_accepted
            self.metrics.speculative_tokens_total += tokens_total
            
            # Decode the response
            response = self.tokenizer.decode(generated_tokens)
            responses.append(response)
            
            # Update token count with correct timing
            prompt_time = time.time() - prompt_start_time
            self.metrics.add_generation_metrics(len(generated_tokens), prompt_time)
        
        return responses

    def batch_generate_standard(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """Generate responses for a batch of prompts using standard generation."""
        responses = []
        
        # Create progress bar for individual prompts if in verbose mode
        prompt_iter = tqdm(prompts, desc="Standard generation", leave=False) if self.verbose else prompts
        
        # Generate for each prompt
        # Note: MLX doesn't support true batch generation yet, so we process sequentially
        for prompt in prompt_iter:
            gen_start = time.time()
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                verbose=False,
                kv_bits=8,
                kv_group_size=32
            )
            gen_time = time.time() - gen_start
            
            # Estimate token count (rough approximation)
            token_count = len(self.tokenizer.encode(response))
            self.metrics.add_generation_metrics(token_count, gen_time)
            
            responses.append(response)
        
        return responses

    def check_relevance_batch(self, text_contents: List[str]) -> List[bool]:
        """Batch process relevance checks."""
        prompts = []
        for text_content in text_contents:
            prompt = self.format_prompt(text_content, task="relevance")
            prompts.append(prompt)
        
        try:
            start_time = time.time()
            
            if self.draft_model:
                responses = self.batch_generate_speculative(prompts, max_tokens=10)
            else:
                responses = self.batch_generate_standard(prompts, max_tokens=10)
            
            self.metrics.relevance_check_time += time.time() - start_time
            
            results = []
            for i, (response, text) in enumerate(zip(responses, text_contents)):
                # Clean and normalize response
                response_clean = response.strip().upper()
                
                # Check for YES/NO in response
                has_yes = "YES" in response_clean
                has_no = "NO" in response_clean
                
                # Determine result
                if has_yes and not has_no:
                    result = True
                elif has_no and not has_yes:
                    result = False
                else:
                    # Ambiguous or no clear answer, default to False
                    result = False
                    if self.verbose:
                        logger.warning(f"Ambiguous response for text {i}: {response_clean[:50]}...")
                
                results.append(result)
                
                # Log sample responses in verbose mode
                if self.verbose and i < 3:  # Show first 3 responses
                    logger.info(f"Sample relevance check {i}:")
                    logger.info(f"  Text: {text[:100]}...")
                    logger.info(f"  Response: {response_clean[:50]}...")
                    logger.info(f"  Result: {result}")
            
            return results
        except Exception as e:
            logger.error(f"Batch relevance check failed: {e}")
            return [False] * len(text_contents)

    def get_enhancement_batch(self, text_contents: List[str]) -> List[Optional[Dict]]:
        """Batch process enhancements."""
        prompts = []
        for text_content in text_contents:
            prompt = self.format_prompt(text_content, task="enhancement")
            prompts.append(prompt)
        
        try:
            start_time = time.time()
            
            if self.draft_model:
                responses = self.batch_generate_speculative(prompts, max_tokens=1024)
            else:
                responses = self.batch_generate_standard(prompts, max_tokens=1024)
            
            self.metrics.enhancement_time += time.time() - start_time
            
            results = []
            for i, response in enumerate(responses):
                try:
                    # Try multiple JSON extraction patterns
                    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                    if not json_match:
                        # Try to find JSON with nested objects
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(0)
                        # Clean up common issues
                        json_str = json_str.replace('\n', ' ').replace('\\n', ' ')
                        enhancement = json.loads(json_str)
                        results.append(enhancement)
                        
                        if self.verbose and i < 2:  # Show first 2 enhancements
                            logger.info(f"Sample enhancement {i}: {json.dumps(enhancement, indent=2)}")
                    else:
                        results.append(None)
                        if self.verbose:
                            logger.warning(f"No JSON found in enhancement response {i}: {response[:100]}...")
                except Exception as e:
                    results.append(None)
                    if self.verbose:
                        logger.warning(f"Failed to parse enhancement {i}: {e}")
            return results
        except Exception as e:
            logger.error(f"Batch enhancement failed: {e}")
            return [None] * len(text_contents)

    def load_data(self, file_path: Path) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json': data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']: data = yaml.safe_load(f)
                else: return []
            if file_path.name.startswith('capec_data'): return self.process_capec_file(data)
            if isinstance(data, dict):
                for key in ['data', 'entries', 'papers', 'objects', 'vulnerabilities', 'value', 'ctftime_events']:
                    if key in data and isinstance(data[key], list): return data[key]
                return [data]
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    def is_relevant_rule_based(self, text: str) -> bool:
        if not isinstance(text, str) or len(text) < self.min_content_length: return False
        if any(re.search(p, text, re.IGNORECASE) for p in self.exclusion_patterns.values()): return False
        text_lower = text.lower()
        score = sum(2 for kw in self.cybersecurity_keywords['high_relevance'] if kw in text_lower)
        score += sum(1 for kw in self.cybersecurity_keywords['medium_relevance'] if kw in text_lower)
        return score >= self.min_keyword_matches

    def _get_text_from_entry(self, entry: Dict) -> str:
        primary_fields = ['title', 'summary', 'description', 'name', 'instruction', 'response']
        text_parts = [str(entry.get(field, '')) for field in primary_fields]
        other_parts = [str(value) for key, value in entry.items() if key not in primary_fields and isinstance(value, str)]
        return " ".join(text_parts + other_parts).strip()

    def process_capec_file(self, data: Dict) -> List[Dict]:
        xml_string = data.get("xml_data")
        if not xml_string: return []
        namespace = {'capec': 'http://capec.mitre.org/capec-3'}
        try:
            root = ET.fromstring(xml_string)
            patterns = [
                {'id': f"CAPEC-{p.get('ID')}", 'name': p.get('Name'), 'description': re.sub(r'\s+', ' ', ''.join(p.find('capec:Description', namespace).itertext()).strip())}
                for p in root.findall('.//capec:Attack_Pattern', namespace) if p.find('capec:Description', namespace) is not None
            ]
            return patterns
        except ET.ParseError as e:
            logger.error(f"Failed to parse CAPEC XML: {e}")
            return []

    def filter_dataset(self, input_file: Path) -> Tuple[List[Dict], List[Dict]]:
        file_start_time = time.time()
        rule_filter_start = time.time()
        
        # Initialize file metrics
        file_metric = FileMetrics()
        
        all_entries = self.load_data(input_file)
        if not all_entries: return [], []
        
        file_metric.items_total = len(all_entries)
        logger.info(f"Processing {len(all_entries)} entries from {input_file.name}...")
        
        relevant_entries, filtered_out_entries = [], []
        
        # Pre-filter with rule-based approach
        relevance_candidates = []
        for entry in all_entries:
            if not isinstance(entry, dict): continue
            text_content = self._get_text_from_entry(entry)
            if not text_content:
                entry['filtered_reason'] = "Empty content"
                filtered_out_entries.append(entry)
            elif self.is_relevant_rule_based(text_content):
                relevance_candidates.append((entry, text_content))
            else:
                entry['filtered_reason'] = "Failed rule-based pre-filter"
                filtered_out_entries.append(entry)
        
        file_metric.rule_filter_time = time.time() - rule_filter_start
        file_metric.items_passed_rules = len(relevance_candidates)
        logger.info(f"Rule-based pre-filter complete. {len(relevance_candidates)} entries passed for LLM processing.")

        # PASS 1: Batch Relevance Check
        tasks_for_enhancement = []
        total_batches_p1 = (len(relevance_candidates) + self.batch_size - 1) // self.batch_size
        
        # Process in batches
        for batch_idx, i in enumerate(tqdm(range(0, len(relevance_candidates), self.batch_size), 
                                          desc="Pass 1/2: Batch Relevance Check", 
                                          total=total_batches_p1)):
            batch = relevance_candidates[i:i + self.batch_size]
            batch_texts = [text for _, text in batch]
            self.metrics.batch_sizes.append(len(batch))
            
            # Log batch start
            logger.info(f"Processing relevance batch {batch_idx + 1}/{total_batches_p1} (size: {len(batch)})")
            batch_start = time.time()
            
            # Batch check relevance
            relevance_results = self.check_relevance_batch(batch_texts)
            
            # Log batch completion with timing
            batch_time = time.time() - batch_start
            if batch_time > 0:
                rate = len(batch) / batch_time
                remaining_items = len(relevance_candidates) - (i + len(batch))
                eta = remaining_items / rate if rate > 0 else 0
                logger.info(f"  Completed batch in {batch_time:.1f}s ({rate:.1f} items/s, ETA: {eta:.1f}s)")
            
            for (entry, text), is_relevant in zip(batch, relevance_results):
                if is_relevant:
                    tasks_for_enhancement.append((entry, text))
                else:
                    entry['filtered_reason'] = "LLM determined not relevant"
                    filtered_out_entries.append(entry)

        file_metric.items_passed_relevance = len(tasks_for_enhancement)
        logger.info(f"Relevance check complete. {len(tasks_for_enhancement)} entries passed for detailed enhancement.")

        # PASS 2: Batch Enhancement
        total_batches_p2 = (len(tasks_for_enhancement) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, i in enumerate(tqdm(range(0, len(tasks_for_enhancement), self.batch_size), 
                                          desc="Pass 2/2: Batch Enhancement",
                                          total=total_batches_p2)):
            batch = tasks_for_enhancement[i:i + self.batch_size]
            batch_texts = [text for _, text in batch]
            
            # Log batch start with more detail
            logger.info(f"Processing enhancement batch {batch_idx + 1}/{total_batches_p2} (size: {len(batch)})")
            if self.verbose and len(batch) > 0:
                logger.info(f"  First item preview: {batch_texts[0][:100]}...")
            batch_start = time.time()
            
            # Batch enhance
            enhancement_results = self.get_enhancement_batch(batch_texts)
            
            # Log batch completion with timing
            batch_time = time.time() - batch_start
            if batch_time > 0:
                rate = len(batch) / batch_time
                remaining_items = len(tasks_for_enhancement) - (i + len(batch))
                eta = remaining_items / rate if rate > 0 else 0
                logger.info(f"  Completed batch in {batch_time:.1f}s ({rate:.1f} items/s, ETA: {eta:.1f}s)")
            
            # Process results
            success_count = 0
            for (entry, _), enhancement in zip(batch, enhancement_results):
                if enhancement:
                    entry.update(enhancement)
                    entry['is_relevant'] = True
                    relevant_entries.append(entry)
                    success_count += 1
                else:
                    entry['filtered_reason'] = "LLM enhancement failed"
                    filtered_out_entries.append(entry)
            
            logger.info(f"  Enhanced {success_count}/{len(batch)} items successfully")
        
        # Update file metrics
        file_metric.items_enhanced = len(relevant_entries)
        file_metric.processing_time = time.time() - file_start_time
        file_metric.relevance_check_time = self.metrics.relevance_check_time - file_metric.rule_filter_time
        file_metric.enhancement_time = self.metrics.enhancement_time
        file_metric.tokens_per_second = self.metrics.get_tokens_per_second()
        file_metric.avg_batch_size = sum(self.metrics.batch_sizes)/len(self.metrics.batch_sizes) if self.metrics.batch_sizes else 0
        file_metric.speculative_acceptance_rate = self.metrics.get_speculative_acceptance_rate()
        
        # Store file metrics
        self.metrics.file_metrics[input_file.name] = file_metric
        self.metrics.file_processing_times[input_file.name] = file_metric.processing_time
        
        # Update aggregate metrics
        self.metrics.total_items_processed += file_metric.items_total
        self.metrics.total_items_retained += file_metric.items_enhanced
        
        # Log detailed benchmark for this file
        self.metrics.log_file_benchmark(input_file.name)
                        
        return relevant_entries, filtered_out_entries

    def save_data(self, data: List[Dict], original_file: Path, suffix: str = ''):
        if not data: return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"{original_file.stem}{suffix}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} entries to {output_file}")

    def process_directory(self, limit: Optional[int] = None):
        files_to_process = [p for p in self.input_dir.glob('*.*') if p.suffix.lower() in ['.json', '.yaml', '.yml']]
        if limit: files_to_process = files_to_process[:limit]
        
        logger.info(f"Found {len(files_to_process)} files to process.")
        overall_start = time.time()
        
        try:
            for file_path in files_to_process:
                relevant, filtered = self.filter_dataset(file_path)
                self.save_data(relevant, file_path, '_filtered')
                self.save_data(filtered, file_path, '_removed')
                logger.info(f"--- Finished processing {file_path.name}: {len(relevant)} retained, {len(filtered)} removed ---")
        except KeyboardInterrupt:
            logger.info("\n\n=== INTERRUPTED BY USER (Ctrl+C) ===")
            logger.info("Saving current benchmark data...")
        
        # Log overall benchmarks (runs for both normal completion and interruption)
        total_time = time.time() - overall_start
        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Files processed: {len(self.metrics.file_metrics)}/{len(files_to_process)}")
        self.metrics.log_summary()
        
        # Save interrupted state benchmark if interrupted
        if len(self.metrics.file_metrics) < len(files_to_process):
            interrupt_file = benchmarks_dir / f"interrupted_{timestamp}.json"
            self.metrics.timestamp_end = datetime.now().isoformat()
            self.metrics.save_to_file(interrupt_file)
            logger.info(f"\nInterrupted state saved to: {interrupt_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter and enhance data using MLX models with batch processing and speculative decoding.")
    parser.add_argument("--input-dir", default="raw_data", help="Directory containing raw data files.")
    parser.add_argument("--output-dir", default="filtered_data", help="Directory to save the filtered data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for all).")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-14B-4bit-DWQ-053125", help="The MLX-compatible model to use.")
    parser.add_argument("--draft-model", type=str, default=None, help="The draft model for speculative decoding (e.g., mlx-community/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="Number of draft tokens for speculative decoding.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    args = parser.parse_args()
    
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        try:
            print("PyYAML not found. Attempting to install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            print("PyYAML installed successfully.")
        except Exception as e:
            logger.error(f"Failed to install PyYAML. Please install it manually using 'pip install pyyaml'. Error: {e}")
            sys.exit(1)

    try:
        data_filter = CyberDataFilter(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            model_path=args.model,
            draft_model_path=args.draft_model,
            batch_size=args.batch_size,
            num_draft_tokens=args.num_draft_tokens,
            verbose=args.verbose
        )
        
        data_filter.process_directory(limit=args.limit if args.limit > 0 else None)
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Benchmarks have been saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()