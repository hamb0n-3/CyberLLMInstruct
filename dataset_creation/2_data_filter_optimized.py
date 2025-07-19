#!/usr/bin/env python3

import json
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from datetime import datetime
import re
import argparse
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm

# We are now loading the model directly in this script.
from mlx_lm import load, generate
import mlx.core as mx

# Import performance logger
try:
    from performance_logger import get_performance_logger
    PERFORMANCE_LOGGING = True
except ImportError:
    PERFORMANCE_LOGGING = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedCyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, model_path: str, 
                 batch_size: int = 32, enable_streaming: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.enable_streaming = enable_streaming
        
        # Load the model directly into the class. This happens only once.
        logger.info(f"Loading model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        logger.info("Model loaded successfully.")
        mx.eval(self.model.parameters())
        
        # Create optimized batch generation function
        self._create_batch_functions()
        
        self.trusted_sources = ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft']
        self.cybersecurity_keywords = {'high_relevance': {'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security', 'attack', 'threat', 'breach', 'cve-', 'patch', 'authentication', 'authorization', 'encryption', 'cryptography', 'backdoor', 'botnet', 'phishing', 'injection', 'zero-day', '0day', 'penetration', 'pentest', 'firewall', 'malicious'}, 'medium_relevance': {'network', 'system', 'software', 'hardware', 'protocol', 'server', 'client', 'database', 'web', 'application', 'code', 'programming', 'access', 'control', 'monitoring', 'detection', 'response', 'incident'}}
        self.exclusion_patterns = {'generic_terms': r'\b(test|sample|example|dummy|todo)\b', 'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b', 'empty_content': r'^\s*$'}
        self.min_content_length = 20
        self.min_keyword_matches = 2
        
        # Performance tracking
        self.perf_logger = get_performance_logger() if PERFORMANCE_LOGGING else None
    
    def _create_batch_functions(self):
        """Create optimized batch processing functions"""
        
        # Batch relevance check function
        def batch_relevance_check(prompts: List[str]) -> List[str]:
            """Check relevance for a batch of prompts"""
            results = []
            
            # Process in parallel using MLX's automatic optimization
            for prompt in prompts:
                response = generate(
                    self.model, 
                    self.tokenizer, 
                    prompt, 
                    max_tokens=5, 
                    verbose=False,
                    kv_bits=8,
                    kv_group_size=32,
                    temp=0.0
                )
                results.append(response)
            
            # Force evaluation of all results at once
            mx.eval(results)
            return results
        
        # Compile the batch function for better performance
        self.batch_relevance = mx.compile(batch_relevance_check, shapeless=True)
        
        # Batch enhancement function
        def batch_enhancement(prompts: List[str]) -> List[str]:
            """Generate enhancements for a batch of prompts"""
            results = []
            
            for prompt in prompts:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=1024,
                    verbose=False,
                    kv_bits=8,
                    kv_group_size=32,
                    temp=0.0
                )
                results.append(response)
            
            mx.eval(results)
            return results
        
        self.batch_enhance = mx.compile(batch_enhancement, shapeless=True)

    def load_data_streaming(self, file_path: Path) -> Iterator[Dict]:
        """Load data in a streaming fashion to reduce memory usage"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    return
                
            # Handle different data structures
            if file_path.name.startswith('capec_data'):
                for item in self.process_capec_file(data):
                    yield item
            elif isinstance(data, dict):
                for key in ['data', 'entries', 'papers', 'objects', 'vulnerabilities', 'value', 'ctftime_events']:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            yield item
                        return
                yield data
            elif isinstance(data, list):
                for item in data:
                    yield item
                    
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return

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

    def filter_dataset_streaming(self, input_file: Path):
        """Process dataset with streaming and true batching"""
        logger.info(f"Processing {input_file.name} with streaming and optimized batching...")
        
        # Output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        relevant_file = self.output_dir / f"{input_file.stem}_filtered_{timestamp}.json"
        filtered_file = self.output_dir / f"{input_file.stem}_removed_{timestamp}.json"
        
        # Open output files for streaming writes
        with open(relevant_file, 'w', encoding='utf-8') as f_relevant, \
             open(filtered_file, 'w', encoding='utf-8') as f_filtered:
            
            f_relevant.write('[\n')
            f_filtered.write('[\n')
            
            first_relevant = True
            first_filtered = True
            
            # Process in batches
            batch = []
            batch_texts = []
            total_processed = 0
            relevant_count = 0
            filtered_count = 0
            
            # Start performance tracking
            if self.perf_logger:
                self.perf_logger.start_pipeline_stage(f"filter_{input_file.name}")
            
            for entry in self.load_data_streaming(input_file):
                if not isinstance(entry, dict):
                    continue
                    
                text_content = self._get_text_from_entry(entry)
                if not text_content:
                    entry['filtered_reason'] = "Empty content"
                    if not first_filtered:
                        f_filtered.write(',\n')
                    json.dump(entry, f_filtered, indent=2, ensure_ascii=False)
                    first_filtered = False
                    filtered_count += 1
                    continue
                
                # Rule-based pre-filter
                if not self.is_relevant_rule_based(text_content):
                    entry['filtered_reason'] = "Failed rule-based pre-filter"
                    if not first_filtered:
                        f_filtered.write(',\n')
                    json.dump(entry, f_filtered, indent=2, ensure_ascii=False)
                    first_filtered = False
                    filtered_count += 1
                    continue
                
                # Add to batch
                batch.append(entry)
                batch_texts.append(text_content)
                
                # Process batch when full
                if len(batch) >= self.batch_size:
                    rel, filt = self._process_batch(batch, batch_texts)
                    
                    # Write results immediately
                    for item in rel:
                        if not first_relevant:
                            f_relevant.write(',\n')
                        json.dump(item, f_relevant, indent=2, ensure_ascii=False)
                        first_relevant = False
                        relevant_count += 1
                    
                    for item in filt:
                        if not first_filtered:
                            f_filtered.write(',\n')
                        json.dump(item, f_filtered, indent=2, ensure_ascii=False)
                        first_filtered = False
                        filtered_count += 1
                    
                    total_processed += len(batch)
                    logger.info(f"Processed {total_processed} items...")
                    
                    # Clear batch
                    batch = []
                    batch_texts = []
            
            # Process remaining items
            if batch:
                rel, filt = self._process_batch(batch, batch_texts)
                
                for item in rel:
                    if not first_relevant:
                        f_relevant.write(',\n')
                    json.dump(item, f_relevant, indent=2, ensure_ascii=False)
                    first_relevant = False
                    relevant_count += 1
                
                for item in filt:
                    if not first_filtered:
                        f_filtered.write(',\n')
                    json.dump(item, f_filtered, indent=2, ensure_ascii=False)
                    first_filtered = False
                    filtered_count += 1
            
            f_relevant.write('\n]')
            f_filtered.write('\n]')
        
        # End performance tracking
        if self.perf_logger:
            self.perf_logger.end_pipeline_stage(f"filter_{input_file.name}")
        
        logger.info(f"Completed {input_file.name}: {relevant_count} relevant, {filtered_count} filtered")

    def _process_batch(self, entries: List[Dict], texts: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of entries with true parallel processing"""
        relevant = []
        filtered = []
        
        # PASS 1: Batch relevance check
        relevance_prompts = [
            f'[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "{text[:500]}"[/INST]'
            for text in texts
        ]
        
        try:
            # Process entire batch at once
            start_time = time.time()
            responses = self.batch_relevance(relevance_prompts)
            relevance_time = time.time() - start_time
            
            if self.perf_logger:
                self.perf_logger.update_stage_progress("relevance_check", len(entries))
            
            # Filter based on relevance
            enhancement_candidates = []
            enhancement_texts = []
            
            for entry, text, response in zip(entries, texts, responses):
                if "YES" in response.upper():
                    enhancement_candidates.append(entry)
                    enhancement_texts.append(text)
                else:
                    entry['filtered_reason'] = "LLM determined not relevant"
                    filtered.append(entry)
            
            # PASS 2: Batch enhancement
            if enhancement_candidates:
                enhancement_prompts = [
                    f"""[INST]
Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".
Text to analyze:
"{text}"
[/INST]```json
"""
                    for text in enhancement_texts
                ]
                
                start_time = time.time()
                enhancement_responses = self.batch_enhance(enhancement_prompts)
                enhancement_time = time.time() - start_time
                
                if self.perf_logger:
                    self.perf_logger.update_stage_progress("enhancement", len(enhancement_candidates))
                
                for entry, response in zip(enhancement_candidates, enhancement_responses):
                    try:
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            llm_result = json.loads(json_match.group(0))
                            entry.update(llm_result)
                            entry['is_relevant'] = True
                            relevant.append(entry)
                        else:
                            entry['filtered_reason'] = "LLM enhancement failed - no JSON"
                            filtered.append(entry)
                    except Exception as e:
                        entry['filtered_reason'] = f"Enhancement parsing failed: {str(e)}"
                        filtered.append(entry)
                
                logger.debug(f"Batch processed in {relevance_time:.2f}s (relevance) + {enhancement_time:.2f}s (enhancement)")
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for entry in entries:
                entry['filtered_reason'] = "Batch processing error"
                filtered.append(entry)
        
        return relevant, filtered

    def process_directory(self, limit: Optional[int] = None):
        files_to_process = [p for p in self.input_dir.glob('*.*') if p.suffix.lower() in ['.json', '.yaml', '.yml']]
        if limit: files_to_process = files_to_process[:limit]
        
        logger.info(f"Found {len(files_to_process)} files to process.")
        
        for file_path in files_to_process:
            start_time = time.time()
            self.filter_dataset_streaming(file_path)
            elapsed = time.time() - start_time
            logger.info(f"--- Finished {file_path.name} in {elapsed:.2f}s ---")

def main():
    parser = argparse.ArgumentParser(description="Optimized filter with true batch processing and streaming.")
    parser.add_argument("--input-dir", default="raw_data", help="Directory containing raw data files.")
    parser.add_argument("--output-dir", default="filtered_data", help="Directory to save the filtered data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for all).")
    parser.add_argument("--model", type=str, default="mlx-community/Phi-3-mini-4k-instruct-4bit", help="The MLX-compatible model to use.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode")
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

    data_filter = OptimizedCyberDataFilter(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        model_path=args.model,
        batch_size=args.batch_size,
        enable_streaming=not args.no_streaming
    )
    
    data_filter.process_directory(limit=args.limit if args.limit > 0 else None)

if __name__ == "__main__":
    main()