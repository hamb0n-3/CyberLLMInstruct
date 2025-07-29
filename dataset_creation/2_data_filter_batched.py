#!/usr/bin/env python3
"""
Optimized data filtering pipeline with parallel LLM enhancement.

This is a batched version of 2_data_filter.py that uses multiple LLM instances
for faster processing while maintaining the same cybersecurity filtering logic.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import yaml
import re
import time
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# MLX imports
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
import mlx.core as mx

# Import shared utilities
try:
    from utils import extract_first_json_object, BenchmarkTracker
except ImportError:
    # Fallback implementations
    from json import JSONDecoder
    
    def extract_first_json_object(text: str) -> Optional[Dict]:
        """Extract the first valid JSON object from text."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        try:
            obj, _ = JSONDecoder().raw_decode(text, start_idx)
            return obj
        except json.JSONDecodeError:
            return None
    
    class BenchmarkTracker:
        """Dummy benchmark tracker."""
        def __init__(self, logger=None):
            self.logger = logger
        def record_entry(self, *args, **kwargs): pass
        def record_stage_time(self, *args, **kwargs): pass
        def record_llm_performance(self, *args, **kwargs): pass
        def log_benchmark_stats(self, *args, **kwargs): pass
        def reset_file_timing(self): pass

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Global variables for worker processes
_worker_model = None
_worker_tokenizer = None
_worker_sampler = None
_worker_logits_processors = None

def init_llm_worker(model_path: str, sampling_params: Dict):
    """Initialize MLX model in worker process."""
    global _worker_model, _worker_tokenizer, _worker_sampler, _worker_logits_processors
    
    logger.info(f"Worker {os.getpid()} initializing with model: {model_path}")
    
    try:
        # Set MLX to use GPU
        mx.set_default_device(mx.gpu)
        
        # Load model
        _worker_model, _worker_tokenizer = load(model_path)
        logger.info(f"Worker {os.getpid()} loaded model")
        
        # Force GPU synchronization after model loading
        mx.synchronize()
        
        # Create sampler
        _worker_sampler = make_sampler(
            temp=sampling_params['temperature'],
            top_p=sampling_params['top_p'],
            top_k=sampling_params['top_k'],
            min_p=sampling_params['min_p'],
            min_tokens_to_keep=1
        )
        
        # Create logits processors
        _worker_logits_processors = []
        if sampling_params.get('repetition_penalty', 1.0) != 1.0:
            rep_penalty = make_repetition_penalty(
                penalty=sampling_params['repetition_penalty'],
                context_size=20
            )
            _worker_logits_processors.append(rep_penalty)
        
        logger.info(f"Worker {os.getpid()} ready")
        
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to initialize: {e}", exc_info=True)
        raise

def process_llm_batch(args):
    """Process a batch of entries with LLM enhancement."""
    entries_with_text, max_tokens = args
    global _worker_model, _worker_tokenizer, _worker_sampler, _worker_logits_processors
    
    if _worker_model is None:
        logger.error(f"Worker {os.getpid()} model not loaded!")
        return [(entry, None) for entry, _ in entries_with_text]
    
    logger.debug(f"Worker {os.getpid()} processing {len(entries_with_text)} entries for enhancement")
    
    results = []
    for entry, text_content in entries_with_text:
        try:
            # Format enhancement prompt
            messages = [
                {"role": "user", "content": f"""Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".
Text to analyze:
"{text_content[:2000]}"
```json
"""}
            ]
            
            # Apply chat template if available
            if hasattr(_worker_tokenizer, 'apply_chat_template'):
                prompt = _worker_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                prompt = messages[0]["content"]
            
            # Synchronize before generation
            mx.synchronize()
            
            # Generate response
            response = generate(
                _worker_model,
                _worker_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                sampler=_worker_sampler,
                logits_processors=_worker_logits_processors,
                kv_bits=8,
                kv_group_size=64
            )
            
            # Synchronize after generation
            mx.synchronize()
            
            results.append((entry, response))
            
        except Exception as e:
            logger.error(f"Worker {os.getpid()} error processing entry: {e}")
            results.append((entry, None))
    
    return results

def process_rule_batch(args):
    """Process a batch of entries for rule-based filtering."""
    entries_batch, filter_params = args
    results = []
    
    for entry in entries_batch:
        if not isinstance(entry, dict):
            results.append((entry, False, {'reason': 'Not a dictionary'}))
            continue
        
        text = get_text_from_entry(entry)
        if not text:
            results.append((entry, False, {'reason': 'Empty content'}))
            continue
        
        # Apply filtering rules (matching the logic from 2_data_filter.py)
        passed, debug_info = apply_relevance_rules(text, entry, filter_params)
        results.append((entry, passed, debug_info))
    
    return results

def get_text_from_entry(entry: Dict) -> str:
    """Extract text content from various entry formats (matches 2_data_filter.py)."""
    text_parts = []
    
    # Handle CVE nested structure
    if 'cve' in entry and isinstance(entry['cve'], dict):
        cve_data = entry['cve']
        
        # Add CVE ID
        if 'id' in cve_data:
            text_parts.append(str(cve_data['id']))
        
        # Extract descriptions
        if 'descriptions' in cve_data and isinstance(cve_data['descriptions'], list):
            for desc in cve_data['descriptions']:
                if isinstance(desc, dict) and 'value' in desc:
                    text_parts.append(str(desc['value']))
        
        # Add weakness descriptions
        if 'weaknesses' in cve_data and isinstance(cve_data['weaknesses'], list):
            for weakness in cve_data['weaknesses']:
                if isinstance(weakness, dict) and 'description' in weakness:
                    for desc in weakness['description']:
                        if isinstance(desc, dict) and 'value' in desc:
                            text_parts.append(str(desc['value']))
        
        # Add any metrics descriptions
        if 'metrics' in cve_data:
            metrics = cve_data['metrics']
            for metric_type in ['cvssMetricV2', 'cvssMetricV3', 'cvssMetricV31']:
                if metric_type in metrics and isinstance(metrics[metric_type], list):
                    for metric in metrics[metric_type]:
                        if 'cvssData' in metric and 'vectorString' in metric['cvssData']:
                            text_parts.append(str(metric['cvssData']['vectorString']))
    
    # Original logic for other data types
    else:
        primary_fields = ['title', 'summary', 'description', 'name', 'instruction', 'response']
        text_parts.extend([str(entry.get(field, '')) for field in primary_fields])
        other_parts = [str(value) for key, value in entry.items() 
                      if key not in primary_fields and isinstance(value, str)]
        text_parts.extend(other_parts)
    
    return " ".join(text_parts).strip()

def apply_relevance_rules(text: str, entry: Dict, filter_params: Dict) -> Tuple[bool, Dict]:
    """Apply cybersecurity relevance rules (matches is_relevant_rule_based from 2_data_filter.py)."""
    debug_info = {
        'text_length': len(text),
        'matched_keywords': [],
        'matched_patterns': [],
        'score': 0,
        'context_scores': {}
    }
    
    # Check minimum length
    if len(text) < filter_params['min_content_length']:
        debug_info['reason'] = f"Text too short: {len(text)} chars (min: {filter_params['min_content_length']})"
        return False, debug_info
    
    # Check exclusion patterns
    for pattern_name, pattern in filter_params['exclusion_patterns'].items():
        if re.search(pattern, text, re.IGNORECASE):
            debug_info['reason'] = f"Contains exclusion pattern: {pattern_name}"
            return False, debug_info
    
    text_lower = text.lower()
    
    # Extract context-specific text
    title_text = str(entry.get('title', '') or entry.get('name', '')).lower()
    summary_text = str(entry.get('summary', '') or entry.get('description', '')).lower()
    
    # Check compound terms first (they score higher)
    for relevance, terms in filter_params['compound_terms'].items():
        score_value = 3 if relevance == 'high' else 2
        for term in terms:
            if term in text_lower:
                debug_info['matched_keywords'].append(f"{term}(compound-{relevance})")
                debug_info['score'] += score_value
                # Bonus if in title/summary
                if term in title_text:
                    debug_info['score'] += 2
                    debug_info['context_scores']['title'] = debug_info['context_scores'].get('title', 0) + 2
                elif term in summary_text:
                    debug_info['score'] += 1
                    debug_info['context_scores']['summary'] = debug_info['context_scores'].get('summary', 0) + 1
    
    # Check single keywords
    for relevance, keywords in filter_params['cybersecurity_keywords'].items():
        score_value = 2 if relevance == 'high_relevance' else 1
        for kw in keywords:
            # Skip if already matched as part of compound term
            if any(kw in match for match in debug_info['matched_keywords']):
                continue
                
            if kw in text_lower:
                debug_info['matched_keywords'].append(f"{kw}({relevance})")
                debug_info['score'] += score_value
                # Context bonus
                if kw in title_text:
                    debug_info['score'] += 1
                    debug_info['context_scores']['title'] = debug_info['context_scores'].get('title', 0) + 1
                elif kw in summary_text:
                    debug_info['score'] += 0.5
                    debug_info['context_scores']['summary'] = debug_info['context_scores'].get('summary', 0) + 0.5
    
    # Check regex patterns
    for pattern, score_value in filter_params['security_patterns']:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            debug_info['matched_patterns'].extend(matches[:3])  # Log first 3 matches
            debug_info['score'] += score_value * len(matches)
    
    # Source-based scoring
    source_file = entry.get('_source_file', '')
    for trusted_source in filter_params['trusted_sources']:
        if trusted_source in source_file.lower():
            debug_info['score'] += 1
            debug_info['context_scores']['trusted_source'] = trusted_source
            break
    
    passed = debug_info['score'] >= filter_params['min_keyword_score']
    if not passed:
        debug_info['reason'] = f"Score {debug_info['score']} < {filter_params['min_keyword_score']}"
    else:
        debug_info['reason'] = f"Passed with score {debug_info['score']}"
        
    return passed, debug_info

class CyberDataFilter:
    """Optimized cybersecurity data filter with parallel LLM enhancement."""
    
    def __init__(self, input_dir: str, output_dir: str, model_path: str = None,
                 no_enhancement: bool = False, batch_size: int = 8,
                 num_llm_instances: int = 2, temperature: float = 0.7,
                 top_p: float = 0.95, top_k: int = 20, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, chunk_size: int = 100):
        """Initialize the filter with configuration."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = model_path
        self.no_enhancement = no_enhancement
        self.batch_size = batch_size
        self.num_llm_instances = num_llm_instances
        self.chunk_size = chunk_size
        
        # Sampling parameters
        self.sampling_params = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty
        }
        
        # Filter parameters (matching 2_data_filter.py)
        self.filter_params = {
            'cybersecurity_keywords': {
                'high_relevance': {
                    'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security', 'attack', 
                    'threat', 'breach', 'cve-', 'patch', 'authentication', 'authorization', 'encryption', 
                    'cryptography', 'backdoor', 'botnet', 'phishing', 'injection', 'zero-day', '0day', 
                    'penetration', 'pentest', 'firewall', 'malicious', 'secure', 'adversary', 'cryptographic', 
                    'cipher', 'confidentiality', 'integrity', 'privacy', 'defense', 'defence', 'vulnerable',
                    'hacker', 'trojan', 'virus', 'worm', 'spyware', 'rootkit', 'keylogger', 'payload',
                    'shellcode', 'buffer overflow', 'sql injection', 'xss', 'csrf', 'ddos', 'dos attack',
                    'brute force', 'privilege escalation', 'lateral movement', 'command injection',
                    'remote code execution', 'rce', 'local file inclusion', 'lfi', 'remote file inclusion', 'rfi'
                },
                'medium_relevance': {
                    'network', 'system', 'software', 'hardware', 'protocol', 'server', 'client', 'database', 
                    'web', 'application', 'code', 'programming', 'access', 'control', 'monitoring', 'detection', 
                    'response', 'incident', 'rsa', 'des', 'aes', 'tls', 'ssl', 'https', 'certificate', 'key', 
                    'algorithm', 'hash', 'signature', 'audit', 'compliance', 'policy', 'endpoint', 'siem',
                    'ids', 'ips', 'waf', 'nat', 'vpn', 'proxy', 'sandbox', 'honeypot', 'forensics',
                    'reverse engineering', 'binary analysis', 'static analysis', 'dynamic analysis',
                    'security operations', 'soc', 'incident response', 'threat intelligence', 'ioc', 'ttp'
                }
            },
            'compound_terms': {
                'high': [
                    'buffer overflow', 'sql injection', 'cross site scripting', 'cross-site scripting',
                    'denial of service', 'dos attack', 'ddos attack', 'brute force', 'privilege escalation',
                    'lateral movement', 'command injection', 'remote code execution', 'local file inclusion',
                    'remote file inclusion', 'zero day', 'advanced persistent threat', 'apt attack',
                    'supply chain attack', 'man in the middle', 'mitm attack', 'session hijacking'
                ],
                'medium': [
                    'reverse engineering', 'binary analysis', 'static analysis', 'dynamic analysis',
                    'security operations', 'incident response', 'threat intelligence', 'security information',
                    'event management', 'intrusion detection', 'intrusion prevention', 'web application firewall',
                    'virtual private network', 'network address translation', 'security orchestration'
                ]
            },
            'security_patterns': [
                (r'\bCVE-\d{4}-\d{4,}\b', 3),  # CVE IDs (high relevance)
                (r'\bMS\d{2}-\d{3}\b', 2),      # Microsoft Security Bulletins
                (r'\bCAPEC-\d+\b', 2),          # CAPEC IDs
                (r'\bCWE-\d+\b', 2),            # CWE IDs
                (r'\bOWASP\s+Top\s+\d+\b', 2), # OWASP references
                (r'\b[A-Z]{2,}-\d{4}-\d+\b', 1) # Generic security advisory format
            ],
            'exclusion_patterns': {
                'generic_terms': r'\b(test|sample|example|dummy|todo|foo|bar|placeholder)\b',
                'placeholder_text': r'\b(lorem ipsum|xxx|placeholder|your text here)\b',
                'empty_content': r'^\s*$',
                'non_security_academic': r'\b(abstract algebra|pure mathematics|theoretical physics|quantum mechanics)\b'
            },
            'trusted_sources': ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft'],
            'min_content_length': 25,
            'min_keyword_score': 1
        }
        
        # Benchmark tracker
        self.benchmark = BenchmarkTracker(logger)
        
        # Stats
        self.stats = {
            'total_files': 0,
            'total_entries': 0,
            'rule_based_passed': 0,
            'enhancement_passed': 0,
            'final_retained': 0,
            'processing_times': {}
        }
    
    def load_data(self, file_path: Path) -> List[Dict]:
        """Load data from various file formats."""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
            elif file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []
            
            # Handle CAPEC files
            if file_path.name.startswith('capec_data'):
                return self.process_capec_file(data)
            
            # Handle wrapped data structures
            if isinstance(data, dict):
                for key in ['data', 'entries', 'papers', 'objects', 'vulnerabilities', 'value', 'ctftime_events']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def process_capec_file(self, data: Dict) -> List[Dict]:
        """Process CAPEC XML data."""
        import xml.etree.ElementTree as ET
        xml_string = data.get("xml_data")
        if not xml_string:
            return []
        
        namespace = {'capec': 'http://capec.mitre.org/capec-3'}
        try:
            root = ET.fromstring(xml_string)
            patterns = []
            for p in root.findall('.//capec:Attack_Pattern', namespace):
                desc_elem = p.find('capec:Description', namespace)
                if desc_elem is not None:
                    patterns.append({
                        'id': f"CAPEC-{p.get('ID')}",
                        'name': p.get('Name'),
                        'description': re.sub(r'\s+', ' ', ''.join(desc_elem.itertext()).strip())
                    })
            return patterns
        except ET.ParseError as e:
            logger.error(f"Failed to parse CAPEC XML: {e}")
            return []
    
    def filter_dataset(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Filter and enhance a dataset file."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {file_path.name}")
        logger.info(f"{'='*60}")
        
        self.benchmark.reset_file_timing()
        start_time = time.time()
        
        # Load data
        all_entries = self.load_data(file_path)
        if not all_entries:
            return [], []
        
        self.stats['total_entries'] += len(all_entries)
        logger.info(f"Loaded {len(all_entries)} entries")
        
        # Add source file info
        for entry in all_entries:
            if isinstance(entry, dict):
                entry['_source_file'] = file_path.name
        
        # Rule-based filtering (this is the ONLY relevance check)
        rule_start = time.time()
        relevant_candidates, filtered = self.apply_rule_filtering(all_entries)
        rule_time = time.time() - rule_start
        self.benchmark.record_stage_time('rule_based_filter', rule_time)
        
        logger.info(f"Rule-based filtering took {rule_time:.2f}s ({len(all_entries)/rule_time:.1f} entries/sec)")
        logger.info(f"Rule-based pre-filter complete:")
        logger.info(f"  - Passed: {len(relevant_candidates)}/{len(all_entries)} ({len(relevant_candidates)/len(all_entries)*100:.1f}%)")
        logger.info(f"  - Failed: {len(filtered)}/{len(all_entries)} ({len(filtered)/len(all_entries)*100:.1f}%)")
        
        # LLM enhancement (NOT a filter, just adds metadata)
        if not self.no_enhancement and self.model_path and relevant_candidates:
            enhancement_start = time.time()
            enhanced = self.apply_llm_enhancement(relevant_candidates)
            enhancement_time = time.time() - enhancement_start
            self.benchmark.record_stage_time('enhancement', enhancement_time)
            logger.info(f"Enhancement took {enhancement_time:.2f}s")
        else:
            enhanced = [entry for entry, _ in relevant_candidates]
            if self.no_enhancement:
                logger.info("Skipping LLM enhancement phase (no-enhancement mode)")
        
        elapsed = time.time() - start_time
        self.stats['processing_times'][file_path.name] = elapsed
        
        logger.info(f"\nFile processing complete for {file_path.name}:")
        logger.info(f"  - Total entries: {len(all_entries)}")
        logger.info(f"  - Final retained: {len(enhanced)} ({len(enhanced)/len(all_entries)*100:.1f}%)")
        logger.info(f"  - Processing time: {elapsed:.2f}s")
        logger.info(f"  - Avg time per entry: {elapsed/len(all_entries):.3f}s" if all_entries else "")
        
        self.benchmark.log_benchmark_stats(force=True)
        
        return enhanced, filtered
    
    def apply_rule_filtering(self, entries: List[Dict]) -> Tuple[List[Tuple[Dict, str]], List[Dict]]:
        """Apply rule-based filtering with multiprocessing."""
        logger.info("Applying rule-based filtering...")
        
        # Prepare batches
        batches = []
        for i in range(0, len(entries), self.batch_size * 10):
            batch = entries[i:i + self.batch_size * 10]
            batches.append((batch, self.filter_params))
        
        retained = []
        filtered = []
        
        # Process with multiprocessing
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = [executor.submit(process_rule_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rule filtering"):
                results = future.result()
                for entry, passed, debug_info in results:
                    self.benchmark.record_entry(passed, 0)
                    
                    if passed:
                        text = get_text_from_entry(entry)
                        retained.append((entry, text))
                    else:
                        entry['filtered_reason'] = f"Failed rule-based pre-filter: {debug_info.get('reason', 'Unknown')}"
                        entry['debug_keywords'] = debug_info.get('matched_keywords', [])
                        entry['debug_patterns'] = debug_info.get('matched_patterns', [])
                        entry['debug_score'] = debug_info.get('score', 0)
                        entry['debug_context_scores'] = debug_info.get('context_scores', {})
                        filtered.append(entry)
        
        self.stats['rule_based_passed'] += len(retained)
        return retained, filtered
    
    def apply_llm_enhancement(self, candidates: List[Tuple[Dict, str]]) -> List[Dict]:
        """Enhance entries with LLM-generated metadata."""
        logger.info(f"Starting LLM enhancement for {len(candidates)} candidates...")
        
        enhanced = []
        failed_enhancements = []
        
        # Process in chunks with multiple attempts per entry
        all_batches = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            all_batches.append(batch)
        
        # Token limits for retries
        token_limits = [512, 1024, 1500]
        
        # Set up multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=self.num_llm_instances,
            mp_context=mp_context,
            initializer=init_llm_worker,
            initargs=(self.model_path, self.sampling_params)
        ) as executor:
            
            # Process chunks with retry logic
            for chunk_start in range(0, len(all_batches), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(all_batches))
                chunk_batches = all_batches[chunk_start:chunk_end]
                
                # Try with increasing token limits
                for attempt, max_tokens in enumerate(token_limits):
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1} with {max_tokens} max tokens")
                    
                    # Submit batches for processing
                    batch_args = [(batch, max_tokens) for batch in chunk_batches]
                    futures = [executor.submit(process_llm_batch, args) for args in batch_args]
                    
                    # Collect results
                    retry_needed = []
                    for future in tqdm(as_completed(futures), total=len(futures),
                                     desc=f"Enhancement chunk {chunk_start//self.chunk_size + 1} (attempt {attempt + 1})"):
                        results = future.result()
                        for (entry, text_content), response in results:
                            if response:
                                json_obj = extract_first_json_object(response)
                                if json_obj:
                                    entry.update(json_obj)
                                    entry['is_relevant'] = True
                                    enhanced.append(entry)
                                    self.stats['enhancement_passed'] += 1
                                else:
                                    # Check if likely truncated
                                    if len(response) >= max_tokens - 10 and attempt < len(token_limits) - 1:
                                        retry_needed.append((entry, text_content))
                                    else:
                                        entry['filtered_reason'] = "Failed to extract JSON from enhancement"
                                        failed_enhancements.append(entry)
                            else:
                                retry_needed.append((entry, text_content))
                    
                    # Update chunk_batches for retry
                    if retry_needed and attempt < len(token_limits) - 1:
                        chunk_batches = [retry_needed[i:i+self.batch_size] 
                                       for i in range(0, len(retry_needed), self.batch_size)]
                    else:
                        break
        
        logger.info(f"Enhanced {self.stats['enhancement_passed']}/{len(candidates)} entries")
        if failed_enhancements:
            logger.warning(f"Failed to enhance {len(failed_enhancements)} entries")
        
        self.stats['final_retained'] += len(enhanced)
        return enhanced
    
    def process_directory(self, sources: Optional[List[str]] = None):
        """Process all files in the input directory."""
        files = list(self.input_dir.glob("*.json")) + \
                list(self.input_dir.glob("*.yaml")) + \
                list(self.input_dir.glob("*.yml")) + \
                list(self.input_dir.glob("*.jsonl"))
        
        # Filter by sources if specified
        if sources:
            filtered_files = []
            for file_path in files:
                file_name_lower = file_path.name.lower()
                for source in sources:
                    if source.lower() in file_name_lower:
                        filtered_files.append(file_path)
                        break
            files = filtered_files
            logger.info(f"Filtered to {len(files)} files matching sources: {sources}")
        
        self.stats['total_files'] = len(files)
        logger.info(f"\nStarting batch processing of {len(files)} files...\n")
        
        for file_path in files:
            retained, filtered = self.filter_dataset(file_path)
            
            # Save results with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if retained:
                output_file = self.output_dir / f"{file_path.stem}_filtered_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(retained, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(retained)} entries to {output_file}")
            
            if filtered:
                removed_file = self.output_dir / f"{file_path.stem}_removed_{timestamp}.json"
                with open(removed_file, 'w') as f:
                    json.dump(filtered, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(filtered)} filtered entries to {removed_file}")
        
        # Print final statistics
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print comprehensive statistics summary."""
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL PROCESSING STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Total entries processed: {self.stats['total_entries']}")
        logger.info(f"\nFiltering funnel:")
        logger.info(f"  1. Rule-based filter: {self.stats['rule_based_passed']}/{self.stats['total_entries']} passed ({self.stats['rule_based_passed']/self.stats['total_entries']*100:.1f}%)" if self.stats['total_entries'] > 0 else "  1. Rule-based filter: 0/0 passed")
        logger.info(f"  2. LLM enhancement: {self.stats['enhancement_passed']}/{self.stats['rule_based_passed']} enhanced ({self.stats['enhancement_passed']/self.stats['rule_based_passed']*100:.1f}%)" if self.stats['rule_based_passed'] > 0 else "  2. LLM enhancement: 0/0 enhanced")
        logger.info(f"\nFinal retention rate: {self.stats['final_retained']}/{self.stats['total_entries']} ({self.stats['final_retained']/self.stats['total_entries']*100:.1f}%)" if self.stats['total_entries'] > 0 else "Final retention rate: 0/0")
        
        if self.stats['processing_times']:
            total_time = sum(self.stats['processing_times'].values())
            logger.info(f"\nPerformance metrics:")
            logger.info(f"  Total processing time: {total_time:.2f}s")
            logger.info(f"  Average time per file: {total_time/len(self.stats['processing_times']):.2f}s")
            logger.info(f"  Average time per entry: {total_time/self.stats['total_entries']:.3f}s" if self.stats['total_entries'] > 0 else "")
        
        logger.info(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Filter and enhance cybersecurity data with parallel processing")
    parser.add_argument("--input-dir", default="raw_data", help="Input directory")
    parser.add_argument("--output-dir", default="filtered_data", help="Output directory")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit-DWQ-053125",
                        help="MLX model path or HuggingFace ID")
    parser.add_argument("--no-enhancement", action="store_true",
                        help="Skip LLM enhancement phase")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for LLM processing")
    parser.add_argument("--num-llm-instances", type=int, default=2,
                        help="Number of parallel LLM instances")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of batches per chunk")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--sources", nargs="+", help="Filter specific sources")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = []
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
    else:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'filter_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    
    # Force spawn method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create filter instance
    filter_instance = CyberDataFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model if not args.no_enhancement else None,
        no_enhancement=args.no_enhancement,
        batch_size=args.batch_size,
        num_llm_instances=args.num_llm_instances,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        chunk_size=args.chunk_size
    )
    
    # Process files
    filter_instance.process_directory(sources=args.sources)

if __name__ == "__main__":
    main()