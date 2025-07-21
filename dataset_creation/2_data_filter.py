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
from json import JSONDecoder

# We are now loading the model directly in this script.
from mlx_lm import load, generate
import mlx.core as mx
mx.set_default_device(mx.gpu)

# Import benchmark module
from benchmark import BenchmarkTracker

# Module logger - users can configure this externally
logger = logging.getLogger(__name__)


def extract_first_json_object(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from text, ignoring any trailing content."""
    # First try to find JSON object boundaries
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Use JSONDecoder to handle proper JSON parsing
    decoder = JSONDecoder()
    try:
        # This will parse the first valid JSON object and return its end position
        obj, end_idx = decoder.raw_decode(text, start_idx)
        return obj
    except json.JSONDecodeError:
        # Fallback: try to extract with balanced braces
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start_idx:i+1])
                        except json.JSONDecodeError:
                            return None
        
        return None

class CyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, model_path: str = None, 
                 no_enhancement: bool = False, logger: Optional[logging.Logger] = None,
                 benchmark_tracker: Optional[BenchmarkTracker] = None):
        """
        Initialize CyberDataFilter.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for filtered output
            model_path: Path to MLX model (optional if no_enhancement=True)
            no_enhancement: Skip LLM enhancement phase
            logger: Optional logger instance
            benchmark_tracker: Optional BenchmarkTracker instance
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.no_enhancement = no_enhancement
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize or use provided benchmark tracker
        self.benchmark = benchmark_tracker or BenchmarkTracker(logger=self.logger)

        # Load model only if enhancement is enabled
        if not self.no_enhancement and model_path:
            self.logger.info(f"Loading model: {model_path}...")
            self.model, self.tokenizer = load(model_path)
            self.logger.info("Model loaded successfully.")
            mx.eval(self.model.parameters())
            def _gen(prompt:str, **kw):
                return generate(self.model,self.tokenizer, prompt, **kw)
            self.fast_generate = mx.compile(_gen,shapeless=True)
        else:
            self.logger.info("Running without LLM enhancement (rule-based filtering only)")
            self.model = None
            self.tokenizer = None
            self.fast_generate = None

        self.trusted_sources = ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft']
        
        # Enhanced keyword lists with compound terms
        self.cybersecurity_keywords = {
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
        }
        
        # Compound terms that should be matched as phrases
        self.compound_terms = {
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
        }
        
        # Regex patterns for security identifiers
        self.security_patterns = [
            (r'\bCVE-\d{4}-\d{4,}\b', 3),  # CVE IDs (high relevance)
            (r'\bMS\d{2}-\d{3}\b', 2),      # Microsoft Security Bulletins
            (r'\bCAPEC-\d+\b', 2),          # CAPEC IDs
            (r'\bCWE-\d+\b', 2),            # CWE IDs
            (r'\bOWASP\s+Top\s+\d+\b', 2), # OWASP references
            (r'\b[A-Z]{2,}-\d{4}-\d+\b', 1) # Generic security advisory format
        ]
        
        self.exclusion_patterns = {
            'generic_terms': r'\b(test|sample|example|dummy|todo|foo|bar|placeholder)\b',
            'placeholder_text': r'\b(lorem ipsum|xxx|placeholder|your text here)\b',
            'empty_content': r'^\s*$',
            'non_security_academic': r'\b(abstract algebra|pure mathematics|theoretical physics|quantum mechanics)\b'
        }
        
        self.min_content_length = 25  # Increased minimum length
        self.min_keyword_score = 1     # Minimum score needed (was min_keyword_matches)
        
        # Initialize statistics
        self.stats = {
            'total_files': 0,
            'total_entries': 0,
            'rule_based_passed': 0,
            'enhancement_passed': 0,
            'final_retained': 0,
            'processing_times': {}
        }


    def get_enhancement(self, text_content: str) -> Optional[Dict]:
        """Pass 2: A slow, detailed enhancement for confirmed relevant items."""
        prompt = f"""
Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".
Text to analyze:
"{text_content}"
```json
"""
        try:
            start_time = time.time()
            response = self.fast_generate(prompt, max_tokens=1024, verbose=False,kv_bits=8,kv_group_size=32)
            generation_time = time.time() - start_time
            
            # Try to extract JSON using the robust extractor
            json_obj = extract_first_json_object(response)
            
            if json_obj:
                # Log successful enhancement timing
                self.logger.debug(f"LLM enhancement took {generation_time:.2f}s for {len(text_content)} chars")
                return json_obj
            else:
                # Log the raw response for debugging
                self.logger.error(f"Failed to extract JSON from LLM response. Raw response (first 500 chars):")
                self.logger.error(f"{response[:500]}...")
                self.logger.error(f"Response length: {len(response)} chars")
                
        except Exception as e:
            self.logger.error(f"Enhancement failed for an item: {e}")
            self.logger.error(f"Text preview (first 200 chars): {text_content[:200]}...")
            
        return None

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
            self.logger.error(f"Error loading file {file_path}: {e}")
            return []

    def is_relevant_rule_based(self, entry: Dict, text: str) -> Tuple[bool, Dict]:
        debug_info = {
            'text_length': len(text) if isinstance(text, str) else 0, 
            'matched_keywords': [], 
            'matched_patterns': [],
            'score': 0,
            'context_scores': {}
        }
        
        if not isinstance(text, str) or len(text) < self.min_content_length: 
            debug_info['reason'] = f"Text too short: {debug_info['text_length']} chars (min: {self.min_content_length})"
            return False, debug_info
            
        # Check exclusion patterns
        for pattern_name, pattern in self.exclusion_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                debug_info['reason'] = f"Contains exclusion pattern: {pattern_name}"
                return False, debug_info
        
        text_lower = text.lower()
        
        # Extract context-specific text
        title_text = str(entry.get('title', '') or entry.get('name', '')).lower()
        summary_text = str(entry.get('summary', '') or entry.get('description', '')).lower()
        
        # Check compound terms first (they score higher)
        for relevance, terms in self.compound_terms.items():
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
        for relevance, keywords in self.cybersecurity_keywords.items():
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
        for pattern, score_value in self.security_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                debug_info['matched_patterns'].extend(matches[:3])  # Log first 3 matches
                debug_info['score'] += score_value * len(matches)
        
        # Source-based scoring
        source_file = entry.get('_source_file', '')
        for trusted_source in self.trusted_sources:
            if trusted_source in source_file.lower():
                debug_info['score'] += 1
                debug_info['context_scores']['trusted_source'] = trusted_source
                break
        
        passed = debug_info['score'] >= self.min_keyword_score
        if not passed:
            debug_info['reason'] = f"Score {debug_info['score']} < {self.min_keyword_score}"
        else:
            debug_info['reason'] = f"Passed with score {debug_info['score']}"
            
        return passed, debug_info

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
            self.logger.error(f"Failed to parse CAPEC XML: {e}")
            return []

    def filter_dataset(self, input_file: Path) -> Tuple[List[Dict], List[Dict]]:
        start_time = time.time()
        all_entries = self.load_data(input_file)
        if not all_entries: return [], []
        
        self.stats['total_entries'] += len(all_entries)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing {len(all_entries)} entries from {input_file.name}")
        self.logger.info(f"{'='*60}")
        
        relevant_entries, filtered_out_entries = [], []
        
        relevance_candidates = []
        rule_based_start = time.time()
        
        for entry in all_entries:
            entry_start = time.time()
            if not isinstance(entry, dict): continue
            # Add source file info for scoring
            entry['_source_file'] = input_file.name
            text_content = self._get_text_from_entry(entry)
            if not text_content:
                entry['filtered_reason'] = "Empty content"
                filtered_out_entries.append(entry)
                self.benchmark.record_entry(False, {'reason': 'Empty content'}, time.time() - entry_start)
            else:
                passed, debug_info = self.is_relevant_rule_based(entry, text_content)
                if passed:
                    relevance_candidates.append((entry, text_content))
                else:
                    entry['filtered_reason'] = f"Failed rule-based pre-filter: {debug_info.get('reason', 'Unknown')}"
                    entry['debug_keywords'] = debug_info.get('matched_keywords', [])
                    entry['debug_patterns'] = debug_info.get('matched_patterns', [])
                    entry['debug_score'] = debug_info.get('score', 0)
                    entry['debug_context_scores'] = debug_info.get('context_scores', {})
                    filtered_out_entries.append(entry)
                self.benchmark.record_entry(passed, debug_info, time.time() - entry_start)
            
            # Check for periodic logging
            self.benchmark.log_benchmark_stats()
        
        rule_based_time = time.time() - rule_based_start
        self.benchmark.record_stage_time('rule_based_filter', rule_based_time)
        
        self.stats['rule_based_passed'] += len(relevance_candidates)
        self.logger.info(f"Rule-based pre-filter complete:")
        self.logger.info(f"  - Passed: {len(relevance_candidates)}/{len(all_entries)} ({len(relevance_candidates)/len(all_entries)*100:.1f}%)")
        self.logger.info(f"  - Failed: {len(filtered_out_entries)}/{len(all_entries)} ({len(filtered_out_entries)/len(all_entries)*100:.1f}%)")

        # Enhancement phase (skip if no_enhancement flag is set)
        if self.no_enhancement or not self.fast_generate:
            # If no enhancement, all candidates pass
            relevant_entries = [entry for entry, _ in relevance_candidates]
            self.logger.info("Skipping LLM enhancement phase (no-enhancement mode)")
        else:
            enhancement_start = time.time()
            total_candidates = len(relevance_candidates)
            self.logger.info(f"Starting LLM enhancement for {total_candidates} candidates...")
            
            # Track timing stats
            enhancement_times = []
            
            for idx, (original_entry, text_content) in enumerate(tqdm(relevance_candidates, desc="Enhancement")):
                entry_start = time.time()
                llm_result = self.get_enhancement(text_content)
                entry_time = time.time() - entry_start
                enhancement_times.append(entry_time)
                
                if llm_result:
                    original_entry.update(llm_result)
                    original_entry['is_relevant'] = True
                    relevant_entries.append(original_entry)
                    self.benchmark.record_entry(True, {'enhanced': True}, entry_time)
                else:
                    original_entry['filtered_reason'] = "LLM enhancement failed"
                    filtered_out_entries.append(original_entry)
                    self.benchmark.record_entry(False, {'reason': 'Enhancement failed'}, entry_time)
                
                # Log progress every 10 entries
                if (idx + 1) % 10 == 0:
                    avg_time = sum(enhancement_times[-10:]) / len(enhancement_times[-10:])
                    eta = avg_time * (total_candidates - idx - 1)
                    self.logger.info(f"Progress: {idx+1}/{total_candidates} - Avg time: {avg_time:.2f}s/entry - ETA: {eta/60:.1f} min")
                
                # Check for periodic logging
                self.benchmark.log_benchmark_stats()
            
            enhancement_time = time.time() - enhancement_start
            self.benchmark.record_stage_time('enhancement', enhancement_time)
            
            # Log enhancement summary
            if enhancement_times:
                self.logger.info(f"\nEnhancement Summary:")
                self.logger.info(f"  - Total time: {enhancement_time:.2f}s")
                self.logger.info(f"  - Average time per entry: {sum(enhancement_times)/len(enhancement_times):.2f}s")
                self.logger.info(f"  - Min/Max times: {min(enhancement_times):.2f}s / {max(enhancement_times):.2f}s")
                
        self.stats['enhancement_passed'] += len(relevant_entries)
        self.stats['final_retained'] += len(relevant_entries)
        
        # Log final statistics for this file
        processing_time = time.time() - start_time
        self.stats['processing_times'][input_file.name] = processing_time
        self.benchmark.metrics['files_completed'] += 1
        
        self.logger.info(f"\nFile processing complete for {input_file.name}:")
        self.logger.info(f"  - Total entries: {len(all_entries)}")
        self.logger.info(f"  - Final retained: {len(relevant_entries)} ({len(relevant_entries)/len(all_entries)*100:.1f}%)")
        self.logger.info(f"  - Processing time: {processing_time:.2f}s")
        self.logger.info(f"  - Avg time per entry: {processing_time/len(all_entries):.3f}s" if all_entries else "")
        
        # Log benchmark stats after each file
        self.benchmark.log_benchmark_stats(force=True)
                        
        return relevant_entries, filtered_out_entries

    def save_data(self, data: List[Dict], original_file: Path, suffix: str = ''):
        if not data: return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"{original_file.stem}{suffix}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(data)} entries to {output_file}")

    def print_final_statistics(self):
        """Print comprehensive statistics summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"FINAL PROCESSING STATISTICS")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total files processed: {self.stats['total_files']}")
        self.logger.info(f"Total entries processed: {self.stats['total_entries']}")
        self.logger.info(f"\nFiltering funnel:")
        self.logger.info(f"  1. Rule-based filter: {self.stats['rule_based_passed']}/{self.stats['total_entries']} passed ({self.stats['rule_based_passed']/self.stats['total_entries']*100:.1f}%)" if self.stats['total_entries'] > 0 else "  1. Rule-based filter: 0/0 passed")
        self.logger.info(f"  2. LLM enhancement: {self.stats['enhancement_passed']}/{self.stats['rule_based_passed']} passed ({self.stats['enhancement_passed']/self.stats['rule_based_passed']*100:.1f}%)" if self.stats['rule_based_passed'] > 0 else "  2. LLM enhancement: 0/0 passed")
        self.logger.info(f"\nFinal retention rate: {self.stats['final_retained']}/{self.stats['total_entries']} ({self.stats['final_retained']/self.stats['total_entries']*100:.1f}%)" if self.stats['total_entries'] > 0 else "Final retention rate: 0/0")
        
        if self.stats['processing_times']:
            total_time = sum(self.stats['processing_times'].values())
            self.logger.info(f"\nPerformance metrics:")
            self.logger.info(f"  Total processing time: {total_time:.2f}s")
            self.logger.info(f"  Average time per file: {total_time/len(self.stats['processing_times']):.2f}s")
            self.logger.info(f"  Average time per entry: {total_time/self.stats['total_entries']:.3f}s" if self.stats['total_entries'] > 0 else "")
            
        # Log file info removed - handled by user's logging config
        self.logger.info(f"{'='*80}\n")
        
    def process_directory(self, limit: Optional[int] = None):
        files_to_process = [p for p in self.input_dir.glob('*.*') if p.suffix.lower() in ['.json', '.yaml', '.yml']]
        if limit: files_to_process = files_to_process[:limit]
        
        self.stats['total_files'] = len(files_to_process)
        self.logger.info(f"\nStarting batch processing of {len(files_to_process)} files...\n")
        
        for file_path in files_to_process:
            relevant, filtered = self.filter_dataset(file_path)
            self.save_data(relevant, file_path, '_filtered')
            self.save_data(filtered, file_path, '_removed')
            
        self.print_final_statistics()
        
        # Final benchmark summary
        self.logger.info("\nFINAL BENCHMARK SUMMARY:")
        self.benchmark.log_benchmark_stats(force=True)

def main():
    parser = argparse.ArgumentParser(description="Filter and enhance data using a local MLX model with a stable, two-pass sequential method.")
    parser.add_argument("--input-dir", default="raw_data", help="Directory containing raw data files.")
    parser.add_argument("--output-dir", default="filtered_data", help="Directory to save the filtered data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for all).")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-14B-4bit-DWQ-053125", help="The MLX-compatible model to use. Smaller is faster.")
    parser.add_argument("--no-enhancement", action="store_true", help="Skip LLM enhancement and use rule-based filtering only.")
    parser.add_argument("--log-file", help="Log file path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging for CLI usage
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = []
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    else:
        # Default log file
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'filter_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger for CLI
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        try:
            print("PyYAML not found. Attempting to install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            print("PyYAML installed successfully.")
        except Exception as e:
            logging.error(f"Failed to install PyYAML. Please install it manually using 'pip install pyyaml'. Error: {e}")
            sys.exit(1)

    data_filter = CyberDataFilter(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        model_path=args.model if not args.no_enhancement else None,
        no_enhancement=args.no_enhancement
    )
    
    data_filter.process_directory(limit=args.limit if args.limit > 0 else None)

if __name__ == "__main__":
    main()