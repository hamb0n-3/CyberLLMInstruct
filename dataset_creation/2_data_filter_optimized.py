#!/usr/bin/env python3
"""
Optimized Data Filter - Efficient streaming and rule-based filtering
"""

import json
import logging
import yaml
import ijson  # For streaming JSON
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from datetime import datetime
import re
import argparse
import time
from lxml import etree  # More efficient XML parsing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import the MLX client
from mlx_client import MLXClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedCyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, config_path: str = "pipeline_config.yaml"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pipeline configuration
        self.config = self._load_config(config_path)
        
        # Initialize MLX client only when needed
        self.mlx_client = None
        self._mlx_initialized = False
        
        # Enhanced keyword patterns for better rule-based filtering
        self.cybersecurity_patterns = {
            'definite_cyber': re.compile(
                r'\b(CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+|exploit|malware|ransomware|'
                r'vulnerability|zero-day|0day|backdoor|rootkit|trojan|phishing|'
                r'SQL injection|XSS|CSRF|RCE|remote code execution)\b', 
                re.IGNORECASE
            ),
            'likely_cyber': re.compile(
                r'\b(security|attack|threat|breach|patch|authentication|encryption|'
                r'firewall|penetration test|pentest|incident response|SIEM|IDS|IPS)\b',
                re.IGNORECASE
            ),
            'technical': re.compile(
                r'\b(buffer overflow|heap overflow|stack overflow|memory corruption|'
                r'privilege escalation|denial of service|DoS|DDoS|cryptography)\b',
                re.IGNORECASE
            )
        }
        
        self.exclusion_patterns = re.compile(
            r'\b(test|sample|example|dummy|todo|lorem ipsum|xxx|placeholder)\b',
            re.IGNORECASE
        )
        
        self.min_content_length = 50
        self.batch_size = 32  # Larger batch size for efficiency
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {
                "mlx_server": {"base_url": "http://localhost:8080", "use_server": True},
                "mlx_model": {"path": "mlx-community/Phi-3-mini-4k-instruct-4bit"},
                "batching": {"batch_size": 32, "batch_timeout_ms": 100}
            }
    
    def _init_mlx_client(self):
        """Lazy initialization of MLX client"""
        if not self._mlx_initialized:
            self.mlx_client = MLXClient(
                server_url=self.config["mlx_server"]["base_url"],
                model_path=self.config["mlx_model"]["path"],
                batch_size=self.batch_size,
                use_server=self.config["mlx_server"]["use_server"]
            )
            self._mlx_initialized = True
            logger.info(f"MLX client initialized")
    
    def is_relevant_rule_based(self, text: str) -> Tuple[bool, int]:
        """
        Enhanced rule-based relevance check with confidence scoring.
        Returns (is_relevant, confidence_score)
        """
        if not text or len(text) < self.min_content_length:
            return False, 0
        
        if self.exclusion_patterns.search(text):
            return False, 0
        
        # Calculate confidence score
        score = 0
        
        # Definite cyber terms = high confidence
        definite_matches = len(self.cybersecurity_patterns['definite_cyber'].findall(text))
        score += definite_matches * 3
        
        # Likely cyber terms = medium confidence
        likely_matches = len(self.cybersecurity_patterns['likely_cyber'].findall(text))
        score += likely_matches * 2
        
        # Technical terms = low confidence
        technical_matches = len(self.cybersecurity_patterns['technical'].findall(text))
        score += technical_matches * 1
        
        # High confidence = skip LLM check
        if score >= 6:
            return True, 100
        # Medium confidence = needs LLM verification
        elif score >= 3:
            return True, 50
        # Low confidence = likely not relevant
        else:
            return False, score * 10
    
    def stream_json_file(self, file_path: Path) -> Iterator[Dict]:
        """Stream large JSON files instead of loading into memory"""
        try:
            with open(file_path, 'rb') as f:
                # Try to detect JSON structure
                parser = ijson.items(f, 'item')
                for item in parser:
                    yield item
        except:
            # Fallback to regular loading for small files
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    # Check for common data keys
                    for key in ['data', 'entries', 'items', 'vulnerabilities']:
                        if key in data and isinstance(data[key], list):
                            yield from data[key]
                            return
                    yield data
    
    def parse_xml_streaming(self, file_path: Path) -> Iterator[Dict]:
        """Stream XML parsing for memory efficiency"""
        namespace = {'capec': 'http://capec.mitre.org/capec-3'}
        
        try:
            for event, elem in etree.iterparse(file_path, events=('start', 'end')):
                if event == 'end' and elem.tag.endswith('Attack_Pattern'):
                    pattern = {
                        'id': f"CAPEC-{elem.get('ID')}",
                        'name': elem.get('Name'),
                        'description': ' '.join(elem.itertext()).strip()
                    }
                    yield pattern
                    elem.clear()  # Free memory
                    elem.getparent().remove(elem)
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
    
    def process_batch_with_llm(self, candidates: List[Tuple[Dict, str, int]]) -> List[Dict]:
        """Process only medium-confidence entries with LLM"""
        # Filter only entries that need LLM verification
        needs_llm = [(entry, text) for entry, text, conf in candidates if conf < 80]
        
        if not needs_llm:
            # All entries are high confidence, return them all
            return [entry for entry, _, _ in candidates]
        
        # Initialize MLX client only when needed
        self._init_mlx_client()
        
        # Create prompts for batch processing
        prompts = []
        for entry, text in needs_llm:
            prompt = f'Is this cybersecurity-related? Answer YES/NO only:\n"{text[:300]}"'
            prompts.append(prompt)
        
        try:
            responses = self.mlx_client.generate_batch(
                prompts,
                max_tokens=5,
                temperature=0.1
            )
            
            # Combine results
            relevant_entries = []
            llm_idx = 0
            
            for entry, text, conf in candidates:
                if conf >= 80:
                    # High confidence, automatically include
                    relevant_entries.append(entry)
                else:
                    # Check LLM response
                    if llm_idx < len(responses) and "YES" in responses[llm_idx].upper():
                        relevant_entries.append(entry)
                    llm_idx += 1
            
            return relevant_entries
            
        except Exception as e:
            logger.error(f"LLM batch processing failed: {e}")
            # On error, include all medium-confidence entries
            return [entry for entry, _, conf in candidates if conf >= 50]
    
    def process_file_streaming(self, file_path: Path) -> Tuple[List[Dict], int]:
        """Process file with streaming to handle large files efficiently"""
        relevant_entries = []
        total_processed = 0
        batch_candidates = []
        
        # Determine file type and stream accordingly
        if file_path.suffix.lower() == '.xml':
            data_stream = self.parse_xml_streaming(file_path)
        else:
            data_stream = self.stream_json_file(file_path)
        
        # Process entries in batches
        for entry in data_stream:
            total_processed += 1
            
            if not isinstance(entry, dict):
                continue
            
            # Extract text content
            text_parts = []
            for field in ['title', 'summary', 'description', 'name', 'instruction', 'response']:
                if field in entry:
                    text_parts.append(str(entry[field]))
            
            text_content = ' '.join(text_parts)
            
            # Rule-based filtering with confidence
            is_relevant, confidence = self.is_relevant_rule_based(text_content)
            
            if is_relevant:
                if confidence >= 80:
                    # High confidence - add directly
                    relevant_entries.append(entry)
                else:
                    # Medium confidence - batch for LLM
                    batch_candidates.append((entry, text_content, confidence))
                    
                    # Process batch when full
                    if len(batch_candidates) >= self.batch_size:
                        batch_results = self.process_batch_with_llm(batch_candidates)
                        relevant_entries.extend(batch_results)
                        batch_candidates = []
        
        # Process remaining candidates
        if batch_candidates:
            batch_results = self.process_batch_with_llm(batch_candidates)
            relevant_entries.extend(batch_results)
        
        return relevant_entries, total_processed
    
    def save_results_streaming(self, entries: List[Dict], output_path: Path):
        """Save results efficiently"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[')
            for i, entry in enumerate(entries):
                if i > 0:
                    f.write(',')
                f.write('\n  ')
                json.dump(entry, f, ensure_ascii=False)
            f.write('\n]')
    
    def process_directory(self, limit: Optional[int] = None):
        """Process directory with parallel file processing"""
        files_to_process = list(self.input_dir.glob('*.*'))
        files_to_process = [f for f in files_to_process 
                           if f.suffix.lower() in ['.json', '.yaml', '.yml', '.xml']]
        
        if limit:
            files_to_process = files_to_process[:limit]
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files with progress bar
        total_relevant = 0
        total_processed = 0
        
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                relevant, processed = self.process_file_streaming(file_path)
                
                if relevant:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_file = self.output_dir / f"{file_path.stem}_filtered_{timestamp}.json"
                    self.save_results_streaming(relevant, output_file)
                    
                    logger.info(f"{file_path.name}: {len(relevant)}/{processed} entries retained")
                    total_relevant += len(relevant)
                    total_processed += processed
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total: {total_relevant}/{total_processed} entries retained")
        
        # Close MLX client if initialized
        if self._mlx_initialized:
            self.mlx_client.close()


def main():
    parser = argparse.ArgumentParser(description="Optimized data filtering with streaming and smart batching")
    parser.add_argument("--input-dir", default="raw_data", help="Input directory")
    parser.add_argument("--output-dir", default="filtered_data", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit files to process")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    filter = OptimizedCyberDataFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    filter.process_directory(limit=args.limit if args.limit > 0 else None)


if __name__ == "__main__":
    main()