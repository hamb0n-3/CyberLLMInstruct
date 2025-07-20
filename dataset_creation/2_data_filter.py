#!/usr/bin/env python3
"""
Simplified Data Filter for Cybersecurity Dataset Creation

This refactored version removes batch processing and speculative decoding,
focusing on efficient single-item processing which benchmarks showed to be faster.
"""

import json
import logging
import yaml
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

# Import MLX components
from mlx_lm import load, generate
import mlx.core as mx

# Setup logging with both console and file handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

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
class ProcessingMetrics:
    """Track processing metrics for the filtering process."""
    # Model information
    model_name: str = ""
    model_type: str = ""
    
    # System information
    timestamp_start: str = ""
    timestamp_end: str = ""
    platform_info: str = platform.platform()
    python_version: str = sys.version.split()[0]
    
    # Performance tracking
    total_tokens_generated: int = 0
    total_generation_time: float = 0.0
    total_items_processed: int = 0
    total_items_retained: int = 0
    
    # Per-file tracking
    file_processing_times: Dict[str, float] = field(default_factory=dict)
    file_items_processed: Dict[str, int] = field(default_factory=dict)
    file_items_retained: Dict[str, int] = field(default_factory=dict)
    
    def add_generation_metrics(self, tokens: int, time_taken: float):
        self.total_tokens_generated += tokens
        self.total_generation_time += time_taken
    
    def get_tokens_per_second(self) -> float:
        if self.total_generation_time > 0:
            return self.total_tokens_generated / self.total_generation_time
        return 0.0
    
    def get_overall_retention_rate(self) -> float:
        if self.total_items_processed > 0:
            return self.total_items_retained / self.total_items_processed
        return 0.0
    
    def save_summary(self, output_path: Path):
        """Save processing summary to JSON file."""
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": self.model_name,
                "type": self.model_type
            },
            "system": {
                "platform": self.platform_info,
                "python_version": self.python_version,
                "start_time": self.timestamp_start,
                "end_time": self.timestamp_end
            },
            "performance": {
                "total_tokens_generated": self.total_tokens_generated,
                "tokens_per_second": self.get_tokens_per_second(),
                "total_generation_time": self.total_generation_time
            },
            "results": {
                "total_items_processed": self.total_items_processed,
                "total_items_retained": self.total_items_retained,
                "overall_retention_rate": self.get_overall_retention_rate()
            },
            "file_details": {
                filename: {
                    "processing_time": self.file_processing_times.get(filename, 0),
                    "items_processed": self.file_items_processed.get(filename, 0),
                    "items_retained": self.file_items_retained.get(filename, 0),
                    "retention_rate": self.file_items_retained.get(filename, 0) / self.file_items_processed.get(filename, 1)
                }
                for filename in self.file_processing_times
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary saved to: {output_path}")


class CyberDataFilter:
    """Simplified data filter focusing on single-item processing."""
    
    def __init__(self, input_dir: str, output_dir: str, model_path: str, verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize metrics
        self.metrics = ProcessingMetrics(
            model_name=model_path,
            timestamp_start=datetime.now().isoformat()
        )
        
        # Load the model
        logger.info(f"Loading model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        logger.info("Model loaded successfully.")
        mx.eval(self.model.parameters())
        
        # Detect model type for prompt formatting
        self.model_type = self._detect_model_type(model_path)
        self.metrics.model_type = self.model_type
        logger.info(f"Detected model type: {self.model_type}")
        
        # Cybersecurity filtering configuration
        self.cybersecurity_keywords = {
            'high_relevance': {
                'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security',
                'attack', 'threat', 'breach', 'cve-', 'patch', 'authentication',
                'authorization', 'encryption', 'cryptography', 'backdoor', 'botnet',
                'phishing', 'injection', 'zero-day', '0day', 'penetration', 'pentest',
                'firewall', 'malicious', 'redteam', 'actor', 'TTP'
            },
            'medium_relevance': {
                'network', 'system', 'software', 'hardware', 'protocol', 'server',
                'client', 'database', 'web', 'application', 'code', 'programming',
                'access', 'control', 'monitoring', 'detection', 'response', 'incident', 'software','version','tech'
            }
        }
        
        self.exclusion_patterns = {
            'generic_terms': r'\b(test|sample|example|dummy|todo)\b',
            'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b',
            'empty_content': r'^\\s*$'
        }
        
        self.min_content_length = 20
        self.min_keyword_matches = 1
    
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
    
    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a single response for a prompt."""
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
        
        # Update metrics
        token_count = len(self.tokenizer.encode(response))
        self.metrics.add_generation_metrics(token_count, gen_time)
        
        return response
    
    def check_relevance(self, text: str) -> bool:
        """Check if text is relevant to cybersecurity using LLM."""
        prompt = self.format_prompt(text, task="relevance")
        
        try:
            response = self.generate_response(prompt, max_tokens=10)
            response_clean = response.strip().upper()
            
            # Check for YES/NO in response
            has_yes = "YES" in response_clean
            has_no = "NO" in response_clean
            
            if has_yes and not has_no:
                return True
            elif has_no and not has_yes:
                return False
            else:
                # Ambiguous response, default to False
                if self.verbose:
                    logger.warning(f"Ambiguous relevance response: {response_clean[:50]}...")
                return False
                
        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            return False
    
    def get_enhancement(self, text: str) -> Optional[Dict]:
        """Get LLM enhancement for the text."""
        prompt = self.format_prompt(text, task="enhancement")
        
        try:
            response = self.generate_response(prompt, max_tokens=1024)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                json_str = json_str.replace('\n', ' ').replace('\\n', ' ')
                enhancement = json.loads(json_str)
                return enhancement
            else:
                if self.verbose:
                    logger.warning(f"No JSON found in enhancement response: {response[:100]}...")
                return None
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"Enhancement failed: {e}")
            return None
    
    def is_relevant_rule_based(self, text: str) -> bool:
        """Check relevance using rule-based approach."""
        if not isinstance(text, str) or len(text) < self.min_content_length:
            return False
        
        if any(re.search(p, text, re.IGNORECASE) for p in self.exclusion_patterns.values()):
            return False
        
        text_lower = text.lower()
        score = sum(2 for kw in self.cybersecurity_keywords['high_relevance'] if kw in text_lower)
        score += sum(1 for kw in self.cybersecurity_keywords['medium_relevance'] if kw in text_lower)
        
        return score >= self.min_keyword_matches
    
    def _get_text_from_entry(self, entry: Dict) -> str:
        """Extract text content from an entry."""
        primary_fields = ['title', 'summary', 'description', 'name', 'instruction', 'response']
        text_parts = [str(entry.get(field, '')) for field in primary_fields]
        other_parts = [str(value) for key, value in entry.items() 
                      if key not in primary_fields and isinstance(value, str)]
        return " ".join(text_parts + other_parts).strip()
    
    def load_data(self, file_path: Path) -> list:
        """Load data from JSON or YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    return []
            
            # Handle CAPEC files specially
            if file_path.name.startswith('capec_data'):
                return self.process_capec_file(data)
            
            # Extract list from common data structures
            if isinstance(data, dict):
                for key in ['data', 'entries', 'papers', 'objects', 'vulnerabilities', 'value', 'ctftime_events']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def process_capec_file(self, data: Dict) -> List[Dict]:
        """Process CAPEC XML data."""
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
    
    def filter_dataset(self, input_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """Filter a single dataset file."""
        file_start_time = time.time()
        
        all_entries = self.load_data(input_file)
        if not all_entries:
            return [], []
        
        logger.info(f"Processing {len(all_entries)} entries from {input_file.name}...")
        
        relevant_entries = []
        filtered_out_entries = []
        
        # Process each entry
        for i, entry in enumerate(tqdm(all_entries, desc=f"Processing {input_file.name}")):
            if not isinstance(entry, dict):
                continue
            
            # Extract text content
            text_content = self._get_text_from_entry(entry)
            
            if not text_content:
                entry['filtered_reason'] = "Empty content"
                filtered_out_entries.append(entry)
                continue
            
            # Rule-based pre-filter
            if not self.is_relevant_rule_based(text_content):
                entry['filtered_reason'] = "Failed rule-based pre-filter"
                filtered_out_entries.append(entry)
                continue
            
            # LLM relevance check
            if not self.check_relevance(text_content):
                entry['filtered_reason'] = "LLM determined not relevant"
                filtered_out_entries.append(entry)
                continue
            
            # LLM enhancement
            enhancement = self.get_enhancement(text_content)
            if enhancement:
                entry.update(enhancement)
                entry['is_relevant'] = True
                relevant_entries.append(entry)
            else:
                entry['filtered_reason'] = "LLM enhancement failed"
                filtered_out_entries.append(entry)
        
        # Update metrics
        processing_time = time.time() - file_start_time
        self.metrics.file_processing_times[input_file.name] = processing_time
        self.metrics.file_items_processed[input_file.name] = len(all_entries)
        self.metrics.file_items_retained[input_file.name] = len(relevant_entries)
        self.metrics.total_items_processed += len(all_entries)
        self.metrics.total_items_retained += len(relevant_entries)
        
        # Log results
        logger.info(f"Completed {input_file.name}:")
        logger.info(f"  - Processing time: {processing_time:.1f}s")
        logger.info(f"  - Items processed: {len(all_entries)}")
        logger.info(f"  - Items retained: {len(relevant_entries)} ({len(relevant_entries)/len(all_entries)*100:.1f}%)")
        logger.info(f"  - Tokens/second: {self.metrics.get_tokens_per_second():.1f}")
        
        return relevant_entries, filtered_out_entries
    
    def save_data(self, data: List[Dict], original_file: Path, suffix: str = ''):
        """Save filtered data to output directory."""
        if not data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"{original_file.stem}{suffix}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} entries to {output_file}")
    
    def process_directory(self, limit: Optional[int] = None):
        """Process all files in the input directory."""
        files_to_process = [p for p in self.input_dir.glob('*.*') 
                           if p.suffix.lower() in ['.json', '.yaml', '.yml']]
        
        if limit:
            files_to_process = files_to_process[:limit]
        
        logger.info(f"Found {len(files_to_process)} files to process.")
        overall_start = time.time()
        
        try:
            for file_path in files_to_process:
                relevant, filtered = self.filter_dataset(file_path)
                self.save_data(relevant, file_path, '_filtered')
                self.save_data(filtered, file_path, '_removed')
                logger.info(f"--- Finished {file_path.name}: {len(relevant)} retained, {len(filtered)} removed ---\n")
                
        except KeyboardInterrupt:
            logger.info("\n\n=== INTERRUPTED BY USER (Ctrl+C) ===")
            logger.info("Saving current progress...")
        
        # Final summary
        total_time = time.time() - overall_start
        self.metrics.timestamp_end = datetime.now().isoformat()
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Total processing time: {total_time:.1f}s")
        logger.info(f"Files processed: {len(self.metrics.file_processing_times)}/{len(files_to_process)}")
        logger.info(f"Total items: {self.metrics.total_items_processed}")
        logger.info(f"Items retained: {self.metrics.total_items_retained} ({self.metrics.get_overall_retention_rate():.1%})")
        logger.info(f"Overall tokens/second: {self.metrics.get_tokens_per_second():.1f}")
        
        # Save summary
        summary_file = logs_dir / f"summary_{timestamp}.json"
        self.metrics.save_summary(summary_file)


def main():
    parser = argparse.ArgumentParser(
        description="Filter and enhance cybersecurity data using MLX models (simplified version)"
    )
    parser.add_argument("--input-dir", default="raw_data", 
                       help="Directory containing raw data files")
    parser.add_argument("--output-dir", default="filtered_data", 
                       help="Directory to save filtered data")
    parser.add_argument("--model", type=str, 
                       default="mlx-community/Qwen3-14B-4bit-DWQ-053125",
                       help="MLX model to use")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of files to process (0 for all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not found. Please install with: pip install pyyaml")
        sys.exit(1)
    
    # Create and run filter
    data_filter = CyberDataFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        verbose=args.verbose
    )
    
    try:
        data_filter.process_directory(limit=args.limit if args.limit > 0 else None)
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()