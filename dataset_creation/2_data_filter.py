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

# We are now loading the model directly in this script.
from mlx_lm import load, generate
import mlx.core as mx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, model_path: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load the model directly into the class. This happens only once.
        logger.info(f"Loading model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        logger.info("Model loaded successfully.")
        mx.eval(self.model.parameters())
        def _gen(prompt:str, **kw):
            return generate(self.model,self.tokenizer, prompt, **kw)
        self.fast_generate = mx.compile(_gen,shapeless=True)

        self.trusted_sources = ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft']
        self.cybersecurity_keywords = {'high_relevance': {'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security', 'attack', 'threat', 'breach', 'cve-', 'patch', 'authentication', 'authorization', 'encryption', 'cryptography', 'backdoor', 'botnet', 'phishing', 'injection', 'zero-day', '0day', 'penetration', 'pentest', 'firewall', 'malicious'}, 'medium_relevance': {'network', 'system', 'software', 'hardware', 'protocol', 'server', 'client', 'database', 'web', 'application', 'code', 'programming', 'access', 'control', 'monitoring', 'detection', 'response', 'incident'}}
        self.exclusion_patterns = {'generic_terms': r'\b(test|sample|example|dummy|todo)\b', 'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b', 'empty_content': r'^\s*$'}
        self.min_content_length = 20
        self.min_keyword_matches = 2

    def check_relevance(self, text_content: str) -> bool:
        """Pass 1: A simple, fast check for relevance."""
        prompt = f'[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "{text_content[:500]}"[/INST]'
        try:
            response = self.fast_generate(prompt, max_tokens=5, verbose=False,kv_bits=8,kv_group_size=32)
            return "YES" in response.upper()
        except Exception as e:
            logger.error(f"Relevance check failed for an item: {e}")
            return False

    def get_enhancement(self, text_content: str) -> Optional[Dict]:
        """Pass 2: A slow, detailed enhancement for confirmed relevant items."""
        prompt = f"""[INST]
Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".
Text to analyze:
"{text_content}"
[/INST]```json
"""
        try:
            response = self.fast_generate(prompt, max_tokens=1024, verbose=False,kv_bits=8,kv_group_size=32)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Enhancement failed for an item: {e}")
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
        all_entries = self.load_data(input_file)
        if not all_entries: return [], []
        logger.info(f"Processing {len(all_entries)} entries from {input_file.name}...")
        
        relevant_entries, filtered_out_entries = [], []
        
        relevance_candidates = []
        for entry in all_entries:
            if not isinstance(entry, dict): continue
            text_content = self._get_text_from_entry(entry)
            if not text_content:
                entry['filtered_reason'] = "Empty content"; filtered_out_entries.append(entry)
            elif self.is_relevant_rule_based(text_content):
                relevance_candidates.append((entry, text_content))
            else:
                entry['filtered_reason'] = "Failed rule-based pre-filter"; filtered_out_entries.append(entry)
        
        logger.info(f"Rule-based pre-filter complete. {len(relevance_candidates)} entries passed for LLM processing.")

        # PASS 1: Sequential Relevance Check
        tasks_for_enhancement = []
        for original_entry, text_content in tqdm(relevance_candidates, desc="Pass 1/2: Relevance Check"):
            if self.check_relevance(text_content):
                tasks_for_enhancement.append((original_entry, text_content))
            else:
                original_entry['filtered_reason'] = "LLM determined not relevant"
                filtered_out_entries.append(original_entry)

        logger.info(f"Relevance check complete. {len(tasks_for_enhancement)} entries passed for detailed enhancement.")

        # PASS 2: Sequential Enhancement
        for original_entry, text_content in tqdm(tasks_for_enhancement, desc="Pass 2/2: Enhancement"):
            llm_result = self.get_enhancement(text_content)
            if llm_result:
                original_entry.update(llm_result)
                original_entry['is_relevant'] = True
                relevant_entries.append(original_entry)
            else:
                original_entry['filtered_reason'] = "LLM enhancement failed"
                filtered_out_entries.append(original_entry)
                        
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
        for file_path in files_to_process:
            relevant, filtered = self.filter_dataset(file_path)
            self.save_data(relevant, file_path, '_filtered')
            self.save_data(filtered, file_path, '_removed')
            logger.info(f"--- Finished processing {file_path.name}: {len(relevant)} retained, {len(filtered)} removed ---")

def main():
    parser = argparse.ArgumentParser(description="Filter and enhance data using a local MLX model with a stable, two-pass sequential method.")
    parser.add_argument("--input-dir", default="raw_data", help="Directory containing raw data files.")
    parser.add_argument("--output-dir", default="filtered_data", help="Directory to save the filtered data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for all).")
    parser.add_argument("--model", type=str, default="mlx-community/Phi-3-mini-4k-instruct-4bit", help="The MLX-compatible model to use. Smaller is faster.")
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

    data_filter = CyberDataFilter(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        model_path=args.model
    )
    
    data_filter.process_directory(limit=args.limit if args.limit > 0 else None)

if __name__ == "__main__":
    main()