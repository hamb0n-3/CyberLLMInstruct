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

# Import batching and speculative decoding utilities
try:
    from mlx_batch_utils import BatchedMLXGenerator, DynamicBatcher
    from mlx_speculative import SpeculativeDecoder, SpeculativeConfig
    from performance_logger import get_performance_logger
    BATCHING_AVAILABLE = True
except ImportError:
    BATCHING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDataFilter:
    def __init__(self, input_dir: str, output_dir: str, model_path: str, 
                 use_batching: bool = True, batch_size: int = 32, 
                 batch_timeout: float = 0.1, use_speculative: bool = False,
                 draft_model_path: str = "mlx-community/gemma-3-1b-it-bf16"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_batching = use_batching and BATCHING_AVAILABLE
        self.use_speculative = use_speculative and BATCHING_AVAILABLE

        # Load the model directly into the class. This happens only once.
        logger.info(f"Loading model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        logger.info("Model loaded successfully.")
        mx.eval(self.model.parameters())
        
        # Setup generation methods based on configuration
        if self.use_speculative:
            logger.info(f"Initializing speculative decoding with draft model: {draft_model_path}")
            self.speculative_decoder = SpeculativeDecoder(
                target_model=self.model,
                target_tokenizer=self.tokenizer,
                config=SpeculativeConfig(
                    draft_model_path=draft_model_path,
                    target_model_path=model_path,
                    temperature=0.0
                )
            )
            self.fast_generate = self._generate_speculative
        elif self.use_batching:
            logger.info(f"Initializing batched generator with batch_size={batch_size}")
            self.batch_generator = BatchedMLXGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=batch_size,
                batch_timeout=batch_timeout,
                enable_kv_cache=True,
                kv_bits=8,
                kv_group_size=32
            )
            self.fast_generate = self._generate_batched
        else:
            # Original non-batched generation
            def _gen(prompt:str, **kw):
                return generate(self.model,self.tokenizer, prompt, **kw)
            self.fast_generate = mx.compile(_gen,shapeless=True)

        self.trusted_sources = ['capec', 'mitre', 'nvd', 'opencve', 'ubuntu', 'redhat', 'microsoft']
        self.cybersecurity_keywords = {'high_relevance': {'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security', 'attack', 'threat', 'breach', 'cve-', 'patch', 'authentication', 'authorization', 'encryption', 'cryptography', 'backdoor', 'botnet', 'phishing', 'injection', 'zero-day', '0day', 'penetration', 'pentest', 'firewall', 'malicious'}, 'medium_relevance': {'network', 'system', 'software', 'hardware', 'protocol', 'server', 'client', 'database', 'web', 'application', 'code', 'programming', 'access', 'control', 'monitoring', 'detection', 'response', 'incident'}}
        self.exclusion_patterns = {'generic_terms': r'\b(test|sample|example|dummy|todo)\b', 'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b', 'empty_content': r'^\s*$'}
        self.min_content_length = 20
        self.min_keyword_matches = 2
    
    def _generate_batched(self, prompt: str, **kwargs) -> str:
        """Generate using batched processing"""
        return self.batch_generator.generate(
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 256),
            temperature=kwargs.get('temp', 0.0),
            top_p=kwargs.get('top_p', 1.0)
        )
    
    def _generate_speculative(self, prompt: str, **kwargs) -> str:
        """Generate using speculative decoding"""
        return self.speculative_decoder.generate(
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 256),
            temperature=kwargs.get('temp', 0.0),
            top_p=kwargs.get('top_p', 1.0),
            verbose=kwargs.get('verbose', False)
        )

    def check_relevance(self, text_content: str) -> bool:
        """Pass 1: A simple, fast check for relevance."""
        prompt = f'[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "{text_content[:500]}"[/INST]'
        try:
            response = self.fast_generate(prompt, max_tokens=5, verbose=False,kv_bits=8,kv_group_size=32)
            return "YES" in response.upper()
        except Exception as e:
            logger.error(f"Relevance check failed for an item: {e}")
            return False
    
    def batch_check_relevance(self, text_contents: List[str]) -> List[bool]:
        """Batch process relevance checks for multiple texts"""
        prompts = [
            f'[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.\nText: "{text[:500]}"[/INST]'
            for text in text_contents
        ]
        
        results = []
        
        # Use actual batch processing if available
        if self.use_batching and hasattr(self, 'batch_generator'):
            # Process all prompts in parallel
            try:
                # For now, we'll process them one by one through the batch generator
                # which handles queueing and parallel processing internally
                for prompt in prompts:
                    response = self.batch_generator.generate(
                        prompt=prompt,
                        max_tokens=5,
                        temperature=0.0
                    )
                    results.append("YES" in response.upper())
            except Exception as e:
                logger.error(f"Batch relevance check failed: {e}")
                # Fallback to individual checks
                results = [False] * len(prompts)
        else:
            # Sequential processing
            for prompt in prompts:
                try:
                    response = self.fast_generate(prompt, max_tokens=5, verbose=False, kv_bits=8, kv_group_size=32)
                    results.append("YES" in response.upper())
                except Exception as e:
                    logger.error(f"Relevance check failed: {e}")
                    results.append(False)
        
        return results

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
        """Extract text content from various data formats"""
        text_parts = []
        
        # Handle CVE format (nested structure)
        if 'cve' in entry and isinstance(entry['cve'], dict):
            cve_data = entry['cve']
            # Add CVE ID
            if 'id' in cve_data:
                text_parts.append(str(cve_data['id']))
            
            # Extract descriptions
            for desc in cve_data.get('descriptions', []):
                if isinstance(desc, dict) and 'value' in desc:
                    text_parts.append(desc['value'])
            
            # Add other relevant CVE fields
            if 'sourceIdentifier' in cve_data:
                text_parts.append(str(cve_data['sourceIdentifier']))
        
        # Handle standard fields for other formats
        primary_fields = ['title', 'summary', 'description', 'name', 'instruction', 'response', 
                         'id', 'content', 'text', 'message', 'details']
        for field in primary_fields:
            if field in entry and entry[field]:
                text_parts.append(str(entry[field]))
        
        # For entries that might have nested descriptions
        if 'descriptions' in entry and isinstance(entry['descriptions'], list):
            for desc in entry['descriptions']:
                if isinstance(desc, dict) and 'value' in desc:
                    text_parts.append(desc['value'])
                elif isinstance(desc, str):
                    text_parts.append(desc)
        
        # Add string values from other fields
        for key, value in entry.items():
            if key not in primary_fields and key != 'cve' and isinstance(value, str) and len(value) > 10:
                text_parts.append(value)
        
        return " ".join(text_parts).strip()

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

        # PASS 1: Relevance Check (with batching if enabled)
        tasks_for_enhancement = []
        
        if self.use_batching and hasattr(self, 'batch_generator'):
            # Real batch processing for relevance check
            logger.info("Using batched processing for relevance check")
            batch_size = 32
            
            with tqdm(total=len(relevance_candidates), desc="Pass 1/2: Relevance Check (Batched)") as pbar:
                for i in range(0, len(relevance_candidates), batch_size):
                    batch = relevance_candidates[i:i+batch_size]
                    
                    # Extract texts for batch processing
                    batch_texts = [text for _, text in batch]
                    
                    # Process entire batch at once
                    batch_results = self.batch_check_relevance(batch_texts)
                    
                    # Process results
                    for (original_entry, text_content), is_relevant in zip(batch, batch_results):
                        if is_relevant:
                            tasks_for_enhancement.append((original_entry, text_content))
                        else:
                            original_entry['filtered_reason'] = "LLM determined not relevant"
                            filtered_out_entries.append(original_entry)
                    
                    pbar.update(len(batch))
        else:
            # Original sequential processing
            for original_entry, text_content in tqdm(relevance_candidates, desc="Pass 1/2: Relevance Check"):
                if self.check_relevance(text_content):
                    tasks_for_enhancement.append((original_entry, text_content))
                else:
                    original_entry['filtered_reason'] = "LLM determined not relevant"
                    filtered_out_entries.append(original_entry)

        logger.info(f"Relevance check complete. {len(tasks_for_enhancement)} entries passed for detailed enhancement.")

        # PASS 2: Enhancement (with batching if enabled)
        if self.use_batching and hasattr(self, 'batch_generator'):
            # Batch processing for enhancement
            logger.info("Using batched processing for enhancement")
            batch_size = 16  # Smaller batch for longer generations
            for i in tqdm(range(0, len(tasks_for_enhancement), batch_size), desc="Pass 2/2: Enhancement (Batched)"):
                batch = tasks_for_enhancement[i:i+batch_size]
                
                # Process each item in the batch through the batch generator
                # The batch generator handles internal queueing and parallel processing
                for original_entry, text_content in batch:
                    llm_result = self.get_enhancement(text_content)
                    if llm_result:
                        original_entry.update(llm_result)
                        original_entry['is_relevant'] = True
                        relevant_entries.append(original_entry)
                    else:
                        original_entry['filtered_reason'] = "LLM enhancement failed"
                        filtered_out_entries.append(original_entry)
        else:
            # Original sequential processing
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
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'batch_generator'):
            self.batch_generator.shutdown()
            logger.info("Batch generator shutdown complete")

def main():
    parser = argparse.ArgumentParser(description="Filter and enhance data using a local MLX model with batching and speculative decoding support.")
    parser.add_argument("--input-dir", default="raw_data", help="Directory containing raw data files.")
    parser.add_argument("--output-dir", default="filtered_data", help="Directory to save the filtered data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for all).")
    parser.add_argument("--model", type=str, default="mlx-community/Phi-3-mini-4k-instruct-4bit", help="The MLX-compatible model to use. Smaller is faster.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Batch timeout in seconds (default: 0.1)")
    parser.add_argument("--no-batching", action="store_true", help="Disable batched processing")
    parser.add_argument("--enable-speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--draft-model", type=str, default="mlx-community/gemma-3-1b-it-bf16", help="Draft model for speculative decoding")
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
            use_batching=not args.no_batching,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            use_speculative=args.enable_speculative,
            draft_model_path=args.draft_model
        )
        
        data_filter.process_directory(limit=args.limit if args.limit > 0 else None)
    finally:
        if 'data_filter' in locals():
            data_filter.cleanup()

if __name__ == "__main__":
    main()