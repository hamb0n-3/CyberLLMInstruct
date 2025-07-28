#!/usr/bin/env python3

import os
# Disable tokenizers parallelism to avoid multiprocessing conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CyberDataStructurer:
    def __init__(self, input_dir: str, output_dir: str, llm_model: str, workers: int, disable_llm: bool,
                 temperature: float = 0.7, top_p: float = 1.0, top_k: int = 0,
                 min_p: float = 0.0, repetition_penalty: float = 1.0):
        """Initialize the data structurer."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = workers
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
            logger.warning("LLM disabled or mlx-lm not found. Structuring will not be possible.")

        self.file_handlers = {
            'ctf_data': self._prepare_ctf_tasks,
            'arxiv_papers': self._prepare_arxiv_tasks,
            'ubuntu_security': self._prepare_ubuntu_tasks,
            'microsoft_security': self._prepare_microsoft_tasks,
            'capec_data': self._prepare_capec_tasks,
            'opencve_data': self._prepare_opencve_tasks,
            'mitre_attack': self._prepare_mitre_attack_tasks,
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the local MLX model with retry logic."""
        if not self.llm_available: 
            return None
            
        max_attempts = 3
        token_limits = [512, 1024, 1500]
        
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

    def _load_data(self, file_path: Path) -> Optional[List[Dict]]:
        """Load and normalize data from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list): return data
            if isinstance(data, dict):
                for key in ['data', 'entries', 'papers', 'objects', 'vulnerabilities', 'value', 'ctftime_events']:
                    if key in data and isinstance(data[key], list): return data[key]
                return [data]
            return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    # --- Task Preparation Methods ---
    # These methods create the prompts but DO NOT call the LLM.

    def _prepare_ctf_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            title = entry.get('title', 'N/A')
            context = f"CTF Title: {title}\nDescription: {entry.get('description', 'N/A')}\nFormat: {entry.get('format', 'N/A')}"
            instructions = [f"Provide a concise summary of the CTF event: '{title}'.", f"What are the prizes for the '{title}' CTF?"]
            for instruction in instructions:
                prompt = f"[INST] You are an assistant providing clear information about CTF events.\n\nBased on this information:\n{context}\n\nAnswer the following question:\n{instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'ctf_event', 'source_data': {'id': title, 'type': 'ctf_event'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_arxiv_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            title = entry.get('title', 'N/A').replace('\n', ' ').strip()
            summary = entry.get('summary', 'N/A').replace('\n', ' ').strip()
            authors = ", ".join([author['name'] for author in entry.get('authors', [])])
            context = f"Paper Title: {title}\nAuthors: {authors}\nSummary: {summary}"
            instructions = [f"Summarize the key findings of the research paper titled '{title}'.", f"What is the main contribution of the paper '{title}' by {authors}?"]
            for instruction in instructions:
                prompt = f"[INST] You are a research assistant summarizing academic papers.\n\nBased on the paper details, answer the question.\n\nDetails:\n{context}\n\nQuestion: {instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'research_paper', 'source_data': {'id': entry.get('id'), 'type': 'arxiv_paper'}}
                tasks.append((prompt, metadata))
        return tasks
    
    # This handler uses pre-enhanced data and does not need the LLM
    def _prepare_ubuntu_tasks(self, data: List[Dict]) -> List[Dict]:
        structured_pairs = []
        for entry in data:
            if 'enhanced' in entry and entry['enhanced'].get('description'):
                usn_id = entry.get('title', 'Unknown USN').split(':')[0].strip()
                cves = re.findall(r'CVE-\d{4}-\d{4,7}', entry.get('summary', ''))
                instruction = f"Summarize Ubuntu security notice {usn_id} and the vulnerabilities it addresses."
                # The response is created directly from the enhanced data, no LLM call needed here.
                response = f"Ubuntu Security Notice {usn_id} addresses vulnerabilities including {', '.join(cves) if cves else 'unspecified CVEs'}. The issue involves: {entry['enhanced']['description']}. The risk is rated as '{entry['enhanced']['risk_level']}'. The recommended mitigation is to {entry['enhanced']['mitigations']}."
                structured_pairs.append({'instruction': instruction, 'response': response.strip(), 'type': 'security_advisory', 'source_data': {'id': usn_id, 'type': 'ubuntu_advisory'}})
        return structured_pairs

    def _prepare_microsoft_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            title = entry.get('DocumentTitle', {}).get('Value', 'N/A')
            context = f"Update Title: {title}\nRelease Date: {entry.get('CurrentReleaseDate', 'N/A')}"
            instructions = [f"What is the purpose of the Microsoft security update titled '{title}'?", f"Provide a brief overview of the '{title}' security update."]
            for instruction in instructions:
                prompt = f"[INST] You are an assistant summarizing Microsoft security bulletins.\n\nBased on the metadata, answer the question.\n\nMetadata:\n{context}\n\nQuestion: {instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'security_advisory', 'source_data': {'id': entry.get('ID', {}).get('Value'), 'type': 'microsoft_advisory'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_capec_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            name = entry.get('name', 'N/A')
            capec_id = entry.get('id', 'N/A')
            description = entry.get('description', 'N/A')
            context = f"CAPEC ID: {capec_id}\nName: {name}\nDescription: {description}"
            instructions = [f"Describe the Common Attack Pattern (CAPEC) known as '{name}'.", f"What are the typical mitigations for the attack pattern {capec_id}?"]
            for instruction in instructions:
                prompt = f"[INST] You are a cybersecurity expert explaining attack patterns.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'attack_pattern', 'source_data': {'id': capec_id, 'type': 'capec'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_opencve_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            cve_id = entry.get('id', 'N/A')
            summary = entry.get('summary', 'N/A')
            cvss_v3 = entry.get('cvss', {}).get('v3', 'N/A')
            context = f"CVE ID: {cve_id}\nSummary: {summary}\nCVSSv3 Score: {cvss_v3}"
            instructions = [f"Summarize the vulnerability {cve_id} and its potential impact.", f"What is the severity of CVE {cve_id} based on its CVSS score?"]
            for instruction in instructions:
                prompt = f"[INST] You are a cybersecurity analyst summarizing vulnerability reports.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'vulnerability', 'source_data': {'id': cve_id, 'type': 'opencve'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_mitre_attack_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        attack_patterns = [obj for obj in data if obj.get('type') == 'attack-pattern']
        for entry in attack_patterns:
            name = entry.get('name', 'N/A')
            description = entry.get('description', 'N/A')
            mitre_id = next((ref['external_id'] for ref in entry.get('external_references', []) if ref.get('source_name') == 'mitre-attack'), 'N/A')
            context = f"MITRE ID: {mitre_id}\nName: {name}\nDescription: {description}"
            instructions = [f"Explain the MITRE ATT&CK technique '{name}' ({mitre_id}).", f"What are common detection methods for the attack technique '{name}'?"]
            for instruction in instructions:
                prompt = f"[INST] You are a cybersecurity expert explaining MITRE ATT&CK techniques.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction} [/INST]"
                metadata = {'instruction': instruction, 'type': 'attack_pattern', 'source_data': {'id': mitre_id, 'type': 'mitre_attack'}}
                tasks.append((prompt, metadata))
        return tasks

    def process_directory(self):
        """Process all recognized files in the input directory."""
        all_structured_pairs = []
        input_files = list(self.input_dir.glob('*_filtered_*.json'))
        if not input_files:
            logger.warning(f"No '*_filtered_*.json' files found in {self.input_dir}. Nothing to process.")
            return

        for file_path in input_files:
            logger.info(f"Processing file: {file_path.name}")
            # Reset file timing for benchmark tracking
            self.benchmark.reset_file_timing()
            
            handler = next((func for key, func in self.file_handlers.items() if key in file_path.name), None)
            if not handler:
                logger.warning(f"No handler for file: {file_path.name}. Skipping.")
                continue

            data = self._load_data(file_path)
            if not data: continue
            
            # Special case for Ubuntu data which is pre-structured
            if handler == self._prepare_ubuntu_tasks:
                structured_pairs = handler(data)
                if structured_pairs:
                    all_structured_pairs.extend(structured_pairs)
                    logger.info(f"Generated {len(structured_pairs)} pairs from {file_path.name}")
                continue # Move to the next file
            
            # For all other handlers, prepare and process LLM tasks
            if self.llm_available:
                tasks = handler(data)
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_meta = {executor.submit(self._call_llm, prompt): meta for prompt, meta in tasks}
                    progress_bar = tqdm(as_completed(future_to_meta), total=len(tasks), desc=f"Structuring {file_path.name}")
                    
                    for future in progress_bar:
                        metadata = future_to_meta[future]
                        response = future.result()
                        if response:
                            # Try to extract JSON from response if expected
                            if '{' in response and '}' in response:
                                json_obj = extract_first_json_object(response)
                                if json_obj:
                                    response = json.dumps(json_obj)
                            metadata['response'] = response.strip()
                            all_structured_pairs.append(metadata)
                        
                        # Log benchmark stats periodically
                        self.benchmark.log_benchmark_stats()
            
            # Increment files completed
            self.benchmark.metrics['files_completed'] += 1

        # Log final benchmark stats
        self.benchmark.log_benchmark_stats(force=True)
        
        if all_structured_pairs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"consolidated_cybersecurity_dataset_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {'total_entries': len(all_structured_pairs), 'generation_timestamp': timestamp, 'model_used': self.model.model.name if self.llm_available else "N/A"},
                    'data': all_structured_pairs
                }, f, indent=2)
            
            logger.info(f"Successfully saved {len(all_structured_pairs)} structured pairs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Structure filtered cybersecurity data using an LLM.")
    parser.add_argument("--input-dir", default="filtered_data", help="Directory containing filtered data.")
    parser.add_argument("--output-dir", default="structured_data", help="Output directory.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-8B-4bit", help="The MLX-compatible model to use from Hugging Face.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for LLM calls.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM usage.")
    # Add sampling parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter (default: 0)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling parameter (default: 0.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    args = parser.parse_args()

    try:
        structurer = CyberDataStructurer(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            llm_model=args.model, 
            workers=args.workers, 
            disable_llm=args.disable_llm,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty
        )
        structurer.process_directory()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()