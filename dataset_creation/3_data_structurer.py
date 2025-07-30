#!/usr/bin/env python3

import os
# Disable tokenizers parallelism to avoid multiprocessing conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
import sys
import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import argparse
import xml.etree.ElementTree as ET
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

# Initial logging setup (will be reconfigured in setup_logging)
logger = logging.getLogger(__name__)


def setup_logging(log_file_arg: str, model_name: str):
    """Setup logging configuration with optional file logging."""
    # Base logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file_arg:
        try:
            if log_file_arg == 'default':
                # Extract model name from path
                model_basename = model_name.split('/')[-1] if '/' in model_name else model_name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create logs directory relative to script
                script_dir = Path(__file__).parent
                logs_dir = script_dir / 'logs'
                logs_dir.mkdir(exist_ok=True)
                
                log_file_path = logs_dir / f"{model_basename}_{timestamp}.log"
            else:
                log_file_path = Path(log_file_arg)
                # Create parent directory if needed
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file_path}")
            
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")


class CyberDataStructurer:
    def __init__(self, input_dir: str, output_dir: str, llm_model: str,
                 temperature: float = 0.7, top_p: float = 1.0, top_k: int = 0,
                 min_p: float = 0.0, repetition_penalty: float = 1.0,
                 two_prompts: bool = False):
        """Initialize the data structurer."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer = None, None
        self.sampler = None
        self.llm_model_name = llm_model  # Store the model name
        self.two_prompts = two_prompts  # Whether to generate 2 prompts per entry
        
        # Store sampling parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        
        # Initialize benchmark tracker
        self.benchmark = BenchmarkTracker(logger=logger)
        
        # Initialize state management
        self.state_file = self.output_dir / ".structurer_state.pkl"
        self.state = self._load_state()

        if not MLX_AVAILABLE:
            logger.error("MLX not available. Please install mlx-lm package.")
            raise RuntimeError("MLX not available. Cannot proceed without LLM.")
            
        try:
            logger.info(f"Loading MLX model '{llm_model}'...")
            self.model, self.tokenizer = load(llm_model)
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
            raise RuntimeError(f"Failed to load MLX model: {e}")

        self.file_handlers = {
            'ctf_data': self._prepare_ctf_tasks,
            'arxiv_papers': self._prepare_arxiv_tasks,
            'ubuntu_security': self._prepare_ubuntu_tasks,
            'microsoft_security': self._prepare_microsoft_tasks,
            'capec_data': self._prepare_capec_tasks,
            'opencve_data': self._prepare_opencve_tasks,
            'mitre_attack': self._prepare_mitre_attack_tasks,
        }
    
    def _load_state(self) -> Dict:
        """Load processing state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                # Rebuild processed_items from partial_results if needed
                if 'processed_items' not in state or not state['processed_items']:
                    state['processed_items'] = set()
                    source_type_counts = {}
                    for result in state.get('partial_results', []):
                        source_id = result.get('source_data', {}).get('id', '')
                        source_type = result.get('source_data', {}).get('type', 'unknown')
                        instruction = result.get('instruction', '')
                        if source_id and instruction:
                            # Use SHA256 for deterministic hashing
                            instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
                            key = (source_id, instruction_hash)
                            state['processed_items'].add(key)
                            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                    logger.info(f"Rebuilt processed_items set with {len(state['processed_items'])} entries from partial results")
                    logger.info(f"Source types in partial results: {source_type_counts}")
                
                # Ensure completed_files exists in state
                if 'completed_files' not in state:
                    state['completed_files'] = set()
                
                logger.info(f"Loaded state with {len(state.get('processed_items', set()))} processed items, "
                           f"{len(state.get('partial_results', []))} partial results, "
                           f"{len(state.get('completed_files', set()))} completed files")
                
                # Check if we need to migrate from old hash format
                if state.get('processed_items') and isinstance(next(iter(state['processed_items']), None), tuple):
                    sample_item = next(iter(state['processed_items']))
                    if len(sample_item) == 2 and isinstance(sample_item[1], int):
                        logger.warning("Detected old hash format in state file. State will be rebuilt with new SHA256 hashes.")
                        # Clear processed_items to force rebuild from partial_results
                        state['processed_items'] = set()
                        # Rebuild with new hash format
                        source_type_counts = {}
                        for result in state.get('partial_results', []):
                            source_id = result.get('source_data', {}).get('id', '')
                            source_type = result.get('source_data', {}).get('type', 'unknown')
                            instruction = result.get('instruction', '')
                            if source_id and instruction:
                                # Use SHA256 for deterministic hashing
                                instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
                                key = (source_id, instruction_hash)
                                state['processed_items'].add(key)
                                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                        logger.info(f"Migrated {len(state['processed_items'])} entries to new SHA256 hash format")
                        logger.info(f"Source types migrated: {source_type_counts}")
                
                return state
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        
        return {
            'processed_items': set(),  # Set of (source_id, instruction_hash) tuples
            'partial_results': [],     # Results processed but not yet saved
            'last_file': None,        # Last file being processed (deprecated, kept for compatibility)
            'completed_files': set(),  # Set of fully processed file names
            'start_time': time.time()
        }
    
    def _load_consolidated_dataset(self, dataset_path: str):
        """Load an existing consolidated dataset and populate state from it."""
        try:
            logger.info(f"Loading consolidated dataset from {dataset_path}")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            if 'data' not in dataset:
                logger.error("Invalid dataset format: missing 'data' field")
                return
            
            # Extract entries and rebuild state
            entries = dataset['data']
            source_type_counts = {}
            file_source_counts = {}
            
            for entry in entries:
                # Add to partial results
                self.state['partial_results'].append(entry)
                
                # Extract metadata
                source_id = entry.get('source_data', {}).get('id', '')
                source_type = entry.get('source_data', {}).get('type', 'unknown')
                instruction = entry.get('instruction', '')
                
                # Track source types
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                
                # Map source types to file patterns
                source_to_file_map = {
                    'capec': 'capec_data',
                    'arxiv_paper': 'arxiv_papers',
                    'ubuntu_advisory': 'ubuntu_security',
                    'microsoft_advisory': 'microsoft_security',
                    'opencve': 'opencve_data',
                    'mitre_attack': 'mitre_attack',
                    'ctf_event': 'ctf_data'
                }
                
                file_pattern = source_to_file_map.get(source_type, source_type)
                file_source_counts[file_pattern] = file_source_counts.get(file_pattern, 0) + 1
                
                # Add to processed items
                if source_id and instruction:
                    instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
                    key = (source_id, instruction_hash)
                    self.state['processed_items'].add(key)
            
            logger.info(f"Loaded {len(entries)} entries from consolidated dataset")
            logger.info(f"Source type distribution: {source_type_counts}")
            logger.info(f"Estimated file distribution: {file_source_counts}")
            
            # Try to determine which files might be fully completed
            # This is an approximation since we don't have exact file completion info
            for file_pattern, count in file_source_counts.items():
                # Check if this looks like a complete file based on typical counts
                # This is a heuristic and may need adjustment
                if file_pattern == 'capec' and count >= 500:
                    self.state['completed_files'].add('capec_data')
                elif file_pattern == 'mitre_attack' and count >= 500:
                    self.state['completed_files'].add('mitre_attack')
                # Add other heuristics as needed
            
            logger.info(f"Marked as potentially completed: {self.state['completed_files']}")
            
        except Exception as e:
            logger.error(f"Failed to load consolidated dataset: {e}")
            raise
    
    def _save_state(self):
        """Save current processing state to disk."""
        try:
            # Create temporary file first for atomic write
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(self.state, f)
            # Atomic rename
            temp_file.replace(self.state_file)
            logger.debug(f"Saved state with {len(self.state['processed_items'])} processed items")
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def _mark_processed(self, source_id: str, instruction: str):
        """Mark an item as processed."""
        # Use SHA256 for deterministic hashing across Python sessions
        instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
        key = (source_id, instruction_hash)
        self.state['processed_items'].add(key)
    
    def _is_processed(self, source_id: str, instruction: str) -> bool:
        """Check if an item has been processed."""
        # Use SHA256 for deterministic hashing across Python sessions
        instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
        key = (source_id, instruction_hash)
        return key in self.state['processed_items']

    def _call_llm(self, prompt: str, use_chat_template: bool = True) -> Optional[str]:
        """Call the local MLX model with retry logic."""
            
        # Apply chat template if supported and requested
        formatted_prompt = prompt
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # Parse the prompt to extract the instruction part
            # The prompts are in format: [INST] System message + instruction [/INST]
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # We want direct responses, not thinking process
            )
        
        max_attempts = 1
        token_limits = [1500]
        
        for attempt in range(max_attempts):
            try:
                # Count input tokens
                input_tokens = len(self.tokenizer.encode(formatted_prompt)) if self.tokenizer else len(formatted_prompt.split())
                
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
                    prompt=formatted_prompt, 
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
            
            if self.two_prompts:
                instructions = [f"Provide a concise summary of the CTF event: '{title}'.", f"What are the prizes for the '{title}' CTF?"]
            else:
                # Single comprehensive instruction
                instructions = [f"Provide a comprehensive summary of the CTF event '{title}', including its format, description, and prize information if available."]
            
            for instruction in instructions:
                prompt = f"You are an assistant providing clear information about CTF events.\n\nBased on this information:\n{context}\n\nAnswer the following question:\n{instruction}"
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
            
            if self.two_prompts:
                instructions = [f"Summarize the key findings of the research paper titled '{title}'.", f"What is the main contribution of the paper '{title}' by {authors}?"]
            else:
                # Single comprehensive instruction
                instructions = [f"Provide a comprehensive summary of the research paper '{title}' by {authors}, highlighting the key findings, main contributions, and significance of the work."]
            
            for instruction in instructions:
                prompt = f"You are a research assistant summarizing academic papers.\n\nBased on the paper details, answer the question.\n\nDetails:\n{context}\n\nQuestion: {instruction}"
                metadata = {'instruction': instruction, 'type': 'research_paper', 'source_data': {'id': entry.get('id'), 'type': 'arxiv_paper'}}
                tasks.append((prompt, metadata))
        return tasks
    
    # This handler uses pre-enhanced data and does not need the LLM
    def _prepare_ubuntu_tasks(self, data: List[Dict]) -> List[Dict]:
        structured_pairs = []
        for entry in data:
            start_time = time.time()
            # Check for enhanced fields at the top level (from 2_data_filter.py)
            if 'technical_description' in entry:
                usn_id = entry.get('title', 'Unknown USN').split(':')[0].strip()
                cves = re.findall(r'CVE-\d{4}-\d{4,7}', entry.get('summary', ''))
                instruction = f"Summarize Ubuntu security notice {usn_id} and the vulnerabilities it addresses."
                # The response is created directly from the enhanced data, no LLM call needed here.
                response = f"Ubuntu Security Notice {usn_id} addresses vulnerabilities including {', '.join(cves) if cves else 'unspecified CVEs'}. The issue involves: {entry['technical_description']}. The risk is rated as '{entry['risk_level']}'. The recommended mitigation is to {entry['mitigations']}."
                structured_pairs.append({'instruction': instruction, 'response': response.strip(), 'type': 'security_advisory', 'source_data': {'id': usn_id, 'type': 'ubuntu_advisory'}})
                
                # Record benchmark entry
                processing_time = time.time() - start_time
                entry_size = len(json.dumps(entry))
                self.benchmark.record_entry(passed=True, processing_time=processing_time, entry_size=entry_size)
        return structured_pairs

    def _prepare_microsoft_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            # Handle nested structure for DocumentTitle and ID
            if isinstance(entry.get('DocumentTitle'), dict):
                title = entry.get('DocumentTitle', {}).get('Value', 'N/A')
                doc_id = entry.get('ID', {}).get('Value', 'N/A') if isinstance(entry.get('ID'), dict) else entry.get('ID', 'N/A')
            else:
                title = entry.get('DocumentTitle', 'N/A')
                doc_id = entry.get('ID', 'N/A')
            
            # Check if enhanced fields are available
            if 'technical_description' in entry:
                context = f"Update Title: {title}\nRelease Date: {entry.get('CurrentReleaseDate', 'N/A')}\nTechnical Details: {entry['technical_description']}\nRisk Level: {entry.get('risk_level', 'N/A')}\nAffected Systems: {entry.get('affected_systems', 'N/A')}"
                if self.two_prompts:
                    instructions = [
                        f"Provide a comprehensive analysis of the Microsoft security update '{title}' including technical details and impact.",
                        f"What are the specific vulnerabilities addressed in '{title}' and their recommended mitigations?"
                    ]
                else:
                    instructions = [f"Provide a comprehensive analysis of the Microsoft security update '{title}', including technical details, impact assessment, specific vulnerabilities addressed, and recommended mitigations."]
            else:
                context = f"Update Title: {title}\nRelease Date: {entry.get('CurrentReleaseDate', 'N/A')}"
                if self.two_prompts:
                    instructions = [
                        f"What is the purpose of the Microsoft security update titled '{title}'?",
                        f"Provide a brief overview of the '{title}' security update."
                    ]
                else:
                    instructions = [f"Provide a comprehensive overview of the Microsoft security update '{title}', including its purpose, scope, and key information."]
            
            for instruction in instructions:
                prompt = f"You are an assistant summarizing Microsoft security bulletins.\n\nBased on the metadata, answer the question.\n\nMetadata:\n{context}\n\nQuestion: {instruction}"
                metadata = {'instruction': instruction, 'type': 'security_advisory', 'source_data': {'id': doc_id, 'type': 'microsoft_advisory'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_capec_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        for entry in data:
            name = entry.get('name', 'N/A')
            capec_id = entry.get('id', 'N/A')
            description = entry.get('description', 'N/A')
            
            # Check if enhanced fields are available
            if 'technical_description' in entry:
                context = f"CAPEC ID: {capec_id}\nName: {name}\nDescription: {description}\nTechnical Analysis: {entry['technical_description']}\nRisk Level: {entry.get('risk_level', 'N/A')}\nAffected Systems: {entry.get('affected_systems', 'N/A')}"
                if self.two_prompts:
                    instructions = [
                        f"Provide a detailed explanation of the Common Attack Pattern (CAPEC) '{name}' including technical implementation details.",
                        f"What are the comprehensive mitigations and detection strategies for the attack pattern {capec_id}?"
                    ]
                else:
                    instructions = [f"Provide a comprehensive analysis of the Common Attack Pattern (CAPEC) '{name}' ({capec_id}), including technical implementation details, affected systems, and complete mitigation and detection strategies."]
            else:
                context = f"CAPEC ID: {capec_id}\nName: {name}\nDescription: {description}"
                if self.two_prompts:
                    instructions = [
                        f"Describe the Common Attack Pattern (CAPEC) known as '{name}'.",
                        f"What are the typical mitigations for the attack pattern {capec_id}?"
                    ]
                else:
                    instructions = [f"Provide a comprehensive description of the Common Attack Pattern (CAPEC) '{name}' ({capec_id}), including how it works and typical mitigations."]
            
            for instruction in instructions:
                prompt = f"You are a cybersecurity expert explaining attack patterns.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction}"
                metadata = {'instruction': instruction, 'type': 'attack_pattern', 'source_data': {'id': capec_id, 'type': 'capec'}}
                tasks.append((prompt, metadata))
        return tasks

    def _prepare_opencve_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        
        # Handle the special structure of opencve filtered data
        # It's wrapped in an array with one object containing 'summary' array and enhanced fields
        if isinstance(data, list) and len(data) > 0 and 'summary' in data[0]:
            # Extract the CVE entries from the summary array
            cve_entries = data[0].get('summary', [])
            # Get enhanced fields from the top-level object
            enhanced_tech_desc = data[0].get('technical_description', '')
            enhanced_risk = data[0].get('risk_level', '')
            enhanced_affected = data[0].get('affected_systems', [])
            enhanced_mitigations = data[0].get('mitigations', [])
            
            for entry in cve_entries:
                cve_id = entry.get('cve_id', 'N/A')
                description = entry.get('description', 'N/A')
                
                # Create enhanced context if we have enhanced data
                if enhanced_tech_desc:
                    context = f"CVE ID: {cve_id}\nDescription: {description}\nTechnical Analysis: {enhanced_tech_desc}\nRisk Level: {enhanced_risk}"
                    if enhanced_affected:
                        context += f"\nAffected Systems: {', '.join(enhanced_affected[:5])}"  # Limit to first 5
                    if enhanced_mitigations:
                        context += f"\nMitigations: {'; '.join(enhanced_mitigations[:3])}"  # Limit to first 3
                    
                    if self.two_prompts:
                        instructions = [
                            f"Provide a detailed analysis of vulnerability {cve_id} including its technical impact and severity.",
                            f"What are the recommended mitigations for CVE {cve_id} and which systems are affected?"
                        ]
                    else:
                        instructions = [f"Provide a comprehensive analysis of vulnerability {cve_id}, including its technical impact, severity assessment, affected systems, and recommended mitigations."]
                else:
                    context = f"CVE ID: {cve_id}\nDescription: {description}"
                    if self.two_prompts:
                        instructions = [
                            f"Summarize the vulnerability {cve_id} and its potential impact.",
                            f"What is the severity of CVE {cve_id} based on available information?"
                        ]
                    else:
                        instructions = [f"Provide a comprehensive summary of vulnerability {cve_id}, including its potential impact and severity assessment."]
                
                for instruction in instructions:
                    prompt = f"You are a cybersecurity analyst summarizing vulnerability reports.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction}"
                    metadata = {'instruction': instruction, 'type': 'vulnerability', 'source_data': {'id': cve_id, 'type': 'opencve'}}
                    tasks.append((prompt, metadata))
        else:
            # Fallback to original logic if structure is different
            for entry in data:
                cve_id = entry.get('id', entry.get('cve_id', 'N/A'))
                summary = entry.get('summary', entry.get('description', 'N/A'))
                cvss_v3 = entry.get('cvss', {}).get('v3', 'N/A') if isinstance(entry.get('cvss'), dict) else 'N/A'
                context = f"CVE ID: {cve_id}\nSummary: {summary}\nCVSSv3 Score: {cvss_v3}"
                if self.two_prompts:
                    instructions = [f"Summarize the vulnerability {cve_id} and its potential impact.", f"What is the severity of CVE {cve_id} based on its CVSS score?"]
                else:
                    instructions = [f"Provide a comprehensive summary of vulnerability {cve_id}, including its potential impact and severity based on the CVSS score."]
                for instruction in instructions:
                    prompt = f"You are a cybersecurity analyst summarizing vulnerability reports.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction}"
                    metadata = {'instruction': instruction, 'type': 'vulnerability', 'source_data': {'id': cve_id, 'type': 'opencve'}}
                    tasks.append((prompt, metadata))
        return tasks

    def _prepare_mitre_attack_tasks(self, data: List[Dict]) -> List[Tuple[str, Dict]]:
        tasks = []
        
        # Check if we have enhanced fields at the top level
        enhanced_data = None
        if isinstance(data, list) and len(data) > 0 and 'technical_description' in data[0]:
            enhanced_data = data[0]
            # Extract attack patterns from the nested structure if present
            if 'objects' in data[0]:
                attack_patterns = [obj for obj in data[0]['objects'] if obj.get('type') == 'attack-pattern']
            else:
                attack_patterns = [obj for obj in data if obj.get('type') == 'attack-pattern']
        else:
            attack_patterns = [obj for obj in data if obj.get('type') == 'attack-pattern']
        
        for entry in attack_patterns:
            name = entry.get('name', 'N/A')
            description = entry.get('description', 'N/A')
            mitre_id = next((ref['external_id'] for ref in entry.get('external_references', []) if ref.get('source_name') == 'mitre-attack'), 'N/A')
            
            # Use enhanced description if available
            if enhanced_data and enhanced_data.get('technical_description'):
                context = f"MITRE ID: {mitre_id}\nName: {name}\nDescription: {description}\nEnhanced Analysis: {enhanced_data['technical_description']}"
                if self.two_prompts:
                    instructions = [
                        f"Provide a comprehensive explanation of the MITRE ATT&CK technique '{name}' ({mitre_id}) including its technical implementation.",
                        f"What are the detection methods and mitigations for the attack technique '{name}'?"
                    ]
                else:
                    instructions = [f"Provide a comprehensive analysis of the MITRE ATT&CK technique '{name}' ({mitre_id}), including its technical implementation, detection methods, and mitigation strategies."]
            else:
                context = f"MITRE ID: {mitre_id}\nName: {name}\nDescription: {description}"
                if self.two_prompts:
                    instructions = [
                        f"Explain the MITRE ATT&CK technique '{name}' ({mitre_id}).",
                        f"What are common detection methods for the attack technique '{name}'?"
                    ]
                else:
                    instructions = [f"Provide a comprehensive explanation of the MITRE ATT&CK technique '{name}' ({mitre_id}), including how it works and common detection methods."]
            
            for instruction in instructions:
                prompt = f"You are a cybersecurity expert explaining MITRE ATT&CK techniques.\n\nBased on the provided information, answer the question.\n\nInformation:\n{context}\n\nQuestion: {instruction}"
                metadata = {'instruction': instruction, 'type': 'attack_pattern', 'source_data': {'id': mitre_id, 'type': 'mitre_attack'}}
                tasks.append((prompt, metadata))
        return tasks

    def process_directory(self, sources: Optional[List[str]] = None, resume: bool = True):
        """Process all recognized files in the input directory."""
        all_structured_pairs = []
        
        # Load partial results if resuming
        if resume and self.state.get('partial_results'):
            all_structured_pairs.extend(self.state['partial_results'])
            # Count source types in partial results
            source_type_counts = {}
            for result in self.state['partial_results']:
                source_type = result.get('source_data', {}).get('type', 'unknown')
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
            logger.info(f"Resuming with {len(all_structured_pairs)} existing entries")
            logger.info(f"Existing entries by type: {source_type_counts}")
            if self.state.get('completed_files'):
                logger.info(f"Files marked as completed: {self.state['completed_files']}")
            if self.state.get('last_file'):
                logger.info(f"Last processed file: {self.state['last_file']}")
        
        input_files = list(self.input_dir.glob('*_filtered_*.json'))
        
        # Filter by sources if specified
        if sources:
            filtered_files = []
            for file_path in input_files:
                file_name_lower = file_path.name.lower()
                for source in sources:
                    if source.lower() in file_name_lower:
                        filtered_files.append(file_path)
                        break
            input_files = filtered_files
            logger.info(f"Filtered to {len(input_files)} files matching sources: {sources}")
        
        if not input_files:
            logger.warning(f"No '*_filtered_*.json' files found in {self.input_dir}. Nothing to process.")
            return

        try:
            for file_path in input_files:
                # Skip files that have been fully completed
                if resume and file_path.name in self.state.get('completed_files', set()):
                    logger.info(f"Skipping already completed file: {file_path.name}")
                    continue
                
                logger.info(f"Processing file: {file_path.name}")
                self.state['last_file'] = file_path.name  # Keep for backward compatibility
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
                    new_pairs = []
                    for pair in structured_pairs:
                        source_id = pair.get('source_data', {}).get('id', '')
                        instruction = pair.get('instruction', '')
                        if not self._is_processed(source_id, instruction):
                            new_pairs.append(pair)
                            self._mark_processed(source_id, instruction)
                    
                    if new_pairs:
                        all_structured_pairs.extend(new_pairs)
                        logger.info(f"Generated {len(new_pairs)} new pairs from {file_path.name} (skipped {len(structured_pairs) - len(new_pairs)} already processed)")
                    else:
                        logger.info(f"All entries from {file_path.name} already processed")
                    continue # Move to the next file
                
                # For all other handlers, prepare and process LLM tasks
                tasks = handler(data)
                
                # Filter out already processed tasks
                new_tasks = []
                skipped_count = 0
                # Log first few IDs for debugging
                first_few_ids = []
                for i, (prompt, metadata) in enumerate(tasks):
                    source_id = metadata.get('source_data', {}).get('id', '')
                    source_type = metadata.get('source_data', {}).get('type', '')
                    instruction = metadata.get('instruction', '')
                    if i < 3:  # Log first 3 for debugging
                        first_few_ids.append(f"{source_type}:{source_id}")
                    if not self._is_processed(source_id, instruction):
                        new_tasks.append((prompt, metadata))
                    else:
                        skipped_count += 1
                
                if first_few_ids:
                    logger.info(f"Sample IDs being checked from {file_path.name}: {first_few_ids}")
                
                if not new_tasks:
                    logger.info(f"All {len(tasks)} tasks from {file_path.name} already processed")
                    continue
                
                logger.info(f"Processing {len(new_tasks)} new tasks from {file_path.name} (skipped {skipped_count} already processed)")
                
                progress_bar = tqdm(new_tasks, desc=f"Structuring {file_path.name}")
                
                for i, (prompt, metadata) in enumerate(progress_bar):
                    response = self._call_llm(prompt)
                    if response:
                        # Try to extract JSON from response if expected
                        if '{' in response and '}' in response:
                            json_obj = extract_first_json_object(response)
                            if json_obj:
                                response = json.dumps(json_obj)
                        metadata['response'] = response.strip()
                        all_structured_pairs.append(metadata)
                        
                        # Mark as processed
                        source_id = metadata.get('source_data', {}).get('id', '')
                        instruction = metadata.get('instruction', '')
                        self._mark_processed(source_id, instruction)
                    
                    # Save state periodically (every 10 items)
                    if (i + 1) % 10 == 0:
                        self.state['partial_results'] = all_structured_pairs
                        self._save_state()
                        logger.debug(f"Saved state after {i + 1} items")
                    
                    # Log benchmark stats periodically (every 100 items instead of every item)
                    if (i + 1) % 100 == 0:
                        self.benchmark.log_benchmark_stats()
                # Increment files completed
                self.benchmark.metrics['files_completed'] += 1
                # Mark this file as completed
                if 'completed_files' not in self.state:
                    self.state['completed_files'] = set()
                self.state['completed_files'].add(file_path.name)
                logger.info(f"Completed processing {file_path.name}")
                
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted! Saving state...")
            self.state['partial_results'] = all_structured_pairs
            self._save_state()
            logger.info(f"State saved. Processed {len(all_structured_pairs)} items so far.")
            logger.info("Run the script again to resume from where you left off.")
            return

        # Log final benchmark stats
        self.benchmark.log_benchmark_stats(force=True)
        
        if all_structured_pairs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"consolidated_cybersecurity_dataset_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {'total_entries': len(all_structured_pairs), 'generation_timestamp': timestamp, 'model_used': self.llm_model_name},
                    'data': all_structured_pairs
                }, f, indent=2)
            
            logger.info(f"Successfully saved {len(all_structured_pairs)} structured pairs to {output_file}")
            
            # Clean up state file on successful completion
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info("Processing completed successfully, removed state file")

def main():
    parser = argparse.ArgumentParser(description="Structure filtered cybersecurity data using an LLM.")
    parser.add_argument("--input-dir", default="filtered_data", help="Directory containing filtered data.")
    parser.add_argument("--output-dir", default="structured_data", help="Output directory.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-8B-4bit-DWQ-053125", help="The MLX-compatible model to use from Hugging Face.")
    # Add sampling parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling parameter (default: 0)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling parameter (default: 0.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    # Source filtering
    parser.add_argument("--sources", nargs='+', help="Filter files by source names (e.g., opencve mitre_attack ubuntu_security)")
    # Two prompts mode
    parser.add_argument("--two-prompts", action="store_true", help="Generate two prompts per entry instead of one comprehensive prompt (default: False)")
    # Resume control
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring any saved state (default: resume if state exists)")
    parser.add_argument("--resume-from", type=str, help="Resume from an existing consolidated dataset JSON file")
    # Logging
    parser.add_argument("-L", "--log-file", type=str, nargs='?', const='default', help="Log to file. If no path provided, uses logs/[model_name]_[datetime].log")
    args = parser.parse_args()
    
    # Setup logging configuration
    setup_logging(args.log_file, args.model)
    
    #Temperature=0.6, TopP=0.95, TopK=20, and MinP=0
    try:
        structurer = CyberDataStructurer(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            llm_model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            two_prompts=args.two_prompts
        )
        
        # Handle resume-from consolidated dataset
        if args.resume_from:
            structurer._load_consolidated_dataset(args.resume_from)
            # Force resume mode when loading from consolidated dataset
            resume = True
        else:
            resume = not args.no_resume
        
        structurer.process_directory(sources=args.sources, resume=resume)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()