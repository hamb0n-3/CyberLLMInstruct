#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import re
import yaml
from tqdm import tqdm

# Import MLX client
from mlx_client import MLXClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    def __init__(self, input_dir: str, output_dir: str, config_path: str = "pipeline_config.yaml", disable_llm: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        if not disable_llm:
            # Initialize MLX client
            self.mlx_client = MLXClient(
                server_url=self.config["mlx_server"]["base_url"],
                model_path=self.config["mlx_model"]["path"],
                batch_size=self.config["batching"]["batch_size"],
                batch_timeout_ms=self.config["batching"]["batch_timeout_ms"],
                use_server=self.config["mlx_server"]["use_server"],
                max_retries=self.config["mlx_server"]["max_retries"]
            )
            self.llm_available = True
            logger.info(f"MLX client initialized in {self.mlx_client.mode} mode")
        else:
            self.mlx_client = None
            self.llm_available = False
            logger.warning("LLM disabled. Classification will use rule-based method only.")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                "mlx_server": {"base_url": "http://localhost:8080", "use_server": True, "max_retries": 3},
                "mlx_model": {"path": "mlx-community/Phi-3-mini-4k-instruct-4bit"},
                "batching": {"batch_size": 16, "batch_timeout_ms": 100},
                "pipeline": {
                    "domain_classifier": {"max_tokens": 512, "temperature": 0.3}
                }
            }

        self.domains = ['malware', 'phishing', 'zero_day', 'iot_security', 'web_security', 'network_security', 'vulnerability_management', 'cloud_security', 'cryptography', 'identity_access_management', 'incident_response', 'threat_intelligence', 'compliance_and_frameworks', 'social_engineering']

    def classify_entry_with_llm(self, entry: Dict) -> Dict:
        prompt = f"""[INST]
You are a cybersecurity domain classification expert. Analyze the following content and classify it into one or more of these domains: {', '.join(self.domains)}.
Return your classification as a JSON array of objects with 'domain' and 'confidence' (from 0.0 to 1.0) fields, sorted by confidence. Only include domains with confidence > 0.2.

Content:
Instruction: {entry.get('instruction', '')}
Response: {entry.get('response', '')}

Respond with ONLY the JSON array.
[/INST]"""
        
        try:
            # Get configuration for domain classifier
            config = self.config["pipeline"]["domain_classifier"]
            response = self.mlx_client.generate(
                prompt,
                max_tokens=config.get("max_tokens", 512),
                temperature=config.get("temperature", 0.3)
            )
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group(0))
                entry['domains'] = classifications
                entry['primary_domain'] = classifications[0]['domain'] if classifications else 'uncategorized'
            else:
                entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
        except Exception as e:
            logger.error(f"Error during LLM classification: {e}")
            entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
            
        return entry
    
    def classify_entries_batch(self, entries: List[Dict]) -> List[Dict]:
        """Classify multiple entries using batch processing"""
        prompts = []
        for entry in entries:
            prompt = f"""[INST]
You are a cybersecurity domain classification expert. Analyze the following content and classify it into one or more of these domains: {', '.join(self.domains)}.
Return your classification as a JSON array of objects with 'domain' and 'confidence' (from 0.0 to 1.0) fields, sorted by confidence. Only include domains with confidence > 0.2.

Content:
Instruction: {entry.get('instruction', '')}
Response: {entry.get('response', '')}

Respond with ONLY the JSON array.
[/INST]"""
            prompts.append(prompt)
        
        try:
            # Get batch responses
            config = self.config["pipeline"]["domain_classifier"]
            responses = self.mlx_client.generate_batch(
                prompts,
                max_tokens=config.get("max_tokens", 512),
                temperature=config.get("temperature", 0.3)
            )
            
            # Process responses
            for entry, response in zip(entries, responses):
                try:
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        classifications = json.loads(json_match.group(0))
                        entry['domains'] = classifications
                        entry['primary_domain'] = classifications[0]['domain'] if classifications else 'uncategorized'
                    else:
                        entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
                except:
                    entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
                    
        except Exception as e:
            logger.error(f"Error during batch classification: {e}")
            # Fallback to all uncategorized
            for entry in entries:
                entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
                
        return entries

    def process_directory(self):
        files_to_process = list(self.input_dir.glob('*.json'))
        for file_path in files_to_process:
            logger.info(f"Processing {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data_obj = json.load(f)
                entries_to_process = data_obj.get('data', [])
                if not isinstance(entries_to_process, list): continue
                if not self.llm_available:
                    logger.warning("LLM not available. Cannot classify entries."); break

                # Process in batches for better performance
                batch_size = self.config["batching"]["batch_size"]
                classified_entries = []
                
                for i in tqdm(range(0, len(entries_to_process), batch_size), desc=f"Classifying {file_path.name} (batched)"):
                    batch = entries_to_process[i:i + batch_size]
                    classified_batch = self.classify_entries_batch(batch)
                    classified_entries.extend(classified_batch)

                data_obj['data'] = classified_entries
                data_obj['metadata']['classification_timestamp'] = datetime.now().isoformat()
                
                output_file = self.output_dir / f"{file_path.stem}_classified.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_obj, f, indent=2)
                logger.info(f"Saved classified data to {output_file}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
        
        # Close the MLX client when done
        if self.mlx_client:
            self.mlx_client.close()

def main():
    parser = argparse.ArgumentParser(description="Classify cybersecurity data into domains using MLX with advanced server support.")
    parser.add_argument("--input-dir", default="structured_data", help="Directory containing structured data.")
    parser.add_argument("--output-dir", default="domain_classified", help="Directory to save classified data.")
    parser.add_argument("--config", type=str, default="pipeline_config.yaml", help="Path to pipeline configuration file.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM usage entirely.")
    args = parser.parse_args()

    classifier = CyberDomainClassifier(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        config_path=args.config,
        disable_llm=args.disable_llm
    )
    classifier.process_directory()

if __name__ == "__main__":
    main()