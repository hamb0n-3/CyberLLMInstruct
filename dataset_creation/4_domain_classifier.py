#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import re
from tqdm import tqdm

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    # CORRECTED: Removed 'workers' from __init__
    def __init__(self, input_dir: str, output_dir: str, llm_model: str, disable_llm: bool):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer, self.llm_available = None, None, False

        if not disable_llm and MLX_AVAILABLE:
            try:
                logger.info(f"Loading MLX model '{llm_model}'...")
                self.model, self.tokenizer = load(llm_model)
                self.llm_available = True
                logger.info("MLX model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
        else:
            logger.warning("LLM disabled or mlx-lm not found.")

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
            # CORRECTED: Removed the temperature/temp argument entirely
            response = generate(self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=256)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group(0))
                entry['domains'] = classifications
                entry['primary_domain'] = classifications['domain'] if classifications else 'uncategorized'
            else:
                entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
        except Exception as e:
            logger.error(f"Error during LLM classification: {e}")
            entry['domains'] = []; entry['primary_domain'] = 'uncategorized'
            
        return entry

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

                # Using a sequential loop with tqdm
                classified_entries = [self.classify_entry_with_llm(entry) for entry in tqdm(entries_to_process, desc=f"Classifying {file_path.name}")]

                data_obj['data'] = classified_entries
                data_obj['metadata']['classification_timestamp'] = datetime.now().isoformat()
                
                output_file = self.output_dir / f"{file_path.stem}_classified.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_obj, f, indent=2)
                logger.info(f"Saved classified data to {output_file}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Classify cybersecurity data into domains using an LLM.")
    parser.add_argument("--input-dir", default="structured_data", help="Directory containing structured data.")
    parser.add_argument("--output-dir", default="domain_classified", help="Directory to save classified data.")
    parser.add_argument("--model", type=str, default="mlx-community/c4ai-command-r-v01-4bit", help="The MLX-compatible model to use.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM usage entirely.")
    args = parser.parse_args()

    # CORRECTED: Removed 'workers' from the constructor call
    classifier = CyberDomainClassifier(input_dir=args.input_dir, output_dir=args.output_dir, llm_model=args.model, disable_llm=args.disable_llm)
    classifier.process_directory()

if __name__ == "__main__":
    main()