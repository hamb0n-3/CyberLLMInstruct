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

# Import batching utilities
try:
    from mlx_batch_utils import BatchedMLXGenerator, DynamicBatcher
    from mlx_speculative import SpeculativeDecoder, SpeculativeConfig
    BATCHING_AVAILABLE = True
except ImportError:
    BATCHING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    def __init__(self, input_dir: str, output_dir: str, llm_model: str, disable_llm: bool,
                 use_batching: bool = True, batch_size: int = 32, batch_timeout: float = 0.1,
                 use_speculative: bool = False, draft_model_path: str = "mlx-community/gemma-3-1b-it-bf16"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer, self.llm_available = None, None, False
        self.use_batching = use_batching and BATCHING_AVAILABLE
        self.use_speculative = use_speculative and BATCHING_AVAILABLE
        self.batch_size = batch_size

        if not disable_llm and MLX_AVAILABLE:
            try:
                logger.info(f"Loading MLX model '{llm_model}'...")
                self.model, self.tokenizer = load(llm_model)
                self.llm_available = True
                logger.info("MLX model loaded successfully.")
                
                # Setup generation method based on configuration
                if self.use_speculative:
                    logger.info(f"Initializing speculative decoding with draft model: {draft_model_path}")
                    self.speculative_decoder = SpeculativeDecoder(
                        target_model=self.model,
                        target_tokenizer=self.tokenizer,
                        config=SpeculativeConfig(
                            draft_model_path=draft_model_path,
                            target_model_path=llm_model,
                            temperature=0.0
                        )
                    )
                elif self.use_batching:
                    logger.info(f"Initializing batched generator with batch_size={batch_size}")
                    self.batch_generator = BatchedMLXGenerator(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_batch_size=batch_size,
                        batch_timeout=batch_timeout,
                        max_sequence_length=1024
                    )
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
        else:
            logger.warning("LLM disabled or mlx-lm not found.")

        self.domains = ['malware', 'phishing', 'zero_day', 'iot_security', 'web_security', 'network_security', 'vulnerability_management', 'cloud_security', 'cryptography', 'identity_access_management', 'incident_response', 'threat_intelligence', 'compliance_and_frameworks', 'social_engineering']

    def _generate_classification(self, prompt: str) -> str:
        """Generate classification using configured method"""
        if self.use_speculative:
            return self.speculative_decoder.generate(
                prompt=prompt,
                max_tokens=256,
                temperature=0.0,
                verbose=False
            )
        elif self.use_batching:
            return self.batch_generator.generate(
                prompt=prompt,
                max_tokens=256,
                temperature=0.0
            )
        else:
            # Original generation
            return generate(self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=256)
    
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
            response = self._generate_classification(prompt)
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

                # Process with batching if enabled
                if self.use_batching and hasattr(self, 'batch_generator'):
                    logger.info(f"Using batched processing for classification")
                    classified_entries = []
                    for i in tqdm(range(0, len(entries_to_process), self.batch_size), desc=f"Classifying {file_path.name} (Batched)"):
                        batch = entries_to_process[i:i+self.batch_size]
                        for entry in batch:
                            classified_entry = self.classify_entry_with_llm(entry)
                            classified_entries.append(classified_entry)
                else:
                    # Original sequential processing
                    classified_entries = [self.classify_entry_with_llm(entry) for entry in tqdm(entries_to_process, desc=f"Classifying {file_path.name}")]

                data_obj['data'] = classified_entries
                data_obj['metadata']['classification_timestamp'] = datetime.now().isoformat()
                
                output_file = self.output_dir / f"{file_path.stem}_classified.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_obj, f, indent=2)
                logger.info(f"Saved classified data to {output_file}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'batch_generator'):
            self.batch_generator.shutdown()
            logger.info("Batch generator shutdown complete")

def main():
    parser = argparse.ArgumentParser(description="Classify cybersecurity data into domains using an LLM with batching support.")
    parser.add_argument("--input-dir", default="structured_data", help="Directory containing structured data.")
    parser.add_argument("--output-dir", default="domain_classified", help="Directory to save classified data.")
    parser.add_argument("--model", type=str, default="mlx-community/c4ai-command-r-v01-4bit", help="The MLX-compatible model to use.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM usage entirely.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Batch timeout in seconds (default: 0.1)")
    parser.add_argument("--no-batching", action="store_true", help="Disable batched processing")
    parser.add_argument("--enable-speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--draft-model", type=str, default="mlx-community/gemma-3-1b-it-bf16", help="Draft model for speculative decoding")
    args = parser.parse_args()

    try:
        classifier = CyberDomainClassifier(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            llm_model=args.model, 
            disable_llm=args.disable_llm,
            use_batching=not args.no_batching,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            use_speculative=args.enable_speculative,
            draft_model_path=args.draft_model
        )
        classifier.process_directory()
    finally:
        if 'classifier' in locals():
            classifier.cleanup()

if __name__ == "__main__":
    main()