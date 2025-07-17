#!/usr/bin/env python3

import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    def __init__(self, input_dir: str = "structured_data", output_dir: str = "domain_classified", 
                 ollama_model: str = "mistral:7b", ollama_port: int = 11434):
        """Initialize the domain classifier with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = f"http://localhost:{ollama_port}/api/generate"
        
        # Define cybersecurity domains
        self.domains = [
            'malware', 'phishing', 'zero_day', 'iot_security', 
            'web_security', 'network_security', 'vulnerability_management',
            'cloud_security', 'cryptography', 'identity_access_management'
        ]

    def call_ollama(self, prompt: str, system_prompt: str, max_retries: int = 2) -> Optional[str]:
        """Call Ollama API with retry mechanism."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for more focused responses
                            "num_predict": 128   # Limit response length
                        }
                    },
                    timeout=15
                )
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt + 1} timed out after 15 seconds")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error in attempt {attempt + 1}: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        return None

    def classify_entry(self, entry: Dict) -> Dict:
        """Classify a single entry using Ollama."""
        # Create prompt for classification
        prompt = f"""Analyze this cybersecurity content and classify it into one or more of these domains: {', '.join(self.domains)}

Content:
Instruction: {entry.get('instruction', '')}
Response: {entry.get('response', '')}

Return your classification as a JSON array of objects with 'domain' and 'confidence' fields, sorted by confidence in descending order.
Example format:
[
    {{"domain": "web_security", "confidence": 0.95}},
    {{"domain": "vulnerability_management", "confidence": 0.75}}
]

Only include domains with confidence > 0.2. Be specific and accurate in your classification."""

        system_prompt = """You are a cybersecurity domain classification expert. Your task is to analyze cybersecurity content and classify it into specific domains. 
        Be precise and technical in your analysis. Only classify into the provided domains. Return results in the exact JSON format specified."""

        response = self.call_ollama(prompt, system_prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    classifications = json.loads(response[json_start:json_end])
                    entry['domains'] = classifications
                    if classifications:
                        entry['primary_domain'] = classifications[0]['domain']
                    else:
                        entry['primary_domain'] = 'uncategorized'
                else:
                    logger.warning(f"Could not parse classification JSON from response: {response}")
                    entry['domains'] = []
                    entry['primary_domain'] = 'uncategorized'
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing classification JSON: {str(e)}")
                entry['domains'] = []
                entry['primary_domain'] = 'uncategorized'
        else:
            entry['domains'] = []
            entry['primary_domain'] = 'uncategorized'
        
        return entry

    def process_directory(self):
        """Process all files in the input directory."""
        all_classified_data = []
        processed_files = 0
        
        for file_path in self.input_dir.glob('*.json'):
            logger.info(f"Processing {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict) or 'data' not in data:
                    logger.warning(f"Invalid data format in {file_path}")
                    continue
                
                # Process each entry
                classified_entries = []
                total_entries = len(data['data'])
                
                for i, entry in enumerate(data['data'], 1):
                    logger.info(f"Classifying entry {i}/{total_entries}")
                    classified_entry = self.classify_entry(entry)
                    classified_entries.append(classified_entry)
                
                # Update the data structure
                data['data'] = classified_entries
                data['metadata']['classification_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save classified data
                output_file = self.output_dir / f"{file_path.stem}_classified.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                processed_files += 1
                all_classified_data.extend(classified_entries)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Log statistics
        domain_counts = {}
        for entry in all_classified_data:
            domain = entry.get('primary_domain', 'uncategorized')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("\nClassification Statistics:")
        logger.info(f"Processed Files: {processed_files}")
        logger.info(f"Total Entries: {len(all_classified_data)}")
        logger.info("\nDomain Distribution:")
        for domain, count in sorted(domain_counts.items()):
            percentage = (count / len(all_classified_data)) * 100
            logger.info(f"{domain}: {count} entries ({percentage:.1f}%)")

def main():
    """Main function to demonstrate usage."""
    classifier = CyberDomainClassifier()
    classifier.process_directory()

if __name__ == "__main__":
    main() 