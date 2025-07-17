#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
from collections import Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCategoriser:
    def __init__(self, input_dir: str = "dataset", output_dir: str = "categorised"):
        """Initialize the data categoriser with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Category patterns
        self.categories = {
            'malware': r'\b(malware|virus|trojan|ransomware|worm|spyware|botnet)\b',
            'phishing': r'\b(phish|social engineering|scam|fraud|impersonation)\b',
            'network_security': r'\b(firewall|network|packet|router|switch|vpn|dns|ddos)\b',
            'web_security': r'\b(xss|sql injection|csrf|web app|website|browser)\b',
            'cryptography': r'\b(encrypt|decrypt|hash|cipher|key|certificate|ssl|tls)\b',
            'authentication': r'\b(auth|password|login|credential|mfa|2fa|access control)\b',
            'forensics': r'\b(forensic|investigation|analysis|incident response|evidence)\b',
            'compliance': r'\b(compliance|regulation|gdpr|hipaa|pci|iso|policy)\b',
            'iot_security': r'\b(iot|device|sensor|embedded|smart|connected device)\b',
            'cloud_security': r'\b(cloud|aws|azure|gcp|saas|iaas|paas)\b'
        }
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'categorised_entries': 0,
            'multi_category': 0,
            'uncategorised': 0,
            'category_distribution': Counter()
        }

    def categorise_entry(self, text: str) -> List[str]:
        """Categorise a single text entry based on pattern matching."""
        text = text.lower()
        matched_categories = []
        
        for category, pattern in self.categories.items():
            if re.search(pattern, text):
                matched_categories.append(category)
        
        return matched_categories

    def process_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process a single file and categorise its entries."""
        try:
            # Load data based on file type
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = pd.DataFrame(json.load(f))
            elif suffix == '.csv':
                data = pd.read_csv(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None

            # Process each entry
            self.stats['total_entries'] += len(data)
            
            # Combine instruction and response for categorisation
            data['combined_text'] = data['instruction'] + ' ' + data['response']
            
            # Categorise entries
            data['categories'] = data['combined_text'].apply(self.categorise_entry)
            
            # Update statistics
            data['category_count'] = data['categories'].apply(len)
            self.stats['multi_category'] += len(data[data['category_count'] > 1])
            self.stats['uncategorised'] += len(data[data['category_count'] == 0])
            
            # Update category distribution
            for categories in data['categories']:
                self.stats['category_distribution'].update(categories)
            
            # Clean up
            data = data.drop(columns=['combined_text', 'category_count'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def save_results(self, data: pd.DataFrame, original_file: Path):
        """Save categorised data in multiple formats."""
        base_name = original_file.stem + '_categorised'
        
        # Save as JSON
        json_path = self.output_dir / f"{base_name}.json"
        data.to_json(json_path, orient='records', indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_name}.csv"
        data.to_csv(csv_path, index=False)
        
        logger.info(f"Saved categorised data to {json_path} and {csv_path}")

    def generate_report(self):
        """Generate a report of categorisation statistics."""
        report = {
            'total_entries': self.stats['total_entries'],
            'entries_with_multiple_categories': self.stats['multi_category'],
            'uncategorised_entries': self.stats['uncategorised'],
            'category_distribution': dict(self.stats['category_distribution'])
        }
        
        report_path = self.output_dir / 'categorisation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated categorisation report at {report_path}")

    def process_directory(self):
        """Process all files in the input directory."""
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() in {'.json', '.csv'}:
                logger.info(f"Processing {file_path}")
                data = self.process_file(file_path)
                if data is not None:
                    self.save_results(data, file_path)
        
        self.generate_report()
        logger.info("Categorisation complete")

def main():
    """Main function to demonstrate usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Categorise cyber security dataset entries')
    parser.add_argument('--input-dir', default='dataset',
                       help='Input directory containing dataset files')
    parser.add_argument('--output-dir', default='categorised',
                       help='Output directory for categorised data')
    args = parser.parse_args()
    
    categoriser = DataCategoriser(args.input_dir, args.output_dir)
    categoriser.process_directory()

if __name__ == "__main__":
    main() 