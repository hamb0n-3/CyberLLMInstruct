#!/usr/bin/env python3

import pandas as pd
import json
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
from huggingface_hub import HfApi, DatasetCard
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetExporter:
    def __init__(self, input_dir: str = "dataset", output_dir: str = "release"):
        """Initialize the dataset exporter."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset metadata
        self.metadata = {
            'name': 'CyberLLMInstruct',
            'version': '1.0.0',
            'description': 'A dataset for analysing safety of fine-tuned LLMs using cyber security data',
            'homepage': 'https://github.com/yourusername/CyberLLMInstruct',
            'license': 'MIT',
            'citation': '',
            'author': '',
            'created_at': datetime.now().isoformat(),
            'stats': {
                'total_entries': 0,
                'file_count': 0,
                'category_distribution': {}
            }
        }

    def load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return pd.DataFrame(json.load(f))
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def generate_dataset_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash of the dataset content."""
        content = data.to_json(orient='records').encode()
        return hashlib.sha256(content).hexdigest()

    def update_metadata(self, data: pd.DataFrame):
        """Update metadata with dataset statistics."""
        self.metadata['stats']['total_entries'] += len(data)
        self.metadata['stats']['file_count'] += 1
        
        if 'categories' in data.columns:
            category_counts = {}
            for categories in data['categories']:
                if isinstance(categories, list):
                    for category in categories:
                        category_counts[category] = category_counts.get(category, 0) + 1
            self.metadata['stats']['category_distribution'].update(category_counts)

    def _read_dataset(self, file_path: Path) -> pd.DataFrame:
        """Read dataset from file."""
        suffix = file_path.suffix.lower()
        if suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def export_dataset(self, data: pd.DataFrame, base_name: str, format: str = 'json'):
        """Export dataset to specified format."""
        if format == 'json':
            json_path = self.output_dir / f"{base_name}.json"
            data.to_json(json_path, orient='records', indent=2)
            logger.info(f"Exported JSON dataset to {json_path}")
        elif format == 'parquet':
            parquet_path = self.output_dir / f"{base_name}.parquet"
            data.to_parquet(parquet_path, index=False)
            logger.info(f"Exported Parquet dataset to {parquet_path}")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def generate_dataset_card(self):
        """Generate a dataset card for Hugging Face."""
        card_content = f"""---
language:
- en
license: {self.metadata['license']}
pretty_name: {self.metadata['name']}
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- text-classification
---

# CyberLLMInstruct Dataset

{self.metadata['description']}

## Dataset Summary

A comprehensive dataset for training and evaluating language models on cyber security tasks.

## Dataset Structure

### Data Fields

- `instruction`: The input prompt or question
- `response`: The corresponding response or answer
- `categories`: List of cyber security categories
- `metadata`: Additional information about the entry

### Data Splits

- Training set: {self.metadata['stats']['total_entries']} examples

### Statistics

- Total entries: {self.metadata['stats']['total_entries']}
- Number of files: {self.metadata['stats']['file_count']}
- Category distribution:
{yaml.dump(self.metadata['stats']['category_distribution'], indent=2)}

## Additional Information

### Licensing Information

{self.metadata['license']}

### Citation Information

{self.metadata['citation']}

### Contributions

{self.metadata['author']}
"""
        card_path = self.output_dir / 'README.md'
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        logger.info(f"Generated dataset card at {card_path}")

    def upload_to_huggingface(self, repo_id: str, token: str):
        """Upload dataset to Hugging Face."""
        try:
            api = HfApi()
            
            # Create or get repository
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                token=token
            )
            
            # Upload files
            for file_path in self.output_dir.glob('*.*'):
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
            
            logger.info(f"Successfully uploaded dataset to {repo_id}")
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {str(e)}")

    def process_directory(self):
        """Process all files and prepare dataset for release."""
        all_data = []
        
        # Load and combine all data
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() in {'.json', '.parquet'}:
                logger.info(f"Processing {file_path}")
                data = self.load_data(file_path)
                if data is not None:
                    all_data.append(data)
                    self.update_metadata(data)
        
        if not all_data:
            logger.error("No valid data files found")
            return
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Generate dataset hash
        self.metadata['hash'] = self.generate_dataset_hash(combined_data)
        
        # Export in all formats
        for format in ['json', 'parquet']:
            self.export_dataset(combined_data, format)
        
        # Generate dataset card
        self.generate_dataset_card()
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("Dataset export complete")

def main():
    """Main function to demonstrate usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Export cyber security dataset for release')
    parser.add_argument('--input-dir', default='dataset',
                       help='Input directory containing dataset files')
    parser.add_argument('--output-dir', default='release',
                       help='Output directory for release files')
    parser.add_argument('--hf-repo', help='Hugging Face repository ID')
    parser.add_argument('--hf-token', help='Hugging Face API token')
    args = parser.parse_args()
    
    exporter = DatasetExporter(args.input_dir, args.output_dir)
    exporter.process_directory()
    
    if args.hf_repo and args.hf_token:
        exporter.upload_to_huggingface(args.hf_repo, args.hf_token)

if __name__ == "__main__":
    main()