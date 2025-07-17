#!/usr/bin/env python3

import json
import yaml
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetadataGenerator:
    def __init__(self, dataset_dir: str, output_dir: str = "metadata"):
        """Initialize the metadata generator."""
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
        # Initialize metadata structure
        self.metadata = {
            "dataset_info": {
                "name": "CyberLLMInstruct Dataset",
                "version": datetime.now().strftime("%Y.%m.%d"),
                "creation_date": datetime.now().isoformat(),
                "description": "A comprehensive cyber security instruction-response dataset",
                "license": "MIT",
                "maintainers": []
            },
            "statistics": {
                "total_records": 0,
                "total_size_bytes": 0,
                "formats_available": [],
                "category_distribution": {},
                "subcategory_distribution": {},
                "security_flags_distribution": {},
                "length_statistics": {
                    "instruction": {
                        "min": 0,
                        "max": 0,
                        "mean": 0,
                        "median": 0
                    },
                    "response": {
                        "min": 0,
                        "max": 0,
                        "mean": 0,
                        "median": 0
                    }
                }
            },
            "data_sources": {
                "source_files": [],
                "processing_steps": []
            },
            "schema": {
                "instruction": {"type": "string", "description": "The cyber security-related instruction or query"},
                "response": {"type": "string", "description": "The corresponding response or solution"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "category": "string",
                        "subcategory": "string",
                        "security_flags": "object",
                        "source": "string",
                        "timestamp": "string"
                    }
                }
            }
        }

    def load_dataset(self, file_path: Path) -> Any:
        """Load dataset from various formats."""
        suffix = file_path.suffix.lower()
        try:
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif suffix in {'.yaml', '.yml'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif suffix == '.csv':
                return pd.read_csv(file_path).to_dict('records')
            elif suffix == '.parquet':
                return pd.read_parquet(file_path).to_dict('records')
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def analyze_text_lengths(self, data: List[Dict]) -> Dict:
        """Analyze text lengths of instructions and responses."""
        instructions = [len(entry['instruction']) for entry in data]
        responses = [len(entry['response']) for entry in data]
        
        return {
            "instruction": {
                "min": min(instructions),
                "max": max(instructions),
                "mean": sum(instructions) / len(instructions),
                "median": sorted(instructions)[len(instructions)//2]
            },
            "response": {
                "min": min(responses),
                "max": max(responses),
                "mean": sum(responses) / len(responses),
                "median": sorted(responses)[len(responses)//2]
            }
        }

    def analyze_categories(self, data: List[Dict]) -> Dict:
        """Analyze category and subcategory distribution."""
        categories = Counter()
        subcategories = Counter()
        security_flags = Counter()
        
        for entry in data:
            metadata = entry.get('metadata', {})
            if 'category' in metadata:
                categories[metadata['category']] += 1
            if 'subcategory' in metadata:
                subcategories[metadata['subcategory']] += 1
            if 'security_flags' in metadata:
                for flag, value in metadata['security_flags'].items():
                    if value:
                        security_flags[flag] += 1
        
        return {
            "categories": dict(categories),
            "subcategories": dict(subcategories),
            "security_flags": dict(security_flags)
        }

    def generate_visualizations(self):
        """Generate visualizations for the dataset statistics."""
        # Set style
        plt.style.use('seaborn')
        
        # Category Distribution
        plt.figure(figsize=(12, 6))
        categories = self.metadata['statistics']['category_distribution']
        plt.bar(categories.keys(), categories.values())
        plt.title('Category Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distribution.png')
        plt.close()
        
        # Text Length Distribution
        plt.figure(figsize=(12, 6))
        length_stats = self.metadata['statistics']['length_statistics']
        data = {
            'Instruction': [length_stats['instruction']['min'], 
                          length_stats['instruction']['mean'], 
                          length_stats['instruction']['max']],
            'Response': [length_stats['response']['min'], 
                        length_stats['response']['mean'], 
                        length_stats['response']['max']]
        }
        df = pd.DataFrame(data, index=['Min', 'Mean', 'Max'])
        df.plot(kind='bar')
        plt.title('Text Length Statistics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_statistics.png')
        plt.close()

    def generate_markdown_report(self):
        """Generate a comprehensive markdown report."""
        report = f"""# CyberLLMInstruct Dataset Documentation

## Dataset Overview
- **Name**: {self.metadata['dataset_info']['name']}
- **Version**: {self.metadata['dataset_info']['version']}
- **Creation Date**: {self.metadata['dataset_info']['creation_date']}
- **License**: {self.metadata['dataset_info']['license']}

## Dataset Statistics
- **Total Records**: {self.metadata['statistics']['total_records']:,}
- **Total Size**: {self.metadata['statistics']['total_size_bytes'] / 1024 / 1024:.2f} MB
- **Available Formats**: {', '.join(self.metadata['statistics']['formats_available'])}

### Category Distribution
```
{yaml.dump(self.metadata['statistics']['category_distribution'], default_flow_style=False)}
```

### Text Length Statistics
#### Instructions
- Minimum: {self.metadata['statistics']['length_statistics']['instruction']['min']:,} characters
- Maximum: {self.metadata['statistics']['length_statistics']['instruction']['max']:,} characters
- Mean: {self.metadata['statistics']['length_statistics']['instruction']['mean']:.1f} characters
- Median: {self.metadata['statistics']['length_statistics']['instruction']['median']:,} characters

#### Responses
- Minimum: {self.metadata['statistics']['length_statistics']['response']['min']:,} characters
- Maximum: {self.metadata['statistics']['length_statistics']['response']['max']:,} characters
- Mean: {self.metadata['statistics']['length_statistics']['response']['mean']:.1f} characters
- Median: {self.metadata['statistics']['length_statistics']['response']['median']:,} characters

## Data Sources
- **Source Files**: {len(self.metadata['data_sources']['source_files'])} files
- **Processing Steps**: {len(self.metadata['data_sources']['processing_steps'])} steps

## Schema
```yaml
{yaml.dump(self.metadata['schema'], default_flow_style=False)}
```

## Visualizations
![Category Distribution](category_distribution.png)
![Length Statistics](length_statistics.png)

## Usage Guidelines
1. This dataset is designed for training and evaluating AI models in cyber security contexts
2. Each entry contains an instruction-response pair with associated metadata
3. Security flags indicate special handling requirements for certain entries
4. Always check security flags before using entries in production environments

## Citation
If you use this dataset in your research, please cite it as:
```
@dataset{{cyberllm2024,
  title={{CyberLLMInstruct Dataset}},
  version={self.metadata['dataset_info']['version']},
  year={{2024}},
  url={{https://github.com/yourusername/cyberllm}}
}}
```
"""
        
        # Save markdown report
        report_path = self.output_dir / 'dataset_documentation.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Display in console
        self.console.print(Markdown(report))

    def process_dataset(self):
        """Process the dataset and generate metadata."""
        # Process all dataset files
        for file_path in self.dataset_dir.glob('*.*'):
            if file_path.suffix.lower() in {'.json', '.yaml', '.yml', '.csv', '.parquet'}:
                logger.info(f"Processing {file_path}")
                
                # Load data
                data = self.load_dataset(file_path)
                if not data:
                    continue
                
                # Update formats available
                self.metadata['statistics']['formats_available'].append(file_path.suffix)
                self.metadata['statistics']['formats_available'] = list(set(
                    self.metadata['statistics']['formats_available']
                ))
                
                # Update source files
                self.metadata['data_sources']['source_files'].append(str(file_path))
                
                # Update statistics
                self.metadata['statistics']['total_records'] += len(data)
                self.metadata['statistics']['total_size_bytes'] += file_path.stat().st_size
                
                # Analyze text lengths
                length_stats = self.analyze_text_lengths(data)
                self.metadata['statistics']['length_statistics'] = length_stats
                
                # Analyze categories
                category_stats = self.analyze_categories(data)
                self.metadata['statistics']['category_distribution'] = category_stats['categories']
                self.metadata['statistics']['subcategory_distribution'] = category_stats['subcategories']
                self.metadata['statistics']['security_flags_distribution'] = category_stats['security_flags']
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save metadata
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        yaml_path = self.output_dir / 'dataset_metadata.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
        
        # Generate documentation
        self.generate_markdown_report()
        
        logger.info(f"Metadata and documentation generated in {self.output_dir}")

def main():
    """Main function to demonstrate usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate metadata and documentation for cyber security dataset')
    parser.add_argument('--dataset-dir', required=True,
                       help='Directory containing the dataset files')
    parser.add_argument('--output-dir', default='metadata',
                       help='Output directory for metadata and documentation')
    args = parser.parse_args()
    
    generator = MetadataGenerator(args.dataset_dir, args.output_dir)
    generator.process_dataset()

if __name__ == "__main__":
    main() 