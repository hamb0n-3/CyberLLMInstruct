#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Tuple
from datetime import datetime
import hashlib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import pandas as pd
import argparse
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetAssembler:
    def __init__(self, input_dir: str = "structured_data", output_dir: str = "final_dataset"):
        """Initialize the dataset assembler with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rich console
        self.console = Console()
        
        # Assembly statistics
        self.stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'duplicate_entries': 0,
            'invalid_entries': 0,
            'sources': set(),
            'entry_lengths': {
                'instruction': [],
                'response': []
            }
        }

    def load_data(self, file_path: Path) -> Optional[Dict]:
        """Load data from JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def generate_entry_hash(self, entry: Dict) -> str:
        """Generate a unique hash for an entry based on its content."""
        # Create a normalized string representation of instruction and response
        content = (
            entry.get('instruction', '').lower().strip() +
            entry.get('response', '').lower().strip()
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def process_consolidated_file(self, file_path: Path) -> List[Dict]:
        """Process a consolidated dataset file from 3_data_structurer.py."""
        processed_entries = []
        
        data = self.load_data(file_path)
        if not data:
            return processed_entries
        
        # Extract metadata
        metadata = data.get('metadata', {})
        self.stats['sources'].add(metadata.get('model_used', 'unknown'))
        
        # Extract entries from the data field
        entries = data.get('data', [])
        if not isinstance(entries, list):
            logger.warning(f"Data field is not a list in {file_path}")
            return processed_entries
        
        for entry in entries:
            try:
                if not isinstance(entry, dict):
                    self.stats['invalid_entries'] += 1
                    continue
                
                instruction = self.clean_text(entry.get('instruction', ''))
                response = self.clean_text(entry.get('response', ''))
                
                # Skip entries without instruction or response
                if not instruction or not response:
                    self.stats['invalid_entries'] += 1
                    continue
                
                # Track statistics
                self.stats['entry_lengths']['instruction'].append(len(instruction))
                self.stats['entry_lengths']['response'].append(len(response))
                
                # Create clean entry with only instruction and response
                clean_entry = {
                    'instruction': instruction,
                    'response': response
                }
                
                processed_entries.append(clean_entry)
                self.stats['total_entries'] += 1
                
            except Exception as e:
                logger.error(f"Error processing entry: {str(e)}")
                self.stats['invalid_entries'] += 1
        
        return processed_entries

    def remove_duplicates(self, entries: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on content hash."""
        unique_entries = {}
        duplicates = 0
        
        for entry in entries:
            entry_hash = self.generate_entry_hash(entry)
            if entry_hash not in unique_entries:
                unique_entries[entry_hash] = entry
            else:
                duplicates += 1
        
        self.stats['duplicate_entries'] = duplicates
        self.stats['valid_entries'] = len(unique_entries)
        return list(unique_entries.values())

    def save_dataset_json(self, data: List[Dict], timestamp: str):
        """Save dataset as JSON."""
        json_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON dataset to {json_path}")
        return json_path

    def save_dataset_jsonl(self, data: List[Dict], timestamp: str):
        """Save dataset as JSONL (one JSON object per line)."""
        jsonl_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}.jsonl"
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved JSONL dataset to {jsonl_path}")
        return jsonl_path

    def save_dataset_csv(self, data: List[Dict], timestamp: str):
        """Save dataset as CSV."""
        csv_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}.csv"
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV with proper escaping
        df.to_csv(csv_path, index=False, encoding='utf-8', escapechar='\\')
        
        logger.info(f"Saved CSV dataset to {csv_path}")
        return csv_path

    def save_dataset_huggingface(self, data: List[Dict], timestamp: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Save dataset in HuggingFace format with train/validation/test splits."""
        hf_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}_hf"
        
        # Validate split ratios
        if not abs(sum(split_ratios) - 1.0) < 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Calculate split indices
        total_size = len(data)
        train_size = int(total_size * split_ratios[0])
        val_size = int(total_size * split_ratios[1])
        
        # Create splits
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        # Save to disk
        dataset_dict.save_to_disk(str(hf_path))
        
        # Log split information
        logger.info(f"Saved HuggingFace dataset to {hf_path}")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return hf_path

    def save_dataset_mlx(self, data: List[Dict], timestamp: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Save dataset in MLX format with train/valid/test JSONL files."""
        mlx_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}_mlx"
        mlx_path.mkdir(parents=True, exist_ok=True)
        
        # Validate split ratios
        if not abs(sum(split_ratios) - 1.0) < 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Calculate split indices
        total_size = len(data)
        train_size = int(total_size * split_ratios[0])
        val_size = int(total_size * split_ratios[1])
        
        # Create splits
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Save each split as JSONL with MLX expected names
        train_path = mlx_path / "train.jsonl"
        valid_path = mlx_path / "valid.jsonl"
        test_path = mlx_path / "test.jsonl"
        
        # Write train.jsonl
        with open(train_path, 'w', encoding='utf-8') as f:
            for entry in train_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Write valid.jsonl
        with open(valid_path, 'w', encoding='utf-8') as f:
            for entry in val_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Write test.jsonl
        with open(test_path, 'w', encoding='utf-8') as f:
            for entry in test_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Log split information
        logger.info(f"Saved MLX dataset to {mlx_path}")
        logger.info(f"  train.jsonl: {len(train_data)} samples")
        logger.info(f"  valid.jsonl: {len(val_data)} samples")
        logger.info(f"  test.jsonl: {len(test_data)} samples")
        
        return mlx_path

    def generate_summary_report(self, data: List[Dict]) -> Dict:
        """Generate a summary report of the dataset."""
        report = {
            'total_entries': len(data),
            'duplicates_removed': self.stats['duplicate_entries'],
            'invalid_entries': self.stats['invalid_entries'],
            'average_instruction_length': sum(self.stats['entry_lengths']['instruction']) / len(self.stats['entry_lengths']['instruction']) if self.stats['entry_lengths']['instruction'] else 0,
            'average_response_length': sum(self.stats['entry_lengths']['response']) / len(self.stats['entry_lengths']['response']) if self.stats['entry_lengths']['response'] else 0,
            'models_used': list(self.stats['sources'])
        }
        return report

    def assemble_dataset(self, output_formats: List[str] = ['json'], split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Assemble the final dataset from consolidated files in structured_data."""
        all_entries = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            # Get all consolidated dataset files
            file_pattern = 'consolidated_cybersecurity_dataset_*.json'
            file_paths = list(self.input_dir.glob(file_pattern))
            
            if not file_paths:
                logger.warning(f"No '{file_pattern}' files found in {self.input_dir}")
                return
            
            task = progress.add_task(f"Processing {len(file_paths)} files...", total=len(file_paths))
            
            # Process each file
            for file_path in file_paths:
                entries = self.process_consolidated_file(file_path)
                all_entries.extend(entries)
                progress.update(task, advance=1)
        
        if not all_entries:
            logger.warning("No valid entries found to process")
            return
        
        # Remove duplicates
        unique_entries = self.remove_duplicates(all_entries)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save in requested formats
        saved_files = []
        if 'json' in output_formats:
            saved_files.append(self.save_dataset_json(unique_entries, timestamp))
        if 'jsonl' in output_formats:
            saved_files.append(self.save_dataset_jsonl(unique_entries, timestamp))
        if 'csv' in output_formats:
            saved_files.append(self.save_dataset_csv(unique_entries, timestamp))
        if 'huggingface' in output_formats:
            saved_files.append(self.save_dataset_huggingface(unique_entries, timestamp, split_ratios))
        if 'mlx' in output_formats:
            saved_files.append(self.save_dataset_mlx(unique_entries, timestamp, split_ratios))
        
        # Generate and display summary report
        report = self.generate_summary_report(unique_entries)
        
        # Display summary
        self.console.print(f"\n[green]Dataset Assembly Complete![/green]")
        
        # Create summary table
        table = Table(title="Assembly Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total entries processed", str(self.stats['total_entries']))
        table.add_row("Valid entries", str(report['total_entries']))
        table.add_row("Duplicate entries removed", str(report['duplicates_removed']))
        table.add_row("Invalid entries", str(report['invalid_entries']))
        table.add_row("Average instruction length", f"{report['average_instruction_length']:.0f} chars")
        table.add_row("Average response length", f"{report['average_response_length']:.0f} chars")
        table.add_row("Models used", ", ".join(report['models_used']))
        
        self.console.print(table)
        
        # Display saved files
        self.console.print("\n[bold]Output files:[/bold]")
        for file_path in saved_files:
            self.console.print(f"  • {file_path}")
        
        # Save summary report
        report_path = self.output_dir / f"assembly_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        self.console.print(f"  • {report_path} (summary report)")
        
        return unique_entries

def main():
    """Main function to assemble the dataset."""
    parser = argparse.ArgumentParser(description='Assemble final cybersecurity dataset from structured data')
    parser.add_argument('--input-dir', default='structured_data',
                       help='Input directory containing consolidated datasets from 3_data_structurer.py')
    parser.add_argument('--output-dir', default='final_dataset',
                       help='Output directory for final dataset')
    parser.add_argument('--formats', nargs='+', default=['json'],
                       choices=['json', 'jsonl', 'csv', 'huggingface', 'mlx'],
                       help='Output formats (default: json)')
    parser.add_argument('--split-ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                       help='Train/validation/test split ratios for HuggingFace format (default: 0.8 0.1 0.1)')
    args = parser.parse_args()
    
    # Validate split ratios
    if args.split_ratios and abs(sum(args.split_ratios) - 1.0) > 0.001:
        parser.error("Split ratios must sum to 1.0")
    
    assembler = DatasetAssembler(args.input_dir, args.output_dir)
    assembler.assemble_dataset(output_formats=args.formats, split_ratios=tuple(args.split_ratios))

if __name__ == "__main__":
    main()