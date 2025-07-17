#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Union
from datetime import datetime
import hashlib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import jsonschema
from concurrent.futures import ThreadPoolExecutor
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetAssembler:
    def __init__(self, input_dir: str = "security_aligned", output_dir: str = "final_dataset"):
        """Initialize the dataset assembler with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rich console
        self.console = Console()
        
        # Dataset schema for validation
        self.dataset_schema = {
            "type": "object",
            "required": ["instruction", "response"],
            "properties": {
                "instruction": {"type": "string", "minLength": 1},
                "response": {"type": "string", "minLength": 1},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "subcategory": {"type": "string"},
                        "security_flags": {
                            "type": "object",
                            "properties": {
                                "review_required": {"type": "boolean"},
                                "isolation_required": {"type": "boolean"}
                            }
                        },
                        "enhancement_history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "enhancement_type": {"type": "string"},
                                    "timestamp": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Assembly statistics
        self.stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'duplicate_entries': 0,
            'invalid_entries': 0,
            'sources': set(),
            'categories': set(),
            'entry_lengths': {
                'instruction': [],
                'response': []
            }
        }

    def load_data(self, file_path: Path) -> Union[Dict, List, None]:
        """Load data from JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both direct lists and data field
                if isinstance(data, dict) and 'data' in data:
                    return data['data']
                return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def validate_entry(self, entry: Dict) -> bool:
        """Validate a single entry against the schema."""
        try:
            # Handle entries that are nested in a 'data' field
            if isinstance(entry, dict) and 'data' in entry and isinstance(entry['data'], list):
                return True
                
            # Handle entries that are nested in a 'data' field as a single item
            if isinstance(entry, dict) and 'data' in entry and isinstance(entry['data'], dict):
                return True
                
            # Basic validation for direct entries
            if not isinstance(entry, dict):
                return False
                
            # Check for required fields
            if 'instruction' not in entry or 'response' not in entry:
                return False
                
            # Validate instruction and response are non-empty strings
            if not isinstance(entry['instruction'], str) or not entry['instruction'].strip():
                return False
            if not isinstance(entry['response'], str) or not entry['response'].strip():
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating entry: {str(e)}")
            return False

    def generate_entry_hash(self, entry: Dict) -> str:
        """Generate a unique hash for an entry based on its content."""
        # Create a normalized string representation of instruction and response
        content = (
            entry.get('instruction', '').lower().strip() +
            entry.get('response', '').lower().strip()
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def process_entry(self, entry: Dict, source_file: Path) -> List[Dict]:
        """Process and validate a single entry."""
        try:
            processed_entries = []
            
            # Handle entries that are nested in a 'data' field
            if isinstance(entry, dict) and 'data' in entry:
                if isinstance(entry['data'], list):
                    # Process each item in the data list
                    for item in entry['data']:
                        processed = self.process_single_entry(item, source_file)
                        if processed:
                            processed_entries.extend(processed if isinstance(processed, list) else [processed])
                elif isinstance(entry['data'], dict):
                    # Process the single data item
                    processed = self.process_single_entry(entry['data'], source_file)
                    if processed:
                        processed_entries.extend(processed if isinstance(processed, list) else [processed])
            else:
                # Process direct entry
                processed = self.process_single_entry(entry, source_file)
                if processed:
                    processed_entries.extend(processed if isinstance(processed, list) else [processed])
            
            return processed_entries
            
        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            self.stats['invalid_entries'] += 1
            return []

    def process_single_entry(self, entry: Dict, source_file: Path) -> Optional[Dict]:
        """Process a single entry and normalize it to instruction-response format."""
        try:
            # Skip entries without instruction or response
            if not isinstance(entry, dict):
                return None
                
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            # Handle complex response structures
            if isinstance(response, dict):
                # Convert dictionary response to formatted string
                formatted_response = []
                for key, value in response.items():
                    if isinstance(value, (list, dict)):
                        formatted_response.append(f"**{key}:**\n{json.dumps(value, indent=2)}")
                    else:
                        formatted_response.append(f"**{key}:** {value}")
                response = "\n\n".join(formatted_response)
            
            # Clean and validate text
            instruction = self.clean_text(instruction)
            response = self.clean_text(response)
            
            if not instruction or not response:
                self.stats['invalid_entries'] += 1
                return None
            
            # Update statistics
            if 'category' in entry.get('metadata', {}):
                self.stats['categories'].add(entry['metadata']['category'])
            
            self.stats['entry_lengths']['instruction'].append(len(instruction))
            self.stats['entry_lengths']['response'].append(len(response))
            
            return {
                'instruction': instruction,
                'response': response
            }
            
        except Exception as e:
            logger.error(f"Error processing single entry: {str(e)}")
            self.stats['invalid_entries'] += 1
            return None

    def remove_duplicates(self, entries: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on content hash."""
        unique_entries = {}
        duplicates = 0
        
        # Flatten the list of entries if needed
        flattened_entries = []
        for entry in entries:
            if isinstance(entry, list):
                flattened_entries.extend(entry)
            else:
                flattened_entries.append(entry)
        
        for entry in flattened_entries:
            if not isinstance(entry, dict):
                continue
                
            entry_hash = self.generate_entry_hash(entry)
            if entry_hash not in unique_entries:
                unique_entries[entry_hash] = entry
            else:
                duplicates += 1
        
        self.stats['duplicate_entries'] = duplicates
        return list(unique_entries.values())

    def save_dataset(self, data: List[Dict]):
        """Save dataset as JSON with only instruction-response pairs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"final_cybersecurity_dataset_{timestamp}.json"
        
        # Flatten the list of entries if needed
        flattened_data = []
        for entry in data:
            if isinstance(entry, list):
                flattened_data.extend(entry)
            else:
                flattened_data.append(entry)
        
        # Save the clean dataset
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(flattened_data, f, indent=2)
        logger.info(f"Saved final dataset to {json_path}")

    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file."""
        data = self.load_data(file_path)
        if not data:
            return []
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        self.stats['total_entries'] += len(data)
        self.stats['sources'].add(str(file_path))
        
        # Process entries
        processed_entries = []
        for entry in data:
            processed_entry = self.process_entry(entry, file_path)
            if processed_entry:
                processed_entries.extend(processed_entry)
                self.stats['valid_entries'] += 1
        
        return processed_entries

    def assemble_dataset(self):
        """Assemble the final dataset from all JSON files in security_aligned."""
        all_entries = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            # Get all JSON files
            file_paths = list(self.input_dir.glob('*.json'))
            
            task = progress.add_task("Processing files...", total=len(file_paths))
            
            # Process files in parallel
            with ThreadPoolExecutor() as executor:
                for entries in executor.map(self.process_file, file_paths):
                    if isinstance(entries, list):
                        all_entries.extend(entries)
                    elif entries is not None:
                        all_entries.append(entries)
                    progress.update(task, advance=1)
        
        # Remove duplicates
        unique_entries = self.remove_duplicates(all_entries)
        
        # Save final dataset
        self.save_dataset(unique_entries)
        
        # Print summary
        self.console.print(f"\n[green]Dataset Assembly Complete![/green]")
        self.console.print(f"Total entries processed: {self.stats['total_entries']}")
        self.console.print(f"Valid entries: {len(unique_entries)}")
        self.console.print(f"Duplicate entries removed: {self.stats['duplicate_entries']}")
        self.console.print(f"Invalid entries: {self.stats['invalid_entries']}")
        self.console.print(f"Number of categories: {len(self.stats['categories'])}")
        
        return unique_entries

def main():
    """Main function to demonstrate usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Assemble final cybersecurity dataset')
    parser.add_argument('--input-dir', default='security_aligned',
                       help='Input directory containing processed data')
    parser.add_argument('--output-dir', default='final_dataset',
                       help='Output directory for final dataset')
    args = parser.parse_args()
    
    assembler = DatasetAssembler(args.input_dir, args.output_dir)
    assembler.assemble_dataset()

if __name__ == "__main__":
    main() 