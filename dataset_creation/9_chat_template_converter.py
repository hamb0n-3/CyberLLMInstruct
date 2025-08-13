#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import argparse
# Removed rich dependencies for simplicity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatTemplateConverter:
    def __init__(self, input_dir: str = "final_dataset", output_dir: str = "chat_templates"):
        """Initialize the chat template converter with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Removed rich console initialization
        
        # Conversion statistics
        self.stats = {
            'total_entries': 0,
            'converted_entries': 0,
            'failed_entries': 0,
            'format_used': '',
            'files_processed': 0
        }

    def load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load entries from JSONL file."""
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line {line_num} in {file_path}: {str(e)}")
                        self.stats['failed_entries'] += 1
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
        
        return entries

    def convert_to_openai_format(self, entry: Dict) -> Dict:
        """Convert entry to OpenAI chat format.
        Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        try:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": entry.get('instruction', '')
                    },
                    {
                        "role": "assistant",
                        "content": entry.get('response', '')
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error converting to OpenAI format: {str(e)}")
            return None

    def convert_to_sharegpt_format(self, entry: Dict) -> Dict:
        """Convert entry to ShareGPT format.
        Format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        try:
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": entry.get('instruction', '')
                    },
                    {
                        "from": "gpt",
                        "value": entry.get('response', '')
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error converting to ShareGPT format: {str(e)}")
            return None

    def convert_to_alpaca_format(self, entry: Dict) -> Dict:
        """Convert entry to Alpaca format.
        Format: {"instruction": "...", "input": "", "output": "..."}
        """
        try:
            return {
                "instruction": entry.get('instruction', ''),
                "input": "",  # Alpaca format includes an input field, empty for our use case
                "output": entry.get('response', '')
            }
        except Exception as e:
            logger.error(f"Error converting to Alpaca format: {str(e)}")
            return None

    def convert_to_chatml_format(self, entry: Dict) -> Dict:
        """Convert entry to ChatML format with special tokens.
        Format: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
        """
        try:
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            chatml_text = (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
            
            return {
                "text": chatml_text
            }
        except Exception as e:
            logger.error(f"Error converting to ChatML format: {str(e)}")
            return None

    def convert_to_vicuna_format(self, entry: Dict) -> Dict:
        """Convert entry to Vicuna format.
        Format: {"conversations": [{"from": "USER", "value": "..."}, {"from": "ASSISTANT", "value": "..."}]}
        """
        try:
            return {
                "conversations": [
                    {
                        "from": "USER",
                        "value": entry.get('instruction', '')
                    },
                    {
                        "from": "ASSISTANT",
                        "value": entry.get('response', '')
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error converting to Vicuna format: {str(e)}")
            return None

    def convert_to_llama2_format(self, entry: Dict) -> Dict:
        """Convert entry to Llama 2 chat format.
        Format: {"text": "<s>[INST] ... [/INST] ... </s>"}
        """
        try:
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            llama2_text = f"<s>[INST] {instruction} [/INST] {response} </s>"
            
            return {
                "text": llama2_text
            }
        except Exception as e:
            logger.error(f"Error converting to Llama 2 format: {str(e)}")
            return None

    def convert_to_mistral_format(self, entry: Dict) -> Dict:
        """Convert entry to Mistral format.
        Format: {"text": "[INST] ... [/INST] ..."}
        """
        try:
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            mistral_text = f"[INST] {instruction} [/INST] {response}"
            
            return {
                "text": mistral_text
            }
        except Exception as e:
            logger.error(f"Error converting to Mistral format: {str(e)}")
            return None

    def convert_to_qwen_format(self, entry: Dict) -> Dict:
        """Convert entry to Qwen format.
        Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        Note: Qwen uses the same format as OpenAI but can include system messages
        """
        try:
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful cybersecurity expert assistant."
                    },
                    {
                        "role": "user",
                        "content": entry.get('instruction', '')
                    },
                    {
                        "role": "assistant",
                        "content": entry.get('response', '')
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error converting to Qwen format: {str(e)}")
            return None

    def convert_entry(self, entry: Dict, format_type: str) -> Optional[Dict]:
        """Convert a single entry to the specified format."""
        converters = {
            'openai': self.convert_to_openai_format,
            'sharegpt': self.convert_to_sharegpt_format,
            'alpaca': self.convert_to_alpaca_format,
            'chatml': self.convert_to_chatml_format,
            'vicuna': self.convert_to_vicuna_format,
            'llama2': self.convert_to_llama2_format,
            'mistral': self.convert_to_mistral_format,
            'qwen': self.convert_to_qwen_format
        }
        
        converter = converters.get(format_type)
        if not converter:
            logger.error(f"Unknown format type: {format_type}")
            return None
        
        return converter(entry)

    def save_converted_data(self, data: List[Dict], output_path: Path, format_type: str):
        """Save converted data to JSONL file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(data)} entries to {output_path}")
            
            # Also save metadata
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            metadata = {
                'format': format_type,
                'total_entries': len(data),
                'conversion_timestamp': datetime.now().isoformat(),
                'source_format': 'instruction-response pairs',
                'description': f'Cybersecurity dataset converted to {format_type} chat template format'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving converted data: {str(e)}")

    def process_file(self, file_path: Path, format_type: str) -> List[Dict]:
        """Process a single JSONL file and convert to specified format."""
        entries = self.load_jsonl(file_path)
        if not entries:
            logger.warning(f"No entries found in {file_path}")
            return []
        
        converted_entries = []
        
        for entry in entries:
            self.stats['total_entries'] += 1
            converted = self.convert_entry(entry, format_type)
            
            if converted:
                converted_entries.append(converted)
                self.stats['converted_entries'] += 1
            else:
                self.stats['failed_entries'] += 1
        
        return converted_entries

    def convert_dataset(self, format_type: str = 'openai', input_pattern: str = '*.jsonl'):
        """Convert dataset files to specified chat template format."""
        self.stats['format_used'] = format_type
        
        # Get all JSONL files matching pattern
        file_paths = list(self.input_dir.glob(input_pattern))
        
        if not file_paths:
            logger.warning(f"No files found matching pattern '{input_pattern}' in {self.input_dir}")
            return
        
        print(f"\nConverting {len(file_paths)} file(s) to {format_type} format...")
        
        for file_path in file_paths:
            # Skip metadata files
            if 'metadata' in file_path.name:
                continue
            
            print(f"Processing {file_path.name}...")
            
            # Process file
            converted_data = self.process_file(file_path, format_type)
            
            if converted_data:
                # Generate output filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{file_path.stem}_{format_type}_{timestamp}.jsonl"
                output_path = self.output_dir / output_filename
                
                # Save converted data
                self.save_converted_data(converted_data, output_path, format_type)
                self.stats['files_processed'] += 1
        
        # Display summary
        self.display_summary()

    def display_summary(self):
        """Display conversion summary."""
        print(f"\nConversion Complete!")
        print("="*50)
        print("Conversion Summary")
        print("="*50)
        
        print(f"Format Used: {self.stats['format_used']}")
        print(f"Files Processed: {self.stats['files_processed']}")
        print(f"Total Entries: {self.stats['total_entries']}")
        print(f"Converted Entries: {self.stats['converted_entries']}")
        print(f"Failed Entries: {self.stats['failed_entries']}")
        
        if self.stats['total_entries'] > 0:
            success_rate = (self.stats['converted_entries'] / self.stats['total_entries']) * 100
            print(f"Success Rate: {success_rate:.2f}%")
        
        print("="*50)
        # Display output directory
        print(f"\nOutput directory: {self.output_dir}")

def main():
    """Main function to run the chat template converter."""
    parser = argparse.ArgumentParser(
        description='Convert cybersecurity dataset JSONL to various chat template formats'
    )
    parser.add_argument(
        '--input-dir',
        default='final_dataset',
        help='Input directory containing JSONL files from 8_final_assembler.py'
    )
    parser.add_argument(
        '--output-dir',
        default='chat_templates',
        help='Output directory for converted chat template files'
    )
    parser.add_argument(
        '--format',
        choices=['openai', 'sharegpt', 'alpaca', 'chatml', 'vicuna', 'llama2', 'mistral', 'qwen'],
        default='openai',
        help='Chat template format to convert to (default: openai)'
    )
    parser.add_argument(
        '--input-pattern',
        default='*.jsonl',
        help='File pattern to match in input directory (default: *.jsonl)'
    )
    parser.add_argument(
        '--all-formats',
        action='store_true',
        help='Convert to all available formats'
    )
    
    args = parser.parse_args()
    
    converter = ChatTemplateConverter(args.input_dir, args.output_dir)
    
    if args.all_formats:
        # Convert to all available formats
        formats = ['openai', 'sharegpt', 'alpaca', 'chatml', 'vicuna', 'llama2', 'mistral', 'qwen']
        for format_type in formats:
            logger.info(f"\n{'='*50}")
            logger.info(f"Converting to {format_type} format...")
            logger.info(f"{'='*50}")
            
            # Reset stats for each format
            converter.stats = {
                'total_entries': 0,
                'converted_entries': 0,
                'failed_entries': 0,
                'format_used': '',
                'files_processed': 0
            }
            
            converter.convert_dataset(format_type, args.input_pattern)
    else:
        # Convert to single specified format
        converter.convert_dataset(args.format, args.input_pattern)

if __name__ == "__main__":
    main()