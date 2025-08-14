#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import argparse
import random

# Try to import MLX - graceful fallback if not available
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    mx.set_default_device(mx.gpu)
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MLX not available - will run without LLM reformulation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatTemplateConverter:
    def __init__(self, input_dir: str = "final_dataset", output_dir: str = "chat_templates", 
                 use_llm_reformulation: bool = True):
        """Initialize the chat template converter with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM for reformulation
        self.use_llm_reformulation = use_llm_reformulation and MLX_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.sampler = None
        self.model_path = "mlx-community/Qwen3-8B-4bit-DWQ-053125"
        
        # Sampling parameters
        self.temperature = 0.5  # Lower temperature for more consistent reformulations
        self.top_p = 0.95
        self.top_k = 20
        self.repetition_penalty = 1.0
        
        if self.use_llm_reformulation:
            self.initialize_llm()
        
        # Conversion statistics
        self.stats = {
            'total_entries': 0,
            'converted_entries': 0,
            'failed_entries': 0,
            'format_used': '',
            'files_processed': 0,
            'reformulated_entries': 0
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

    def initialize_llm(self):
        """Initialize the MLX LLM for reformulation."""
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - cannot initialize LLM")
            self.use_llm_reformulation = False
            return
            
        try:
            logger.info(f"Loading LLM model: {self.model_path}")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={"trust_remote_code": True}
            )
            
            # Create sampler with configured parameters
            self.sampler = make_sampler(
                temp=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=0.0,
                min_tokens_to_keep=1
            )
            
            logger.info(f"LLM model loaded successfully")
            logger.info(f"Sampler configured: temp={self.temperature}, top_p={self.top_p}, "
                       f"top_k={self.top_k}, rep_penalty={self.repetition_penalty}")
                       
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            logger.warning("Continuing without LLM reformulation")
            self.use_llm_reformulation = False
            self.model = None
            self.tokenizer = None
            self.sampler = None
    
    def reformulate_instruction(self, instruction: str, max_retries: int = 2) -> str:
        """Reformulate an instruction using the LLM while maintaining technical accuracy."""
        if not self.use_llm_reformulation or not self.model:
            return instruction
         
        prompt = f"""You are a cybersecurity expert. Rephrase the following cybersecurity question while maintaining the exact same technical meaning, complexity, and scope. Make it sound natural but different.

Original question: {instruction}

Rephrased question (provide only the rephrased question, no explanation):"""
        
        try:
            # Apply chat template if available
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # We want direct response, not thinking process
                )
            else:
                formatted_prompt = prompt
            
            # Build logits processors for repetition penalty if needed
            logits_processors = []
            if self.repetition_penalty != 1.0:
                rep_penalty_processor = make_repetition_penalty(
                    penalty=self.repetition_penalty,
                    context_size=20
                )
                logits_processors.append(rep_penalty_processor)
            
            # Generate reformulation
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=1500,  # Reformulated questions should be concise
                verbose=False,
                sampler=self.sampler,
                logits_processors=logits_processors if logits_processors else None,
                kv_bits=8,
                kv_group_size=64
            )
            
            # Extract the reformulated instruction
            reformulated = response.strip()
            
            # Basic validation - ensure it's not empty and different from original
            if reformulated and reformulated != instruction and len(reformulated) > 20:
                self.stats['reformulated_entries'] += 1
                return reformulated
            else:
                print(f"    → Keeping original (reformulation failed validation)")
                return instruction
                
        except Exception as e:
            logger.warning(f"Failed to reformulate instruction: {str(e)}")
            print(f"    → Error during reformulation, keeping original")
            return instruction
    
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

    def create_validation_test_sets(self, data: List[Dict], validation_size: float = 0.1, 
                                   test_size: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train, validation, and test sets with reformulated questions for val/test."""
        random.seed(seed)
        
        # Calculate sizes
        total_size = len(data)
        val_size = int(total_size * validation_size)
        test_size = int(total_size * test_size)
        
        # Sample indices for validation and test (non-overlapping)
        all_indices = list(range(total_size))
        random.shuffle(all_indices)
        
        val_indices = set(all_indices[:val_size])
        test_indices = set(all_indices[val_size:val_size + test_size])
        
        # Create validation set with reformulated instructions
        validation_data = []
        for i, idx in enumerate(val_indices, 1):
            entry = data[idx].copy()
            
            # Reformulate instruction for validation set
            if 'messages' in entry:
                # For formats with messages structure
                for msg in entry['messages']:
                    if msg.get('role') == 'user':
                        original = msg['content']
                        msg['content'] = self.reformulate_instruction(original)
            elif 'conversations' in entry:
                # For ShareGPT/Vicuna formats
                for conv in entry['conversations']:
                    if conv.get('from') in ['human', 'USER']:
                        original = conv['value']
                        conv['value'] = self.reformulate_instruction(original)
            elif 'instruction' in entry:
                # For Alpaca format
                original = entry['instruction']
                entry['instruction'] = self.reformulate_instruction(original)
            elif 'text' in entry:
                # For text-based formats (ChatML, Llama2, Mistral)
                # These need special handling as they're formatted strings
                pass  # Keep original for now, reformulation would break formatting
            
            validation_data.append(entry)
            
            # Show progress
            if i % 5 == 0 or i == len(val_indices):
                percent = (i / len(val_indices)) * 100
                bar_length = 40
                filled = int(bar_length * i / len(val_indices))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  Validation: [{bar}] {percent:.1f}% ({i}/{len(val_indices)})")
        
        # Create test set with reformulated instructions
        test_data = []
        for i, idx in enumerate(test_indices, 1):
            entry = data[idx].copy()
            
            # Reformulate instruction for test set
            if 'messages' in entry:
                for msg in entry['messages']:
                    if msg.get('role') == 'user':
                        original = msg['content']
                        msg['content'] = self.reformulate_instruction(original)
            elif 'conversations' in entry:
                for conv in entry['conversations']:
                    if conv.get('from') in ['human', 'USER']:
                        original = conv['value']
                        conv['value'] = self.reformulate_instruction(original)
            elif 'instruction' in entry:
                original = entry['instruction']
                entry['instruction'] = self.reformulate_instruction(original)
            
            test_data.append(entry)
            
            # Show progress
            if i % 5 == 0 or i == len(test_indices):
                percent = (i / len(test_indices)) * 100
                bar_length = 40
                filled = int(bar_length * i / len(test_indices))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  Test:       [{bar}] {percent:.1f}% ({i}/{len(test_indices)})")
        
        # Show reformulation summary
        if self.use_llm_reformulation:
            print(f"\nReformulation complete!")
            print(f"  Total reformulated entries: {self.stats['reformulated_entries']}")
        else:
            print(f"\nNo reformulation performed (LLM not available)")
        
        # Training data is all the original data
        train_data = data
        
        return train_data, validation_data, test_data
    
    def save_split_data(self, train_data: List[Dict], val_data: List[Dict], 
                       test_data: List[Dict], format_dir: Path, format_type: str):
        """Save train, validation, and test splits to separate files."""
        try:
            # Save training data
            train_path = format_dir / "train.jsonl"
            with open(train_path, 'w', encoding='utf-8') as f:
                for entry in train_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(train_data)} training entries to {train_path}")
            
            # Save validation data
            valid_path = format_dir / "valid.jsonl"
            with open(valid_path, 'w', encoding='utf-8') as f:
                for entry in val_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(val_data)} validation entries to {valid_path}")
            
            # Save test data
            test_path = format_dir / "test.jsonl"
            with open(test_path, 'w', encoding='utf-8') as f:
                for entry in test_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(test_data)} test entries to {test_path}")
            
            # Save metadata
            metadata_path = format_dir / "metadata.json"
            metadata = {
                'format': format_type,
                'train_entries': len(train_data),
                'validation_entries': len(val_data),
                'test_entries': len(test_data),
                'total_entries': len(train_data),
                'reformulated_entries': self.stats['reformulated_entries'],
                'conversion_timestamp': datetime.now().isoformat(),
                'source_format': 'instruction-response pairs',
                'description': f'Cybersecurity dataset converted to {format_type} chat template format with train/val/test splits'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving split data: {str(e)}")

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

    def convert_dataset(self, format_type: str = 'openai', input_pattern: str = '*.jsonl',
                        validation_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
        """Convert dataset files to specified chat template format with train/val/test splits."""
        self.stats['format_used'] = format_type
        self.stats['reformulated_entries'] = 0
        
        # Get all JSONL files matching pattern
        file_paths = list(self.input_dir.glob(input_pattern))
        
        if not file_paths:
            logger.warning(f"No files found matching pattern '{input_pattern}' in {self.input_dir}")
            return
        
        print(f"\nConverting {len(file_paths)} file(s) to {format_type} format...")
        
        # Create format-specific directory
        format_dir = self.output_dir / format_type
        format_dir.mkdir(parents=True, exist_ok=True)
        
        all_converted_data = []
        
        for file_path in file_paths:
            # Skip metadata files
            if 'metadata' in file_path.name:
                continue
            
            print(f"Processing {file_path.name}...")
            
            # Process file
            converted_data = self.process_file(file_path, format_type)
            
            if converted_data:
                all_converted_data.extend(converted_data)
                self.stats['files_processed'] += 1
        
        if all_converted_data:
            print(f"\nCreating train/validation/test splits...")
            print(f"Training: 100% of data ({len(all_converted_data)} entries)")
            print(f"Validation: {validation_size*100:.0f}% sample with reformulated questions")
            print(f"Test: {test_size*100:.0f}% sample with reformulated questions")
            
            # Create splits with reformulation
            train_data, val_data, test_data = self.create_validation_test_sets(
                all_converted_data, validation_size, test_size, seed
            )
            
            # Save split data
            self.save_split_data(train_data, val_data, test_data, format_dir, format_type)
        
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
        print(f"Reformulated Entries: {self.stats['reformulated_entries']}")
        
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
    parser.add_argument(
        '--validation-size',
        type=float,
        default=0.1,
        help='Proportion of data to use for validation set (default: 0.1)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Proportion of data to use for test set (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM reformulation for validation/test sets'
    )
    
    args = parser.parse_args()
    
    converter = ChatTemplateConverter(
        args.input_dir, 
        args.output_dir,
        use_llm_reformulation=not args.no_llm
    )
    
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
            
            converter.convert_dataset(
                format_type, 
                args.input_pattern,
                args.validation_size,
                args.test_size,
                args.seed
            )
    else:
        # Convert to single specified format
        converter.convert_dataset(
            args.format, 
            args.input_pattern,
            args.validation_size,
            args.test_size,
            args.seed
        )

if __name__ == "__main__":
    main()