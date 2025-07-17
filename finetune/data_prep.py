import os
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pyarrow as pa
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'mistral-7b': 'mistralai/Mistral-7B-v0.3',
    'llama-2-70b': 'meta-llama/Llama-2-70b',
    'qwen-coder-7b': 'Qwen/Qwen2.5-Coder-7B',
    'gemma-2-9b': 'google/gemma-2-9b',
    'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
    'phi-3-mini': 'microsoft/Phi-3.5-mini-instruct'
}

@dataclass
class DataArguments:
    """Arguments for data preprocessing."""
    input_path: str = field(
        default="dataset/CyberLLM_full_dataset.csv",
        metadata={"help": "Path to the input dataset"}
    )
    output_dir: str = field(
        default="processed_data",
        metadata={"help": "Directory to save processed datasets"}
    )
    train_ratio: float = field(
        default=0.8,
        metadata={"help": "Proportion of data for training"}
    )
    val_ratio: float = field(
        default=0.1,
        metadata={"help": "Proportion of data for validation"}
    )
    test_ratio: float = field(
        default=0.1,
        metadata={"help": "Proportion of data for testing"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )

class DataPreprocessor:
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        model_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """Initialize the data preprocessor."""
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIGS[model_name],
                trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

    def load_data(self) -> List[Dict]:
        """Load and validate the input dataset."""
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 10000
            all_data = []
            
            logger.info(f"Loading dataset from {self.input_path}")
            for chunk in pd.read_csv(self.input_path, chunksize=chunk_size):
                # Ensure required columns exist
                if 'instruction' not in chunk.columns or 'response' not in chunk.columns:
                    raise ValueError("Dataset must contain 'instruction' and 'response' columns")
                
                # Convert chunk to list of dictionaries
                chunk_data = chunk[['instruction', 'response']].to_dict('records')
                all_data.extend(chunk_data)
            
            logger.info(f"Loaded {len(all_data)} examples")
            return all_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def format_prompt(self, instruction: str, response: str) -> str:
        """Format the prompt based on the model's requirements."""
        if 'llama' in self.model_name:
            return f"<s>[INST] {instruction} [/INST] {response} </s>"
        elif 'mistral' in self.model_name:
            return f"<s>[INST] {instruction} [/INST] {response} </s>"
        elif 'qwen' in self.model_name:
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        elif 'gemma' in self.model_name:
            return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
        elif 'phi' in self.model_name:
            return f"Instruction: {instruction}\nResponse: {response}"
        else:
            return f"{instruction}\n{response}"

    def tokenize_data(self, examples: List[Dict]) -> List[Dict]:
        """Tokenize the dataset using the model's tokenizer."""
        tokenized_data = []
        
        for example in tqdm(examples, desc="Tokenizing data"):
            try:
                formatted_text = self.format_prompt(
                    example['instruction'],
                    example['response']
                )
                
                tokenized = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=2048,
                    padding=False,
                    return_tensors=None
                )
                
                tokenized_data.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'original_instruction': example['instruction'],
                    'original_response': example['response']
                })
                
            except Exception as e:
                logger.warning(f"Error tokenizing example: {str(e)}")
                continue
        
        logger.info(f"Successfully tokenized {len(tokenized_data)} examples")
        return tokenized_data

    def split_data(self, data: List[Dict]) -> Tuple[Dataset, Dataset, Dataset]:
        """Split the data into train, validation, and test sets."""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(data))
        
        train_size = int(len(data) * self.train_ratio)
        val_size = int(len(data) * self.val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_dataset, val_dataset, test_dataset

    def save_datasets(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset
    ) -> None:
        """Save the processed datasets."""
        output_path = self.output_dir / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        dataset_dict.save_to_disk(output_path)
        logger.info(f"Saved processed datasets to {output_path}")

    def process(self) -> None:
        """Main processing pipeline."""
        logger.info(f"Starting data preprocessing for {self.model_name}")
        
        # Load data
        raw_data = self.load_data()
        
        # Tokenize data
        tokenized_data = self.tokenize_data(raw_data)
        
        # Split data
        train_dataset, val_dataset, test_dataset = self.split_data(tokenized_data)
        
        # Save processed datasets
        self.save_datasets(train_dataset, val_dataset, test_dataset)
        
        logger.info("Data preprocessing completed successfully")

def main():
    """Main entry point for data preprocessing."""
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    preprocessor = DataPreprocessor(
        input_path=data_args.input_path,
        output_dir=data_args.output_dir,
        model_name=model_args.model_name,
        train_ratio=data_args.train_ratio,
        val_ratio=data_args.val_ratio,
        test_ratio=data_args.test_ratio,
        seed=data_args.seed
    )
    
    preprocessor.process()

if __name__ == "__main__":
    main() 