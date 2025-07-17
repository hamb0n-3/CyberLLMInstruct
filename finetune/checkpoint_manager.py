import os
import json
import shutil
import logging
import torch
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    HfArgumentParser
)
from peft import PeftModel, PeftConfig
import datetime
import yaml
from tqdm import tqdm
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations (same as in other scripts)
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
class CheckpointArguments:
    """Arguments for checkpoint management."""
    model_name: str = field(
        metadata={"help": "Model identifier", "choices": MODEL_CONFIGS.keys()}
    )
    checkpoint_dir: str = field(
        metadata={"help": "Directory containing checkpoints"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save processed checkpoints"}
    )
    action: str = field(
        metadata={
            "help": "Action to perform",
            "choices": ["list", "compress", "convert", "clean", "merge"]
        }
    )
    checkpoint_id: Optional[str] = field(
        default=None,
        metadata={"help": "Specific checkpoint ID to process"}
    )
    keep_n_latest: Optional[int] = field(
        default=None,
        metadata={"help": "Number of latest checkpoints to keep when cleaning"}
    )

class CheckpointInfo:
    """Class to store checkpoint metadata."""
    def __init__(
        self,
        checkpoint_id: str,
        model_name: str,
        timestamp: str,
        step: int,
        is_lora: bool,
        metrics: Optional[Dict] = None
    ):
        self.checkpoint_id = checkpoint_id
        self.model_name = model_name
        self.timestamp = timestamp
        self.step = step
        self.is_lora = is_lora
        self.metrics = metrics or {}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointInfo':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return {
            'checkpoint_id': self.checkpoint_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'step': self.step,
            'is_lora': self.is_lora,
            'metrics': self.metrics
        }

class CheckpointManager:
    def __init__(self, args: CheckpointArguments):
        """Initialize the checkpoint manager."""
        self.args = args
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.output_dir = Path(args.output_dir) if args.output_dir else None
        self.model_name = args.model_name
        
        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint registry
        self.registry_path = self.checkpoint_dir / "checkpoint_registry.yaml"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, CheckpointInfo]:
        """Load the checkpoint registry from disk."""
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return {
                k: CheckpointInfo.from_dict(v) for k, v in data.items()
            }
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            return {}

    def _save_registry(self):
        """Save the checkpoint registry to disk."""
        try:
            data = {
                k: v.to_dict() for k, v in self.registry.items()
            }
            with open(self.registry_path, 'w') as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        metrics: Optional[Dict] = None,
        is_lora: bool = False
    ) -> str:
        """Save a model checkpoint."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_{timestamp}_step{step}"
        save_path = self.checkpoint_dir / checkpoint_id
        
        try:
            # Save model and tokenizer
            if is_lora:
                model.save_pretrained(save_path)
            else:
                model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            
            # Save training state if available
            if hasattr(model, 'trainer') and model.trainer is not None:
                model.trainer.save_state()
            
            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                model_name=self.model_name,
                timestamp=timestamp,
                step=step,
                is_lora=is_lora,
                metrics=metrics
            )
            
            # Update registry
            self.registry[checkpoint_id] = checkpoint_info
            self._save_registry()
            
            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def load_checkpoint(
        self,
        checkpoint_id: str,
        device_map: str = "auto"
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a model checkpoint."""
        if checkpoint_id not in self.registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry")
        
        checkpoint_info = self.registry[checkpoint_id]
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
            # Load model
            if checkpoint_info.is_lora:
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_CONFIGS[self.model_name],
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True
                )
                # Load LoRA adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True
                )
            
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def compress_checkpoint(self, checkpoint_id: str):
        """Compress a checkpoint for storage or transfer."""
        if checkpoint_id not in self.registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry")
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        output_path = self.output_dir / f"{checkpoint_id}.zip"
        
        try:
            logger.info(f"Compressing checkpoint: {checkpoint_id}")
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(checkpoint_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, checkpoint_path)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Compressed checkpoint saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error compressing checkpoint: {str(e)}")
            raise

    def convert_to_deployment(self, checkpoint_id: str):
        """Convert a checkpoint for deployment (merge LoRA if applicable)."""
        if checkpoint_id not in self.registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry")
        
        checkpoint_info = self.registry[checkpoint_id]
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        output_path = self.output_dir / f"{checkpoint_id}_deployment"
        
        try:
            if checkpoint_info.is_lora:
                logger.info("Converting LoRA checkpoint to full model...")
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_CONFIGS[self.model_name],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                # Load and merge LoRA
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                model = model.merge_and_unload()
            else:
                logger.info("Copying full model checkpoint...")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # Save converted model
            model.save_pretrained(output_path, safe_serialization=True)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Converted checkpoint saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error converting checkpoint: {str(e)}")
            raise

    def clean_checkpoints(self, keep_n: int):
        """Clean old checkpoints, keeping only the N most recent ones."""
        if not self.registry:
            logger.info("No checkpoints to clean")
            return
        
        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.registry.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep the N most recent checkpoints
        checkpoints_to_remove = sorted_checkpoints[keep_n:]
        
        for checkpoint_id, _ in checkpoints_to_remove:
            try:
                checkpoint_path = self.checkpoint_dir / checkpoint_id
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                del self.registry[checkpoint_id]
                logger.info(f"Removed checkpoint: {checkpoint_id}")
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint_id}: {str(e)}")
        
        self._save_registry()
        logger.info(f"Kept {keep_n} most recent checkpoints")

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints with their information."""
        checkpoints = []
        for checkpoint_id, info in self.registry.items():
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoints.append({
                'id': checkpoint_id,
                'model': info.model_name,
                'timestamp': info.timestamp,
                'step': info.step,
                'type': 'LoRA' if info.is_lora else 'Full',
                'size': self._get_dir_size(checkpoint_path),
                'metrics': info.metrics
            })
        return checkpoints

    def _get_dir_size(self, path: Path) -> str:
        """Get the size of a directory in human-readable format."""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.1f}{unit}"
            total_size /= 1024
        return f"{total_size:.1f}TB"

def main():
    """Main entry point for checkpoint management."""
    parser = HfArgumentParser(CheckpointArguments)
    args = parser.parse_args()
    
    manager = CheckpointManager(args)
    
    if args.action == "list":
        checkpoints = manager.list_checkpoints()
        print("\nAvailable Checkpoints:")
        print("=" * 80)
        for cp in checkpoints:
            print(f"ID: {cp['id']}")
            print(f"Model: {cp['model']}")
            print(f"Timestamp: {cp['timestamp']}")
            print(f"Step: {cp['step']}")
            print(f"Type: {cp['type']}")
            print(f"Size: {cp['size']}")
            if cp['metrics']:
                print("Metrics:")
                for k, v in cp['metrics'].items():
                    print(f"  {k}: {v}")
            print("-" * 80)
    
    elif args.action == "compress":
        if not args.checkpoint_id:
            raise ValueError("Must specify --checkpoint_id for compress action")
        if not args.output_dir:
            raise ValueError("Must specify --output_dir for compress action")
        manager.compress_checkpoint(args.checkpoint_id)
    
    elif args.action == "convert":
        if not args.checkpoint_id:
            raise ValueError("Must specify --checkpoint_id for convert action")
        if not args.output_dir:
            raise ValueError("Must specify --output_dir for convert action")
        manager.convert_to_deployment(args.checkpoint_id)
    
    elif args.action == "clean":
        if not args.keep_n_latest:
            raise ValueError("Must specify --keep_n_latest for clean action")
        manager.clean_checkpoints(args.keep_n_latest)

if __name__ == "__main__":
    main() 