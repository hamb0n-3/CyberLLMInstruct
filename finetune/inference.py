import os
import json
import logging
import torch
import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    HfArgumentParser
)
from peft import PeftModel
import threading
from queue import Queue
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations (same as in train.py)
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
class InferenceArguments:
    """Arguments for inference."""
    model_name: str = field(
        metadata={"help": "Model identifier", "choices": MODEL_CONFIGS.keys()}
    )
    model_path: str = field(
        default="finetuned_models",
        metadata={"help": "Path to the fine-tuned model"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )
    input_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to input file containing prompts (one per line)"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save outputs"}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling probability"}
    )
    interactive: bool = field(
        default=False,
        metadata={"help": "Run in interactive mode"}
    )

class ModelInference:
    def __init__(self, args: InferenceArguments):
        """Initialize the inference class."""
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.args.model_path}")
        
        # Load tokenizer
        base_model = MODEL_CONFIGS[self.args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Determine quantization config
        quantization_config = None
        if self.args.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.args.use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned weights
        if os.path.exists(self.args.model_path):
            if os.path.exists(os.path.join(self.args.model_path, "adapter_config.json")):
                logger.info("Loading LoRA adapter...")
                model = PeftModel.from_pretrained(model, self.args.model_path)
            else:
                logger.info("Loading full fine-tuned model...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_path,
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
        
        model.eval()
        return model, tokenizer
    
    def format_prompt(self, instruction: str) -> str:
        """Format the prompt based on the model's requirements."""
        if 'llama' in self.args.model_name:
            return f"<s>[INST] {instruction} [/INST]"
        elif 'mistral' in self.args.model_name:
            return f"<s>[INST] {instruction} [/INST]"
        elif 'qwen' in self.args.model_name:
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        elif 'gemma' in self.args.model_name:
            return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        elif 'phi' in self.args.model_name:
            return f"Instruction: {instruction}\nResponse:"
        else:
            return instruction

    def generate_response(self, instruction: str, stream: bool = False) -> Union[str, TextIteratorStreamer]:
        """Generate a response for a given instruction."""
        try:
            formatted_prompt = self.format_prompt(instruction)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            generation_config = {
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            if stream:
                streamer = TextIteratorStreamer(self.tokenizer)
                generation_config["streamer"] = streamer
                
                thread = threading.Thread(
                    target=self.model.generate,
                    kwargs={
                        **inputs,
                        **generation_config
                    }
                )
                thread.start()
                return streamer
            else:
                outputs = self.model.generate(**inputs, **generation_config)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                response = response[len(formatted_prompt):].strip()
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "[ERROR] Failed to generate response"

    def process_batch(self, input_file: str, output_file: str):
        """Process a batch of instructions from a file."""
        try:
            # Read input file
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
                instructions = df['instruction'].tolist()
            else:
                with open(input_file, 'r', encoding='utf-8') as f:
                    instructions = [line.strip() for line in f if line.strip()]
            
            results = []
            for instruction in tqdm(instructions, desc="Processing instructions"):
                response = self.generate_response(instruction)
                results.append({
                    'instruction': instruction,
                    'response': response,
                    'error': response.startswith('[ERROR]')
                })
            
            # Save results
            output_df = pd.DataFrame(results)
            if output_file.endswith('.csv'):
                output_df.to_csv(output_file, index=False)
            else:
                output_df.to_json(output_file, orient='records', lines=True)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def interactive_mode(self):
        """Run interactive mode for generating responses."""
        print("\nEntering interactive mode. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            try:
                instruction = input("\nEnter your instruction: ").strip()
                if instruction.lower() == 'quit':
                    break
                
                if not instruction:
                    continue
                
                print("\nGenerating response...")
                streamer = self.generate_response(instruction, stream=True)
                
                print("\nResponse:")
                for text in streamer:
                    print(text, end="", flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print("\nAn error occurred. Please try again.")

def main():
    """Main entry point for inference."""
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args()
    
    inference = ModelInference(args)
    
    if args.interactive:
        inference.interactive_mode()
    elif args.input_file and args.output_file:
        inference.process_batch(args.input_file, args.output_file)
    else:
        logger.error("Must specify either --interactive or both --input_file and --output_file")

if __name__ == "__main__":
    main() 