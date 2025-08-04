#!/usr/bin/env python3
"""
Simple generation script for testing trained cybersecurity models.
Supports both base models and LoRA checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer


def load_model_and_tokenizer(model_path: str, tokenizer_path: str = None, strict_compatibility: bool = False):
    """Load model and tokenizer, handling both base and LoRA models."""
    model_path = Path(model_path)
    
    # Check if this is a LoRA checkpoint
    adapter_config_file = model_path / "adapter_config.json"
    training_config_file = model_path / "training_config.json"
    
    if adapter_config_file.exists() or (model_path / "training_state.pkl").exists():
        # This is a LoRA checkpoint
        
        # Try to load adapter config first (new format)
        if adapter_config_file.exists():
            with open(adapter_config_file, 'r') as f:
                adapter_config = json.load(f)
            
            print(f"Loading adapter: {adapter_config.get('adapter_name', 'Unknown')}")
            print(f"Adapter ID: {adapter_config.get('adapter_id', 'Unknown')}")
            print(f"Trained on: {adapter_config['trained_on']['base_model']}")
            
            base_model_name = adapter_config['trained_on']['base_model']
            lora_config_dict = adapter_config['lora_config']
            
        # Fall back to old format
        elif training_config_file.exists():
            with open(training_config_file, 'r') as f:
                training_config = json.load(f)
            base_model_name = training_config.get('model_name')
            lora_config_dict = training_config.get('lora_config', {})
            
            if not base_model_name:
                raise ValueError("Cannot determine base model from checkpoint")
        else:
            raise ValueError("Invalid LoRA checkpoint: missing config files")
        
        # Load the actual model specified by user (might be different from training)
        if model_path.name != base_model_name:
            # User is trying to load adapter onto a different model
            actual_model_name = str(model_path)
            
            if strict_compatibility:
                raise ValueError(
                    f"Strict mode: Adapter was trained on {base_model_name} "
                    f"but trying to load on {actual_model_name}. "
                    "Disable strict mode to attempt loading anyway."
                )
            else:
                print(f"\nWARNING: Loading adapter trained on '{base_model_name}'")
                print(f"         onto different model '{actual_model_name}'")
                print("         This may work if architectures are similar, but results may vary!\n")
        else:
            actual_model_name = base_model_name
        
        print(f"Loading base model: {actual_model_name}")
        model, tokenizer = load(actual_model_name)
        
        # Apply LoRA configuration
        from train_cybersecurity_model import apply_lora_to_model, LoRAConfig
        lora_config = LoRAConfig(**lora_config_dict)
        
        # Track which modules we're trying to apply
        target_modules = lora_config.target_modules
        applied_modules = []
        skipped_modules = []
        
        # Custom apply function that tracks success/failure
        original_apply = apply_lora_to_model
        
        def tracking_replace_module(module, name, parent):
            """Track which modules get LoRA applied."""
            if any(target in name for target in target_modules):
                try:
                    # Check if module exists
                    if hasattr(parent, name.split('.')[-1]):
                        # Original replacement logic would go here
                        applied_modules.append(name)
                    else:
                        skipped_modules.append(name)
                except Exception as e:
                    print(f"Warning: Failed to apply LoRA to {name}: {e}")
                    skipped_modules.append(name)
        
        model = apply_lora_to_model(model, lora_config)
        
        # Load LoRA weights
        lora_weights = mx.load(str(model_path / "lora_weights.safetensors"))
        
        # Apply weights (MLX-LM style)
        lora_layer_idx = 0
        def apply_lora_weights(module):
            nonlocal lora_layer_idx
            for name, child in module.__dict__.items():
                if hasattr(child, 'lora_a') and hasattr(child, 'lora_b'):
                    # Found a LoRA layer
                    if f'param_{lora_layer_idx}' in lora_weights:
                        child.lora_a = lora_weights[f'param_{lora_layer_idx}']
                        child.lora_b = lora_weights[f'param_{lora_layer_idx + 1}']
                        lora_layer_idx += 2
                elif isinstance(child, mx.nn.Module):
                    apply_lora_weights(child)
        
        apply_lora_weights(model)
        
        print(f"Successfully loaded LoRA adapter")
        
        # Show adapter info if available
        if adapter_config_file.exists():
            training_info = adapter_config.get('training_info', {})
            print(f"Training stats: {training_info.get('num_epochs', 'N/A')} epochs, "
                  f"final loss: {training_info.get('latest_loss', 'N/A'):.4f}")
            
    else:
        # Regular model loading
        print(f"Loading model: {model_path}")
        model, tokenizer = load(str(model_path))
    
    # Load custom tokenizer if specified
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    elif adapter_config_file.exists():
        # Try to load tokenizer from checkpoint
        checkpoint_tokenizer_path = model_path / "tokenizer"
        if checkpoint_tokenizer_path.exists():
            print(f"Loading tokenizer from checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_tokenizer_path))
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    system_prompt: str = None
):
    """Generate a response for the given prompt."""
    # Format prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        if system_prompt:
            formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
    
    # Generate
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        verbose=False,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p)
    )
    
    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer, system_prompt: str = None):
    """Interactive chat mode."""
    print("Interactive mode - Type 'exit' or 'quit' to stop")
    print("=" * 60)
    
    if system_prompt:
        print(f"System: {system_prompt}")
        print("=" * 60)
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if prompt.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, prompt, 
                system_prompt=system_prompt
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with trained cybersecurity models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or name (can be base model or LoRA checkpoint)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Custom tokenizer path (defaults to model)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate response for"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive chat mode"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful cybersecurity expert assistant.",
        help="System prompt to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show example prompts"
    )
    parser.add_argument(
        "--strict-compatibility",
        action="store_true",
        help="Enable strict compatibility checking for adapters"
    )
    
    args = parser.parse_args()
    
    # Show examples if requested
    if args.examples:
        print("Example cybersecurity prompts:")
        examples = [
            "What are the key indicators of a ransomware attack?",
            "Explain the MITRE ATT&CK framework.",
            "How can I protect against SQL injection attacks?",
            "What is the difference between symmetric and asymmetric encryption?",
            "Describe the steps in incident response.",
            "What are common privilege escalation techniques?",
            "How do I analyze a suspicious email for phishing?",
            "Explain zero-trust security architecture.",
            "What tools are used for network vulnerability scanning?",
            "How can machine learning detect malware?"
        ]
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        return
    
    # Load model
    try:
        # Add parent directory to path for imports
        sys.path.append(str(Path(__file__).parent.parent))
        model, tokenizer = load_model_and_tokenizer(
            args.model, 
            args.tokenizer,
            strict_compatibility=args.strict_compatibility
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate response
    if args.interactive:
        interactive_mode(model, tokenizer, args.system_prompt)
    elif args.prompt:
        print(f"Prompt: {args.prompt}")
        print("\nGenerating response...")
        response = generate_response(
            model, tokenizer, args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt
        )
        print(f"\nResponse: {response}")
    else:
        print("Error: Specify either --prompt or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()