#!/usr/bin/env python3
"""
Example: Integrating Advanced Inference into Data Processing Pipeline

This example shows how to modify the existing data processing scripts
to use the new adaptive speculative decoding and continuous batching features.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Import the unified inference engine
from mlx_parallm.inference_engine import create_inference_engine, InferenceMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedDataFilter:
    """
    Enhanced version of CyberDataFilter that uses advanced inference features
    """
    
    def __init__(self, input_dir: str, output_dir: str, model_path: str, 
                 use_advanced: bool = True, draft_model_path: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference engine with advanced features
        logger.info("Initializing advanced inference engine...")
        self.inference_engine = create_inference_engine(
            model_path=model_path,
            use_advanced=use_advanced,
            draft_model_path=draft_model_path,
            batch_size=16  # Larger batch size for better throughput
        )
        
        # Example: Using the engine for single inference
        self.check_relevance_prompt = """[INST]Is the following text relevant to cybersecurity? Answer ONLY with YES or NO.
Text: "{text}"[/INST]"""
        
        # Example: Using the engine for batch inference
        self.enhancement_prompt = """[INST]
Analyze the following text. Respond ONLY with a single, clean JSON object containing "technical_description", "risk_level", "affected_systems", and "mitigations".
Text to analyze:
"{text}"
[/INST]```json
"""
    
    def check_relevance_batch(self, texts: List[str]) -> List[bool]:
        """Check relevance for multiple texts using batch inference"""
        prompts = [self.check_relevance_prompt.format(text=text[:500]) for text in texts]
        
        # Use batch generation for better throughput
        responses = self.inference_engine.generate_batch(
            prompts,
            max_tokens=5,
            temperature=0.1  # Low temperature for consistent YES/NO answers
        )
        
        return ["YES" in response.upper() for response in responses]
    
    def enhance_entries_batch(self, texts: List[str]) -> List[Dict]:
        """Enhance multiple entries using batch inference"""
        prompts = [self.enhancement_prompt.format(text=text) for text in texts]
        
        # Generate enhancements in batch
        responses = self.inference_engine.generate_batch(
            prompts,
            max_tokens=1024,
            temperature=0.7
        )
        
        # Parse responses
        enhancements = []
        for response in responses:
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    enhancement = json.loads(json_match.group(0))
                    enhancements.append(enhancement)
                else:
                    enhancements.append(None)
            except:
                enhancements.append(None)
        
        return enhancements
    
    def process_file_advanced(self, file_path: Path):
        """Process a file using advanced batch inference"""
        logger.info(f"Processing {file_path.name} with advanced inference...")
        
        # Load data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            entries = data['data']
        elif isinstance(data, list):
            entries = data
        else:
            logger.warning(f"Unexpected data format in {file_path}")
            return
        
        # Extract text content from entries
        texts = []
        for entry in entries:
            if isinstance(entry, dict):
                text = " ".join([
                    str(entry.get('title', '')),
                    str(entry.get('summary', '')),
                    str(entry.get('description', '')),
                    str(entry.get('instruction', '')),
                    str(entry.get('response', ''))
                ]).strip()
                texts.append(text)
            else:
                texts.append(str(entry))
        
        # Batch relevance checking
        logger.info(f"Checking relevance for {len(texts)} entries...")
        relevance_results = self.check_relevance_batch(texts)
        
        # Filter relevant entries
        relevant_entries = []
        relevant_texts = []
        for entry, text, is_relevant in zip(entries, texts, relevance_results):
            if is_relevant:
                relevant_entries.append(entry)
                relevant_texts.append(text)
        
        logger.info(f"Found {len(relevant_entries)} relevant entries")
        
        # Batch enhancement
        if relevant_texts:
            logger.info(f"Enhancing {len(relevant_texts)} relevant entries...")
            enhancements = self.enhance_entries_batch(relevant_texts)
            
            # Apply enhancements
            enhanced_entries = []
            for entry, enhancement in zip(relevant_entries, enhancements):
                if enhancement:
                    entry.update(enhancement)
                    entry['enhanced'] = True
                enhanced_entries.append(entry)
            
            # Save results
            output_file = self.output_dir / f"{file_path.stem}_advanced_filtered.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'total_processed': len(entries),
                        'relevant_entries': len(enhanced_entries),
                        'inference_mode': self.inference_engine.mode
                    },
                    'data': enhanced_entries
                }, f, indent=2)
            
            logger.info(f"Saved {len(enhanced_entries)} entries to {output_file}")
    
    def shutdown(self):
        """Clean up resources"""
        self.inference_engine.shutdown()


def benchmark_inference_modes():
    """Benchmark different inference modes"""
    import time
    
    test_prompts = [
        "What is SQL injection?",
        "Explain buffer overflow attacks",
        "How does encryption work?",
        "What is a firewall?",
        "Describe phishing attacks"
    ] * 10  # 50 prompts total
    
    results = {}
    
    # Test configurations
    configs = [
        ("Direct MLX", False, None),
        ("Continuous Batching", True, None),
        ("Speculative Decoding", True, "mlx-community/Phi-3-mini-4k-instruct-4bit")
    ]
    
    model_path = "mlx-community/c4ai-command-r-v01-4bit"
    
    for name, use_advanced, draft_model in configs:
        logger.info(f"\nBenchmarking {name}...")
        
        try:
            # Create engine
            engine = create_inference_engine(
                model_path=model_path,
                use_advanced=use_advanced,
                draft_model_path=draft_model,
                batch_size=8
            )
            
            # Benchmark batch processing
            start_time = time.time()
            responses = engine.generate_batch(test_prompts, max_tokens=50)
            batch_time = time.time() - start_time
            
            # Benchmark single requests
            single_times = []
            for prompt in test_prompts[:5]:  # Test 5 single requests
                start = time.time()
                engine.generate(prompt, max_tokens=50)
                single_times.append(time.time() - start)
            
            avg_single_time = sum(single_times) / len(single_times)
            
            results[name] = {
                'batch_time': batch_time,
                'batch_throughput': len(test_prompts) / batch_time,
                'avg_single_latency': avg_single_time,
                'responses_generated': len(responses)
            }
            
            logger.info(f"  Batch time: {batch_time:.2f}s")
            logger.info(f"  Throughput: {results[name]['batch_throughput']:.2f} prompts/sec")
            logger.info(f"  Avg single latency: {avg_single_time:.3f}s")
            
            engine.shutdown()
            
        except Exception as e:
            logger.error(f"Error benchmarking {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Print comparison
    print("\n=== Benchmark Results ===")
    print(f"{'Mode':<20} {'Batch Time':<12} {'Throughput':<15} {'Single Latency':<15}")
    print("-" * 62)
    
    for name, stats in results.items():
        if 'error' not in stats:
            print(f"{name:<20} {stats['batch_time']:<12.2f} "
                  f"{stats['batch_throughput']:<15.2f} {stats['avg_single_latency']:<15.3f}")


def main():
    parser = argparse.ArgumentParser(description="Example: Advanced Inference Integration")
    parser.add_argument("--mode", choices=["filter", "benchmark"], default="benchmark",
                       help="Run mode: filter data or benchmark inference")
    parser.add_argument("--input-dir", default="raw_data", help="Input directory")
    parser.add_argument("--output-dir", default="advanced_filtered", help="Output directory")
    parser.add_argument("--model", default="mlx-community/c4ai-command-r-v01-4bit",
                       help="Main model path")
    parser.add_argument("--draft-model", help="Draft model for speculative decoding")
    parser.add_argument("--use-advanced", action="store_true", 
                       help="Use advanced inference features")
    
    args = parser.parse_args()
    
    if args.mode == "benchmark":
        benchmark_inference_modes()
    else:
        # Run advanced filtering
        filter = AdvancedDataFilter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model,
            use_advanced=args.use_advanced,
            draft_model_path=args.draft_model
        )
        
        # Process all JSON files
        for file_path in Path(args.input_dir).glob("*.json"):
            filter.process_file_advanced(file_path)
        
        filter.shutdown()


if __name__ == "__main__":
    main() 