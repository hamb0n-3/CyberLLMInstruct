#!/usr/bin/env python3
"""
Comprehensive evaluation script for cybersecurity instruction-tuned models.
Includes perplexity, generation quality, and domain-specific metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from data_loader import load_cybersec_datasets, DatasetConfig, CyberSecDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Model
    model_path: str
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = True
    
    # Data
    eval_data_paths: List[str] = field(default_factory=list)
    max_samples: Optional[int] = None
    batch_size: int = 1
    max_length: int = 2048
    
    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 512
    do_sample: bool = True
    
    # Metrics
    compute_perplexity: bool = True
    compute_generation_quality: bool = True
    compute_domain_metrics: bool = True
    
    # Output
    output_file: str = "evaluation_results.json"
    save_generations: bool = True
    verbose: bool = True


class CyberSecEvaluator:
    """Evaluator for cybersecurity instruction-tuned models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config: EvaluationConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup sampler for generation
        self.sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        ) if config.do_sample else None
        
        # Metrics storage
        self.results = {
            'perplexity': {},
            'generation_quality': {},
            'domain_metrics': {},
            'examples': []
        }
    
    def compute_perplexity(self, dataset: CyberSecDataset) -> float:
        """Compute perplexity on the dataset."""
        logger.info("Computing perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        with mx.no_grad():
            for i in tqdm(range(len(dataset)), desc="Perplexity"):
                if self.config.max_samples and i >= self.config.max_samples:
                    break
                
                example = dataset[i]
                input_ids = example['input_ids']
                labels = example['labels']
                
                # Forward pass
                logits = self.model(input_ids.reshape(1, -1))
                
                # Compute loss on valid tokens
                valid_mask = labels != -100
                if valid_mask.sum() == 0:
                    continue
                
                # Cross entropy loss
                loss = nn.losses.cross_entropy(
                    logits[0][:-1][valid_mask[1:]],
                    labels[1:][valid_mask[1:]],
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += valid_mask[1:].sum().item()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        self.results['perplexity'] = {
            'value': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_examples': min(len(dataset), self.config.max_samples or len(dataset))
        }
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def evaluate_generation_quality(self, dataset: CyberSecDataset) -> Dict[str, float]:
        """Evaluate generation quality with various metrics."""
        logger.info("Evaluating generation quality...")
        
        generation_times = []
        response_lengths = []
        examples = []
        
        # Cybersecurity-specific test prompts
        test_prompts = [
            "What are the main indicators of a ransomware attack?",
            "Explain the MITRE ATT&CK framework and its purpose.",
            "How can organizations protect against SQL injection attacks?",
            "Describe the steps in a typical incident response process.",
            "What is zero-trust security architecture?",
            "Explain the difference between IDS and IPS systems.",
            "How do you identify and mitigate phishing attacks?",
            "What are the best practices for secure API development?",
            "Describe common privilege escalation techniques.",
            "How can machine learning be used in cybersecurity?"
        ]
        
        # Evaluate on test prompts
        for prompt in tqdm(test_prompts, desc="Test prompts"):
            start_time = time.time()
            response = self._generate_response(prompt)
            gen_time = time.time() - start_time
            
            generation_times.append(gen_time)
            response_lengths.append(len(response.split()))
            
            examples.append({
                'prompt': prompt,
                'response': response,
                'generation_time': gen_time,
                'response_length': len(response.split())
            })
        
        # Evaluate on dataset samples
        num_samples = min(
            50, 
            self.config.max_samples or len(dataset),
            len(dataset)
        )
        
        for i in tqdm(range(num_samples), desc="Dataset samples"):
            example = dataset.examples[i]
            prompt = example['instruction']
            expected = example['response']
            
            start_time = time.time()
            response = self._generate_response(prompt)
            gen_time = time.time() - start_time
            
            generation_times.append(gen_time)
            response_lengths.append(len(response.split()))
            
            # Calculate similarity metrics
            similarity = self._calculate_similarity(response, expected)
            
            examples.append({
                'prompt': prompt,
                'response': response,
                'expected': expected,
                'generation_time': gen_time,
                'response_length': len(response.split()),
                'similarity': similarity
            })
        
        # Calculate metrics
        metrics = {
            'avg_generation_time': np.mean(generation_times),
            'avg_response_length': np.mean(response_lengths),
            'min_response_length': int(np.min(response_lengths)),
            'max_response_length': int(np.max(response_lengths)),
            'total_examples': len(examples)
        }
        
        # Calculate average similarity for dataset examples
        similarities = [ex.get('similarity', 0) for ex in examples if 'similarity' in ex]
        if similarities:
            metrics['avg_similarity'] = np.mean(similarities)
        
        self.results['generation_quality'] = metrics
        self.results['examples'] = examples[:20]  # Save first 20 examples
        
        logger.info(f"Average generation time: {metrics['avg_generation_time']:.2f}s")
        logger.info(f"Average response length: {metrics['avg_response_length']:.1f} words")
        
        return metrics
    
    def evaluate_domain_metrics(self, dataset: CyberSecDataset) -> Dict[str, Any]:
        """Evaluate domain-specific cybersecurity metrics."""
        logger.info("Evaluating domain-specific metrics...")
        
        # Domain-specific evaluation prompts
        domain_tests = {
            'vulnerability_analysis': [
                "Analyze CVE-2021-44228 (Log4Shell) and its impact.",
                "What are the security implications of CVE-2014-0160 (Heartbleed)?",
                "Explain the CVSS scoring system and how to interpret scores."
            ],
            'incident_response': [
                "Outline the steps to respond to a data breach incident.",
                "How would you handle a ransomware attack on critical infrastructure?",
                "Describe the incident response lifecycle according to NIST."
            ],
            'threat_intelligence': [
                "Explain the cyber kill chain model.",
                "What are TTPs in threat intelligence?",
                "How do you track and attribute APT groups?"
            ],
            'security_tools': [
                "Compare Nmap and Masscan for network scanning.",
                "When would you use Wireshark vs tcpdump?",
                "Explain how SIEM systems work."
            ],
            'secure_coding': [
                "What are the OWASP Top 10 vulnerabilities?",
                "How do you prevent XSS attacks in web applications?",
                "Explain input validation best practices."
            ]
        }
        
        domain_results = {}
        
        for domain, prompts in domain_tests.items():
            domain_scores = []
            domain_examples = []
            
            for prompt in prompts:
                response = self._generate_response(prompt)
                
                # Score based on keywords and concepts
                score = self._score_domain_response(response, domain)
                domain_scores.append(score)
                
                domain_examples.append({
                    'prompt': prompt,
                    'response': response,
                    'score': score
                })
            
            domain_results[domain] = {
                'avg_score': np.mean(domain_scores),
                'examples': domain_examples
            }
        
        # Overall domain expertise score
        overall_score = np.mean([
            results['avg_score'] 
            for results in domain_results.values()
        ])
        
        domain_results['overall_score'] = overall_score
        
        self.results['domain_metrics'] = domain_results
        
        logger.info(f"Overall domain expertise score: {overall_score:.2f}")
        
        return domain_results
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response for a given prompt."""
        # Format prompt with chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors=None,
            truncation=True,
            max_length=self.config.max_length
        )
        
        input_ids = mx.array(inputs['input_ids'])
        
        # Generate
        generated = input_ids.tolist()
        prompt_len = len(generated)
        
        for _ in range(self.config.max_new_tokens):
            # Get model prediction
            logits = self.model(mx.array([generated]))
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            if self.sampler:
                next_token = self.sampler(next_token_logits)
            else:
                next_token = mx.argmax(next_token_logits)
            
            generated.append(next_token.item())
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            generated[prompt_len:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _calculate_similarity(self, generated: str, reference: str) -> float:
        """Calculate similarity between generated and reference text."""
        # Simple word overlap similarity
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words & ref_words)
        similarity = overlap / len(ref_words)
        
        return similarity
    
    def _score_domain_response(self, response: str, domain: str) -> float:
        """Score a response based on domain-specific criteria."""
        # Domain-specific keywords and concepts
        domain_keywords = {
            'vulnerability_analysis': [
                'cve', 'cvss', 'vulnerability', 'exploit', 'patch', 
                'risk', 'impact', 'mitigation', 'attack vector', 'severity'
            ],
            'incident_response': [
                'incident', 'response', 'containment', 'eradication', 
                'recovery', 'investigation', 'forensics', 'timeline', 
                'stakeholder', 'communication'
            ],
            'threat_intelligence': [
                'threat', 'intelligence', 'ioc', 'indicator', 'ttp', 
                'attribution', 'apt', 'campaign', 'actor', 'malware'
            ],
            'security_tools': [
                'scan', 'monitor', 'detect', 'analyze', 'tool', 
                'configuration', 'output', 'command', 'feature', 'capability'
            ],
            'secure_coding': [
                'validation', 'sanitization', 'encryption', 'authentication',
                'authorization', 'owasp', 'vulnerability', 'secure', 
                'practice', 'input'
            ]
        }
        
        # Check for keyword presence
        keywords = domain_keywords.get(domain, [])
        response_lower = response.lower()
        
        keyword_score = sum(
            1 for keyword in keywords 
            if keyword in response_lower
        ) / len(keywords)
        
        # Check response length (longer, detailed responses score higher)
        length_score = min(len(response.split()) / 100, 1.0)
        
        # Check for technical depth (presence of technical terms)
        technical_terms = [
            'cve-', 'tcp', 'udp', 'http', 'https', 'ssl', 'tls',
            'api', 'sql', 'xss', 'csrf', 'dos', 'ddos', 'firewall',
            'ids', 'ips', 'siem', 'encryption', 'hash', 'certificate'
        ]
        
        technical_score = sum(
            1 for term in technical_terms 
            if term in response_lower
        ) / len(technical_terms)
        
        # Combine scores
        final_score = (
            keyword_score * 0.4 + 
            length_score * 0.3 + 
            technical_score * 0.3
        )
        
        return final_score
    
    def run_evaluation(self, dataset: CyberSecDataset):
        """Run complete evaluation suite."""
        logger.info("Starting comprehensive evaluation...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Run evaluations
        if self.config.compute_perplexity:
            self.compute_perplexity(dataset)
        
        if self.config.compute_generation_quality:
            self.evaluate_generation_quality(dataset)
        
        if self.config.compute_domain_metrics:
            self.evaluate_domain_metrics(dataset)
        
        # Save results
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        if 'perplexity' in self.results:
            print(f"\nPerplexity: {self.results['perplexity']['value']:.4f}")
        
        if 'generation_quality' in self.results:
            metrics = self.results['generation_quality']
            print(f"\nGeneration Quality:")
            print(f"  - Avg generation time: {metrics['avg_generation_time']:.2f}s")
            print(f"  - Avg response length: {metrics['avg_response_length']:.1f} words")
            if 'avg_similarity' in metrics:
                print(f"  - Avg similarity to reference: {metrics['avg_similarity']:.2%}")
        
        if 'domain_metrics' in self.results:
            print(f"\nDomain Expertise:")
            domain_metrics = self.results['domain_metrics']
            for domain, results in domain_metrics.items():
                if domain != 'overall_score' and isinstance(results, dict):
                    print(f"  - {domain}: {results['avg_score']:.2f}")
            print(f"  - Overall score: {domain_metrics['overall_score']:.2f}")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cybersecurity instruction-tuned models"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or name"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer path (defaults to model path)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to evaluation data"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    # Generation arguments
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
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    
    # Evaluation options
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity calculation"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation quality evaluation"
    )
    parser.add_argument(
        "--skip-domain",
        action="store_true",
        help="Skip domain-specific evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        eval_data_paths=args.data_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        compute_perplexity=not args.skip_perplexity,
        compute_generation_quality=not args.skip_generation,
        compute_domain_metrics=not args.skip_domain,
        output_file=args.output
    )
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_path}")
    model, tokenizer = load(
        config.model_path,
        tokenizer_config={"trust_remote_code": config.trust_remote_code}
    )
    
    # Use custom tokenizer if specified
    if config.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=config.trust_remote_code
        )
    
    # Load evaluation dataset
    logger.info("Loading evaluation dataset...")
    dataset_config = DatasetConfig(
        paths=config.eval_data_paths,
        format="cybersec",
        max_length=config.max_length,
        shuffle=False
    )
    
    eval_dataset = CyberSecDataset(
        dataset_config,
        tokenizer,
        split="validation"
    )
    
    # Create evaluator
    evaluator = CyberSecEvaluator(model, tokenizer, config)
    
    # Run evaluation
    evaluator.run_evaluation(eval_dataset)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()