# Dataset Creation Pipeline

This directory contains a series of Python scripts that form a comprehensive pipeline for creating, processing, and validating a cyber security dataset for the CyberLLMInstruct paper ([arXiv:2503.09334](https://arxiv.org/abs/2503.09334)). Each script performs a specific function in the pipeline, designed to be run sequentially.

## Prerequisites

1. **Python Environment Setup**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

2. **Ollama Setup**
```bash
# Install Ollama (macOS/Linux)
curl https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma:2b
ollama pull mistral:7b
```

3. **API Setup**
Before running the data collection script, you need to set up API access for your data sources. This includes:
- Obtaining API keys
- Setting up authentication
- Configuring rate limits
- Handling API-specific requirements

Example API configuration in `config.json`:
```json
{
  "apis": {
    "source1": {
      "api_key": "YOUR_API_KEY",
      "base_url": "https://api.source1.com",
      "rate_limit": 100
    },
    "source2": {
      "api_key": "YOUR_API_KEY",
      "base_url": "https://api.source2.com",
      "rate_limit": 50
    }
  }
}
```

## Pipeline Overview

1. **Data Collection** (`1_data_collector.py`)
   - Collects raw data from various sources
   - Supports multiple input formats (JSON, CSV, YAML)
   - Handles API rate limiting and error recovery
   - Saves raw data with source metadata
   
   Example output folder:
   ```
   raw_data/
   ├── source1_data_20250507_165224.json
   ├── source2_data_20250507_165224.json
   └── collection_stats.json
   ```

2. **Data Filtering** (`2_data_filter.py`)
   - Removes irrelevant or low-quality entries
   - Applies content filters and quality checks
   - Uses Ollama for content quality assessment
   - Handles duplicate detection
   - Generates filtering statistics
   
   Example output folder:
   ```
   filtered_data/
   ├── filtered_dataset_20250507_165926.json
   └── filtering_stats.json
   ```

3. **Data Structuring** (`3_data_structurer.py`)
   - Standardises data format
   - Validates data structure
   - Uses Ollama for text normalization
   - Ensures consistent metadata fields
   
   Example output folder:
   ```
   structured_data/
   ├── structured_dataset_20250507_165926.json
   └── structure_validation_report.json
   ```

4. **Domain Classification** (`4_domain_classifier.py`)
   - Categorises entries into cyber security domains
   - Uses Ollama for domain classification
   - Supports manual classification corrections
   - Tracks classification confidence scores
   
   Example output folder:
   ```
   classified_data/
   ├── classified_dataset_20250507_170156.json
   └── classification_metrics.json
   ```

5. **Manual Review** (`5_manual_reviewer.py`)
   - Interactive interface for data review
   - Quality assessment tools
   - Uses Ollama for automated quality checks
   - Progress tracking and reporting
   
   Example output folder:
   ```
   reviewed_data/
   ├── reviewed_dataset_20250507_170404.json
   └── review_notes.json
   ```

6. **Security Alignment & Enhancement** (`6_security_aligner.py`)
   - Adds security-focused examples
   - Generates adversarial cases
   - Uses Ollama for instruction-response enhancement
   - Implements compliance testing
   - Flags sensitive content
   - Enhances clarity and coherence
   - Tracks enhancement history
   
   Example output folder:
   ```
   security_aligned/
   ├── consolidated_cybersecurity_dataset_20250507_165224_classified_reviewed_20250507_170156_security_aligned_20250507_172532.json
   └── enhancement_log.json
   ```

7. **Final Assembly** (`8_final_assembler.py`)
   - Merges processed data
   - Performs final validation
   - Removes duplicates
   - Generates clean instruction-response pairs
   - Exports final dataset in JSON format
   
   Example output folder:
   ```
   final_dataset/
   └── final_cybersecurity_dataset_20250507_173441.json
   ```

## Usage

1. Ensure Ollama is running:
```bash
# Start Ollama service
ollama serve
```

2. Run scripts in sequence:
```bash
# 1. Collect data
python 1_data_collector.py --sources source1 source2 --output-dir raw_data

# 2. Filter data
python 2_data_filter.py --input-dir raw_data --output-dir filtered_data

# 3. Structure data
python 3_data_structurer.py --input-dir filtered_data --output-dir structured_data

# 4. Classify domains
python 4_domain_classifier.py --input-dir structured_data --output-dir classified_data

# 5. Manual review
python 5_manual_reviewer.py --input-dir classified_data --output-dir reviewed_data

# 6. Security alignment and enhancement
python 6_security_aligner.py --input-dir reviewed_data --output-dir security_aligned

# 7. Final assembly
python 8_final_assembler.py --input-dir security_aligned --output-dir final_dataset
```

## Output Format

The final dataset is available in JSON format with the following structure:
```json
[
  {
    "instruction": "text of the instruction",
    "response": "text of the response"
  }
]
```

Each entry contains only the instruction-response pair, with all metadata and complex structures converted to clean text format.

## Configuration

The pipeline uses the following configuration files:
- `config.json`: General pipeline configuration
- `ollama_config.json`: Ollama API settings and model parameters

Example `ollama_config.json`:
```json
{
  "base_url": "http://localhost:11434",
  "models": {
    "classification": "gemma:2b",
    "enhancement": "mistral:7b"
  },
  "timeout": 60,
  "max_tokens": 2000
}
```

## Notes

- The pipeline requires Ollama to be running locally
- Each script includes progress tracking and error handling
- The final dataset contains only clean instruction-response pairs
- All complex responses are converted to readable text format
- Duplicate entries are automatically removed
- Invalid entries are logged and excluded from the final dataset
- Each stage creates timestamped output files for tracking and reproducibility
- API keys and credentials should be kept secure and never committed to version control