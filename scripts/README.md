# Scripts

This directory contains utility scripts for processing and managing the CyberLLMInstruct dataset.

## Available Scripts

### `categorise.py`
Categorises dataset entries into cyber security domains using pattern matching.

```bash
python categorise.py --input-dir dataset --output-dir categorised
```

Options:
- `--input-dir`: Directory containing dataset files (default: dataset)
- `--output-dir`: Output directory for categorised data (default: categorised)

Outputs:
- JSON files with categorised entries
- Categorisation report with statistics

### `dataset_export.py`
Prepares the dataset for release and handles platform uploads.

```bash
# Export dataset
python dataset_export.py --input-dir dataset --output-dir release

# Export and upload to Hugging Face
python dataset_export.py --input-dir dataset --output-dir release --hf-repo username/repo --hf-token YOUR_TOKEN
```

Options:
- `--input-dir`: Directory containing dataset files (default: dataset)
- `--output-dir`: Output directory for release files (default: release)
- `--hf-repo`: Hugging Face repository ID (optional)
- `--hf-token`: Hugging Face API token (optional)

Outputs:
- Dataset files in JSON format
- Dataset card (README.md)
- Metadata file with statistics

## Common Usage

1. First, categorise the processed data:
```bash
python categorise.py --input-dir processed_data --output-dir categorised_data
```

2. Then, prepare for release:
```bash
python dataset_export.py --input-dir categorised_data --output-dir release
```

Note: For dataset creation, please refer to the complete pipeline in the `dataset_creation/` directory.

## Dependencies

All required packages are listed in the root `requirements.txt` file.