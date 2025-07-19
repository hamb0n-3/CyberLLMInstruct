#!/usr/bin/env python3
"""
Test script to verify data extraction fixes for CVE entries
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Import the filter class
import importlib.util
spec = importlib.util.spec_from_file_location("data_filter", "2_data_filter.py")
data_filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_filter_module)
CyberDataFilter = data_filter_module.CyberDataFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_cve_data() -> Dict:
    """Create test CVE data with nested structure"""
    return {
        "cve": {
            "id": "CVE-2024-12345",
            "sourceIdentifier": "security@nvd.nist.gov",
            "published": "2024-01-15T10:00:00.000",
            "lastModified": "2024-01-16T15:30:00.000",
            "vulnStatus": "Analyzed",
            "descriptions": [
                {
                    "lang": "en",
                    "value": "A critical vulnerability in the authentication mechanism allows remote attackers to bypass security controls and gain unauthorized access to sensitive data. This vulnerability affects all versions prior to 2.5.0."
                }
            ],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "source": "nvd@nist.gov",
                        "type": "Primary",
                        "cvssData": {
                            "version": "3.1",
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                            "baseScore": 9.8,
                            "baseSeverity": "CRITICAL"
                        }
                    }
                ]
            },
            "weaknesses": [
                {
                    "source": "nvd@nist.gov",
                    "type": "Primary",
                    "description": [
                        {
                            "lang": "en",
                            "value": "CWE-287"
                        }
                    ]
                }
            ],
            "references": [
                {
                    "url": "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
                    "source": "nvd@nist.gov"
                }
            ]
        }
    }

def create_test_regular_data() -> Dict:
    """Create test data with regular structure"""
    return {
        "id": "CAPEC-123",
        "title": "SQL Injection Attack",
        "description": "SQL injection is a code injection technique that exploits security vulnerabilities in an application's software.",
        "summary": "Attackers can use SQL injection to access, modify, or delete data in the database.",
        "source": "capec"
    }

def test_text_extraction():
    """Test the _get_text_from_entry method"""
    logger.info("Testing text extraction from different data formats...")
    
    # Create dummy filter instance (we won't use the model)
    filter_instance = CyberDataFilter.__new__(CyberDataFilter)
    filter_instance.cybersecurity_keywords = {
        'high_relevance': {'vulnerability', 'exploit', 'attack', 'security', 'authentication'},
        'medium_relevance': {'access', 'data', 'control'}
    }
    filter_instance.min_keyword_matches = 2
    filter_instance.min_content_length = 20
    filter_instance.exclusion_patterns = {
        'generic_terms': r'\b(test|sample|example|dummy|todo)\b',
        'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b',
        'empty_content': r'^\s*$'
    }
    
    # Test CVE data extraction
    logger.info("\n=== Testing CVE Data Extraction ===")
    cve_data = create_test_cve_data()
    cve_text = filter_instance._get_text_from_entry(cve_data)
    logger.info(f"Extracted CVE text length: {len(cve_text)}")
    logger.info(f"CVE text preview: {cve_text[:200]}...")
    
    # Check if CVE ID and description were extracted
    assert "CVE-2024-12345" in cve_text, "CVE ID not found in extracted text"
    assert "critical vulnerability" in cve_text.lower(), "CVE description not found"
    assert "authentication mechanism" in cve_text.lower(), "Key vulnerability details not found"
    logger.info("✓ CVE data extraction successful")
    
    # Test regular data extraction
    logger.info("\n=== Testing Regular Data Extraction ===")
    regular_data = create_test_regular_data()
    regular_text = filter_instance._get_text_from_entry(regular_data)
    logger.info(f"Extracted regular text length: {len(regular_text)}")
    logger.info(f"Regular text preview: {regular_text[:200]}...")
    
    assert "CAPEC-123" in regular_text, "ID not found in extracted text"
    assert "SQL Injection" in regular_text, "Title not found"
    assert "code injection technique" in regular_text, "Description not found"
    logger.info("✓ Regular data extraction successful")
    
    # Test rule-based relevance on extracted text
    logger.info("\n=== Testing Rule-Based Relevance ===")
    cve_relevant = filter_instance.is_relevant_rule_based(cve_text)
    regular_relevant = filter_instance.is_relevant_rule_based(regular_text)
    
    logger.info(f"CVE data rule-based relevance: {cve_relevant}")
    logger.info(f"Regular data rule-based relevance: {regular_relevant}")
    
    assert cve_relevant, "CVE data should be marked as relevant by rule-based filter"
    assert regular_relevant, "Regular security data should be marked as relevant"
    
    return True

def test_batch_processing():
    """Test batch processing with sample data"""
    logger.info("\n=== Testing Batch Processing ===")
    
    # Create test dataset
    test_data = {
        "entries": [
            create_test_cve_data(),
            create_test_regular_data(),
            {
                "id": "test-003",
                "description": "This is a non-security related entry about general software updates"
            },
            {
                "cve": {
                    "id": "CVE-2024-99999",
                    "descriptions": [
                        {
                            "lang": "en",
                            "value": "Remote code execution vulnerability in network protocol handler allows attackers to execute arbitrary code with elevated privileges."
                        }
                    ]
                }
            }
        ]
    }
    
    # Save test data
    test_dir = Path("test_raw_data")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_cve_batch.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Created test file with {len(test_data['entries'])} entries")
    
    # Count how many should pass
    expected_relevant = 3  # Two CVE entries and one CAPEC entry
    logger.info(f"Expected relevant entries: {expected_relevant}")
    
    return test_file

def verify_fixed_extraction():
    """Verify that the fixed extraction properly handles nested CVE data"""
    logger.info("\n=== Verifying Fixed CVE Extraction ===")
    
    # Test with deeply nested CVE structure
    nested_cve = {
        "cve": {
            "id": "CVE-2024-NESTED",
            "descriptions": [
                {"lang": "en", "value": "First description about vulnerability"},
                {"lang": "es", "value": "Segunda descripción"},
                {"lang": "en", "value": "Additional details about the exploit"}
            ],
            "sourceIdentifier": "test@security.org",
            "metrics": {
                "cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]
            }
        },
        "random_field": "This should also be captured if long enough"
    }
    
    filter_instance = CyberDataFilter.__new__(CyberDataFilter)
    text = filter_instance._get_text_from_entry(nested_cve)
    
    logger.info(f"Extracted text from nested CVE: {text[:300]}...")
    
    # Verify all descriptions were extracted
    assert "First description" in text
    assert "Additional details" in text
    assert "CVE-2024-NESTED" in text
    assert "test@security.org" in text
    
    logger.info("✓ Nested CVE extraction verified successfully")

def main():
    """Run all tests"""
    logger.info("Starting data extraction tests...")
    
    try:
        # Test basic extraction
        test_text_extraction()
        
        # Test nested CVE handling
        verify_fixed_extraction()
        
        # Create test batch data
        test_file = test_batch_processing()
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*60)
        
        logger.info(f"\nTest data created at: {test_file}")
        logger.info("You can now run the filter script on this test data:")
        logger.info(f"  python 2_data_filter.py --input-dir test_raw_data --output-dir test_filtered --limit 1")
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)