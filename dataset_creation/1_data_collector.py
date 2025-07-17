#!/usr/bin/env python3

import os
import json
import logging
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from pathlib import Path
import feedparser
import pandas as pd
import yaml
import time
import shutil
import argparse
from github import Github
import io
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDataCollector:
    def __init__(self, output_dir: str = "raw_data"):
        """Initialize the data collector with output directory configuration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients and keys
        self.github_client = Github(os.getenv('GITHUB_TOKEN'))
        self.opencve_auth = (os.getenv('OPENCVE_EMAIL'), os.getenv('OPENCVE_PASSWORD'))
        self.nvd_api_key = os.getenv('NVD_API_KEY')
        
        self.api_keys = {
            'virustotal': os.getenv('VIRUSTOTAL_API_KEY'),
            'alienvault': os.getenv('ALIENVAULT_API_KEY'),
            'hackthebox': os.getenv('HTB_API_KEY'),
            'malpedia': os.getenv('MALPEDIA_API_KEY'),
            'malshare': os.getenv('MALSHARE_API_KEY'),
            'shodan': os.getenv('SHODAN_API_KEY'),
            'phishtank': os.getenv('PHISHTANK_API_KEY'),
        }
        
        # Rate limiting and timeouts
        self.rate_limits = {
            'nvd_cve': {'requests': 5, 'period': 30},
            'ctftime': {'requests': 30, 'period': 60},
            'github': {'requests': 60, 'period': 3600},
            'virustotal': {'requests': 4, 'period': 60},
            'shodan': {'requests': 1, 'period': 1},
            'malshare': {'requests': 25, 'period': 60},
        }
        self.last_request_time = {}
        self.timeouts = {'default': 30, 'download': 180, 'scraping': 60}
        
        # API endpoints
        self.endpoints = {
            'nvd_cve': 'https://services.nvd.nist.gov/rest/json/cves/2.0',
            'opencve': 'https://app.opencve.io/api/cve',
            'mitre_attack': 'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
            'mitre_capec': 'https://capec.mitre.org/data/xml/views/3000.xml', # This URL provides a ZIP file
            'alienvault_otx': 'https://otx.alienvault.com/api/v1/pulses/subscribed',
            'threatfox_api': 'https://threatfox-api.abuse.ch/api/v1/',
            'microsoft_security': 'https://api.msrc.microsoft.com/cvrf/v2.0/updates',
            'ubuntu_usn': 'https://ubuntu.com/security/notices/rss.xml',
            'redhat_security': 'https://access.redhat.com/labs/securitydataapi/cve.json',
            'arxiv_cs_crypto': 'http://export.arxiv.org/api/query?search_query=cat:cs.CR&max_results=100',
            'exploit_db': 'https://www.exploit-db.com/download/',
            'malware_bazaar': 'https://bazaar.abuse.ch/api/v1/',
            # Add other endpoints as needed
        }
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CyberLLMInstruct-DataCollector/1.0'})

    def _check_rate_limit(self, endpoint: str):
        """Sleeps if necessary to respect rate limits."""
        if endpoint not in self.rate_limits:
            return
        current_time = time.time()
        if endpoint in self.last_request_time:
            elapsed = current_time - self.last_request_time[endpoint]
            limit = self.rate_limits[endpoint]
            if elapsed < (limit['period'] / limit['requests']):
                sleep_time = (limit['period'] / limit['requests']) - elapsed
                time.sleep(sleep_time)
        self.last_request_time[endpoint] = current_time

    def _make_request(self, endpoint: str, url: str, params: Dict = None, headers: Dict = None, timeout: int = None, method: str = 'get', data: Dict = None, auth=None) -> Optional[requests.Response]:
        """Makes a web request with rate limiting and error handling."""
        self._check_rate_limit(endpoint)
        try:
            if method.lower() == 'get':
                response = self.session.get(url, params=params, headers=headers, timeout=timeout or self.timeouts['default'], auth=auth)
            elif method.lower() == 'post':
                response = self.session.post(url, params=params, headers=headers, json=data, timeout=timeout or self.timeouts['default'], auth=auth)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def fetch_capec_data(self) -> Optional[Dict]:
        """
        Fetches and correctly extracts MITRE CAPEC data from the ZIP archive.
        """
        logger.info("Fetching CAPEC data from the ZIP archive...")
        try:
            # 1. Get the binary content from the URL.
            response = self.session.get(self.endpoints['mitre_capec'], timeout=self.timeouts['download'])
            response.raise_for_status()

            # 2. Use 'io.BytesIO' to treat the downloaded bytes as an in-memory file.
            zip_file_in_memory = io.BytesIO(response.content)

            # 3. Open the ZIP archive.
            with zipfile.ZipFile(zip_file_in_memory) as zf:
                # The name of the XML file inside the archive is '3000.xml'.
                xml_filename = '3000.xml'
                if xml_filename not in zf.namelist():
                    logger.error(f"Could not find '{xml_filename}' in the downloaded ZIP archive.")
                    return None

                # 4. Extract and decode the XML content into a readable string.
                xml_content_bytes = zf.read(xml_filename)
                xml_content_string = xml_content_bytes.decode('utf-8')
                logger.info("Successfully extracted and decoded the CAPEC XML data.")

                # 5. Return the clean XML data.
                return {'xml_data': xml_content_string}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CAPEC data: {str(e)}")
            return None
        except zipfile.BadZipFile:
            logger.error("The file downloaded from the CAPEC URL is not a valid ZIP archive.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing CAPEC data: {str(e)}")
            return None

    # --- Keep all your other fetch_* functions here ---
    def fetch_cve_data(self, start_index: int = 0, results_per_page: int = 2000) -> Optional[Dict]:
        """Fetch CVE data from NVD database."""
        try:
            params = {'startIndex': start_index, 'resultsPerPage': results_per_page}
            response = self._make_request('nvd_cve', self.endpoints['nvd_cve'], params=params)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CVE data: {str(e)}")
            return None

    def fetch_opencve_data(self, limit: int = 100) -> Optional[Dict]:
        """Fetch CVE data from the OpenCVE API."""
        try:
            params = {'page': 1}
            response = self._make_request('opencve', self.endpoints['opencve'], params=params)
            if not response: return None
            data = response.json()
            return {'summary': data.get('results', [])[:limit]}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching OpenCVE data: {str(e)}")
            return None
    
    def fetch_mitre_attack(self) -> Optional[Dict]:
        """Fetch MITRE ATT&CK framework data."""
        try:
            response = self.session.get(self.endpoints['mitre_attack'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MITRE ATT&CK data: {str(e)}")
            return None

    def fetch_ubuntu_security_notices(self) -> Optional[Dict]:
        """Fetch Ubuntu Security Notices."""
        try:
            feed = feedparser.parse(self.endpoints['ubuntu_usn'])
            return {'entries': feed.entries}
        except Exception as e:
            logger.error(f"Error fetching Ubuntu Security Notices: {str(e)}")
            return None

    def fetch_arxiv_papers(self) -> Optional[Dict]:
        """Fetch recent cyber security papers from arXiv."""
        try:
            response = self.session.get(self.endpoints['arxiv_cs_crypto'])
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            return {'papers': feed.entries}
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {str(e)}")
            return None

    def fetch_redhat_security(self) -> Optional[Dict]:
        """Fetch Red Hat Security Data."""
        try:
            response = self.session.get(self.endpoints['redhat_security'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Red Hat Security data: {str(e)}")
            return None

    def fetch_microsoft_security(self) -> Optional[Dict]:
        """Fetch Microsoft Security Updates."""
        try:
            response = self.session.get(self.endpoints['microsoft_security'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Microsoft Security Updates: {str(e)}")
            return None
    
    def fetch_ctf_data(self) -> Optional[Dict]:
        """Fetch CTF event data from CTFtime."""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(days=90)
            params = {'start': int(start_time.timestamp()), 'finish': int(end_time.timestamp()), 'limit': 100}
            response = self.session.get(self.endpoints['ctftime'], params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CTF data: {str(e)}")
            return None

    def save_data(self, data: Union[Dict, List], source: str, format: str = 'json'):
        """Saves the collected data to a file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f"{source}_{timestamp}.{format}"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved data to {filename}")
        except Exception as e:
            logger.error(f"Error saving data for {source}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Collect cybersecurity data.")
    parser.add_argument("--sources", nargs="+", help="List of sources to fetch data from")
    parser.add_argument("--output-dir", default="raw_data", help="Directory to save collected data")
    args = parser.parse_args()
    
    collector = CyberDataCollector(output_dir=args.output_dir)
    
    all_sources = {
        'cve_data': collector.fetch_cve_data,
        'opencve_data': collector.fetch_opencve_data,
        'mitre_attack': collector.fetch_mitre_attack,
        'capec_data': collector.fetch_capec_data,
        'ubuntu_security': collector.fetch_ubuntu_security_notices,
        'arxiv_papers': collector.fetch_arxiv_papers,
        'redhat_security': collector.fetch_redhat_security,
        'microsoft_security': collector.fetch_microsoft_security,
        'ctf_data': collector.fetch_ctf_data,
    }

    sources_to_fetch = all_sources if not args.sources or "all" in args.sources else {s: all_sources[s] for s in args.sources if s in all_sources}
    
    logger.info(f"Collecting data from sources: {', '.join(sources_to_fetch.keys())}")
    
    for source_name, fetch_function in sources_to_fetch.items():
        logger.info(f"Fetching data from {source_name}...")
        data = fetch_function()
        if data:
            collector.save_data(data, source_name)
        else:
            logger.warning(f"No data retrieved from {source_name}")

if __name__ == "__main__":
    main()

