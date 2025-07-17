#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import curses
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import threading
import queue
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDataReviewer:
    def __init__(self, input_dir: str = "domain_classified", output_dir: str = "reviewed_data"):
        """Initialize the manual reviewer with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Review session tracking
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.review_progress_file = self.output_dir / f"review_progress_{self.session_id}.json"
        self.review_stats_file = self.output_dir / f"review_stats_{self.session_id}.json"
        
        # Initialize rich console for better UI
        self.console = Console()
        
        # Review statistics
        self.stats = {
            'total_entries': 0,
            'reviewed_entries': 0,
            'modified_entries': 0,
            'review_time': 0,
            'reviewers': set(),
            'domains_modified': {},
            'quality_scores': []
        }
        
        # Quality assessment criteria
        self.quality_criteria = {
            'relevance': {
                'question': 'How relevant is this entry to cybersecurity?',
                'options': ['High', 'Medium', 'Low', 'Not Relevant']
            },
            'accuracy': {
                'question': 'How accurate is the information?',
                'options': ['Very Accurate', 'Mostly Accurate', 'Somewhat Accurate', 'Inaccurate']
            },
            'completeness': {
                'question': 'How complete is the information?',
                'options': ['Complete', 'Mostly Complete', 'Partially Complete', 'Incomplete']
            }
        }

    def load_data(self, file_path: Path) -> Union[Dict, List, None]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract the 'data' field if it exists
                    if isinstance(data, dict) and 'data' in data:
                        return data['data']
                    return data
            elif suffix in {'.yaml', '.yml'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif suffix == '.csv':
                return pd.read_csv(file_path).to_dict('records')
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def save_progress(self):
        """Save review progress and statistics."""
        # Convert set to list for JSON serialization
        stats_copy = self.stats.copy()
        stats_copy['reviewers'] = list(stats_copy['reviewers'])
        
        with open(self.review_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2)
        
        logger.info(f"Saved review statistics to {self.review_stats_file}")

    def display_entry(self, entry: Dict):
        """Display an entry for review in a formatted manner."""
        self.console.clear()
        
        # Create a table for the entry
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Content", style="white")
        
        # Add basic information
        table.add_row("Instruction", entry.get('instruction', 'N/A'))
        table.add_row("Response", entry.get('response', 'N/A'))
        
        # Add type information
        table.add_row("Type", entry.get('type', 'N/A'))
        
        # Add source data
        if 'source_data' in entry:
            source_str = json.dumps(entry['source_data'], indent=2)
            table.add_row("Source Data", source_str)
        
        # Add domain information
        if 'domains' in entry:
            domains_str = '\n'.join([
                f"{d['domain']} (confidence: {d['confidence']:.2f})"
                for d in entry['domains']
            ])
            table.add_row("Domains", domains_str)
            table.add_row("Primary Domain", entry.get('primary_domain', 'N/A'))
        
        self.console.print(Panel(table, title="Entry Review", border_style="green"))

    def get_quality_assessment(self) -> Dict:
        """Get quality assessment from reviewer."""
        assessment = {}
        
        for criterion, details in self.quality_criteria.items():
            answer = questionary.select(
                details['question'],
                choices=details['options']
            ).ask()
            
            assessment[criterion] = answer
        
        return assessment

    def get_reviewer_feedback(self, entry: Dict) -> Dict:
        """Get feedback from reviewer for an entry."""
        self.display_entry(entry)
        
        # Get quality assessment
        quality = self.get_quality_assessment()
        
        # Get additional feedback
        needs_modification = questionary.confirm(
            "Does this entry need modifications?",
            default=False
        ).ask()
        
        if needs_modification:
            # Get specific modifications
            modifications = {}
            
            # Modify instruction
            if questionary.confirm("Modify instruction?").ask():
                modifications['instruction'] = questionary.text(
                    "Enter modified instruction:",
                    default=entry.get('instruction', '')
                ).ask()
            
            # Modify response
            if questionary.confirm("Modify response?").ask():
                modifications['response'] = questionary.text(
                    "Enter modified response:",
                    default=entry.get('response', '')
                ).ask()
            
            # Modify domains
            if questionary.confirm("Modify domains?").ask():
                domains_input = questionary.text(
                    "Enter domains (comma-separated):",
                    default=','.join(d['name'] for d in entry.get('domains', []))
                ).ask()
                
                domains = [
                    {'name': d.strip(), 'confidence': 1.0}
                    for d in domains_input.split(',')
                    if d.strip()
                ]
                
                modifications['domains'] = domains
            
            # Add reviewer notes
            notes = questionary.text("Add any additional notes:").ask()
            if notes:
                modifications['reviewer_notes'] = notes
        else:
            modifications = None
        
        return {
            'quality_assessment': quality,
            'modifications': modifications,
            'review_timestamp': datetime.now().isoformat(),
            'needs_modification': needs_modification
        }

    def apply_modifications(self, entry: Dict, feedback: Dict) -> Dict:
        """Apply reviewer modifications to an entry."""
        if not feedback.get('needs_modification'):
            return entry
        
        modifications = feedback.get('modifications', {})
        
        # Apply modifications
        for field, value in modifications.items():
            entry[field] = value
        
        # Add review metadata
        entry['review_info'] = {
            'quality_assessment': feedback['quality_assessment'],
            'review_timestamp': feedback['review_timestamp'],
            'reviewer_notes': modifications.get('reviewer_notes', '')
        }
        
        return entry

    def update_statistics(self, feedback: Dict, entry: Dict):
        """Update review statistics."""
        self.stats['reviewed_entries'] += 1
        
        if feedback.get('needs_modification'):
            self.stats['modified_entries'] += 1
            
            # Track domain modifications
            if 'domains' in entry:
                for domain in entry['domains']:
                    domain_name = domain['name']
                    if domain_name not in self.stats['domains_modified']:
                        self.stats['domains_modified'][domain_name] = 0
                    self.stats['domains_modified'][domain_name] += 1
        
        # Track quality scores
        quality = feedback['quality_assessment']
        self.stats['quality_scores'].append(quality)

    def generate_review_report(self):
        """Generate a detailed review report."""
        report = Table(title="Review Session Report", show_header=True)
        report.add_column("Metric", style="cyan")
        report.add_column("Value", style="green")
        
        # Add statistics
        report.add_row("Total Entries", str(self.stats['total_entries']))
        report.add_row("Reviewed Entries", str(self.stats['reviewed_entries']))
        report.add_row("Modified Entries", str(self.stats['modified_entries']))
        
        # Calculate percentages
        if self.stats['total_entries'] > 0:
            review_percentage = (self.stats['reviewed_entries'] / self.stats['total_entries']) * 100
            modification_percentage = (self.stats['modified_entries'] / self.stats['reviewed_entries']) * 100 if self.stats['reviewed_entries'] > 0 else 0
            
            report.add_row("Review Progress", f"{review_percentage:.1f}%")
            report.add_row("Modification Rate", f"{modification_percentage:.1f}%")
        
        # Domain modifications
        if self.stats['domains_modified']:
            domain_stats = "\n".join(
                f"{domain}: {count} modifications"
                for domain, count in self.stats['domains_modified'].items()
            )
            report.add_row("Domain Modifications", domain_stats)
        
        self.console.print(report)
        
        # Save report
        report_path = self.output_dir / f"review_report_{self.session_id}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(str(report))

    def save_reviewed_data(self, data: List[Dict], original_file: Path):
        """Save reviewed data with modifications."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{original_file.stem}_reviewed_{timestamp}"
        
        # Save as JSON
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_name}.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        logger.info(f"Saved reviewed data to {self.output_dir}")

    def review_entries(self, entries: List[Dict], reviewer_id: str):
        """Review a list of entries."""
        self.stats['reviewers'].add(reviewer_id)
        self.stats['total_entries'] = len(entries)
        
        reviewed_entries = []
        start_time = datetime.now()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"Reviewing entries...", total=len(entries))
                
                for i, entry in enumerate(entries, 1):
                    self.console.clear()
                    progress.update(task, advance=1, description=f"Reviewing entry {i}/{len(entries)}")
                    
                    # Get reviewer feedback
                    feedback = self.get_reviewer_feedback(entry)
                    
                    # Apply modifications if needed
                    modified_entry = self.apply_modifications(entry.copy(), feedback)
                    reviewed_entries.append(modified_entry)
                    
                    # Update statistics
                    self.update_statistics(feedback, modified_entry)
                    
                    # Save progress periodically
                    if i % 10 == 0:
                        self.save_progress()
                    
                    # Check if user wants to continue
                    if i < len(entries) and not questionary.confirm(
                        "Continue to next entry?",
                        default=True
                    ).ask():
                        break
        
        except KeyboardInterrupt:
            self.console.print("[yellow]Review session interrupted by user[/yellow]")
        
        finally:
            # Update review time
            self.stats['review_time'] = (datetime.now() - start_time).total_seconds()
            self.save_progress()
            
            return reviewed_entries

    def process_directory(self):
        """Process all files in the input directory."""
        # Get reviewer ID
        reviewer_id = questionary.text(
            "Enter reviewer ID:",
            validate=lambda text: len(text.strip()) > 0
        ).ask()
        
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() not in {'.json', '.yaml', '.yml', '.csv'}:
                continue
            
            logger.info(f"Processing {file_path}")
            data = self.load_data(file_path)
            
            if not data:
                continue
            
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            
            # Review entries
            reviewed_data = self.review_entries(data, reviewer_id)
            
            # Save reviewed data
            if reviewed_data:
                self.save_reviewed_data(reviewed_data, file_path)
        
        # Generate final report
        self.generate_review_report()

def main():
    """Main function to demonstrate usage."""
    # Update the input directory to point to the correct location
    reviewer = CyberDataReviewer(input_dir="domain_classified")
    
    # Handle graceful exit
    def signal_handler(sig, frame):
        print("\nSaving progress before exit...")
        reviewer.save_progress()
        reviewer.generate_review_report()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    reviewer.process_directory()

if __name__ == "__main__":
    main() 