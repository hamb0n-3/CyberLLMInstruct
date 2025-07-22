#!/usr/bin/env python3
"""
Shared utilities for cybersecurity data processing scripts.
"""

import json
import logging
import time
import os
from typing import Dict, List, Optional
from collections import defaultdict
import statistics
import psutil
from json import JSONDecoder


def extract_first_json_object(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from text, ignoring any trailing content."""
    # First try to find JSON object boundaries
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Use JSONDecoder to handle proper JSON parsing
    decoder = JSONDecoder()
    try:
        # This will parse the first valid JSON object and return its end position
        obj, end_idx = decoder.raw_decode(text, start_idx)
        return obj
    except json.JSONDecodeError:
        # Fallback: try to extract with balanced braces
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start_idx:i+1])
                        except json.JSONDecodeError:
                            return None
        
        return None


class BenchmarkTracker:
    """Tracks detailed performance metrics with comprehensive statistics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize benchmark tracker."""
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Per-file timing configuration
        self.file_start_time = None
        self.file_log_intervals = [120, 1800]  # 2 min, 30 min, then hourly
        self.file_log_index = 0
        self.next_log_time = None
        
        # Performance metrics
        self.metrics = {
            'entries_processed': 0,
            'entries_passed': 0,
            'entries_failed': 0,
            'files_completed': 0,
            'stage_times': defaultdict(list),
            'rejection_reasons': defaultdict(int),
            'memory_usage': [],
            'processing_times': [],
            'llm_calls': 0,
            'llm_tokens_per_second': [],
            'llm_input_tokens': [],
            'llm_output_tokens': [],
            'llm_generation_times': []
        }
        
        # Initialize process for memory tracking
        try:
            self.process = psutil.Process(os.getpid())
            self._record_memory()
        except Exception as e:
            self.logger.warning(f"Could not initialize memory tracking: {e}")
            self.process = None
    
    def _record_memory(self):
        """Record current memory usage."""
        if not self.process:
            return
            
        try:
            mem_info = self.process.memory_info()
            self.metrics['memory_usage'].append({
                'timestamp': time.time(),
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024)
            })
        except Exception as e:
            self.logger.debug(f"Memory recording failed: {e}")
    
    def record_entry(self, passed: bool, processing_time: float = 0):
        """Record metrics for a single entry."""
        self.metrics['entries_processed'] += 1
        
        if passed:
            self.metrics['entries_passed'] += 1
        else:
            self.metrics['entries_failed'] += 1
        
        if processing_time > 0:
            self.metrics['processing_times'].append(processing_time)
    
    def record_stage_time(self, stage: str, duration: float):
        """Record time taken for a processing stage."""
        self.metrics['stage_times'][stage].append(duration)
    
    def record_llm_performance(self, tokens_per_second: float, input_tokens: int, output_tokens: int, generation_time: float):
        """Record LLM performance metrics."""
        self.metrics['llm_calls'] += 1
        self.metrics['llm_tokens_per_second'].append(tokens_per_second)
        self.metrics['llm_input_tokens'].append(input_tokens)
        self.metrics['llm_output_tokens'].append(output_tokens)
        self.metrics['llm_generation_times'].append(generation_time)
    
    def reset_file_timing(self):
        """Reset timing for a new file."""
        self.file_start_time = time.time()
        self.file_log_index = 0
        self.next_log_time = self.file_start_time + self.file_log_intervals[0]
    
    def should_log(self) -> bool:
        """Check if it's time to log benchmark stats."""
        if self.file_start_time is None:
            return False
            
        current_time = time.time()
        if self.next_log_time and current_time >= self.next_log_time:
            # Move to next interval
            if self.file_log_index < len(self.file_log_intervals) - 1:
                self.file_log_index += 1
                self.next_log_time = self.file_start_time + self.file_log_intervals[self.file_log_index]
            else:
                # After initial intervals, log hourly
                self.next_log_time = current_time + 3600  # 1 hour
            return True
        return False
    
    def get_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        return statistics.quantiles(data, n=100)[int(percentile)-1] if len(data) > 1 else data[0]
    
    def log_benchmark_stats(self, force: bool = False):
        """Log comprehensive benchmark statistics."""
        if not force and not self.should_log():
            return
        
        self._record_memory()
        elapsed_time = time.time() - self.start_time
        
        lines = [
            f"\n{'='*80}",
            f"BENCHMARK STATISTICS - Elapsed: {elapsed_time/60:.1f} minutes",
            f"{'='*80}",
            f"Progress Summary:",
            f"  - Total entries processed: {self.metrics['entries_processed']}",
            f"  - Entries succeeded: {self.metrics['entries_passed']}",
            f"  - Entries failed: {self.metrics['entries_failed']}",
            f"  - Files completed: {self.metrics['files_completed']}"
        ]
        
        # Processing speed
        if self.metrics['processing_times']:
            times = self.metrics['processing_times']
            lines.extend([
                f"\nProcessing Speed:",
                f"  - Mean time per entry: {statistics.mean(times):.3f}s",
                f"  - Median time per entry: {statistics.median(times):.3f}s",
                f"  - Throughput: {len(times)/sum(times):.2f} entries/second" if sum(times) > 0 else ""
            ])
        
        # Memory usage
        if self.metrics['memory_usage']:
            latest_mem = self.metrics['memory_usage'][-1]
            all_rss = [m['rss_mb'] for m in self.metrics['memory_usage']]
            lines.extend([
                f"\nMemory Usage:",
                f"  - Current RSS: {latest_mem['rss_mb']:.1f} MB",
                f"  - Peak RSS: {max(all_rss):.1f} MB"
            ])
        
        # LLM performance
        if self.metrics['llm_tokens_per_second']:
            tps = self.metrics['llm_tokens_per_second']
            gen_times = self.metrics['llm_generation_times']
            lines.extend([
                f"\nLLM Performance:",
                f"  - Total calls: {self.metrics['llm_calls']}",
                f"  - Tokens per second: {statistics.median(tps):.1f} (median), {statistics.mean(tps):.1f} (mean)",
                f"  - Generation time: {statistics.median(gen_times):.2f}s (median), {statistics.mean(gen_times):.2f}s (mean)",
                f"  - Avg output tokens: {statistics.mean(self.metrics['llm_output_tokens']):.0f}"
            ])
        
        lines.append(f"{'='*80}\n")
        self.logger.info('\n'.join(lines))