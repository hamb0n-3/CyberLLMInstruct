#!/usr/bin/env python3
"""
Shared utilities for cybersecurity data processing scripts.
"""

import json
import logging
import time
import os
from typing import Dict, List, Optional, Any
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
        """
        Initialize benchmark tracker.
        
        Args:
            logger: Optional logger instance. If None, uses module logger.
        """
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
            'keyword_histogram': defaultdict(int),
            'score_distribution': defaultdict(int),
            'memory_usage': [],
            'processing_times': [],
            'entry_sizes': [],
            'success_by_score': defaultdict(lambda: {'passed': 0, 'failed': 0}),
            # LLM performance metrics
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
                'rss_mb': mem_info.rss / (1024 * 1024),  # Convert to MB
                'vms_mb': mem_info.vms / (1024 * 1024)   # Convert to MB
            })
        except Exception as e:
            self.logger.debug(f"Memory recording failed: {e}")
    
    def record_entry(self, passed: bool, debug_info: Dict = None, processing_time: float = 0, entry_size: int = 0):
        """Record metrics for a single entry."""
        self.metrics['entries_processed'] += 1
        
        if passed:
            self.metrics['entries_passed'] += 1
        else:
            self.metrics['entries_failed'] += 1
            if debug_info:
                reason = debug_info.get('reason', 'Unknown')
                self.metrics['rejection_reasons'][reason] += 1
        
        # Track keyword distribution
        if debug_info:
            for keyword in debug_info.get('matched_keywords', []):
                self.metrics['keyword_histogram'][keyword] += 1
            
            # Track score distribution with more granular buckets
            score = debug_info.get('score', 0)
            if isinstance(score, (int, float)):
                score_bucket = f"{int(score)}"  # Individual score tracking
                self.metrics['score_distribution'][score_bucket] += 1
                
                # Track success rate by score
                if passed:
                    self.metrics['success_by_score'][int(score)]['passed'] += 1
                else:
                    self.metrics['success_by_score'][int(score)]['failed'] += 1
        
        # Track processing speed and size
        if processing_time > 0:
            self.metrics['processing_times'].append(processing_time)
        if entry_size > 0:
            self.metrics['entry_sizes'].append(entry_size)
    
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics as a dictionary."""
        self._record_memory()
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_minutes': elapsed_time / 60,
            'total_entries': self.metrics['entries_processed'],
            'passed_entries': self.metrics['entries_passed'],
            'failed_entries': self.metrics['entries_failed'],
            'pass_rate': self.metrics['entries_passed'] / self.metrics['entries_processed'] * 100 if self.metrics['entries_processed'] > 0 else 0,
            'files_completed': self.metrics['files_completed'],
            'rejection_reasons': dict(self.metrics['rejection_reasons']),
            'top_keywords': dict(sorted(self.metrics['keyword_histogram'].items(), key=lambda x: x[1], reverse=True)[:20]),
            'score_distribution': dict(sorted(self.metrics['score_distribution'].items(), key=lambda x: int(x[0]))),
        }
        
        # Processing time statistics
        if self.metrics['processing_times']:
            times = self.metrics['processing_times']
            stats['processing_times'] = {
                'count': len(times),
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times),
                'p90': self.get_percentile(times, 90),
                'p95': self.get_percentile(times, 95),
                'p99': self.get_percentile(times, 99),
                'total': sum(times),
                'entries_per_second': len(times) / sum(times) if sum(times) > 0 else 0
            }
        
        # Memory statistics
        if self.metrics['memory_usage']:
            latest_mem = self.metrics['memory_usage'][-1]
            all_rss = [m['rss_mb'] for m in self.metrics['memory_usage']]
            stats['memory'] = {
                'current_rss_mb': latest_mem['rss_mb'],
                'current_vms_mb': latest_mem['vms_mb'],
                'peak_rss_mb': max(all_rss),
                'min_rss_mb': min(all_rss),
                'samples': len(self.metrics['memory_usage'])
            }
        
        # Stage timing statistics
        stage_stats = {}
        for stage, times in self.metrics['stage_times'].items():
            if times:
                stage_stats[stage] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times) if len(times) > 1 else times[0],
                    'min': min(times),
                    'max': max(times)
                }
        stats['stage_times'] = stage_stats
        
        # Success rate by score
        score_success = {}
        for score, counts in sorted(self.metrics['success_by_score'].items()):
            total = counts['passed'] + counts['failed']
            if total > 0:
                score_success[score] = {
                    'total': total,
                    'passed': counts['passed'],
                    'failed': counts['failed'],
                    'pass_rate': counts['passed'] / total * 100
                }
        stats['success_by_score'] = score_success
        
        # LLM performance statistics
        if self.metrics['llm_tokens_per_second']:
            tps = self.metrics['llm_tokens_per_second']
            input_tokens = self.metrics['llm_input_tokens']
            output_tokens = self.metrics['llm_output_tokens']
            gen_times = self.metrics['llm_generation_times']
            
            stats['llm_performance'] = {
                'total_calls': self.metrics['llm_calls'],
                'total_input_tokens': sum(input_tokens),
                'total_output_tokens': sum(output_tokens),
                'total_generation_time': sum(gen_times),
                'tokens_per_second': {
                    'mean': statistics.mean(tps),
                    'median': statistics.median(tps),
                    'min': min(tps),
                    'max': max(tps),
                    'p90': self.get_percentile(tps, 90),
                    'p95': self.get_percentile(tps, 95)
                },
                'generation_times': {
                    'mean': statistics.mean(gen_times),
                    'median': statistics.median(gen_times),
                    'min': min(gen_times),
                    'max': max(gen_times)
                },
                'avg_input_tokens': statistics.mean(input_tokens),
                'avg_output_tokens': statistics.mean(output_tokens)
            }
        
        return stats
    
    def get_summary(self) -> str:
        """Get formatted summary string."""
        stats = self.get_statistics()
        
        lines = [
            f"\n{'='*80}",
            f"BENCHMARK STATISTICS - Elapsed: {stats['elapsed_time_minutes']:.1f} minutes",
            f"{'='*80}",
            f"Progress Summary:",
            f"  - Total entries processed: {stats['total_entries']}",
            f"  - Entries passed filter: {stats['passed_entries']} ({stats['pass_rate']:.1f}%)",
            f"  - Entries failed filter: {stats['failed_entries']}",
            f"  - Files completed: {stats['files_completed']}"
        ]
        
        # Processing speed
        if 'processing_times' in stats:
            pt = stats['processing_times']
            lines.extend([
                f"\nProcessing Speed:",
                f"  - Mean time per entry: {pt['mean']:.3f}s",
                f"  - Median time per entry: {pt['median']:.3f}s",
                f"  - 90th percentile: {pt['p90']:.3f}s",
                f"  - 95th percentile: {pt['p95']:.3f}s",
                f"  - Throughput: {pt['entries_per_second']:.2f} entries/second"
            ])
        
        # Memory usage
        if 'memory' in stats:
            mem = stats['memory']
            lines.extend([
                f"\nMemory Usage:",
                f"  - Current RSS: {mem['current_rss_mb']:.1f} MB",
                f"  - Peak RSS: {mem['peak_rss_mb']:.1f} MB"
            ])
        
        # Top rejection reasons
        if stats['rejection_reasons']:
            lines.append(f"\nTop Rejection Reasons:")
            sorted_reasons = sorted(stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_reasons:
                pct = count / stats['failed_entries'] * 100 if stats['failed_entries'] > 0 else 0
                lines.append(f"  - {reason}: {count} ({pct:.1f}%)")
        
        # Score distribution
        if stats['score_distribution']:
            lines.append(f"\nScore Distribution:")
            for score, count in sorted(stats['score_distribution'].items(), key=lambda x: int(x[0])):
                lines.append(f"  - Score {score}: {count} entries")
        
        # LLM performance
        if 'llm_performance' in stats:
            llm = stats['llm_performance']
            lines.extend([
                f"\nLLM Performance:",
                f"  - Total calls: {llm['total_calls']}",
                f"  - Tokens per second: {llm['tokens_per_second']['median']:.1f} (median), {llm['tokens_per_second']['mean']:.1f} (mean)",
                f"  - 90th percentile: {llm['tokens_per_second']['p90']:.1f} tokens/sec",
                f"  - Generation time: {llm['generation_times']['median']:.2f}s (median), {llm['generation_times']['mean']:.2f}s (mean)",
                f"  - Avg tokens: {llm['avg_input_tokens']:.0f} input, {llm['avg_output_tokens']:.0f} output"
            ])
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)
    
    def log_benchmark_stats(self, force: bool = False):
        """Log comprehensive benchmark statistics."""
        if not force and not self.should_log():
            return
        
        self.logger.info(self.get_summary())
    
    def export_json(self, filepath: str):
        """Export statistics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.file_start_time = None
        self.file_log_index = 0
        self.next_log_time = None
        
        # Reset all metrics
        self.metrics = {
            'entries_processed': 0,
            'entries_passed': 0,
            'entries_failed': 0,
            'files_completed': 0,
            'stage_times': defaultdict(list),
            'rejection_reasons': defaultdict(int),
            'keyword_histogram': defaultdict(int),
            'score_distribution': defaultdict(int),
            'memory_usage': [],
            'processing_times': [],
            'entry_sizes': [],
            'success_by_score': defaultdict(lambda: {'passed': 0, 'failed': 0}),
            # LLM performance metrics
            'llm_calls': 0,
            'llm_tokens_per_second': [],
            'llm_input_tokens': [],
            'llm_output_tokens': [],
            'llm_generation_times': []
        }
        
        self._record_memory()