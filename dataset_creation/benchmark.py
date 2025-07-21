"""
Benchmark tracking module for performance monitoring and statistics collection.
"""

import time
import logging
import psutil
import os
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
import statistics


class BenchmarkTracker:
    """Tracks detailed performance metrics with comprehensive statistics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 log_interval_initial: int = 120, 
                 log_interval_periodic: int = 1800):
        """
        Initialize benchmark tracker.
        
        Args:
            logger: Optional logger instance. If None, uses module logger.
            log_interval_initial: Initial interval for logging stats (seconds)
            log_interval_periodic: Periodic interval for logging stats (seconds)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = time.time()
        self.log_interval_initial = log_interval_initial
        self.log_interval_periodic = log_interval_periodic
        self.next_log_time = self.start_time + log_interval_initial
        self.initial_logged = False
        
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
            'success_by_score': defaultdict(lambda: {'passed': 0, 'failed': 0})
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
    
    def record_entry(self, passed: bool, debug_info: Dict, processing_time: float = 0, entry_size: int = 0):
        """Record metrics for a single entry."""
        self.metrics['entries_processed'] += 1
        
        if passed:
            self.metrics['entries_passed'] += 1
        else:
            self.metrics['entries_failed'] += 1
            reason = debug_info.get('reason', 'Unknown')
            self.metrics['rejection_reasons'][reason] += 1
        
        # Track keyword distribution
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
    
    def should_log(self) -> bool:
        """Check if it's time to log benchmark stats."""
        current_time = time.time()
        if current_time >= self.next_log_time:
            if not self.initial_logged:
                self.initial_logged = True
                self.next_log_time = self.start_time + self.log_interval_periodic
            else:
                self.next_log_time = current_time + self.log_interval_periodic
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
        self.next_log_time = self.start_time + self.log_interval_initial
        self.initial_logged = False
        
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
            'success_by_score': defaultdict(lambda: {'passed': 0, 'failed': 0})
        }
        
        self._record_memory()