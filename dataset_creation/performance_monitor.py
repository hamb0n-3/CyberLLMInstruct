#!/usr/bin/env python3
"""
Performance monitoring utilities for the data pipeline
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics for pipeline operations"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Processing metrics
    files_processed: int = 0
    entries_processed: int = 0
    entries_retained: int = 0
    entries_filtered: int = 0
    
    # LLM metrics
    llm_calls: int = 0
    llm_tokens_used: int = 0
    rule_based_filtered: int = 0
    
    # System metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Timing breakdown
    file_io_time: float = 0.0
    rule_filter_time: float = 0.0
    llm_processing_time: float = 0.0
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.avg_cpu_percent = process.cpu_percent(interval=0.1)
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate final statistics"""
        if not self.end_time:
            self.end_time = time.time()
        
        total_time = self.end_time - self.start_time
        
        return {
            "summary": {
                "total_time_seconds": round(total_time, 2),
                "files_processed": self.files_processed,
                "entries_processed": self.entries_processed,
                "entries_retained": self.entries_retained,
                "retention_rate": round(self.entries_retained / max(1, self.entries_processed) * 100, 2),
            },
            "performance": {
                "entries_per_second": round(self.entries_processed / max(1, total_time), 2),
                "avg_time_per_file": round(total_time / max(1, self.files_processed), 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            },
            "filtering": {
                "rule_based_filtered": self.rule_based_filtered,
                "llm_verified": self.llm_calls,
                "llm_reduction_percent": round((1 - self.llm_calls / max(1, self.entries_processed)) * 100, 2),
            },
            "timing_breakdown": {
                "file_io_percent": round(self.file_io_time / max(1, total_time) * 100, 2),
                "rule_filter_percent": round(self.rule_filter_time / max(1, total_time) * 100, 2),
                "llm_processing_percent": round(self.llm_processing_time / max(1, total_time) * 100, 2),
            }
        }
    
    def save_report(self, output_dir: Path):
        """Save performance report"""
        stats = self.calculate_stats()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f"performance_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
        return stats


class ProgressTracker:
    """Real-time progress tracking with ETA"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.description = description
        
    def update(self, items: int = 1):
        """Update progress"""
        self.processed_items += items
        
        if self.processed_items > 0:
            elapsed = time.time() - self.start_time
            rate = self.processed_items / elapsed
            eta = (self.total_items - self.processed_items) / rate if rate > 0 else 0
            
            percent = (self.processed_items / self.total_items) * 100
            
            # Log progress every 10%
            if self.processed_items % max(1, self.total_items // 10) == 0:
                logger.info(
                    f"{self.description}: {percent:.1f}% "
                    f"({self.processed_items}/{self.total_items}) "
                    f"Rate: {rate:.1f}/s, ETA: {eta:.1f}s"
                )
    
    def finish(self):
        """Mark as complete"""
        elapsed = time.time() - self.start_time
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.description} completed: {self.processed_items} items "
            f"in {elapsed:.1f}s ({rate:.1f} items/s)"
        )


def benchmark_function(func):
    """Decorator to benchmark function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper