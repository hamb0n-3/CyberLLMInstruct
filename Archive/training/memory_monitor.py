#!/usr/bin/env python3
"""
Memory monitoring utility for training scripts.
Provides real-time memory usage tracking for macOS.
"""

import subprocess
import time
import psutil
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.get_memory_usage()
        self.peak_memory = self.start_memory
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        # Process-specific memory
        process_info = self.process.memory_info()
        
        # System-wide memory
        vm = psutil.virtual_memory()
        
        # macOS specific memory info
        try:
            # Get more detailed memory stats using vm_stat
            vm_stat = subprocess.check_output(['vm_stat'], text=True)
            page_size = 16384  # Default page size on Apple Silicon
            
            stats = {}
            for line in vm_stat.split('\n'):
                if 'page size' in line:
                    page_size = int(line.split()[-2])
                elif ':' in line:
                    parts = line.split(':')
                    key = parts[0].strip()
                    value = parts[1].strip().rstrip('.')
                    if value.isdigit():
                        stats[key] = int(value) * page_size / (1024**3)  # Convert to GB
        except:
            stats = {}
        
        memory_info = {
            'process_rss_gb': process_info.rss / (1024**3),
            'process_vms_gb': process_info.vms / (1024**3),
            'system_used_gb': vm.used / (1024**3),
            'system_available_gb': vm.available / (1024**3),
            'system_percent': vm.percent,
            **stats
        }
        
        return memory_info
    
    def log_memory_usage(self, step: Optional[int] = None):
        """Log current memory usage."""
        current = self.get_memory_usage()
        
        # Update peak memory
        if current['process_rss_gb'] > self.peak_memory['process_rss_gb']:
            self.peak_memory = current
        
        # Calculate deltas
        delta_rss = current['process_rss_gb'] - self.start_memory['process_rss_gb']
        
        # Format message
        msg = f"Memory Usage"
        if step is not None:
            msg += f" (Step {step})"
        msg += f": Process={current['process_rss_gb']:.1f}GB "
        msg += f"(+{delta_rss:.1f}GB from start), "
        msg += f"System={current['system_percent']:.1f}% used, "
        msg += f"Available={current['system_available_gb']:.1f}GB"
        
        logger.info(msg)
        
        # Warn if memory usage is high
        if current['system_percent'] > 90:
            logger.warning(f"High system memory usage: {current['system_percent']:.1f}%")
        if current['process_rss_gb'] > 50:
            logger.warning(f"High process memory usage: {current['process_rss_gb']:.1f}GB")
    
    def get_summary(self) -> str:
        """Get memory usage summary."""
        current = self.get_memory_usage()
        
        summary = f"""
Memory Usage Summary:
- Start: {self.start_memory['process_rss_gb']:.1f}GB
- Current: {current['process_rss_gb']:.1f}GB
- Peak: {self.peak_memory['process_rss_gb']:.1f}GB
- System Available: {current['system_available_gb']:.1f}GB
"""
        return summary


def monitor_training_memory(log_interval: int = 60):
    """Background memory monitoring during training."""
    monitor = MemoryMonitor()
    
    logger.info("Starting memory monitoring...")
    monitor.log_memory_usage()
    
    step = 0
    try:
        while True:
            time.sleep(log_interval)
            step += 1
            monitor.log_memory_usage(step)
    except KeyboardInterrupt:
        logger.info("Memory monitoring stopped")
        print(monitor.get_summary())


if __name__ == "__main__":
    # Test memory monitoring
    logging.basicConfig(level=logging.INFO)
    monitor_training_memory(10)  # Log every 10 seconds for testing