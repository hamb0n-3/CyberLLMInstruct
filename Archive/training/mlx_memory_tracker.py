#!/usr/bin/env python3
"""
MLX Memory Tracking Utility
Provides detailed memory tracking for MLX operations during training.
"""

import mlx.core as mx
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    step: int
    active_gb: float
    peak_gb: float
    delta_gb: float
    operation: str


class MLXMemoryTracker:
    """Track MLX memory usage with detailed logging and leak detection."""
    
    def __init__(self, log_interval: int = 10, leak_threshold_gb: float = 0.5):
        self.log_interval = log_interval
        self.leak_threshold_gb = leak_threshold_gb
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()
        self._last_memory_gb = 0.0
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current MLX memory statistics."""
        try:
            active_bytes = mx.metal.get_active_memory()
            peak_bytes = mx.metal.get_peak_memory()
            
            return {
                'active_gb': active_bytes / (1024**3),
                'peak_gb': peak_bytes / (1024**3),
                'active_mb': active_bytes / (1024**2),
                'peak_mb': peak_bytes / (1024**2),
            }
        except Exception as e:
            logger.debug(f"Could not get MLX memory stats: {e}")
            return {
                'active_gb': 0.0,
                'peak_gb': 0.0,
                'active_mb': 0.0,
                'peak_mb': 0.0,
            }
    
    def log_memory(self, step: int, operation: str = "training"):
        """Log current memory usage and check for leaks."""
        stats = self.get_memory_stats()
        active_gb = stats['active_gb']
        peak_gb = stats['peak_gb']
        delta_gb = active_gb - self._last_memory_gb
        
        # Create snapshot
        snapshot = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            step=step,
            active_gb=active_gb,
            peak_gb=peak_gb,
            delta_gb=delta_gb,
            operation=operation
        )
        self.snapshots.append(snapshot)
        
        # Log if at interval or significant change
        if step % self.log_interval == 0 or abs(delta_gb) > self.leak_threshold_gb:
            logger.info(
                f"[MLX Memory] Step {step} ({operation}): "
                f"Active={active_gb:.2f}GB, Peak={peak_gb:.2f}GB, "
                f"Delta={delta_gb:+.2f}GB"
            )
            
            # Warn about potential memory leak
            if delta_gb > self.leak_threshold_gb:
                logger.warning(
                    f"⚠️  Memory increased by {delta_gb:.2f}GB - potential memory leak!"
                )
                
        self._last_memory_gb = active_gb
    
    def check_memory_available(self, required_gb: float) -> bool:
        """Check if enough memory is available for an operation."""
        stats = self.get_memory_stats()
        # Rough estimate - MLX doesn't expose total memory easily
        # Assume 48GB GPU memory slice on M4 Max
        estimated_available = 48.0 - stats['active_gb']
        
        if estimated_available < required_gb:
            logger.warning(
                f"Insufficient memory: {estimated_available:.1f}GB available, "
                f"{required_gb:.1f}GB required"
            )
            return False
        return True
    
    def reset_peak_memory(self):
        """Reset MLX peak memory counter."""
        try:
            mx.metal.reset_peak_memory()
            logger.info("MLX peak memory counter reset")
        except Exception as e:
            logger.debug(f"Could not reset peak memory: {e}")
    
    def get_summary(self) -> str:
        """Get a summary of memory usage."""
        if not self.snapshots:
            return "No memory snapshots recorded"
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        max_snapshot = max(self.snapshots, key=lambda s: s.active_gb)
        
        # Detect leak pattern
        leak_detected = False
        if len(self.snapshots) > 10:
            # Check if memory consistently increases
            recent = self.snapshots[-10:]
            increases = sum(1 for s in recent if s.delta_gb > 0.1)
            if increases > 7:
                leak_detected = True
        
        summary = f"""
MLX Memory Usage Summary:
------------------------
Duration: {last.timestamp:.1f} seconds
Steps: {last.step}

Memory Usage:
- Initial: {first.active_gb:.2f}GB
- Final: {last.active_gb:.2f}GB  
- Peak: {max_snapshot.active_gb:.2f}GB (at step {max_snapshot.step})
- Total Increase: {last.active_gb - first.active_gb:.2f}GB

Memory Leak Detection: {'⚠️  LIKELY LEAK DETECTED' if leak_detected else '✓ No leak detected'}
"""
        return summary
    
    def save_report(self, filepath: str):
        """Save detailed memory report."""
        report = {
            'summary': self.get_summary(),
            'snapshots': [
                {
                    'step': s.step,
                    'time': s.timestamp,
                    'active_gb': s.active_gb,
                    'peak_gb': s.peak_gb,
                    'delta_gb': s.delta_gb,
                    'operation': s.operation
                }
                for s in self.snapshots
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Memory report saved to {filepath}")


def track_memory(func: Callable) -> Callable:
    """Decorator to track memory usage of a function."""
    def wrapper(*args, **kwargs):
        tracker = MLXMemoryTracker()
        
        # Log before
        step = kwargs.get('step', 0)
        tracker.log_memory(step, f"before_{func.__name__}")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Log after
        tracker.log_memory(step, f"after_{func.__name__}")
        
        return result
    
    return wrapper


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test memory tracking
    tracker = MLXMemoryTracker(log_interval=1)
    
    # Simulate some operations
    for i in range(5):
        # Allocate some memory
        x = mx.random.normal((1000, 1000))
        y = mx.random.normal((1000, 1000))
        z = x @ y
        mx.eval(z)
        
        tracker.log_memory(i, "matrix_multiply")
        time.sleep(1)
    
    print(tracker.get_summary())