#!/usr/bin/env python3
"""
Performance Logging Module for CyberLLMInstruct Pipeline
Tracks and logs performance metrics for AI models and data pipelines
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import psutil
import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for model performance"""
    model_name: str
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    average_batch_size: float
    peak_memory_gb: float
    acceptance_rate: Optional[float] = None  # For speculative decoding
    
    
@dataclass
class PipelineMetrics:
    """Metrics for pipeline stage performance"""
    stage_name: str
    items_processed: int
    total_time_seconds: float
    items_per_second: float
    model_metrics: Optional[ModelMetrics] = None
    error_count: int = 0
    

class PerformanceLogger:
    """Centralized performance logging for the pipeline"""
    
    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Current session metrics
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_metrics: List[PipelineMetrics] = []
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Timing state
        self.stage_start_times: Dict[str, float] = {}
        self.stage_item_counts: Dict[str, int] = {}
        self.stage_error_counts: Dict[str, int] = {}
        
        # Model state
        self.model_token_counts: Dict[str, int] = {}
        self.model_start_times: Dict[str, float] = {}
        self.model_batch_sizes: Dict[str, List[int]] = {}
        
        logger.info(f"Performance logger initialized. Session ID: {self.session_id}")
    
    def start_pipeline_stage(self, stage_name: str):
        """Start timing a pipeline stage"""
        self.stage_start_times[stage_name] = time.time()
        self.stage_item_counts[stage_name] = 0
        self.stage_error_counts[stage_name] = 0
        logger.info(f"Started pipeline stage: {stage_name}")
    
    def update_stage_progress(self, stage_name: str, items_processed: int = 1):
        """Update progress for a pipeline stage"""
        if stage_name in self.stage_item_counts:
            self.stage_item_counts[stage_name] += items_processed
    
    def record_stage_error(self, stage_name: str):
        """Record an error in a pipeline stage"""
        if stage_name in self.stage_error_counts:
            self.stage_error_counts[stage_name] += 1
    
    def end_pipeline_stage(self, stage_name: str, model_name: Optional[str] = None):
        """End timing a pipeline stage and record metrics"""
        if stage_name not in self.stage_start_times:
            logger.warning(f"Stage {stage_name} was not started")
            return
        
        # Calculate stage metrics
        elapsed_time = time.time() - self.stage_start_times[stage_name]
        items_processed = self.stage_item_counts.get(stage_name, 0)
        items_per_second = items_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Get model metrics if applicable
        model_metrics = None
        if model_name and model_name in self.model_metrics:
            model_metrics = self.model_metrics[model_name]
        
        # Create pipeline metrics
        metrics = PipelineMetrics(
            stage_name=stage_name,
            items_processed=items_processed,
            total_time_seconds=elapsed_time,
            items_per_second=items_per_second,
            model_metrics=model_metrics,
            error_count=self.stage_error_counts.get(stage_name, 0)
        )
        
        self.pipeline_metrics.append(metrics)
        
        logger.info(f"Completed pipeline stage: {stage_name}")
        logger.info(f"  Items processed: {items_processed}")
        logger.info(f"  Time: {elapsed_time:.2f}s")
        logger.info(f"  Throughput: {items_per_second:.2f} items/s")
        if metrics.error_count > 0:
            logger.warning(f"  Errors: {metrics.error_count}")
        
        # Clean up
        del self.stage_start_times[stage_name]
    
    def start_model_inference(self, model_name: str):
        """Start timing model inference"""
        self.model_start_times[model_name] = time.time()
        if model_name not in self.model_token_counts:
            self.model_token_counts[model_name] = 0
            self.model_batch_sizes[model_name] = []
    
    def update_model_tokens(self, model_name: str, tokens_generated: int, batch_size: int = 1):
        """Update token count for a model"""
        if model_name in self.model_token_counts:
            self.model_token_counts[model_name] += tokens_generated
            self.model_batch_sizes[model_name].append(batch_size)
    
    def end_model_inference(self, model_name: str, acceptance_rate: Optional[float] = None):
        """End timing model inference and record metrics"""
        if model_name not in self.model_start_times:
            logger.warning(f"Model {model_name} inference was not started")
            return
        
        # Calculate metrics
        elapsed_time = time.time() - self.model_start_times[model_name]
        total_tokens = self.model_token_counts.get(model_name, 0)
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate average batch size
        batch_sizes = self.model_batch_sizes.get(model_name, [1])
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 1
        
        # Get memory usage
        memory_gb = psutil.virtual_memory().used / (1024 ** 3)
        
        # Create model metrics
        metrics = ModelMetrics(
            model_name=model_name,
            total_tokens_generated=total_tokens,
            total_time_seconds=elapsed_time,
            tokens_per_second=tokens_per_second,
            average_batch_size=avg_batch_size,
            peak_memory_gb=memory_gb,
            acceptance_rate=acceptance_rate
        )
        
        self.model_metrics[model_name] = metrics
        
        logger.info(f"Model inference completed: {model_name}")
        logger.info(f"  Tokens generated: {total_tokens}")
        logger.info(f"  Time: {elapsed_time:.2f}s")
        logger.info(f"  Tokens/s: {tokens_per_second:.2f}")
        logger.info(f"  Avg batch size: {avg_batch_size:.2f}")
        if acceptance_rate is not None:
            logger.info(f"  Acceptance rate: {acceptance_rate:.2%}")
    
    def save_session_log(self):
        """Save performance metrics to a JSON file"""
        log_file = self.log_dir / f"performance_{self.session_id}.json"
        
        # Prepare data for serialization
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "pipeline_metrics": [asdict(m) for m in self.pipeline_metrics],
            "model_metrics": {k: asdict(v) for k, v in self.model_metrics.items()},
            "summary": self._generate_summary()
        }
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Performance log saved to: {log_file}")
        return log_file
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the session performance"""
        total_items = sum(m.items_processed for m in self.pipeline_metrics)
        total_time = sum(m.total_time_seconds for m in self.pipeline_metrics)
        total_errors = sum(m.error_count for m in self.pipeline_metrics)
        
        summary = {
            "total_pipeline_stages": len(self.pipeline_metrics),
            "total_items_processed": total_items,
            "total_pipeline_time_seconds": total_time,
            "overall_throughput": total_items / total_time if total_time > 0 else 0,
            "total_errors": total_errors,
            "models_used": list(self.model_metrics.keys())
        }
        
        # Add model summary
        if self.model_metrics:
            total_tokens = sum(m.total_tokens_generated for m in self.model_metrics.values())
            avg_tokens_per_second = sum(m.tokens_per_second for m in self.model_metrics.values()) / len(self.model_metrics)
            
            summary["model_summary"] = {
                "total_tokens_generated": total_tokens,
                "average_tokens_per_second": avg_tokens_per_second,
                "peak_memory_gb": max(m.peak_memory_gb for m in self.model_metrics.values())
            }
        
        return summary
    
    def print_summary(self):
        """Print a summary of the session performance"""
        summary = self._generate_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Total pipeline stages: {summary['total_pipeline_stages']}")
        print(f"Total items processed: {summary['total_items_processed']}")
        print(f"Total time: {summary['total_pipeline_time_seconds']:.2f}s")
        print(f"Overall throughput: {summary['overall_throughput']:.2f} items/s")
        if summary['total_errors'] > 0:
            print(f"Total errors: {summary['total_errors']}")
        
        if "model_summary" in summary:
            print("\nModel Performance:")
            print(f"  Total tokens: {summary['model_summary']['total_tokens_generated']}")
            print(f"  Avg tokens/s: {summary['model_summary']['average_tokens_per_second']:.2f}")
            print(f"  Peak memory: {summary['model_summary']['peak_memory_gb']:.2f} GB")
        
        print("\nDetailed Stage Metrics:")
        for metrics in self.pipeline_metrics:
            print(f"\n  {metrics.stage_name}:")
            print(f"    Items: {metrics.items_processed}")
            print(f"    Time: {metrics.total_time_seconds:.2f}s")
            print(f"    Rate: {metrics.items_per_second:.2f} items/s")
            if metrics.error_count > 0:
                print(f"    Errors: {metrics.error_count}")
        
        print("="*60 + "\n")


# Global performance logger instance
_performance_logger: Optional[PerformanceLogger] = None


def get_performance_logger() -> PerformanceLogger:
    """Get or create the global performance logger"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def log_pipeline_stage(stage_name: str):
    """Decorator to automatically log pipeline stage performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_performance_logger()
            logger.start_pipeline_stage(stage_name)
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.record_stage_error(stage_name)
                raise
            finally:
                logger.end_pipeline_stage(stage_name)
        return wrapper
    return decorator