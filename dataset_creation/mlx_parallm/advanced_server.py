#!/usr/bin/env python3
"""
Advanced MLX Inference Server with Speculative Decoding and Continuous Batching

This server provides high-performance inference with:
- Adaptive speculative decoding for lower latency
- Continuous batching for higher throughput
- Async API endpoints
"""

import argparse
import json
import logging
import asyncio
import time
from typing import List, Dict, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from mlx_lm import load

from .speculative_decoding import SpeculativeConfig, SpeculativeDecodingEngine
from .continuous_batching import BatchConfig, AsyncContinuousBatchingEngine
from .combined_inference import CombinedConfig, AsyncCombinedEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
app = FastAPI(title="Advanced MLX Inference Server")
inference_engine: Optional[AsyncContinuousBatchingEngine] = None
speculative_engine: Optional[SpeculativeDecodingEngine] = None
combined_engine: Optional[AsyncCombinedEngine] = None
use_speculative = False
use_combined = False


# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    use_speculative: Optional[bool] = None  # Override global setting


class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float
    method: str  # "speculative" or "continuous_batching"


class BatchGenerationResponse(BaseModel):
    results: List[GenerationResponse]
    total_time: float
    batch_size: int


class ServerStats(BaseModel):
    total_requests: int
    average_latency: float
    speculative_stats: Optional[Dict] = None
    batching_stats: Optional[Dict] = None
    combined_stats: Optional[Dict] = None
    mode: str


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/v1/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text for a single prompt"""
    start_time = time.time()
    
    # Decide which engine to use
    use_spec = request.use_speculative if request.use_speculative is not None else use_speculative
    
    try:
        if use_combined and combined_engine:
            # Use combined speculative + continuous batching
            text = await combined_engine.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            method = "combined"
        elif use_spec and speculative_engine:
            # Use speculative decoding only
            if request.stream:
                # Return streaming response
                return StreamingResponse(
                    speculative_stream_generator(request),
                    media_type="text/event-stream"
                )
            else:
                text = speculative_engine.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    stream=False
                )
                method = "speculative"
        else:
            # Use continuous batching only
            if not inference_engine:
                raise HTTPException(status_code=503, detail="Inference engine not initialized")
            text = await inference_engine.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            method = "continuous_batching"
        
        generation_time = time.time() - start_time
        
        # Handle token counting for both string and generator types
        if isinstance(text, str):
            tokens_generated = len(text.split())
        else:
            tokens_generated = 0  # Cannot count tokens from generator
        
        return GenerationResponse(
            text=text if isinstance(text, str) else "[Streaming response]",
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            method=method
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate_batch", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    start_time = time.time()
    
    try:
        # Use combined engine if available, otherwise fall back to continuous batching
        if use_combined and combined_engine:
            texts = await combined_engine.generate_batch(
                request.prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            method = "combined"
        else:
            texts = await inference_engine.generate_batch(
                request.prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            method = "continuous_batching"
        
        results = []
        for i, text in enumerate(texts):
            results.append(GenerationResponse(
                text=text,
                tokens_generated=len(text.split()),
                generation_time=0,  # Individual times not tracked in batch
                method=method
            ))
        
        total_time = time.time() - start_time
        
        return BatchGenerationResponse(
            results=results,
            total_time=total_time,
            batch_size=len(request.prompts)
        )
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats", response_model=ServerStats)
async def get_stats():
    """Get server statistics"""
    stats = {
        "total_requests": 0,
        "average_latency": 0.0,
        "mode": "unknown"
    }
    
    if use_combined and combined_engine:
        combined_stats = combined_engine.get_stats()
        stats["combined_stats"] = combined_stats
        stats["total_requests"] = combined_stats.get("total_requests", 0)
        stats["average_latency"] = combined_stats.get("avg_latency", 0.0)
        stats["mode"] = "combined"
    elif inference_engine:
        batching_stats = inference_engine.engine.get_stats()
        stats["batching_stats"] = batching_stats
        stats["total_requests"] = batching_stats.get("total_requests", 0)
        stats["average_latency"] = batching_stats.get("avg_latency", 0.0)
        stats["mode"] = "continuous_batching"
    
    if speculative_engine and not use_combined:
        # Add speculative decoding stats if available
        stats["speculative_stats"] = {
            "acceptance_rate": speculative_engine.decoder.acceptance_rate,
            "draft_length": speculative_engine.decoder.draft_length
        }
        if not inference_engine:
            stats["mode"] = "speculative"
    
    return ServerStats(**stats)


@app.post("/v1/configure")
async def configure_server(
    use_speculative_decoding: Optional[bool] = None,
    use_combined_mode: Optional[bool] = None,
    max_batch_size: Optional[int] = None,
    batch_timeout_ms: Optional[int] = None
):
    """Configure server parameters"""
    global use_speculative, use_combined
    
    if use_speculative_decoding is not None:
        use_speculative = use_speculative_decoding
    
    if use_combined_mode is not None:
        use_combined = use_combined_mode
    
    if inference_engine and (max_batch_size or batch_timeout_ms):
        if max_batch_size:
            inference_engine.engine.config.max_batch_size = max_batch_size
        if batch_timeout_ms:
            inference_engine.engine.config.timeout_ms = batch_timeout_ms
    
    if combined_engine and (max_batch_size or batch_timeout_ms):
        if max_batch_size:
            combined_engine.engine.config.max_batch_size = max_batch_size
        if batch_timeout_ms:
            combined_engine.engine.config.batch_timeout_ms = batch_timeout_ms
    
    return {
        "use_speculative": use_speculative,
        "use_combined": use_combined,
        "max_batch_size": combined_engine.engine.config.max_batch_size if combined_engine else (inference_engine.engine.config.max_batch_size if inference_engine else None),
        "batch_timeout_ms": combined_engine.engine.config.batch_timeout_ms if combined_engine else (inference_engine.engine.config.timeout_ms if inference_engine else None)
    }


# Streaming helper
async def speculative_stream_generator(request: GenerationRequest):
    """Generate streaming response with speculative decoding"""
    try:
        for chunk in speculative_engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            stream=True
        ):
            if chunk:  # Don't send empty chunks
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize models and engines on startup"""
    global inference_engine, speculative_engine, combined_engine, use_speculative, use_combined
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Main model path")
    parser.add_argument("--draft-model", type=str, help="Draft model for speculative decoding")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=int, default=50)
    parser.add_argument("--use-speculative", action="store_true")
    parser.add_argument("--use-combined", action="store_true", help="Use combined speculative + continuous batching")
    parser.add_argument("--kv-bits", type=int, default=4, help="KV cache quantization bits (4 or 8)")
    parser.add_argument("--kv-group-size", type=int, default=32, help="KV cache group size")
    
    # Parse known args to handle uvicorn args
    args, _ = parser.parse_known_args()
    
    # Determine mode
    use_combined = args.use_combined and args.draft_model
    
    if use_combined:
        # Initialize combined engine
        logger.info(f"Initializing combined engine with model: {args.model} and draft: {args.draft_model}")
        combined_config = CombinedConfig(
            target_model_path=args.model,
            draft_model_path=args.draft_model,
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
            enable_speculative=True,
            enable_kv_cache=True,  # Enable KV cache by default
            kv_bits=8,  # Use 4-bit quantization
            kv_group_size=32  # Standard group size
        )
        combined_engine = AsyncCombinedEngine(combined_config)
        logger.info("Combined speculative + continuous batching engine initialized")
    else:
        # Load main model and tokenizer for separate engines
        logger.info(f"Loading main model: {args.model}")
        model, tokenizer = load(args.model)
        
        # Initialize continuous batching engine
        batch_config = BatchConfig(
            max_batch_size=args.max_batch_size,
            timeout_ms=args.batch_timeout_ms,
            padding_token_id=tokenizer.pad_token_id or 0
        )
        inference_engine = AsyncContinuousBatchingEngine(model, tokenizer, batch_config)
        logger.info("Continuous batching engine initialized")
        
        # Initialize speculative decoding if draft model provided
        if args.draft_model:
            logger.info(f"Initializing speculative decoding with draft model: {args.draft_model}")
            spec_config = SpeculativeConfig(
                draft_model_path=args.draft_model,
                target_model_path=args.model,
                max_draft_tokens=5
            )
            speculative_engine = SpeculativeDecodingEngine(spec_config)
            use_speculative = args.use_speculative
            logger.info("Speculative decoding engine initialized")
    
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if inference_engine:
        inference_engine.stop()
    if combined_engine:
        combined_engine.stop()
    logger.info("Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Advanced MLX Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Main model path")
    parser.add_argument("--draft-model", type=str, help="Draft model for speculative decoding")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=int, default=50)
    parser.add_argument("--use-speculative", action="store_true")
    parser.add_argument("--use-combined", action="store_true", help="Use combined speculative + continuous batching")
    parser.add_argument("--kv-bits", type=int, default=4, help="KV cache quantization bits (4 or 8)")
    parser.add_argument("--kv-group-size", type=int, default=32, help="KV cache group size")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "mlx_parallm.advanced_server:app",
        host=args.host,
        port=args.port,
        log_config=None
    )


if __name__ == "__main__":
    main() 