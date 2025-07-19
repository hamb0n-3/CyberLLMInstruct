#!/usr/bin/env python3
"""
Simplified MLX Inference Server - Fast and efficient
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_creation.mlx_parallm.simple_inference import create_engine, AsyncInferenceEngine, InferenceConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MLX Inference Server")
engine: Optional[AsyncInferenceEngine] = None


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/v1/generate")
async def generate(request: GenerationRequest):
    """Generate text for a single prompt"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start_time = time.time()
    
    try:
        text = await engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "text": text,
            "tokens_generated": len(text.split()),  # Approximate
            "generation_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate_batch")
async def generate_batch(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start_time = time.time()
    
    try:
        results = await engine.generate_batch(
            request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "results": [{"text": text, "tokens_generated": len(text.split())} for text in results],
            "total_time": time.time() - start_time,
            "batch_size": len(request.prompts)
        }
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Simple MLX Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Path to MLX model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--batch-size", type=int, default=8, help="Max batch size")
    
    args = parser.parse_args()
    
    # Initialize engine
    global engine
    config = InferenceConfig(
        model_path=args.model,
        batch_size=args.batch_size
    )
    engine = AsyncInferenceEngine(config)
    
    logger.info(f"Starting server with model: {args.model}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()