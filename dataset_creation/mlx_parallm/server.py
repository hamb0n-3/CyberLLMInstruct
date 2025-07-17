# Save this full content as mlx_parallm/server.py

import argparse
import json
import logging
import re
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mlx_lm import load, generate
from pydantic import BaseModel

# --- Globals and Setup ---
MODEL = None
TOKENIZER = None
app = FastAPI()
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Model for API Data Validation ---
class CompletionsRequest(BaseModel):
    prompt: str

# --- The ONLY API Endpoint ---
# CORRECTED: Removed 'async'. FastAPI will now run this in a thread pool.
def completions(request: CompletionsRequest):
    """
    A single, simple endpoint that takes a prompt and returns the generated text.
    By defining it as a normal 'def' instead of 'async def', FastAPI automatically
    runs this blocking code in a separate thread, preventing the server from freezing.
    """
    try:
        # This is the blocking call that will now run safely in the background.
        response_text = generate(MODEL, TOKENIZER, request.prompt, max_tokens=1024, verbose=False)
        return JSONResponse(content={"text": response_text})
        
    except Exception as e:
        logging.error(f"Error during generation: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# Register the path after the function is defined
app.post("/v1/completions")(completions)

# --- Server Startup ---
@app.on_event("startup")
def startup_event():
    """Load the model when the server starts."""
    global MODEL, TOKENIZER
    # This is a bit of a hack to get args to the startup event, but it's effective for this script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    # Use parse_known_args to ignore uvicorn-specific args
    args, _ = parser.parse_known_args()
    
    logging.info(f"Loading model: {args.model}...")
    MODEL, TOKENIZER = load(args.model)
    logging.info("Model loaded successfully.")

def main():
    parser = argparse.ArgumentParser(description="High-Performance MLX Inference Server")
    parser.add_argument("--model", type=str, required=True, help="The Hugging Face repo or path to the model.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    # Pass arguments to uvicorn, which will then be available for the startup event
    uvicorn.run("mlx_parallm.server:app", host=args.host, port=args.port, log_config=None)

if __name__ == "__main__":
    # This main block allows running with `python -m mlx_parallm.server ...`
    # The server startup logic is now handled by uvicorn and FastAPI's startup event.
    main()