import os
from typing import (  # Keep necessary typing imports
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import configuration and initialize LLM Factory ---
# This import triggers the LLM client initialization based on .env
from .config import DEFAULT_MODEL_NAME, LLM_BACKEND

# --- Import Routers (Now safe to import as config/context are loaded) ---
from .entrypoint_layer.chat import chat_router
from .entrypoint_layer.completions import completions_router
from .reporitory_layer.llm import (
    llm_factory,  # Import the module to ensure initialization
)

# --- Import logger first ---
from .utils.logger import logger

# --- Create FastAPI app ---
app = FastAPI(title=f"OpenAI-Compatible Proxy ({LLM_BACKEND.upper()} Backend)")

# --- Include routers ---
app.include_router(chat_router)
app.include_router(completions_router)

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Add a root endpoint for health check / info
@app.get("/")
async def read_root():
    # Access the initialized client via the getter for status check
    llm_status = "Unavailable"
    try:
        client = llm_factory.get_llm_client()
        llm_status = f"Available ({type(client).__name__})"
    except Exception as e:
        llm_status = f"Error ({e})"

    return {
        "message": f"OpenAI-Compatible Proxy is running.",
        "llm_backend": LLM_BACKEND.upper(),
        "default_model": DEFAULT_MODEL_NAME,
        "llm_client_status": llm_status,
    }


logger.info(f"✅ FastAPI application configured successfully for {LLM_BACKEND.upper()} backend.")

# --- Expose core components if needed (less common for __init__) ---
# __all__ = ["app", "logger", "LLM_BACKEND", "DEFAULT_MODEL_NAME"]