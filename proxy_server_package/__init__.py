import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional, List, Dict, Union, Iterator # Keep necessary typing imports

# --- Import logger first ---
from .utils.logger import logger

# --- Import configuration and context (which initializes LLM) ---
# These imports should happen early and should not depend on entrypoints/services yet.
from .config import GEMINI_MODEL_NAME # Import only what's needed directly here, if anything
from .context import gemini_llm # Import the initialized LLM client

# --- Import Routers (Now safe to import as config/context are loaded) ---
from .entrypoint_layer.chat import chat_router
from .entrypoint_layer.completions import completions_router # Corrected name

# --- Create FastAPI app ---
app = FastAPI(title="Gemini OpenAI-Compatible Proxy")

# --- Include routers ---
app.include_router(chat_router)
app.include_router(completions_router)

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    # Allow all origins for development, restrict in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # Allow all standard methods
    allow_headers=["*"], # Allow all standard headers
)

# Optional: Assign logger/context to app state if needed by middleware/dependencies later
# app.state.logger = logger
# app.state.gemini_llm = gemini_llm # Can be useful for dependency injection

logger.info("✅ FastAPI application configured successfully.")

# --- Expose core components if needed for external use (less common for __init__) ---
# __all__ = ["app", "gemini_llm", "logger", "GEMINI_MODEL_NAME"]
