import os
import time
import uuid
import asyncio
import json
import traceback
import threading # Keep if needed elsewhere, not directly used here

# --- Third-party Imports ---
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from plugin_proxy_server.utils.logger import logger
from plugin_proxy_server import app # Import the app instance
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from dotenv import load_dotenv
# --- Load environment variables ---
load_dotenv() # This should ideally be done once, potentially in __init__.py or here if running this file directly

# --- Server Start Function ---
def start() -> None:
    logger.info("Starting FastAPI proxy...")
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 1234))
    reload = os.environ.get("RELOAD", "true").lower() == "true"

    if workers > 1 and reload:
        logger.warning("⚠️ WARNING: Running multiple workers with reload=True is not recommended. Setting workers to 1.")
        workers = 1

    logger.info(f"Starting Uvicorn server on {host}:{port} with {workers} worker(s)")
    logger.info(f"Reloading enabled: {reload}")

    uvicorn.run(
        app="fast_api_proxy:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None, # Use loguru
        workers=workers,
    )

if __name__ == "__main__":
    start()