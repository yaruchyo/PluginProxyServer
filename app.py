import os

import uvicorn
from dotenv import load_dotenv

# --- Load environment variables early ---
# This ensures config is loaded before proxy_package is imported
load_dotenv()

from proxy_package import app  # Import the app instance from __init__.py

# --- Import logger and the app instance ---
# Logger setup should happen within the package now
from proxy_package.utils.logger import logger


# --- Server Start Function ---
def start() -> None:
    # Configuration reading is now handled within the package's config.py
    # We just read host/port/workers/reload for uvicorn itself
    host = os.environ.get("HOST", "127.0.0.1") # Default to 127.0.0.1 for broader compatibility
    port = int(os.environ.get("PORT", 1234))
    reload = os.environ.get("RELOAD", "true").lower() == "true"
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))

    # Log level for uvicorn can be set if needed, but loguru handles app logs
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    if workers > 1 and reload:
        logger.warning("⚠️ WARNING: Running multiple workers with reload=True is not recommended. Setting workers to 1.")
        workers = 1

    logger.info(f"Starting Uvicorn server on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}, Log Level: {log_level}")
    logger.info(f"Proxy Backend configured via LLM_BACKEND env var.") # Reminder

    uvicorn.run(
        # Use the imported app instance directly
        app="proxy_package:app", # Point to the app object within the package
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        # Let loguru handle application logging format
        log_config=None, # Disable uvicorn's default logging config if using loguru intercept
    )

if __name__ == "__main__":
    start()