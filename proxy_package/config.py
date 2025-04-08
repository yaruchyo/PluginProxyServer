# --- Configuration Loading ---
import os
from dotenv import load_dotenv
from .utils.logger import logger

# --- Load environment variables ---
# Load .env file if it exists. This should ideally happen once.
if load_dotenv():
    logger.info("✅ Loaded environment variables from .env file.")
else:
    logger.info("ℹ️ No .env file found, relying on system environment variables.")


# --- Read Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GOOGLE_MODEL", 'gemini-1.5-pro-latest')
LLM_MODEL = os.environ.get("LLM_MODEL", 'gemini')

# --- Log loaded configuration (optional, avoid logging API keys directly in production) ---
logger.info(f"🔧 Configuration Loaded:")
logger.info(f"   - GOOGLE_MODEL: {GEMINI_MODEL_NAME}")
if not API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY environment variable not set!")
else:
    # Mask the key partially if logging
    masked_key = API_KEY[:4] + "****" + API_KEY[-4:] if len(API_KEY) > 8 else "****"
    logger.info(f"   - GOOGLE_API_KEY: Loaded (ends in {masked_key[-4:]})") # Be careful logging keys
