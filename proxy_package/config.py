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
LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini").lower() # Default to gemini if not set

# --- Gemini Configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_DEFAULT_MODEL = os.environ.get("GOOGLE_MODEL", 'gemini-1.5-pro-latest')

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview") # Use a reasonable default
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_MODEL") # Your primary deployment name
AZURE_OPENAI_MAX_RETRIES = int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", 3))

# --- Log loaded configuration ---
logger.info(f"🔧 Configuration Loaded:")
logger.info(f"   - Selected LLM Backend: {LLM_BACKEND.upper()}")

if LLM_BACKEND == 'gemini':
    logger.info(f"   - Gemini Default Model: {GEMINI_DEFAULT_MODEL}")
    if not GEMINI_API_KEY:
        logger.warning("⚠️ GOOGLE_API_KEY environment variable not set (required for Gemini backend)!")
    else:
        masked_key = GEMINI_API_KEY[:4] + "****" + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "****"
        logger.info(f"   - GOOGLE_API_KEY: Loaded (ends in {masked_key[-4:]})")

elif LLM_BACKEND == 'azure':
    logger.info(f"   - Azure Endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"   - Azure Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    logger.info(f"   - Azure API Version: {AZURE_OPENAI_API_VERSION}")
    if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
        logger.warning("⚠️ Missing one or more Azure OpenAI environment variables (required for Azure backend)!")
    else:
        logger.info(f"   - AZURE_OPENAI_KEY: Loaded")
else:
    logger.error(f"❌ Invalid LLM_BACKEND specified: '{LLM_BACKEND}'. Use 'gemini' or 'azure'.")
    # Decide whether to exit or default
    # exit(1) # Or raise ConfigurationError

# --- Define Default Model based on Backend ---
# This helps the entrypoints know which model name to use if none is specified in the request
DEFAULT_MODEL_NAME = GEMINI_DEFAULT_MODEL if LLM_BACKEND == 'gemini' else AZURE_OPENAI_DEPLOYMENT_NAME