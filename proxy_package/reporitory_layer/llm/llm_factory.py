import os
from typing import Union
from fastapi import HTTPException
# --- Application Context / Shared Resources ---
from ...config import ( # Use relative imports
    LLM_BACKEND,
    GEMINI_API_KEY, GEMINI_DEFAULT_MODEL,
    AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_MAX_RETRIES
)
from .gemini_llm import GeminiLLM
from .azure_llm import AzureLLM
from ...utils.logger import logger # Use relative import

# --- Define the type hint for the LLM client ---
LLMClient = Union[GeminiLLM, AzureLLM, None] # Allow None initially

_llm: LLMClient = None

# --- Initialize the selected LLM client ---
try:
    if LLM_BACKEND == "gemini":
        if not GEMINI_API_KEY or not GEMINI_DEFAULT_MODEL:
             logger.error("❌ Cannot initialize GeminiLLM: Missing GOOGLE_API_KEY or GOOGLE_MODEL in config.")
             raise ValueError("Missing Gemini configuration.")
        _llm = GeminiLLM(api_key=GEMINI_API_KEY, model_name=GEMINI_DEFAULT_MODEL)
        logger.info(f"✅ Initialized LLM client: Gemini ({GEMINI_DEFAULT_MODEL})")

    elif LLM_BACKEND == "azure":
        if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
            logger.error("❌ Cannot initialize AzureLLM: Missing one or more Azure OpenAI environment variables.")
            raise ValueError("Missing Azure OpenAI configuration.")
        _llm = AzureLLM(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            max_retries=AZURE_OPENAI_MAX_RETRIES,
        )
        logger.info(f"✅ Initialized LLM client: Azure ({AZURE_OPENAI_DEPLOYMENT_NAME})")

    else:
        logger.error(f"❌ Invalid LLM_BACKEND '{LLM_BACKEND}' specified in configuration. Cannot initialize LLM client.")
        # Optionally raise an error to prevent startup with no LLM
        raise ValueError(f"Invalid LLM_BACKEND: {LLM_BACKEND}")

except Exception as e:
    logger.error(f"❌ Failed to initialize LLM client for backend '{LLM_BACKEND}': {e}")
    logger.exception(e)
    # Depending on desired behavior, you might exit, raise, or leave _llm as None
    # Leaving _llm as None will likely cause errors later when it's used.
    # raise RuntimeError("LLM Client initialization failed.") from e
    _llm = None # Explicitly set to None on failure

# --- Expose the client ---
# You can access the initialized client using `from .llm_factory import _llm`
# Add a getter function if preferred for encapsulation or testing mocks
def get_llm_client() -> LLMClient:
    """Returns the initialized LLM client instance."""
    if _llm is None:
        logger.critical("❌ LLM Client was not initialized successfully!")
        # This should ideally not happen if startup checks are robust
        raise RuntimeError("LLM Client is not available.")
    return _llm


async def get_current_llm() -> LLMClient:
    try:
        return get_llm_client()
    except RuntimeError as e:
        logger.critical(f"LLM Client unavailable: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

