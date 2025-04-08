# --- Application Context / Shared Resources ---
from .config import API_KEY, GEMINI_MODEL_NAME
from .reporitory_layer.llm.gemini_llm import GeminiLLM
from .utils.logger import logger

# --- Initialize the GeminiLLM client ---
# This will now happen when context.py is first imported.

gemini_llm = GeminiLLM(api_key=API_KEY, model_name=GEMINI_MODEL_NAME)
logger.info("✅ Application context initialized (LLM client ready).")
