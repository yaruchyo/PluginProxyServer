# --- Application Context / Shared Resources ---
from proxy_package.config import API_KEY, GEMINI_MODEL_NAME, LLM_MODEL
from proxy_package.reporitory_layer.llm.gemini_llm import GeminiLLM
from proxy_package.utils.logger import logger
from openai import AzureOpenAI

# --- Initialize the GeminiLLM client ---
# This will now happen when context.py is first imported.

if LLM_MODEL == "gemini":
    _llm = GeminiLLM(api_key=API_KEY, model_name=GEMINI_MODEL_NAME)
else:
    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")  # Example version
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_MODEL")  # Your deployment name
    AZURE_OPENAI_MAX_RETRIES = int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", 3))
    _llm = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        max_retries=AZURE_OPENAI_MAX_RETRIES,
    )

logger.info("✅ Application context initialized (LLM client ready).")
