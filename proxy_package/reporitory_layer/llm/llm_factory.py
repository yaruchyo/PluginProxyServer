import os
from typing import Union, Dict, Any, Optional
from fastapi import HTTPException
import re # Import regex for model matching

# --- Application Context / Shared Resources ---
from ...config import ( # Use relative imports
    # Keep ALL credentials
    GEMINI_API_KEY, GEMINI_DEFAULT_MODEL,
    AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_MAX_RETRIES,
    # Add known Azure deployments if needed for mapping, or rely on AZURE_OPENAI_DEPLOYMENT_NAME as one possibility
    # Example: KNOWN_AZURE_DEPLOYMENTS = ["gpt-4o", "gpt-35-turbo", AZURE_OPENAI_DEPLOYMENT_NAME]
)
from .gemini_llm import GeminiLLM
from .azure_llm import AzureLLM
from ...utils.logger import logger # Use relative import

# --- Define the type hint for the LLM client ---
# Base class approach (if GeminiLLM and AzureLLM inherit from a common base)
# from .base_llm import BaseLLM # Assuming you create a base class
# LLMClient = BaseLLM
# Or keep Union if no strict base class
LLMClient = Union[GeminiLLM, AzureLLM]

# --- Store Credentials (Loaded at startup) ---
# It's good practice to validate these at startup even if not creating a client yet
_gemini_creds = {
    "api_key": GEMINI_API_KEY,
    "default_model": GEMINI_DEFAULT_MODEL
}
_azure_creds = {
    "api_key": AZURE_OPENAI_KEY,
    "api_version": AZURE_OPENAI_API_VERSION,
    "endpoint": AZURE_OPENAI_ENDPOINT,
    "deployment_name": AZURE_OPENAI_DEPLOYMENT_NAME, # Default/primary deployment
    "max_retries": AZURE_OPENAI_MAX_RETRIES,
    # Consider adding a list of *all* valid deployment names if needed for matching
    # "valid_deployments": [AZURE_OPENAI_DEPLOYMENT_NAME, "other-deployment-name"]
}

# --- Optional: Validate credentials on import ---
def _validate_credentials():
    missing_gemini = not all([_gemini_creds["api_key"], _gemini_creds["default_model"]])
    missing_azure = not all([
        _azure_creds["api_key"], _azure_creds["api_version"],
        _azure_creds["endpoint"], _azure_creds["deployment_name"]
    ])

    if missing_gemini:
        logger.warning("⚠️ Gemini credentials (GOOGLE_API_KEY, GOOGLE_MODEL) are missing or incomplete. Gemini models will be unavailable.")
    else:
        logger.info("✅ Gemini credentials loaded.")

    if missing_azure:
        logger.warning("⚠️ Azure OpenAI credentials (AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME) are missing or incomplete. Azure models will be unavailable.")
    else:
        logger.info("✅ Azure OpenAI credentials loaded.")

_validate_credentials() # Run validation when module is imported

# --- Factory Function ---
def create_llm_client(requested_model_name: str) -> LLMClient:
    """
    Creates and returns an LLM client instance based on the requested model name.

    Args:
        requested_model_name: The model name specified in the request.

    Returns:
        An instance of GeminiLLM or AzureLLM.

    Raises:
        ValueError: If the model name is not recognized or credentials for the
                    corresponding backend are missing.
    """
    logger.info(f"Attempting to create LLM client for model: {requested_model_name}")

    # --- Logic to determine backend based on model name ---
    # Example: Use prefix matching for Gemini, assume others are Azure
    # (Adjust this logic based on your actual model naming conventions)

    is_gemini_request = requested_model_name.lower().startswith("gemini")

    # More robust: Check against known Azure deployment names if available
    # is_azure_request = requested_model_name in _azure_creds.get("valid_deployments", [_azure_creds["deployment_name"]])

    # Simple approach: If not Gemini, assume Azure for now
    is_azure_request = not is_gemini_request

    # --- Instantiate Gemini Client ---
    if is_gemini_request:
        logger.debug(f"Model '{requested_model_name}' identified as Gemini.")
        if not all([_gemini_creds["api_key"], _gemini_creds["default_model"]]):
            logger.error(f"❌ Cannot create GeminiLLM: Missing GOOGLE_API_KEY or GOOGLE_MODEL in config for requested model '{requested_model_name}'.")
            raise ValueError(f"Missing or incomplete Gemini credentials in configuration. Cannot serve model '{requested_model_name}'.")
        try:
            # Pass the *requested* model name to the Gemini client
            client = GeminiLLM(api_key=_gemini_creds["api_key"], model_name=_gemini_creds["default_model"])
            logger.info(f"✅ Created GeminiLLM client for model: {requested_model_name}")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to initialize GeminiLLM for model {requested_model_name}: {e}")
            raise ValueError(f"Failed to initialize Gemini client: {e}")

    # --- Instantiate Azure Client ---
    elif is_azure_request:
        logger.debug(f"Model '{requested_model_name}' identified as Azure.")
        if not all([_azure_creds["api_key"], _azure_creds["api_version"], _azure_creds["endpoint"]]):
             logger.error(f"❌ Cannot create AzureLLM: Missing one or more Azure OpenAI environment variables for requested model '{requested_model_name}'.")
             raise ValueError(f"Missing or incomplete Azure OpenAI credentials in configuration. Cannot serve model '{requested_model_name}'.")

        # Use the requested_model_name as the deployment_name for Azure
        # Assumption: The 'model' field in the request corresponds to the Azure 'deployment_name'
        azure_deployment_to_use = requested_model_name
        logger.info(f"Using Azure deployment name: {azure_deployment_to_use}")

        try:
            client = AzureLLM(
                api_key=_azure_creds["api_key"],
                api_version=_azure_creds["api_version"],
                endpoint=_azure_creds["endpoint"],
                deployment_name=_azure_creds['deployment_name'], # Use the requested model name here
                max_retries=_azure_creds["max_retries"],
            )
            logger.info(f"✅ Created AzureLLM client for deployment: {azure_deployment_to_use}")
            return client
        except Exception as e:
            # Catch potential errors during Azure client initialization (e.g., invalid endpoint)
            logger.error(f"❌ Failed to initialize AzureLLM for deployment {azure_deployment_to_use}: {e}")
            raise ValueError(f"Failed to initialize Azure client: {e}")

    # --- Model Not Recognized ---
    else:
        logger.error(f"❌ Unknown model requested: '{requested_model_name}'. Cannot determine backend.")
        raise ValueError(f"Model '{requested_model_name}' is not recognized or supported by this proxy.")