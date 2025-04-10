import os
from typing import Union, Dict, Any, Optional
from fastapi import HTTPException
from proxy_package.domain_layer.llm_domain import LLMResponseModel
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
def create_llm_client(requested_model: LLMResponseModel) -> LLMClient:
    """
    Creates and returns an LLM client instance based on the requested model name
    using structural pattern matching.

    Args:
        requested_model: LLMResponseModel
    Returns:
        An instance of GeminiLLM or AzureLLM.

    Raises:
        HTTPException: If the model name is not recognized, credentials for the
                       corresponding backend are missing, or client initialization fails.
                       Uses status codes 400 for client errors (bad request/config)
                       and 500 for server-side initialization errors.
    """
    logger.info(f"Attempting to create LLM client for model: {requested_model.llm_params.provider}")

    if requested_model.llm_params.provider.lower().startswith("gemini"):
        backend_type = "gemini"
    else:
        # Assuming any non-Gemini request is intended for Azure
        backend_type = "azure"
        logger.debug(f"Model '{requested_model}' does not start with 'gemini', assuming Azure backend.")

    # --- Use match statement to create the appropriate client ---
    match backend_type:
        case "gemini":
            logger.debug(f"Model '{requested_model.llm_params}' identified as Gemini.")
            # Validate Gemini credentials
            #if not all([_gemini_creds["api_key"], _gemini_creds["default_model"]]):
            if not all([requested_model.llm_params.apiKey, requested_model.llm_params.model]): # Keep check for default model as fallback? Or remove? Let's keep for now.
                error_msg = f"Missing or incomplete Gemini credentials (GOOGLE_API_KEY/GOOGLE_MODEL) in configuration. Cannot serve model '{requested_model.llm_params.provider}'."
                logger.error(f"❌ Cannot create GeminiLLM: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg) # 400 Bad Request (config issue)
            try:
                # Pass the *requested* model name to the Gemini client constructor
                client = GeminiLLM(
                    api_key=requested_model.llm_params.apiKey,
                    model_name=requested_model.llm_params.model # Use the specific requested model
            )
                logger.info(f"✅ Created GeminiLLM client for model: {requested_model.llm_params.provider}")
                return client
            except Exception as e:
                error_msg = f"Failed to initialize Gemini client for model {requested_model.llm_params.provider}: {e}"
                logger.error(f"❌ {error_msg}")
                # 500 Internal Server Error for initialization failures
                raise HTTPException(status_code=500, detail=error_msg)

        case "azure":
            logger.debug(f"Model '{requested_model}' identified as Azure.")
            # Validate Azure credentials
            if not all([requested_model.llm_params.apiKey, requested_model.llm_params.version, requested_model.llm_params.endpoint, requested_model.llm_params.model]): # Check all required Azure creds
                error_msg = f"Missing or incomplete Azure OpenAI credentials in configuration. Cannot serve model '{requested_model.llm_params.provider}'."
                logger.error(f"❌ Cannot create AzureLLM: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg) # 400 Bad Request (config issue)

            try:
                client = AzureLLM(
                    api_key=requested_model.llm_params.apiKey,
                    api_version=requested_model.llm_params.version,
                    endpoint=requested_model.llm_params.endpoint,
                    deployment_name=requested_model.llm_params.model, # Use the requested model name here
                    max_retries=requested_model.llm_params.max_retries,
                )
                logger.info(f"✅ Created AzureLLM client for deployment: {requested_model.llm_params.provider}")
                return client
            except Exception as e:
                # Catch potential errors during Azure client initialization (e.g., invalid endpoint, deployment not found)
                error_msg = f"Failed to initialize Azure client for deployment {requested_model.llm_params.provider}: {e}"
                logger.error(f"❌ {error_msg}")
                 # 500 Internal Server Error for initialization failures
                raise HTTPException(status_code=500, detail=error_msg)

        case _:
            # This case should theoretically not be reached with the current logic,
            # but it's good practice for completeness.
            error_msg = f"Unable to determine backend for model: {requested_model.llm_params.provider}"
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg) # 400 Bad Request (unrecognized model)
