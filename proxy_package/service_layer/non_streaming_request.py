import asyncio
from typing import Any, List, Dict, Union
from fastapi import HTTPException
from google.generativeai.types import generation_types as gemini_generation_types
from openai import APIError as AzureAPIError, AuthenticationError as AzureAuthenticationError
from openai.types.chat import ChatCompletion as AzureChatCompletion

# Use relative imports
from ..utils.logger import logger
# Import the specific LLM classes for type checking
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
from ..reporitory_layer.llm.llm_factory import get_llm_client # Use getter
# Import all necessary formatters
from .formating import (
    format_gemini_to_openai_chat, format_gemini_to_openai_completion,
    format_azure_to_openai_chat, format_azure_to_openai_completion
)

# Define a union type for the possible raw responses
LLMResponse = Union[gemini_generation_types.GenerateContentResponse, AzureChatCompletion]

async def handle_non_streaming_request(
    backend_messages: List[Dict[str, Any]],
    generation_config_dict: Dict[str, Any],
    requested_model: str,
    request_id: str,
    is_chat_format: bool = True
) -> Dict[str, Any]:
    """Handles non-streaming requests using the configured LLM client."""
    llm_client = get_llm_client() # Get the initialized client instance
    loop = asyncio.get_event_loop()

    try:
        # Use run_in_executor as the underlying generate_content might be blocking
        # The llm_client instance handles calling the correct backend's method
        raw_response: LLMResponse = await loop.run_in_executor(
            None, # Use default executor
            lambda: llm_client.generate_content(
                contents=backend_messages,
                generation_config_dict=generation_config_dict
            )
        )

        # --- Format the response based on the backend and requested format ---
        if isinstance(llm_client, GeminiLLM):
            logger.debug("Formatting Gemini non-streaming response.")
            if is_chat_format:
                return format_gemini_to_openai_chat(raw_response, requested_model, request_id)
            else:
                return format_gemini_to_openai_completion(raw_response, requested_model, request_id)
        elif isinstance(llm_client, AzureLLM):
            logger.debug("Formatting Azure non-streaming response.")
            if is_chat_format:
                return format_azure_to_openai_chat(raw_response, requested_model, request_id)
            else:
                # Azure uses ChatCompletion even for completion-style requests via this proxy
                return format_azure_to_openai_completion(raw_response, requested_model, request_id)
        else:
            # This should not happen if llm_factory initializes correctly
            logger.error("❌ LLM client type is unknown during non-streaming handling.")
            raise HTTPException(status_code=500, detail="Internal Server Error: Unknown LLM backend.")

    # --- Handle Backend Specific Exceptions ---
    except (gemini_generation_types.BlockedPromptException, gemini_generation_types.StopCandidateException) as gemini_error:
         logger.error(f"❌ Gemini API Generation Error: {type(gemini_error).__name__} - {gemini_error}")
         # For StopCandidateException, Gemini might return a partial response
         if isinstance(gemini_error, gemini_generation_types.StopCandidateException) and hasattr(gemini_error, 'response'):
             logger.warning("Attempting to format partial Gemini response due to StopCandidateException.")
             if isinstance(llm_client, GeminiLLM): # Double check, should be true
                 if is_chat_format:
                     return format_gemini_to_openai_chat(gemini_error.response, requested_model, request_id)
                 else:
                     return format_gemini_to_openai_completion(gemini_error.response, requested_model, request_id)
         # For BlockedPrompt or StopCandidate without response, raise HTTP 400
         raise HTTPException(status_code=400, detail=f"LLM Backend Error: {type(gemini_error).__name__}")

    except (AzureAPIError, AzureAuthenticationError) as azure_error:
        logger.error(f"❌ Azure API Error: {type(azure_error).__name__} - Status={getattr(azure_error, 'status_code', 'N/A')} Body={getattr(azure_error, 'body', 'N/A')}")
        status_code = getattr(azure_error, 'status_code', 500) or 500 # Ensure status_code is not None
        detail = f"Azure API Error: {azure_error}"
        # Map specific Azure status codes if needed (e.g., 401 for auth, 429 for rate limit)
        if isinstance(azure_error, AzureAuthenticationError):
            status_code = 401
            detail = "Azure Authentication Error: Invalid API key or endpoint."
        elif status_code == 429:
             detail = "Azure API Error: Rate limit exceeded or quota reached."
        # You might want to extract more specific details from azure_error.body if available
        raise HTTPException(status_code=status_code, detail=detail)

    except Exception as e:
        logger.error(f"❌ Unexpected error in non-streaming request handler: {e}")
        logger.exception(e)
        # Determine backend for more specific error message if possible
        backend_name = type(llm_client).__name__ if llm_client else "Unknown"
        raise HTTPException(status_code=500, detail=f"Internal Server Error ({backend_name} Backend): {e}")