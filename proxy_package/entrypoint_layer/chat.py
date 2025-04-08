from fastapi import APIRouter, HTTPException, Request, Depends # Add Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

# Use relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response, FilesToUpdate # Import domain models
# Import specific LLM classes for type checking
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
from ..reporitory_layer.llm.llm_factory import get_llm_client, LLMClient # Import getter and type hint
# Import only necessary formatters and config
from ..service_layer.formating import format_openai_to_gemini
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..service_layer.streaming_request import stream_response
from ..config import DEFAULT_MODEL_NAME # Import default model based on backend

chat_router = APIRouter()

# Dependency to get the LLM client instance
# This ensures the client is ready before the endpoint logic runs
async def get_current_llm() -> LLMClient:
    try:
        return get_llm_client()
    except RuntimeError as e:
        logger.critical(f"LLM Client unavailable: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")


@chat_router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    llm_client: LLMClient = Depends(get_current_llm) # Inject LLM client
    ):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    logger.info(f"\n--- Chat Request Received ({type(llm_client).__name__} Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        logger.info(f"➡️ Chat Request Body:\n{json.dumps(body, indent=2)}")

        # Determine backend type for conditional logic
        is_gemini = isinstance(llm_client, GeminiLLM)
        is_azure = isinstance(llm_client, AzureLLM)

        # Use the backend-specific default model if none is provided in the request
        requested_model = body.get('model', DEFAULT_MODEL_NAME)
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None) # Handles dict or string format

        openai_messages = body.get('messages', [])
        if not openai_messages:
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        # --- Prepare messages for the specific backend ---
        backend_messages: List[Dict[str, Any]]
        if is_gemini:
            logger.debug("Formatting messages for Gemini backend.")
            backend_messages = format_openai_to_gemini(openai_messages)
            if not backend_messages:
                # format_openai_to_gemini logs warnings, raise error if empty
                raise HTTPException(status_code=400, detail="Failed to format messages for Gemini, possibly due to invalid input structure.")
        elif is_azure:
            logger.debug("Using messages directly for Azure backend.")
            # Azure client expects standard OpenAI format
            backend_messages = openai_messages
            # Optional: Add JSON instruction for Azure if JSON mode requested and not handled by client param
        else:
             # Should be caught by Depends(get_current_llm)
             raise HTTPException(status_code=500, detail="Internal Server Error: Unknown LLM backend type.")


        # --- Build Generation Config Dictionary (Common Params) ---
        # The LLM client's internal methods (_create_config/_prepare_params) will map these
        generation_config_dict: Dict[str, Any] = {}
        if body.get('max_tokens') is not None:
            # Use 'max_tokens' as the common key, let LLM classes map it
            generation_config_dict['max_tokens'] = body['max_tokens']
        if body.get('temperature') is not None:
            generation_config_dict['temperature'] = body['temperature']
        if body.get('top_p') is not None:
            generation_config_dict['top_p'] = body['top_p']
        if body.get('presence_penalty') is not None: # Supported by Azure, ignored by Gemini class
             generation_config_dict['presence_penalty'] = body['presence_penalty']
        if body.get('frequency_penalty') is not None: # Supported by Azure, ignored by Gemini class
             generation_config_dict['frequency_penalty'] = body['frequency_penalty']
        if body.get('logit_bias') is not None: # Supported by Azure, ignored by Gemini class
             generation_config_dict['logit_bias'] = body['logit_bias']
        if body.get('user') is not None: # Supported by Azure, ignored by Gemini class
             generation_config_dict['user'] = body['user']

        # Use 'stop' as the common key, let LLM classes map it
        stop_val = body.get('stop')
        if stop_val:
             generation_config_dict['stop'] = stop_val

        # --- Handle JSON Mode ---
        parse_to_files = False # Flag for post-processing our custom format
        json_mode_requested = False

        # Standard OpenAI JSON mode
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
             logger.info("ℹ️ OpenAI JSON mode requested ('json_object').")
             json_mode_requested = True
             if is_gemini:
                 if "1.5-pro" not in llm_client.model_name:
                     logger.warning(f"⚠️ Gemini model {llm_client.model_name} might not fully support JSON mode. Use 1.5-pro.")
                 generation_config_dict['response_mime_type'] = 'application/json'
                 # Gemini doesn't need prompt injection for basic JSON mode when mime_type is set
        # Custom 'files_to_update' mode
        elif response_format == "files_to_update":
             logger.info("ℹ️ Custom JSON mode requested ('files_to_update').")
             json_mode_requested = True
             parse_to_files = True # Signal post-processing attempt
             if is_gemini:
                 if "1.5-pro" not in llm_client.model_name:
                     logger.warning(f"⚠️ Gemini model {llm_client.model_name} might not fully support JSON mode with schema. Use 1.5-pro.")
                 # Pass the Pydantic model class as the schema for Gemini
                 generation_config_dict = {}
                 generation_config_dict['response_mime_type'] = 'application/json'
                 generation_config_dict['response_schema'] = Response # Pass the class

        # --- Call appropriate service function ---
        if should_stream:
            logger.info("🌊 Handling STREAMING chat request...")
            return StreamingResponse(
                stream_response( # Use the backend-agnostic streaming function
                    backend_messages=backend_messages,
                    generation_config_dict=generation_config_dict,
                    requested_model=requested_model,
                    request_id=request_id,
                    is_chat_format=True,
                    parse_to_files=parse_to_files # Pass flag for potential post-stream parsing log
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info("📄 Handling NON-STREAMING chat request...")
            response_content = await handle_non_streaming_request( # Use the backend-agnostic non-streaming function
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model,
                request_id=request_id,
                is_chat_format=True
            )

            # --- Post-process for 'files_to_update' in non-streaming ---
            if parse_to_files and response_content.get("choices"):
                 full_text = response_content["choices"][0].get("message", {}).get("content", "")
                 if full_text:
                     try:
                         # Attempt to parse the response text using the Response model
                         list_files_adapter = TypeAdapter(Response)
                         parsed_response_model = list_files_adapter.validate_json(full_text)
                         # Add parsed data to the response if successful and files exist
                         if parsed_response_model.files_to_update:
                             response_content["parsed_data"] = [f.model_dump() for f in parsed_response_model.files_to_update]
                             logger.info("✅ Successfully parsed non-streaming response into FilesToUpdate model.")
                         else:
                             logger.info("ℹ️ Non-streaming response parsed, but no 'files_to_update' found.")
                     except (ValidationError, json.JSONDecodeError) as e:
                         logger.warning(f"⚠️ Failed to parse non-streaming response into FilesToUpdate model: {e}")
                         # Add parsing error information to the response
                         response_content["parsing_error"] = f"Failed to validate response against FilesToUpdate schema: {e}"
                 else:
                      logger.info("ℹ️ Non-streaming response has no content to parse for FilesToUpdate.")

            return JSONResponse(content=response_content)

    except json.JSONDecodeError:
        logger.error(f"❌ Error decoding JSON request body for {request_id}.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e:
         # Re-raise HTTP exceptions directly (e.g., from service layer or input validation)
         raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error in /v1/chat/completions for {request_id}: {e}")
        logger.exception(e) # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
