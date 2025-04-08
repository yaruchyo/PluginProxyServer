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

completions_router = APIRouter()

# Dependency to get the LLM client instance
async def get_current_llm() -> LLMClient:
    try:
        return get_llm_client()
    except RuntimeError as e:
        logger.critical(f"LLM Client unavailable: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

@completions_router.post("/v1/completions")
async def completions(
    request: Request,
    llm_client: LLMClient = Depends(get_current_llm) # Inject LLM client
    ):
    request_id = f"cmpl-{uuid.uuid4()}"
    logger.info(f"\n--- Completion Request Received ({type(llm_client).__name__} Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        logger.info(f"➡️ Completion Request Body:\n{json.dumps(body, indent=2)}")

        # Determine backend type
        is_gemini = isinstance(llm_client, GeminiLLM)
        is_azure = isinstance(llm_client, AzureLLM)

        requested_model = body.get('model', DEFAULT_MODEL_NAME)
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None) # Note: Less common for /completions
        prompt = body.get('prompt')

        if not prompt:
             raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
        if not isinstance(prompt, str):
             # Allow list of strings or tokens? For now, enforce single string as per common usage.
             raise HTTPException(status_code=400, detail="'prompt' must be a string for /v1/completions.")

        # --- Prepare messages for the backend ---
        # Even for completions, we often need to wrap the prompt in a user message,
        # especially for chat-based models like those behind Azure OpenAI deployments.
        openai_messages = [{'role': 'user', 'content': prompt}]
        # Handle optional 'suffix' (append to the user content)
        suffix = body.get('suffix')
        if suffix:
            openai_messages[0]['content'] += str(suffix)

        backend_messages: List[Dict[str, Any]]
        if is_gemini:
            logger.debug("Formatting prompt for Gemini backend (completion).")
            # Gemini can sometimes handle a simple prompt better without role wrapping,
            # but using the chat format is generally more robust. Let's use the formatter.
            backend_messages = format_openai_to_gemini(openai_messages)
            if not backend_messages:
                 raise HTTPException(status_code=400, detail="Failed to format prompt message for Gemini.")
        elif is_azure:
            logger.debug("Using prompt as user message for Azure backend (completion).")
            # Azure requires chat format
            backend_messages = openai_messages
        else:
             raise HTTPException(status_code=500, detail="Internal Server Error: Unknown LLM backend type.")

        # --- Build Generation Config Dictionary (Common Params) ---
        generation_config_dict: Dict[str, Any] = {}
        if body.get('max_tokens') is not None:
            generation_config_dict['max_tokens'] = body['max_tokens']
        if body.get('temperature') is not None:
            generation_config_dict['temperature'] = body['temperature']
        if body.get('top_p') is not None:
            generation_config_dict['top_p'] = body['top_p']
        # Params less common in /completions but supported by some models/backends
        if body.get('presence_penalty') is not None:
             generation_config_dict['presence_penalty'] = body['presence_penalty']
        if body.get('frequency_penalty') is not None:
             generation_config_dict['frequency_penalty'] = body['frequency_penalty']
        # 'n' (number of choices) - complicates streaming/non-streaming logic, usually 1 for proxy
        # 'logprobs' - not easily available from Gemini, maybe from Azure choice.logprobs
        # 'echo' - would require manual implementation

        stop_val = body.get('stop')
        if stop_val:
             generation_config_dict['stop'] = stop_val

        # --- Handle JSON Mode (Custom 'files_to_update') ---
        # Standard 'json_object' is not typical for /v1/completions
        parse_to_files = False
        if response_format == "files_to_update":
             logger.info("ℹ️ Custom JSON mode 'files_to_update' requested for /v1/completions.")
             parse_to_files = True
             if is_gemini:
                 if "1.5-pro" not in llm_client.model_name:
                     logger.warning(f"⚠️ Gemini model {llm_client.model_name} might not fully support JSON mode with schema. Use 1.5-pro.")
                 generation_config_dict['response_mime_type'] = 'application/json'
                 generation_config_dict['response_schema'] = Response
             elif is_azure:
                 logger.info("Injecting 'files_to_update' JSON structure instruction for Azure backend (completion).")
                 # Append instruction to the *single* user message content
                 files_instruction = f"\n\nPlease format your response ONLY as a valid JSON object matching this Pydantic schema:\n```json\n{json.dumps(Response.model_json_schema(), indent=2)}\n```"
                 backend_messages[0]['content'] += files_instruction # Append to existing content

        # --- End Build Generation Config Dictionary ---

        # --- Call appropriate service function ---
        if should_stream:
            logger.info("🌊 Handling STREAMING completion request...")
            return StreamingResponse(
                stream_response( # Use the backend-agnostic streaming function
                    backend_messages=backend_messages,
                    generation_config_dict=generation_config_dict,
                    requested_model=requested_model,
                    request_id=request_id,
                    is_chat_format=False, # <<< Ensure output is formatted as completion chunks
                    parse_to_files=parse_to_files
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info("📄 Handling NON-STREAMING completion request...")
            response_content = await handle_non_streaming_request( # Use the backend-agnostic non-streaming function
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model,
                request_id=request_id,
                is_chat_format=False # <<< Ensure output is formatted as completion response
            )

            # --- Post-process for 'files_to_update' in non-streaming ---
            if parse_to_files and response_content.get("choices"):
                 full_text = response_content["choices"][0].get("text", "") # Get text from completion choice
                 if full_text:
                     try:
                         list_files_adapter = TypeAdapter(Response)
                         parsed_response_model = list_files_adapter.validate_json(full_text)
                         if parsed_response_model.files_to_update:
                             response_content["parsed_data"] = [f.model_dump() for f in parsed_response_model.files_to_update]
                             logger.info("✅ Successfully parsed non-streaming completion response into FilesToUpdate model.")
                         else:
                             logger.info("ℹ️ Non-streaming completion response parsed, but no 'files_to_update' found.")
                     except (ValidationError, json.JSONDecodeError) as e:
                         logger.warning(f"⚠️ Failed to parse non-streaming completion response into FilesToUpdate model: {e}")
                         response_content["parsing_error"] = f"Failed to validate response against FilesToUpdate schema: {e}"
                 else:
                      logger.info("ℹ️ Non-streaming completion response has no text content to parse for FilesToUpdate.")


            return JSONResponse(content=response_content)

    except json.JSONDecodeError:
        logger.error(f"❌ Error decoding JSON request body for {request_id}.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e:
         raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error in /v1/completions for {request_id}: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")