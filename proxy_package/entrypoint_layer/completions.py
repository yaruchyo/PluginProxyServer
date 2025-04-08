from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

# Use relative imports for modules within the package
from ..utils.logger import logger
from ..domain_layer.file_responce import Response
from ..service_layer.formating import format_openai_to_gemini # Keep formatters
from ..service_layer.streaming_request import stream_gemini_response
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..config import GEMINI_MODEL_NAME # Import default model name from config
from ..context import gemini_llm # Import the initialized LLM client
completions_router = APIRouter()

@completions_router.post("/v1/completions")
async def completions(request: Request):
    request_id = f"cmpl-{uuid.uuid4()}"
    logger.info(f"\n--- Completion Request Received ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        logger.info(f"➡️ Completion Request Body:\n{json.dumps(body, indent=2)}")

        requested_model = body.get('model', GEMINI_MODEL_NAME) # Use global default if not specified
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None)
        prompt = body.get('prompt')

        if not prompt:
             raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
        if not isinstance(prompt, str):
             raise HTTPException(status_code=400, detail="'prompt' must be a string.")

        openai_messages = [{'role': 'user', 'content': prompt}]
        suffix = body.get('suffix')
        if suffix:
            openai_messages[0]['content'] += suffix

        gemini_messages = format_openai_to_gemini(openai_messages)
        if not gemini_messages:
            raise HTTPException(status_code=400, detail="Failed to format prompt for Gemini.")

        # --- Build Gemini Generation Config Dictionary ---
        config_args: Dict[str, Any] = {} # Explicitly type as dict
        if body.get('max_tokens') is not None:
            config_args['max_output_tokens'] = body['max_tokens']
        if body.get('temperature') is not None:
            config_args['temperature'] = body['temperature']
        if body.get('top_p') is not None:
            config_args['top_p'] = body['top_p']
        stop_val = body.get('stop')
        if stop_val:
            if isinstance(stop_val, str):
                config_args['stop_sequences'] = [stop_val]
            elif isinstance(stop_val, list):
                config_args['stop_sequences'] = stop_val

        parse_to_files = False
        if response_format == "files_to_update":
             logger.info("ℹ️ Applying JSON mode for /v1/completions based on 'response_format'.")
             if "1.5-pro" not in gemini_llm.model_name: # Check the model used by the client
                 logger.warning(f"⚠️ Warning: Model {gemini_llm.model_name} might not fully support JSON mode. Using gemini-1.5-pro-latest is recommended.")
             config_args['response_mime_type'] = 'application/json'
             parse_to_files = True
        # --- End Build Gemini Generation Config Dictionary ---

        if should_stream:
            logger.info("🌊 Handling STREAMING completion request...")
            return StreamingResponse(
                stream_gemini_response(
                    gemini_messages,
                    config_args, # Pass the config dictionary
                    requested_model,
                    request_id,
                    is_chat_format=False,
                    parse_to_files=parse_to_files
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info("📄 Handling NON-STREAMING completion request...")
            response = await handle_non_streaming_request(
                gemini_messages,
                config_args, # Pass the config dictionary
                requested_model,
                request_id,
                is_chat_format=False
            )
            # Attempt parsing for non-streaming JSON mode
            if parse_to_files and response.get("choices"):
                 full_text = response["choices"][0].get("text", "")
                 try:
                     list_files_adapter = TypeAdapter(Response)
                     parsed_response_model = list_files_adapter.validate_json(full_text)
                     if parsed_response_model.files_to_update:
                         response["parsed_data"] = [f.model_dump() for f in parsed_response_model.files_to_update]
                         logger.info("✅ Successfully parsed non-streaming response into FilesToUpdate model.")
                     else:
                         logger.info("ℹ️ Non-streaming response parsed, but no 'files_to_update' found.")
                 except (ValidationError, json.JSONDecodeError) as e:
                     logger.warning(f"⚠️ Failed to parse non-streaming response into FilesToUpdate model: {e}")
                     response["parsing_error"] = str(e)

            return JSONResponse(content=response)

    except json.JSONDecodeError:
        logger.error("❌ Error decoding JSON.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e:
         raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error in /v1/completions: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

