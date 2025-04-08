from fastapi import APIRouter, HTTPException, Request
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import threading
import uuid
import asyncio
from proxy_package.domain_layer.file_responce import Response
from google.generativeai.types import generation_types
from dotenv import load_dotenv
import traceback
import uvicorn
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from proxy_package.utils.logger import logger
from proxy_package.service_layer.formating import format_openai_to_gemini, format_gemini_to_openai_completion, format_gemini_to_openai_chat
from proxy_package import  GEMINI_MODEL_NAME, gemini_llm
from proxy_package.service_layer.non_streaming_request import handle_non_streaming_request
from proxy_package.service_layer.streaming_request import stream_gemini_response
chat_router = APIRouter()

@chat_router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    logger.info(f"\n--- Chat Request Received ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        logger.info(f"➡️ Chat Request Body:\n{json.dumps(body, indent=2)}")

        requested_model = body.get('model', GEMINI_MODEL_NAME) # Use global default
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None)
        response_format_type = None
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
             response_format_type = "json_object"
             logger.info("ℹ️ OpenAI JSON mode requested ('json_object').")
        elif response_format == "files_to_update":
             response_format_type = "files_to_update"
             logger.info("ℹ️ Custom JSON mode requested ('files_to_update').")

        openai_messages = body.get('messages', [])
        if not openai_messages:
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        gemini_messages = format_openai_to_gemini(openai_messages)
        if not gemini_messages:
            raise HTTPException(status_code=400, detail="No valid messages found after formatting.")

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
        if response_format_type == "json_object" or response_format_type == "files_to_update":
             logger.info("ℹ️ Applying JSON mode for /v1/chat/completions.")
             if "1.5-pro" not in gemini_llm.model_name: # Check the model used by the client
                 logger.warning(f"⚠️ Warning: Model {gemini_llm.model_name} might not fully support JSON mode. Using gemini-1.5-pro-latest is recommended.")
             config_args['response_mime_type'] = 'application/json'
             if response_format_type == "files_to_update":
                 # Pass the schema *if* the underlying API supports it directly in config
                 # Note: google-generativeai library uses 'response_schema' within 'config'
                 # Let's try adding it to the dict, the _create_config will handle it.
                 config_args['response_schema'] = Response
                 parse_to_files = True

        # --- End Build Gemini Generation Config Dictionary ---

        if should_stream:
            logger.info("🌊 Handling STREAMING chat request...")
            return StreamingResponse(
                stream_gemini_response(
                    gemini_messages,
                    config_args, # Pass the config dictionary
                    requested_model,
                    request_id,
                    is_chat_format=True,
                    parse_to_files=parse_to_files
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info("📄 Handling NON-STREAMING chat request...")
            response = await handle_non_streaming_request(
                gemini_messages,
                config_args, # Pass the config dictionary
                requested_model,
                request_id,
                is_chat_format=True
            )
            # Attempt parsing for non-streaming JSON mode ('files_to_update')
            if parse_to_files and response.get("choices"):
                 full_text = response["choices"][0].get("message", {}).get("content", "")
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
        logger.error(f"❌ Unexpected error in /v1/chat/completions: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")