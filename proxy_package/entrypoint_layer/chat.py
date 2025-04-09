from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError, BaseModel, Field # Added for potential request model
from typing import Any, Optional, List, Dict, Union, Iterator

# Relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response, FilesToUpdate # Keep if used by non-streaming
from ..reporitory_layer.llm.llm_factory import LLMClient # Use base class for dependency
from ..reporitory_layer.llm.llm_factory import get_current_llm
from ..service_layer.formating import create_generation_config_dict # Keep formatting separate
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..service_layer.streaming_request import stream_response
from proxy_package.domain_layer.chat_domain import ChatMessage, ChatCompletionRequest
from ..config import DEFAULT_MODEL_NAME

chat_router = APIRouter()

@chat_router.post("/v1/chat/completions")
async def chat_completions(
    request: Request, # Keep raw request for potential logging/debugging if needed
    llm_client: LLMClient = Depends(get_current_llm) # Inject the current LLM client
):
    """
    Handles chat completion requests, supporting both streaming and non-streaming modes.
    """
    request_id = f"chatcmpl-{uuid.uuid4()}"
    start_time = time.time()
    backend_name = type(llm_client).__name__
    logger.info(f"\n--- [{request_id}] Chat Request Received ({backend_name} Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")

    try:
        # 1. Parse and Validate Request Body
        try:
            chat_request_data = await request.json()
            logger.info(f"[{request_id}] ➡️ Request Body:\n{json.dumps(chat_request_data, indent=2)}")

        except json.JSONDecodeError:
            logger.error(f"[{request_id}] ❌ Invalid JSON received.")
            raise HTTPException(status_code=400, detail="Invalid JSON.")
        except ValidationError as e:
             logger.error(f"[{request_id}] ❌ Request validation failed: {e}")
             raise HTTPException(status_code=422, detail=e.errors()) # Unprocessable Entity

        # 2. Extract Parameters
        requested_model = chat_request_data.get('model', DEFAULT_MODEL_NAME)
        should_stream = chat_request_data.get('stream', False)
        openai_messages = chat_request_data.get('messages', [])

        if not openai_messages:
             logger.warning(f"[{request_id}] ⚠️ Request received with no messages.")
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        backend_messages = llm_client.create_backend_messages(openai_messages)
        generation_config_dict = create_generation_config_dict(chat_request_data) # Keep using the helper

        # 4. Handle Request (Streaming or Non-Streaming)
        if should_stream:
            logger.info(f"[{request_id}] 🌊 Handling STREAMING chat request...")
            return StreamingResponse(
                stream_response( # Call the refactored streaming function
                    backend_messages=backend_messages,
                    generation_config_dict=generation_config_dict,
                    requested_model=requested_model,
                    request_id=request_id,
                    is_chat_format=True,
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info(f"[{request_id}] 📄 Handling NON-STREAMING chat request...")
            response_content = await handle_non_streaming_request(
                llm_client=llm_client,
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model,
                request_id=request_id,
                is_chat_format=True
            )
            duration = time.time() - start_time
            logger.info(f"[{request_id}] ✅ Non-streaming request completed in {duration:.2f}s.")
            # logger.debug(f"[{request_id}] Non-streaming Response Content:\n{json.dumps(response_content, indent=2)}")
            return JSONResponse(content=response_content)

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        logger.warning(f"[{request_id}] ⚠️ Handled HTTPException: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"[{request_id}] ❌ Unexpected error in /v1/chat/completions: {e}")
        logger.exception(e) # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Internal Server Error. Request ID: {request_id}")
