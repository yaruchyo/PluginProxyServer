from fastapi import APIRouter, HTTPException, Request # Removed Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError, BaseModel, Field
from typing import Any, Optional, List, Dict, Union, Iterator

# Relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response, FilesToUpdate
# Import the factory function, not the base class or getter
from ..reporitory_layer.llm.llm_factory import create_llm_client, LLMClient # Keep LLMClient type hint
from ..service_layer.formating import create_generation_config_dict
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..service_layer.streaming_request import stream_response
from proxy_package.domain_layer.chat_domain import ChatMessage, ChatCompletionRequest
from ..config import DEFAULT_MODEL_NAME

chat_router = APIRouter()

@chat_router.post("/v1/chat/completions")
async def chat_completions(
    request: Request, # Keep raw request
    # llm_client: LLMClient = Depends(get_current_llm) # REMOVED Dependency Injection
):
    """
    Handles chat completion requests, supporting both streaming and non-streaming modes.
    Dynamically selects the LLM backend based on the 'model' field.
    """
    request_id = f"chatcmpl-{uuid.uuid4()}"
    start_time = time.time()
    llm_client: Optional[LLMClient] = None # Initialize llm_client variable

    logger.info(f"\n--- [{request_id}] Chat Request Received ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")

    try:
        # 1. Parse Request Body (Keep as is)
        try:
            chat_request_data = await request.json()
            logger.info(f"[{request_id}] ➡️ Request Body:\n{json.dumps(chat_request_data, indent=2)}")
        except json.JSONDecodeError:
            logger.error(f"[{request_id}] ❌ Invalid JSON received.")
            raise HTTPException(status_code=400, detail="Invalid JSON.")

        # 2. Extract Parameters (Keep as is)
        tools = chat_request_data.get('tools', [])
        requested_model = chat_request_data.get('model', DEFAULT_MODEL_NAME)
        should_stream = chat_request_data.get('stream', False)
        openai_messages = chat_request_data.get('messages', [])

        if not openai_messages:
             logger.warning(f"[{request_id}] ⚠️ Request received with no messages.")
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        # --- 3. Create LLM Client Dynamically ---
        try:
            llm_client = create_llm_client(requested_model)
        except ValueError as e:
            # Errors from create_llm_client (unknown model, missing creds)
            logger.error(f"[{request_id}] ❌ Failed to create LLM client for model '{requested_model}': {e}")
            # Return 400 for unknown model, 503 if backend creds are missing/invalid
            status_code = 400 if "not recognized" in str(e) else 503
            raise HTTPException(status_code=status_code, detail=str(e))
        except Exception as e: # Catch unexpected errors during client creation
            logger.error(f"[{request_id}] ❌ Unexpected error creating LLM client for model '{requested_model}': {e}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Internal error creating LLM client. Request ID: {request_id}")

        # --- Backend specific info after client creation ---
        backend_name = type(llm_client).__name__
        logger.info(f"[{request_id}] Using backend: {backend_name} for model: {requested_model}")

        # 4. Prepare Backend Messages and Config (Use the created client)
        backend_messages = llm_client.create_backend_messages(openai_messages)
        generation_config_dict = create_generation_config_dict(chat_request_data)

        # 5. Handle Request (Streaming or Non-Streaming)
        if should_stream:
            logger.info(f"[{request_id}] 🌊 Handling STREAMING chat request...")
            return StreamingResponse(
                stream_response( # Pass the created client
                    llm_client=llm_client, # Pass the specific client instance
                    backend_messages=backend_messages,
                    generation_config_dict=generation_config_dict,
                    requested_model=requested_model, # Pass requested model for SSE payload
                    request_id=request_id,
                    is_chat_format=True,
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info(f"[{request_id}] 📄 Handling NON-STREAMING chat request...")
            response_content = await handle_non_streaming_request(
                llm_client=llm_client, # Pass the specific client instance
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model, # Pass requested model for response formatting
                request_id=request_id,
                is_chat_format=True
            )
            duration = time.time() - start_time
            logger.info(f"[{request_id}] ✅ Non-streaming request completed in {duration:.2f}s.")
            return JSONResponse(content=response_content)

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        logger.warning(f"[{request_id}] ⚠️ Handled HTTPException: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        # Catch-all for unexpected errors
        backend_info = f" ({type(llm_client).__name__} Backend)" if llm_client else ""
        logger.error(f"[{request_id}] ❌ Unexpected error in /v1/chat/completions{backend_info}: {e}")
        logger.exception(e) # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Internal Server Error. Request ID: {request_id}")
