from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from proxy_package.service_layer.formating import create_generation_config_dict
from ..utils.logger import logger
from ..domain_layer.file_responce import Response, FilesToUpdate
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
from ..reporitory_layer.llm.llm_factory import get_llm_client, get_current_llm, LLMClient
from ..service_layer.formating import format_openai_to_gemini
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..service_layer.streaming_request import stream_response
from ..config import DEFAULT_MODEL_NAME

chat_router = APIRouter()

@chat_router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    llm_client: LLMClient = Depends(get_current_llm)):

    request_id = f"chatcmpl-{uuid.uuid4()}"
    logger.info(f"\n--- Chat Request Received ({type(llm_client).__name__} Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        logger.info(f"➡️ Chat Request Body:\n{json.dumps(body, indent=2)}")

        requested_model = body.get('model', DEFAULT_MODEL_NAME)
        should_stream = body.get('stream', False)
        openai_messages = body.get('messages', [])

        if not openai_messages:
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        backend_messages = llm_client.create_backend_messages(openai_messages)
        generation_config_dict = create_generation_config_dict(body)

        if should_stream:
            logger.info("🌊 Handling STREAMING chat request...")
            return StreamingResponse(
                stream_response(
                    backend_messages=backend_messages,
                    generation_config_dict=generation_config_dict,
                    requested_model=requested_model,
                    request_id=request_id,
                    is_chat_format=True
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info("📄 Handling NON-STREAMING chat request...")
            response_content = await handle_non_streaming_request(
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model,
                request_id=request_id,
                is_chat_format=True
            )

            return JSONResponse(content=response_content)

    except json.JSONDecodeError:
        logger.error(f"❌ Error decoding JSON request body for {request_id}.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e:
         raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error in /v1/chat/completions for {request_id}: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")