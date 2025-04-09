from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import json
import time
import uuid
import asyncio
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

# Relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response, FilesToUpdate
from ..reporitory_layer.llm.llm_factory import LLMClient,create_llm_client
from ..service_layer.formating import create_generation_config_dict
from ..service_layer.non_streaming_request import handle_non_streaming_request
from ..service_layer.streaming_request import stream_response
from ..config import DEFAULT_MODEL_NAME

completions_router = APIRouter()

@completions_router.post("/v1/completions")
async def completions(
    request: Request
):
    """
    Handles completion requests, supporting both streaming and non-streaming modes.
    """
    request_id = f"cmpl-{uuid.uuid4()}"
    try:
        # 1. Parse and Validate Request Body
        try:
            body_raw = await request.json()
            completion_request_data = body_raw
            logger.info(f"[{request_id}] ➡️ Request Body:\n{json.dumps(completion_request_data, indent=2)}")

        except json.JSONDecodeError:
            logger.error(f"[{request_id}] ❌ Invalid JSON received.")
            raise HTTPException(status_code=400, detail="Invalid JSON.")
        except ValidationError as e:
            logger.error(f"[{request_id}] ❌ Request validation failed: {e}")
            raise HTTPException(status_code=422, detail=e.errors())

        # 2. Extract Parameters
        requested_model = completion_request_data.get('model', DEFAULT_MODEL_NAME)
        prompt = completion_request_data.get('prompt')

        llm_client = create_llm_client(requested_model)

        if not prompt:
            logger.warning(f"[{request_id}] ⚠️ Request received with no prompt.")
            raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
        if not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="'prompt' must be a string for /v1/completions.")

        openai_messages = [{'role': 'user', 'content': prompt}]
        suffix = completion_request_data.get('suffix')
        if suffix:
            openai_messages[0]['content'] += str(suffix)

        backend_messages = llm_client.create_backend_messages(openai_messages)
        generation_config_dict = create_generation_config_dict(completion_request_data)

        logger.info(f"[{request_id}] 🌊 Handling STREAMING completion request...")
        return StreamingResponse(
            stream_response(
                llm_client=llm_client,
                backend_messages=backend_messages,
                generation_config_dict=generation_config_dict,
                requested_model=requested_model,
                request_id=request_id,
                is_chat_format=False,
            ),
            media_type="text/event-stream"
        )
    except HTTPException as e:
        logger.warning(f"[{request_id}] ⚠️ Handled HTTPException: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error in /v1/completions: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error. Request ID: {request_id}")