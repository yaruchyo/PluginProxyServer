# Async function for non-streaming requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import threading
import uuid
import asyncio
from plugin_proxy_server.domain_layer.file_responce import Response
from google.generativeai.types import generation_types
from dotenv import load_dotenv
import traceback
import asyncio
from typing import Any, List, Dict
from google.generativeai.types import generation_types

# Use relative imports
from ..utils.logger import logger
from ..context import gemini_llm # Import initialized LLM client from context
from .formating import format_gemini_to_openai_chat, format_gemini_to_openai_completion # Import formatters

async def handle_non_streaming_request(gemini_messages: List[Dict[str, Any]], generation_config_dict: Dict[str, Any], requested_model: str, request_id: str, is_chat_format: bool = True):
    """Handles non-streaming requests using the GeminiLLM instance."""
    loop = asyncio.get_event_loop()
    try:
        # Use run_in_executor as the underlying generate_content might be blocking
        gemini_response = await loop.run_in_executor(
            None,
            lambda: gemini_llm.generate_content(
                contents=gemini_messages,
                generation_config_dict=generation_config_dict
            )
        )

        if is_chat_format:
            return format_gemini_to_openai_chat(gemini_response, requested_model, request_id)
        else:
            return format_gemini_to_openai_completion(gemini_response, requested_model, request_id)

    except generation_types.BlockedPromptException as bpe:
         logger.error(f"❌ Gemini API Blocked Prompt Error: {bpe}")
         raise HTTPException(status_code=400, detail=f"Blocked Prompt: {bpe}")
    except generation_types.StopCandidateException as sce:
         logger.error(f"❌ Gemini API Stop Candidate Error: {sce}")
         # Return the partial response if available
         if is_chat_format:
             return format_gemini_to_openai_chat(sce.response, requested_model, request_id)
         else:
             return format_gemini_to_openai_completion(sce.response, requested_model, request_id)
    except Exception as e:
        logger.error(f"❌ Error in non-streaming request handler: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
