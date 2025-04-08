# --- Helper Functions (Formatting remains the same) ---
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
import uvicorn
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from plugin_proxy_server.utils.logger import logger
from plugin_proxy_server import GEMINI_MODEL_NAME

def format_openai_to_gemini(openai_messages):
    """Converts OpenAI message format to Gemini format."""
    gemini_messages = []
    system_prompt = None
    temp_messages = []

    for message in openai_messages:
        role = message.get('role')
        content = message.get('content')
        if not content:
            continue
        if role == 'system':
            system_prompt = content
            logger.info(f"ℹ️ Found system prompt: {str(content)[:100]}...")
            continue

        gemini_role = 'user' if role == 'user' else 'model'
        parts = []

        if isinstance(content, str):
            parts.append({'text': content})
        elif isinstance(content, list):
            for item in content:
                if item['type'] == 'text':
                    parts.append({'text': item['text']})
                elif item['type'] == 'image_url':
                    url = item['image_url']['url']
                    if url.startswith('data:'):
                        try:
                            header, data = url.split(',', 1)
                            mime_type = header.split(':')[1].split(';')[0]
                            if ';base64' in header:
                                parts.append({'inline_data': {'mime_type': mime_type, 'data': data}})
                        except Exception as e:
                            logger.error(f"❌ Error parsing image_url: {e}")
        temp_messages.append({'role': gemini_role, 'parts': parts})

    first_user_index = next((i for i, msg in enumerate(temp_messages) if msg['role'] == 'user'), -1)
    if first_user_index != -1 and system_prompt:
        system_part = {'text': f"System Prompt:\n{system_prompt}\n\n"}
        if 'parts' not in temp_messages[first_user_index] or not isinstance(temp_messages[first_user_index]['parts'], list):
             temp_messages[first_user_index]['parts'] = []
        temp_messages[first_user_index]['parts'].insert(0, system_part)
    elif system_prompt:
        logger.warning("⚠️ System prompt found, but no user messages. Adding system prompt as initial user message.")
        temp_messages.insert(0, {'role': 'user', 'parts': [{'text': system_prompt}]})

    corrected_messages = []
    last_role = None
    for msg in temp_messages:
        current_role = msg['role']
        if current_role == last_role and corrected_messages:
            if 'parts' not in corrected_messages[-1] or not isinstance(corrected_messages[-1]['parts'], list):
                corrected_messages[-1]['parts'] = []
            if 'parts' in msg and isinstance(msg['parts'], list):
                 corrected_messages[-1]['parts'].extend(msg['parts'])
        else:
            corrected_messages.append(msg)
            last_role = current_role

    if not corrected_messages:
        logger.warning("⚠️ No valid messages after formatting.")
        return []

    if corrected_messages and corrected_messages[0]['role'] == 'model':
        logger.warning("⚠️ First message is from model, inserting dummy user message.")
        corrected_messages.insert(0, {'role': 'user', 'parts': [{'text': 'Start conversation'}]})

    # logger.debug(f"Formatted Gemini Messages: {corrected_messages}") # Optional debug
    return corrected_messages

def format_gemini_to_openai_chat(gemini_response, requested_model_name, request_id):
    """Converts a complete Gemini response to OpenAI Chat format."""
    finish_reason_map = {
        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
        'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown',
        'UNSPECIFIED': 'unknown', None: 'unknown'
    }
    openai_response = {
        "id": request_id, "object": "chat.completion", "created": int(time.time()),
        "model": requested_model_name, "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "system_fingerprint": f"gemini-{GEMINI_MODEL_NAME}" # Use the configured model name
    }
    try:
        content = ""
        finish_reason = "stop" # Default
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                all_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                content = all_text

            finish_reason_gemini = getattr(candidate, 'finish_reason', None)
            finish_reason_key = finish_reason_gemini.name if finish_reason_gemini else None
            finish_reason = finish_reason_map.get(finish_reason_key, "unknown")

        elif hasattr(gemini_response, 'text'): # Fallback for simpler responses
             content = gemini_response.text

        openai_response["choices"].append({
            "index": 0, "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason, "logprobs": None,
        })

        if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
            usage = gemini_response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            completion_tokens = getattr(usage, 'candidates_token_count', 0)
            total_tokens = usage.total_token_count
            openai_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        else:
             logger.warning("⚠️ Usage metadata missing from Gemini response.")
             pass # Estimate usage if needed

    except Exception as e:
        logger.error(f"❌ Error processing Gemini response for chat format: {e}")
        logger.exception(e)
        openai_response["choices"] = [{
            "index": 0, "message": {"role": "assistant", "content": f"[ERROR: {e}]"},
            "finish_reason": "error", "logprobs": None
        }]
    return openai_response

def format_gemini_to_openai_completion(gemini_response, requested_model_name, request_id):
    """Converts a complete Gemini response to OpenAI Completion format."""
    finish_reason_map = {
        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
        'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown',
        'UNSPECIFIED': 'unknown', None: 'unknown'
    }
    openai_response = {
        "id": request_id, "object": "text_completion", "created": int(time.time()),
        "model": requested_model_name, "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    try:
        text_content = ""
        finish_reason = "stop" # Default
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

            finish_reason_gemini = getattr(candidate, 'finish_reason', None)
            finish_reason_key = finish_reason_gemini.name if finish_reason_gemini else None
            finish_reason = finish_reason_map.get(finish_reason_key, "unknown")

        elif hasattr(gemini_response, 'text'): # Fallback
             text_content = gemini_response.text

        openai_response["choices"].append({
            "index": 0,
            "text": text_content,
            "logprobs": None,
            "finish_reason": finish_reason
        })

        if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
            usage = gemini_response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            completion_tokens = getattr(usage, 'candidates_token_count', 0)
            total_tokens = usage.total_token_count
            openai_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        else:
            logger.warning("⚠️ Usage metadata missing from Gemini response.")

    except Exception as e:
        logger.error(f"❌ Error processing Gemini response for completion format: {e}")
        logger.exception(e)
        openai_response["choices"] = [{
            "index": 0, "text": f"[ERROR: {e}]",
            "finish_reason": "error", "logprobs": None
        }]
    return openai_response
