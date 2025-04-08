# --- Helper Functions (Formatting remains the same) ---
from fastapi import HTTPException # Keep only necessary imports
import os
import json
import time
import uuid
import asyncio
from proxy_package.domain_layer.file_responce import Response # Relative import
from google.generativeai.types import generation_types as gemini_generation_types # Alias for clarity
from openai.types.chat import ChatCompletion, ChatCompletionChunk # Azure response types
from dotenv import load_dotenv
import traceback
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from proxy_package.utils.logger import logger # Relative import
from proxy_package.config import GEMINI_DEFAULT_MODEL, AZURE_OPENAI_DEPLOYMENT_NAME, LLM_BACKEND # Import backend info

# --- Gemini Specific Formatting ---

def format_openai_to_gemini(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts OpenAI message format to Gemini format."""
    # Keep the existing implementation (ensure relative imports if needed)
    gemini_messages = []
    system_prompt = None
    temp_messages = []

    for message in openai_messages:
        role = message.get('role')
        content = message.get('content')
        if not content: # Skip empty messages
            continue

        # Handle system prompt separately
        if role == 'system':
            # Gemini prefers the system prompt to be handled differently (often prepended to the first user message or via specific API params if available)
            # We'll store it and prepend it later.
            system_prompt = content
            logger.info(f"ℹ️ Found system prompt: {str(content)[:100]}...")
            continue # Don't add system message directly to temp_messages yet

        # Map roles
        gemini_role = 'user' if role == 'user' else 'model' # Treat 'assistant' and potentially others as 'model'

        # Handle content (text or multimodal parts)
        parts = []
        if isinstance(content, str):
            parts.append({'text': content})
        elif isinstance(content, list): # OpenAI multimodal format
            for item in content:
                item_type = item.get('type')
                if item_type == 'text':
                    parts.append({'text': item['text']})
                elif item_type == 'image_url':
                    # Process image URLs (assuming base64 data URI format for simplicity here)
                    # Production code might need robust URL fetching and base64 encoding
                    url = item.get('image_url', {}).get('url')
                    if url and url.startswith('data:'):
                        try:
                            # Basic parsing for data URI: data:[<mime_type>][;base64],<data>
                            header, data = url.split(',', 1)
                            mime_type_part = header.split(':')[1]
                            mime_type = mime_type_part.split(';')[0]
                            if ';base64' in mime_type_part:
                                parts.append({'inline_data': {'mime_type': mime_type, 'data': data}})
                            else:
                                logger.warning(f"⚠️ Skipping non-base64 image data URI: {url[:50]}...")
                        except Exception as e:
                            logger.error(f"❌ Error parsing image data URI: {e} - URL: {url[:50]}...")
                    elif url:
                         logger.warning(f"⚠️ Skipping non-data URI image URL: {url[:100]}... (Gemini requires inline data or Google Cloud Storage URIs)")
        else:
             logger.warning(f"⚠️ Skipping message with unexpected content type: {type(content)}")
             continue # Skip message if content format is unknown

        if parts: # Only add message if it has content parts
            temp_messages.append({'role': gemini_role, 'parts': parts})

    # --- Gemini Conversation Structure Rules ---
    # 1. Role Toggling: Must alternate between 'user' and 'model'.
    # 2. Starts with 'user': The conversation must begin with a 'user' role.

    # Prepend system prompt to the first user message's text part
    if system_prompt:
        first_user_index = next((i for i, msg in enumerate(temp_messages) if msg['role'] == 'user'), -1)
        if first_user_index != -1:
            system_part = {'text': f"System Prompt:\n{system_prompt}\n\n---\n\nUser Message:\n"} # Add clear separation
            # Ensure the first part is text, otherwise prepend
            if temp_messages[first_user_index]['parts'] and temp_messages[first_user_index]['parts'][0].get('text') is not None:
                 temp_messages[first_user_index]['parts'][0]['text'] = system_part['text'] + temp_messages[first_user_index]['parts'][0]['text']
            else:
                 temp_messages[first_user_index]['parts'].insert(0, system_part)
            logger.info("ℹ️ Prepended system prompt to the first user message.")
        else:
            # If no user message exists, create one with the system prompt
            logger.warning("⚠️ System prompt provided but no user messages found. Creating initial user message with system prompt.")
            temp_messages.insert(0, {'role': 'user', 'parts': [{'text': system_prompt}]})

    # Correct role alternation if needed (merge consecutive messages of the same role)
    corrected_messages = []
    if not temp_messages:
        logger.warning("⚠️ No messages to format for Gemini after initial processing.")
        return []

    last_role = None
    for msg in temp_messages:
        current_role = msg['role']
        if current_role == last_role and corrected_messages:
            # Merge parts into the previous message
            logger.debug(f"Merging consecutive '{current_role}' messages.")
            corrected_messages[-1]['parts'].extend(msg['parts'])
        else:
            # Add new message
            corrected_messages.append(msg)
            last_role = current_role

    # Ensure the first message is 'user'
    if corrected_messages and corrected_messages[0]['role'] != 'user':
        logger.warning("⚠️ First message after merging is not 'user'. Prepending a dummy user message.")
        # This indicates an issue with the input sequence (e.g., starting with 'assistant')
        # Prepending a generic user message might be necessary for the API call to succeed.
        corrected_messages.insert(0, {'role': 'user', 'parts': [{'text': '(Start of conversation)'}]})


    # logger.debug(f"Formatted Gemini Messages: {json.dumps(corrected_messages, indent=2)}") # Optional debug
    return corrected_messages


def format_gemini_to_openai_chat(gemini_response: gemini_generation_types.GenerateContentResponse, requested_model_name: str, request_id: str) -> Dict[str, Any]:
    """Converts a complete Gemini response to OpenAI Chat format."""
    # Keep the existing implementation
    finish_reason_map = {
        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
        'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown',
        'UNSPECIFIED': 'unknown', None: 'unknown' # Handle None case
    }
    openai_response = {
        "id": request_id, "object": "chat.completion", "created": int(time.time()),
        "model": requested_model_name, # Report the model originally requested by the client
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # Use the actual Gemini model used by the backend
        "system_fingerprint": f"gemini-{GEMINI_DEFAULT_MODEL}"
    }
    try:
        content = ""
        finish_reason = "stop" # Default if not found
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Check for response text directly (simpler structure)
        if hasattr(gemini_response, 'text'):
            content = gemini_response.text
            # Cannot determine finish_reason or usage from simple text response

        # Check for candidates (standard structure)
        elif hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            # Extract content
            if candidate.content and candidate.content.parts:
                # Concatenate text parts
                content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

            # Extract finish reason
            finish_reason_gemini = getattr(candidate, 'finish_reason', None)
            # Use .name attribute if it's an Enum, otherwise use the value itself if it's already a string/None
            finish_reason_key = finish_reason_gemini.name if hasattr(finish_reason_gemini, 'name') else finish_reason_gemini
            finish_reason = finish_reason_map.get(finish_reason_key, "unknown")

        else:
             logger.warning("⚠️ Gemini response structure not recognized or empty.")
             content = "[ERROR: Empty or unparseable Gemini response]"
             finish_reason = "error"


        openai_response["choices"].append({
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
            "logprobs": None, # Not provided by Gemini API in this format
        })

        # Extract usage
        if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
            usage = gemini_response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            # Use 'candidates_token_count' if available, otherwise estimate or default to 0
            completion_tokens = getattr(usage, 'candidates_token_count', 0)
            total_tokens = usage.total_token_count
            openai_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        else:
             logger.warning("⚠️ Usage metadata missing from Gemini response.")
             # Optionally estimate usage based on tokenizing 'content' if needed

    except Exception as e:
        logger.error(f"❌ Error processing Gemini response for chat format: {e}")
        logger.exception(e)
        # Provide a fallback error response
        openai_response["choices"] = [{
            "index": 0, "message": {"role": "assistant", "content": f"[ERROR: Failed to parse Gemini response - {e}]"},
            "finish_reason": "error", "logprobs": None
        }]
        openai_response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Reset usage on error

    return openai_response


def format_gemini_to_openai_completion(gemini_response: gemini_generation_types.GenerateContentResponse, requested_model_name: str, request_id: str) -> Dict[str, Any]:
    """Converts a complete Gemini response to OpenAI Completion format."""
    # Keep the existing implementation
    finish_reason_map = {
        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
        'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown',
        'UNSPECIFIED': 'unknown', None: 'unknown'
    }
    openai_response = {
        "id": request_id, "object": "text_completion", "created": int(time.time()),
        "model": requested_model_name,
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # No system_fingerprint in standard completion object
    }
    try:
        text_content = ""
        finish_reason = "stop" # Default
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Check for response text directly
        if hasattr(gemini_response, 'text'):
            text_content = gemini_response.text

        # Check for candidates
        elif hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

            finish_reason_gemini = getattr(candidate, 'finish_reason', None)
            finish_reason_key = finish_reason_gemini.name if hasattr(finish_reason_gemini, 'name') else finish_reason_gemini
            finish_reason = finish_reason_map.get(finish_reason_key, "unknown")
        else:
             logger.warning("⚠️ Gemini response structure not recognized or empty for completion.")
             text_content = "[ERROR: Empty or unparseable Gemini response]"
             finish_reason = "error"


        openai_response["choices"].append({
            "index": 0,
            "text": text_content,
            "logprobs": None, # Not provided
            "finish_reason": finish_reason
        })

        # Extract usage
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
            "index": 0, "text": f"[ERROR: Failed to parse Gemini response - {e}]",
            "finish_reason": "error", "logprobs": None
        }]
        openai_response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return openai_response


# --- NEW: Azure Specific Formatting ---

def format_azure_to_openai_chat(azure_response: ChatCompletion, requested_model_name: str, request_id: str) -> Dict[str, Any]:
    """Converts a complete Azure ChatCompletion response to standard OpenAI Chat format."""
    openai_response = {
        "id": request_id, # Use the proxy-generated ID
        "object": "chat.completion",
        "created": azure_response.created,
        "model": requested_model_name, # Report the model requested by the client
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # Use the actual Azure deployment name from the response or config
        "system_fingerprint": azure_response.system_fingerprint or f"azure-{AZURE_OPENAI_DEPLOYMENT_NAME}"
    }

    try:
        if azure_response.choices:
            choice = azure_response.choices[0]
            openai_response["choices"].append({
                "index": choice.index,
                # Use model_dump() for safe serialization of the Pydantic message model
                "message": choice.message.model_dump() if choice.message else {"role": "assistant", "content": None},
                "finish_reason": choice.finish_reason,
                "logprobs": getattr(choice, 'logprobs', None), # Include logprobs if present
            })
        else:
            logger.warning("⚠️ Azure response has no choices.")
            openai_response["choices"].append({
                "index": 0,
                "message": {"role": "assistant", "content": "[ERROR: No choices in Azure response]"},
                "finish_reason": "error",
                "logprobs": None
            })

        if azure_response.usage:
            # Use model_dump() for safe serialization of the Pydantic usage model
            openai_response["usage"] = azure_response.usage.model_dump()
        else:
            logger.warning("⚠️ Usage information missing from Azure response.")

    except Exception as e:
        logger.error(f"❌ Error processing Azure response for chat format: {e}")
        logger.exception(e)
        openai_response["choices"] = [{
            "index": 0, "message": {"role": "assistant", "content": f"[ERROR: Failed to parse Azure response - {e}]"},
            "finish_reason": "error", "logprobs": None
        }]
        openai_response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return openai_response


def format_azure_to_openai_completion(azure_response: ChatCompletion, requested_model_name: str, request_id: str) -> Dict[str, Any]:
    """
    Converts an Azure ChatCompletion response (used for the /v1/completions endpoint)
    to the standard OpenAI Completion format.
    """
    openai_response = {
        "id": request_id, # Use the proxy-generated ID
        "object": "text_completion",
        "created": azure_response.created,
        "model": requested_model_name, # Report the model requested by the client
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # No system_fingerprint in standard completion object
    }

    try:
        text_content = ""
        finish_reason = "stop" # Default
        logprobs = None

        if azure_response.choices:
            choice = azure_response.choices[0]
            # Extract text content from the assistant's message
            text_content = choice.message.content if choice.message and choice.message.content else ""
            finish_reason = choice.finish_reason
            logprobs = getattr(choice, 'logprobs', None) # Get logprobs if available

            openai_response["choices"].append({
                "index": choice.index,
                "text": text_content,
                "logprobs": logprobs, # Include if present
                "finish_reason": finish_reason
            })
        else:
            logger.warning("⚠️ Azure response has no choices for completion format.")
            openai_response["choices"].append({
                "index": 0,
                "text": "[ERROR: No choices in Azure response]",
                "logprobs": None,
                "finish_reason": "error"
            })

        if azure_response.usage:
            openai_response["usage"] = azure_response.usage.model_dump()
        else:
            logger.warning("⚠️ Usage information missing from Azure response.")

    except Exception as e:
        logger.error(f"❌ Error processing Azure response for completion format: {e}")
        logger.exception(e)
        openai_response["choices"] = [{
            "index": 0, "text": f"[ERROR: Failed to parse Azure response - {e}]",
            "finish_reason": "error", "logprobs": None
        }]
        openai_response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return openai_response