import asyncio
import json
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import loguru
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types  # Import types
from google.generativeai.types import generation_types
from pydantic import BaseModel, TypeAdapter, ValidationError

logger = loguru.logger
# Load environment variables
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY", None)
GEMINI_MODEL_NAME =os.environ.get("GOOGLE_MODEL", 'gemini-1.5-pro-latest')

# --- GeminiLLM Class ---
class GeminiLLM:
    """Encapsulates Google Gemini API interactions."""
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the GeminiLLM client.

        Args:
            api_key: The Google API key.
            model_name: The name of the Gemini model to use (e.g., 'gemini-1.5-pro-latest').
        """
        if not api_key:
            logger.error("❌ Error: GOOGLE_API_KEY environment variable not set.")
            raise ValueError("GOOGLE_API_KEY is required.")
        if not model_name:
            logger.error("❌ Error: GOOGLE_MODEL environment variable not set.")
            raise ValueError("GOOGLE_MODEL is required.")

        self.model_name = model_name
        self.full_model_name = f"models/{model_name}" # Construct the full model path

        try:
            # Configure the client
            self.client = genai.Client(api_key=api_key)
            logger.info(f"✅ Configured Google Generative AI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Error configuring Gemini client: {e}")
            logger.exception(e) # Log traceback
            raise RuntimeError(f"Failed to configure Gemini client: {e}") from e

    def _create_config(self, generation_config_dict: Optional[Dict[str, Any]] = None) -> Optional[types.GenerateContentConfig]:
        """Creates a GenerateContentConfig object from a dictionary."""
        if not generation_config_dict:
            return None # Return None if no config is provided
        try:
            # Filter out None values before creating the config object
            filtered_config_dict = {k: v for k, v in generation_config_dict.items() if v is not None}
            if not filtered_config_dict:
                 return None # Return None if dict becomes empty after filtering
            return types.GenerateContentConfig(**filtered_config_dict)
        except Exception as e:
            logger.warning(f"⚠️ Could not create GenerateContentConfig from dict: {generation_config_dict}. Error: {e}")
            # Decide how to handle: return None, raise error, or return default? Returning None for now.
            return None

    def generate_content(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> generation_types.GenerateContentResponse:
        """
        Generates content using the Gemini API (non-streaming).

        Args:
            contents: The list of messages in Gemini format.
            generation_config_dict: A dictionary containing generation parameters.

        Returns:
            The raw response object from the Gemini API.

        Raises:
            generation_types.BlockedPromptException: If the prompt was blocked.
            generation_types.StopCandidateException: If generation stopped prematurely but has partial response.
            Exception: For other API or configuration errors.
        """
        gemini_config = self._create_config(generation_config_dict)
        logger.info(f"⚙️ Calling Gemini (Non-Streaming) with config: {gemini_config}")
        try:
            response = self.client.models.generate_content(
                model=self.full_model_name,
                contents=contents,
                config=gemini_config # Pass the config object or None
            )
            logger.success("✅ Gemini Raw Response (Non-Streaming) received.")
            # logger.debug(f"Gemini Raw Response (Non-Streaming):\n{response}") # Optional detailed logging
            return response
        except (generation_types.BlockedPromptException, generation_types.StopCandidateException) as e:
             logger.error(f"❌ Gemini API Generation Error (Non-Streaming): {type(e).__name__} - {e}")
             raise # Re-raise specific Gemini exceptions
        except Exception as e:
            logger.error(f"❌ Unexpected Error during Gemini non-streaming call: {e}")
            logger.exception(e)
            raise # Re-raise other exceptions

    def generate_content_streaming(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> Iterator[generation_types.GenerateContentResponse]:
        """
        Generates content using the Gemini API (streaming).

        Args:
            contents: The list of messages in Gemini format.
            generation_config_dict: A dictionary containing generation parameters.

        Returns:
            An iterator yielding response chunks from the Gemini API.

        Raises:
            Exception: For API or configuration errors during stream initiation.
                     Errors during iteration are handled by the consumer.
        """
        gemini_config = self._create_config(generation_config_dict)
        logger.info(f"⚙️ Calling Gemini (Streaming) with config: {gemini_config}")
        try:
            stream = self.client.models.generate_content_stream(
                model=self.full_model_name,
                contents=contents,
                config=gemini_config # Pass the config object or None
            )
            logger.success("✅ Gemini Stream initiated.")
            return stream
        except Exception as e:
            logger.error(f"❌ Error initiating Gemini streaming call: {e}")
            logger.exception(e)
            raise # Re-raise exceptions during stream initiation

# --- FastAPI App Setup ---
app = FastAPI(title="fast_api_proxy")

# Initialize the GeminiLLM client
try:
    gemini_llm = GeminiLLM(api_key=API_KEY, model_name=GEMINI_MODEL_NAME)
except (ValueError, RuntimeError) as e:
    logger.critical(f"❌ Failed to initialize GeminiLLM: {e}")
    exit(1) # Exit if client setup fails

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class FilesToUpdate(BaseModel):
    filename: Optional[str]
    code_to_update: Optional[str]

class Response(BaseModel):
    answer: str
    files_to_update: Optional[List[FilesToUpdate]]

# --- Helper Functions (Formatting remains the same) ---
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

# --- Request Handlers (Modified to use GeminiLLM) ---

# Async function for non-streaming requests
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

# Async generator for streaming responses
async def stream_gemini_response(gemini_messages: List[Dict[str, Any]], generation_config_dict: Dict[str, Any], requested_model: str, request_id: str, is_chat_format: bool = True, parse_to_files: bool = False):
    """Handles streaming requests using the GeminiLLM instance."""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def generator_thread():
        """Runs the blocking stream iteration in a separate thread."""
        try:
            # Get the stream iterator from the GeminiLLM class
            stream = gemini_llm.generate_content_streaming(
                contents=gemini_messages,
                generation_config_dict=generation_config_dict
            )
            for chunk in stream:
                # logger.debug(f"Raw chunk: {chunk}") # Debugging
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None) # Signal end
        except Exception as e:
            logger.error(f"❌ Error in generator thread: {e}")
            logger.exception(e)
            loop.call_soon_threadsafe(queue.put_nowait, e) # Send exception to main loop

    threading.Thread(target=generator_thread, daemon=True).start()

    full_response_text = ""
    finish_reason = None
    try:
        while True:
            item = await queue.get()
            if item is None:
                logger.info("🏁 Stream finished.")
                break # End of stream
            if isinstance(item, Exception):
                logger.error(f"❌ Received exception from generator: {item}")
                error_content = f"[ERROR: {item}]"
                if is_chat_format:
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": "error"}],
                        "system_fingerprint": f"gemini-{GEMINI_MODEL_NAME}"
                    }
                else:
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "text": error_content, "finish_reason": "error"}]
                    }
                yield f"data: {json.dumps(payload)}\n\n"
                return # Stop generation

            # --- Process valid chunk ---
            chunk_text = ""
            try:
                # Extract text content (handle potential variations in chunk structure)
                if hasattr(item, 'text'):
                    chunk_text = item.text
                elif hasattr(item, 'candidates') and item.candidates:
                     candidate = item.candidates[0]
                     if candidate.content and candidate.content.parts:
                         chunk_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                     # Check for finish reason in the chunk's candidate
                     if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                         finish_reason_gemini = getattr(candidate, 'finish_reason', None)
                         finish_reason_key = finish_reason_gemini.name if finish_reason_gemini else None
                         finish_reason_map = { 'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter', 'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown', 'UNSPECIFIED': 'unknown', None: None }
                         current_chunk_finish_reason = finish_reason_map.get(finish_reason_key)
                         if current_chunk_finish_reason: # Store the first non-None finish reason found
                             finish_reason = current_chunk_finish_reason

            except Exception as e:
                 logger.warning(f"⚠️ Error processing chunk content: {e}")
                 chunk_text = f"[CHUNK_ERROR: {e}]"

            if chunk_text:
                full_response_text += chunk_text
                if is_chat_format:
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                        "system_fingerprint": f"gemini-{GEMINI_MODEL_NAME}"
                    }
                else: # Completion format
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "text": chunk_text, "finish_reason": None}]
                    }
                # logger.debug(f"Yielding: {json.dumps(payload)}") # Debugging
                yield f"data: {json.dumps(payload)}\n\n"

        # --- Stream finished, handle final chunk with finish_reason ---
        if finish_reason or parse_to_files: # Send final chunk if finish_reason is known or if parsing needed
            final_delta_content = ""
            parsed_data = None

            if parse_to_files and full_response_text:
                try:
                    list_files_adapter = TypeAdapter(Response)
                    parsed_response = list_files_adapter.validate_json(full_response_text)
                    if parsed_response.files_to_update:
                        parsed_data = [file.model_dump() for file in parsed_response.files_to_update]
                        final_delta_content = f"\n[Parsed {len(parsed_data)} files to update]"
                        logger.info(f"✅ Successfully parsed response into FilesToUpdate model.")
                    else:
                        logger.info("ℹ️ Response parsed, but no 'files_to_update' found.")

                except (ValidationError, json.JSONDecodeError) as e:
                    logger.warning(f"⚠️ Failed to parse full response into FilesToUpdate model: {e}")
                    final_delta_content = f"\n[WARNING: Failed to parse response JSON - {e}]"

            # Send the final chunk with finish_reason
            final_finish_reason = finish_reason or 'stop' # Default to stop if None
            if is_chat_format:
                final_payload = {
                    "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"content": final_delta_content} if final_delta_content else {}, "finish_reason": final_finish_reason}],
                    "system_fingerprint": f"gemini-{GEMINI_MODEL_NAME}"
                }
            else: # Completion format
                final_payload = {
                    "id": request_id, "object": "text_completion", "created": int(time.time()),
                    "model": requested_model,
                    "choices": [{"index": 0, "text": final_delta_content, "finish_reason": final_finish_reason}]
                }

            yield f"data: {json.dumps(final_payload)}\n\n"

    except asyncio.CancelledError:
         logger.info("🚫 Stream cancelled by client.")
    except Exception as e:
         logger.error(f"❌ Unexpected error during stream processing: {e}")
         logger.exception(e)
         # Attempt to yield a final error chunk
         error_content = f"[STREAM_ERROR: {e}]"
         if is_chat_format:
             payload = { "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": "error"}] }
         else:
             payload = { "id": request_id, "object": "text_completion", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "text": error_content, "finish_reason": "error"}] }
         yield f"data: {json.dumps(payload)}\n\n"
    finally:
        # Always yield [DONE] for SSE standard compliance
        yield "data: [DONE]\n\n"
        logger.info("🏁 Sent [DONE]")


# --- API Endpoints (Modified to pass config dict) ---

@app.post("/v1/completions")
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


@app.post("/v1/chat/completions")
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

# --- Server Start Function ---
def start() -> None:
    logger.info("Starting FastAPI proxy...")
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 1234))
    reload = os.environ.get("RELOAD", "true").lower() == "true"

    if workers > 1 and reload:
        logger.warning("⚠️ WARNING: Running multiple workers with reload=True is not recommended. Setting workers to 1.")
        workers = 1

    logger.info(f"Starting Uvicorn server on {host}:{port} with {workers} worker(s)")
    logger.info(f"Reloading enabled: {reload}")

    uvicorn.run(
        app="fast_api_proxy:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None, # Use loguru
        workers=workers,
    )

if __name__ == "__main__":
    start()