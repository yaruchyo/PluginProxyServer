from fastapi.responses import StreamingResponse # Keep only necessary FastAPI imports if any
import json
import time
import uuid
import asyncio
import threading
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

# Use relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response
from ..context import gemini_llm # Import initialized LLM client from context
from ..config import GEMINI_MODEL_NAME # Import configured model name for fingerprint

from google.generativeai.types import generation_types # Keep Gemini types

# Function definition remains largely the same, update imports and potentially fingerprint logic

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
