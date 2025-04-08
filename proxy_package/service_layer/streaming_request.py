import json
import time
import uuid
import asyncio
import threading
import traceback # Import traceback
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator
from fastapi import HTTPException # Import for raising errors if needed

# Use relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response
# Import the specific LLM classes and response types
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
from ..reporitory_layer.llm.llm_factory import get_llm_client # Use getter
from ..config import GEMINI_DEFAULT_MODEL, AZURE_OPENAI_DEPLOYMENT_NAME, LLM_BACKEND # Import backend info

# Import Gemini types with alias
from google.generativeai.types import generation_types as gemini_generation_types
# Import Azure types with alias
from openai.types.chat import ChatCompletionChunk as AzureChatCompletionChunk
from openai import APIError as AzureAPIError, AuthenticationError as AzureAuthenticationError

# Define a union type for the possible stream chunk types
LLMStreamChunk = Union[gemini_generation_types.GenerateContentResponse, AzureChatCompletionChunk, Exception]

async def stream_response(
    backend_messages: List[Dict[str, Any]],
    generation_config_dict: Dict[str, Any],
    requested_model: str,
    request_id: str,
    is_chat_format: bool = True,
    parse_to_files: bool = False
):
    """Handles streaming requests using the configured LLM client."""
    llm_client = get_llm_client()
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    stream_iterator = None # To hold the stream object

    # Determine backend specifics
    is_gemini = isinstance(llm_client, GeminiLLM)
    is_azure = isinstance(llm_client, AzureLLM)
    # Ensure model_name attribute exists on both clients
    backend_model_name = getattr(llm_client, 'model_name', 'unknown-backend-model')
    system_fingerprint = f"gemini-{backend_model_name}" if is_gemini else f"azure-{backend_model_name}"

    def generator_thread():
        """Runs the blocking stream iteration in a separate thread."""
        nonlocal stream_iterator
        try:
            # Initiate the stream using the correct client method
            logger.info(f"Initiating stream from {'Gemini' if is_gemini else 'Azure'}...")
            stream_iterator = llm_client.generate_content_streaming(
                contents=backend_messages,
                generation_config_dict=generation_config_dict
            )
            # Iterate through the stream and put chunks onto the async queue
            for chunk in stream_iterator:
                # logger.debug(f"Raw chunk received: {chunk}") # Verbose debugging
                loop.call_soon_threadsafe(queue.put_nowait, chunk)

            # Signal the end of the stream
            loop.call_soon_threadsafe(queue.put_nowait, None)
            logger.info(f"Generator thread finished for request {request_id}.")

        # --- Handle GEMINI Exceptions during streaming ---
        except (gemini_generation_types.BlockedPromptException, gemini_generation_types.StopCandidateException) as gemini_error:
            logger.error(f"❌ Gemini API Error in generator thread: {type(gemini_error).__name__} - {gemini_error}")
            loop.call_soon_threadsafe(queue.put_nowait, gemini_error) # Send exception

        # --- Handle AZURE Exceptions during streaming ---
        except (AzureAPIError, AzureAuthenticationError) as azure_error:
             logger.error(f"❌ Azure API Error in generator thread: {type(azure_error).__name__} - Status={getattr(azure_error, 'status_code', 'N/A')} Body={getattr(azure_error, 'body', 'N/A')}")
             loop.call_soon_threadsafe(queue.put_nowait, azure_error) # Send exception

        # --- Handle Generic Exceptions ---
        except Exception as e:
            logger.error(f"❌ Error in generator thread ({'Gemini' if is_gemini else 'Azure'}): {e}")
            logger.exception(e) # Log full traceback for generic errors
            loop.call_soon_threadsafe(queue.put_nowait, e) # Send exception to main loop

    # Start the generator thread
    threading.Thread(target=generator_thread, daemon=True, name=f"StreamGen-{request_id}").start()

    full_response_text = ""
    final_finish_reason = None # Store the *actual* finish reason from the stream
    sent_error_chunk = False

    try:
        while True:
            # Make item Optional to handle None from queue gracefully
            item: Optional[LLMStreamChunk] = await queue.get()

            if item is None:
                logger.info(f"🏁 Stream finished normally for request {request_id}.")
                break # End of stream

            # --- Handle Exceptions received from the queue ---
            if isinstance(item, Exception):
                logger.error(f"❌ Received exception from generator for request {request_id}: {item}")
                # Determine specific error type if possible
                error_type_name = type(item).__name__
                error_detail = str(item)
                # Add more specific details for known error types
                if isinstance(item, (AzureAPIError, AzureAuthenticationError)):
                    status_code = getattr(item, 'status_code', 500)
                    body = getattr(item, 'body', {})
                    error_detail = f"Status={status_code}, Detail={error_detail}, Body={body}" # Include original str(e)
                elif isinstance(item, gemini_generation_types.BlockedPromptException):
                    error_detail = "Prompt was blocked by safety settings."
                elif isinstance(item, gemini_generation_types.StopCandidateException):
                    error_detail = "Generation stopped unexpectedly (check finish reason)."

                error_content = f"[BACKEND_ERROR: {error_type_name} - {error_detail}]" # Closed bracket
                finish_reason = "error" # OpenAI finish reason for errors
                if is_chat_format:
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": finish_reason}],
                        "system_fingerprint": system_fingerprint
                    }
                else: # Completion format
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "text": error_content, "finish_reason": finish_reason}]
                    }
                yield f"data: {json.dumps(payload)}\n\n"
                sent_error_chunk = True
                final_finish_reason = finish_reason # Mark as error finish
                # Continue processing queue until None is received
                continue # Explicitly continue to next item

            # --- Process valid chunk based on backend type ---
            chunk_text = ""
            chunk_finish_reason = None
            payload = None
            processing_error = False # Flag for errors during chunk processing

            try:
                # --- Process Gemini Chunk ---
                if is_gemini:
                    # logger.debug("Processing Gemini chunk") # Optional debug
                    try:
                        # Extract text content
                        if hasattr(item, 'text'):
                            chunk_text = item.text
                        elif hasattr(item, 'candidates') and item.candidates:
                            candidate = item.candidates[0]
                            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content,
                                                                                               'parts') and candidate.content.parts:
                                chunk_text = "".join(
                                    part.text for part in candidate.content.parts if hasattr(part, 'text'))
                            else:
                                chunk_text = ""
                        else:
                            chunk_text = ""

                        # Extract and map finish reason
                        if hasattr(item, 'candidates') and item.candidates:
                            candidate = item.candidates[0]
                            if hasattr(candidate, 'finish_reason') and candidate.finish_reason is not None:
                                finish_reason_gemini = candidate.finish_reason
                                finish_reason_key = finish_reason_gemini.name if hasattr(finish_reason_gemini,
                                                                                         'name') else str(
                                    finish_reason_gemini)
                                finish_reason_map = {
                                    'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
                                    'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown',
                                    'UNSPECIFIED': 'unknown'
                                }
                                mapped_reason = finish_reason_map.get(finish_reason_key, 'unknown')
                                if mapped_reason:
                                    chunk_finish_reason = mapped_reason
                                    final_finish_reason = chunk_finish_reason  # Store latest non-null reason
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing Gemini chunk: {e}. Chunk: {item}")
                        chunk_text = f"[GEMINI_CHUNK_PROCESSING_ERROR: {e}]"
                        processing_error = True


                # --- Process Azure Chunk ---
                elif is_azure:
                    if isinstance(item, AzureChatCompletionChunk):
                        # logger.debug("Processing Azure chunk") # Optional debug
                        try:
                            if item.choices:
                                choice = item.choices[0]
                                if choice.delta and choice.delta.content:
                                    chunk_text = choice.delta.content
                                if choice.finish_reason:
                                    chunk_finish_reason = choice.finish_reason
                                    final_finish_reason = chunk_finish_reason # Store latest non-null reason
                        except Exception as azure_ex:
                            logger.warning(f"⚠️ Error processing Azure chunk content: {azure_ex}. Chunk: {item}")
                            chunk_text = "[AZURE_CHUNK_PROCESSING_ERROR]"
                            processing_error = True

                    else: # <<<< Handle unexpected type FOR AZURE
                        logger.error(f"❌ Received unexpected chunk type for Azure backend: {type(item)}. Item: {item}")
                        chunk_text = f"[UNEXPECTED_AZURE_CHUNK_TYPE: {type(item).__name__}"
                        processing_error = True
                        # chunk_finish_reason = "error"
                        # final_finish_reason = "error"

                # --- Handle Unknown Backend ---
                else:
                    logger.critical(f"❌ Internal Error: Unknown LLM backend type in stream processing loop. Client: {type(llm_client)}")
                    chunk_text = "[INTERNAL_ERROR_UNKNOWN_BACKEND]"
                    processing_error = True
                    chunk_finish_reason = "error"
                    final_finish_reason = "error"

            except Exception as e:
                 # Catch errors during the specific backend processing within the try block above
                 logger.error(f"❌ Error processing chunk content: {e}")
                 logger.exception(e) # Log traceback for processing errors
                 chunk_text = f"[CHUNK_PROCESSING_ERROR: {e}]" # Closed bracket
                 processing_error = True
                 # Potentially set chunk_finish_reason to 'error' here? Maybe not fatal.


            # --- Construct and Yield Payload ---
            # Only yield if there's text, a finish reason, or a processing error marker was generated
            if chunk_text or chunk_finish_reason:
                if is_chat_format:
                    delta_content = {"content": chunk_text} if chunk_text else {}
                    # If a processing error occurred, maybe force finish_reason to error?
                    # if processing_error and not chunk_finish_reason:
                    #    chunk_finish_reason = "error"
                    #    final_finish_reason = "error" # Mark stream as ended badly

                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0,
                                     "delta": delta_content,
                                     "finish_reason": chunk_finish_reason
                                     }],
                        "system_fingerprint": system_fingerprint
                    }
                else: # Completion format
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0,
                                     "text": chunk_text,
                                     "finish_reason": chunk_finish_reason
                                     }]
                    }

                if chunk_text and not processing_error: # Avoid accumulating error markers in full text
                    full_response_text += chunk_text

                # logger.debug(f"Yielding chunk: {json.dumps(payload)}") # Debugging
                yield f"data: {json.dumps(payload)}\n\n"

        # --- Stream finished normally (item was None) ---
        # If parsing is needed and no error occurred during the stream itself
        if not sent_error_chunk and final_finish_reason != "error" and parse_to_files and full_response_text:
            try:
                list_files_adapter = TypeAdapter(Response)
                parsed_response = list_files_adapter.validate_json(full_response_text)
                if parsed_response.files_to_update:
                    parsed_data = [file.model_dump() for file in parsed_response.files_to_update]
                    logger.info(f"✅ Successfully parsed stream response into {len(parsed_data)} FilesToUpdate.")
                else:
                    logger.info("ℹ️ Stream response parsed, but no 'files_to_update' found.")

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"⚠️ Failed to parse full stream response into FilesToUpdate model: {e}")
                # Don't yield an extra chunk for this, client should handle parsing errors

        # Ensure a final chunk with finish_reason is sent *if* the stream ended
        # without a reason being sent AND no fatal error occurred.
        if final_finish_reason is None and not sent_error_chunk:
             final_finish_reason = 'stop' # Assume normal stop if nothing else indicated
             logger.warning(f"⚠️ No finish reason received from backend for request {request_id}. Defaulting to 'stop'. Sending final chunk.")
             if is_chat_format:
                 final_payload = {
                     "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": requested_model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": final_finish_reason}],
                     "system_fingerprint": system_fingerprint
                 }
             else: # Completion format
                 final_payload = {
                     "id": request_id, "object": "text_completion", "created": int(time.time()),
                     "model": requested_model,
                     "choices": [{"index": 0, "text": "", "finish_reason": final_finish_reason}]
                 }
             yield f"data: {json.dumps(final_payload)}\n\n"

    except asyncio.CancelledError:
         logger.info(f"🚫 Stream cancelled by client for request {request_id}.")
         # TODO: Consider how to signal cancellation to the generator thread if needed/possible.
    except Exception as e:
         logger.error(f"❌ Unexpected error during stream processing loop for request {request_id}: {e}")
         logger.exception(e)
         # Attempt to yield a final error chunk if not already done
         if not sent_error_chunk:
             error_content = f"[STREAM_PROCESSING_ERROR: {e}]" # Closed bracket
             finish_reason = "error"
             # Construct payload based on is_chat_format
             if is_chat_format:
                 payload = { "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": finish_reason}], "system_fingerprint": system_fingerprint }
             else:
                 payload = { "id": request_id, "object": "text_completion", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "text": error_content, "finish_reason": finish_reason}] }
             try:
                 yield f"data: {json.dumps(payload)}\n\n"
             except Exception as yield_err:
                  logger.error(f"❌ Failed to yield final error chunk after processing error: {yield_err}")
    finally:
        # Always yield [DONE for SSE standard compliance
        yield "data: [DONE]\n\n"
        logger.info(f"🏁 Sent [DONE for request {request_id}")
