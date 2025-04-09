import json
import time
import uuid
import asyncio
import threading
import traceback
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator, Tuple, AsyncGenerator

# Use relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response
from ..reporitory_layer.llm.llm_factory import LLMClient # Import base class for type hinting
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
from ..reporitory_layer.llm.llm_factory import get_llm_client
from ..config import GEMINI_DEFAULT_MODEL, AZURE_OPENAI_DEPLOYMENT_NAME, LLM_BACKEND
from google.genai.types import GenerateContentResponse
# Import backend-specific types with aliases
from google.generativeai.types import generation_types as gemini_generation_types
from openai.types.chat import ChatCompletionChunk as AzureChatCompletionChunk
from openai import APIError as AzureAPIError, AuthenticationError as AzureAuthenticationError

# Define a union type for the possible stream chunk types from the backend generator
BackendStreamItem = Union[GenerateContentResponse, AzureChatCompletionChunk, Exception]

# --- Constants ---
SSE_DATA_PREFIX = "data: "
SSE_DONE_MESSAGE = f"{SSE_DATA_PREFIX}[DONE]\n\n"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
TEXT_COMPLETION_OBJECT = "text_completion" # Assuming this is the identifier for non-chat format

# --- Helper Function for Payload Formatting ---

def _format_sse_payload(
    request_id: str,
    model: str,
    system_fingerprint: Optional[str],
    is_chat_format: bool,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    is_error: bool = False
) -> str:
    """Formats the data payload for Server-Sent Events (SSE)."""
    created = int(time.time())
    payload: Dict[str, Any]

    if is_chat_format:
        delta = {"content": content} if content else {}
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        payload = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": created,
            "model": model,
            "choices": [choice],
        }
        # Add system_fingerprint only if available (OpenAI standard)
        if system_fingerprint:
             payload["system_fingerprint"] = system_fingerprint
    else: # Completion format
        choice = {"index": 0, "text": content or "", "finish_reason": finish_reason}
        payload = {
            "id": request_id,
            "object": TEXT_COMPLETION_OBJECT,
            "created": created,
            "model": model,
            "choices": [choice],
        }

    return f"{SSE_DATA_PREFIX}{json.dumps(payload)}\n\n"

# --- Stream Processor Class ---

class StreamProcessor:
    """
    Manages the process of receiving chunks from an LLM backend in a separate
    thread, processing them, and formatting them as Server-Sent Events.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        backend_messages: List[Dict[str, Any]],
        generation_config_dict: Dict[str, Any],
        requested_model: str,
        request_id: str,
        is_chat_format: bool = True,
        parse_to_files: bool = False
    ):
        self.llm_client = llm_client
        self.backend_messages = backend_messages
        self.generation_config_dict = generation_config_dict
        self.requested_model = requested_model
        self.request_id = request_id
        self.is_chat_format = is_chat_format
        self.parse_to_files = parse_to_files

        self.loop = asyncio.get_event_loop()
        self.queue: asyncio.Queue[Optional[BackendStreamItem]] = asyncio.Queue()
        self.stream_iterator: Optional[Iterator[BackendStreamItem]] = None # To hold the stream object
        self.full_response_text = ""
        self.final_finish_reason: Optional[str] = None
        self.sent_error_chunk = False
        self.is_gemini = isinstance(self.llm_client, GeminiLLM)
        self.is_azure = isinstance(self.llm_client, AzureLLM)
        # Use a method on the client if possible, otherwise construct here
        self.system_fingerprint = self._get_system_fingerprint()

        logger.info(f"[{self.request_id}] Initialized StreamProcessor for {'Gemini' if self.is_gemini else 'Azure' if self.is_azure else 'Unknown'} backend.")

    def _get_system_fingerprint(self) -> Optional[str]:
        """Determines the system fingerprint based on the backend."""
        # Ideally, this would be a method on the LLMClient interface
        if hasattr(self.llm_client, 'get_system_fingerprint'):
             # Assuming LLMClient classes implement this
             return self.llm_client.get_system_fingerprint()
        else:
            # Fallback based on type checking (less ideal)
            backend_model_name = getattr(self.llm_client, 'model_name', 'unknown-model')
            if self.is_gemini:
                return f"gemini-{backend_model_name}"
            elif self.is_azure:
                 # Azure OpenAI API generally includes this in the response chunk itself
                 # We might not need to pre-define it here if we extract it from the chunk later.
                 # For now, let's return None and rely on the chunk if available.
                 # return f"azure-{backend_model_name}" # Or return None
                 return None # Let Azure chunks provide it if they do
            else:
                return "unknown-backend"


    def _generator_thread_target(self):
        """
        Runs the blocking stream iteration in a separate thread and puts
        items (chunks or exceptions) onto the asyncio queue.
        """
        try:
            backend_name = type(self.llm_client).__name__
            logger.info(f"[{self.request_id}] Generator thread starting: Initiating stream from {backend_name}...")
            self.stream_iterator = self.llm_client.generate_content_streaming(
                contents=self.backend_messages,
                generation_config_dict=self.generation_config_dict
            )

            for chunk in self.stream_iterator:
                # logger.debug(f"[{self.request_id}] Raw chunk received: {chunk}")
                self.loop.call_soon_threadsafe(self.queue.put_nowait, chunk)

            # Signal the end of the stream
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
            logger.info(f"[{self.request_id}] Generator thread finished normally.")

        except (gemini_generation_types.BlockedPromptException, gemini_generation_types.StopCandidateException) as gemini_error:
            logger.error(f"[{self.request_id}] ❌ Gemini API Error in generator thread: {type(gemini_error).__name__} - {gemini_error}")
            self.loop.call_soon_threadsafe(self.queue.put_nowait, gemini_error)

        except (AzureAPIError, AzureAuthenticationError) as azure_error:
             logger.error(f"[{self.request_id}] ❌ Azure API Error in generator thread: {type(azure_error).__name__} - Status={getattr(azure_error, 'status_code', 'N/A')} Body={getattr(azure_error, 'body', 'N/A')}")
             self.loop.call_soon_threadsafe(self.queue.put_nowait, azure_error)

        except Exception as e:
            logger.error(f"[{self.request_id}] ❌ Unexpected Error in generator thread ({type(self.llm_client).__name__}): {e}")
            logger.exception(e) # Log full traceback
            self.loop.call_soon_threadsafe(self.queue.put_nowait, e)

    def _parse_gemini_chunk(self, item: gemini_generation_types.GenerateContentResponse) -> Tuple[Optional[str], Optional[str]]:
        """Parses a Gemini chunk into text content and finish reason."""
        chunk_text = None
        chunk_finish_reason = None
        try:
            # Extract text (handle potential variations in structure)
            if hasattr(item, 'text'):
                chunk_text = item.text
            elif hasattr(item, 'candidates') and item.candidates:
                candidate = item.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    chunk_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

            # Extract and map finish reason
            if hasattr(item, 'candidates') and item.candidates:
                candidate = item.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason is not None:
                    finish_reason_gemini = candidate.finish_reason
                    # Use .name for enum, fallback to str()
                    finish_reason_key = finish_reason_gemini.name if hasattr(finish_reason_gemini, 'name') else str(finish_reason_gemini)
                    # Map Gemini reasons to OpenAI-like reasons
                    finish_reason_map = {
                        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
                        'RECITATION': 'recitation', # Or map to 'stop' or 'content_filter' depending on desired behavior
                        'OTHER': 'unknown', 'UNKNOWN': 'unknown', 'UNSPECIFIED': 'unknown'
                    }
                    chunk_finish_reason = finish_reason_map.get(finish_reason_key.upper(), 'unknown') # Use upper for safety

        except Exception as e:
            logger.warning(f"[{self.request_id}] ⚠️ Error parsing Gemini chunk: {e}. Chunk: {item}")
            chunk_text = f"[GEMINI_CHUNK_PARSING_ERROR: {e}]" # Indicate error in content
            # Don't set finish_reason here, let the stream try to continue or end naturally

        return chunk_text, chunk_finish_reason

    def _parse_azure_chunk(self, item: AzureChatCompletionChunk) -> Tuple[Optional[str], Optional[str]]:
        """Parses an Azure chunk into text content and finish reason."""
        chunk_text = None
        chunk_finish_reason = None
        try:
            if item.choices:
                choice = item.choices[0]
                if choice.delta and choice.delta.content:
                    chunk_text = choice.delta.content
                if choice.finish_reason:
                    chunk_finish_reason = choice.finish_reason
                # Potentially extract system_fingerprint if needed and not set globally
                # if hasattr(item, 'system_fingerprint') and item.system_fingerprint:
                #    self.system_fingerprint = item.system_fingerprint

        except Exception as e:
            logger.warning(f"[{self.request_id}] ⚠️ Error parsing Azure chunk: {e}. Chunk: {item}")
            chunk_text = f"[AZURE_CHUNK_PARSING_ERROR: {e}]"

        return chunk_text, chunk_finish_reason

    def _process_chunk(self, item: BackendStreamItem) -> Tuple[Optional[str], Optional[str], bool]:
        """Processes a valid chunk based on the backend type."""
        llm_client = get_llm_client()
        chunk_text: Optional[str] = None
        chunk_finish_reason: Optional[str] = None
        processing_error = False # Flag for errors *during* parsing

        try:
            chunk_text, chunk_finish_reason = llm_client.parse_chunks(item)
        except Exception as e:
             # Catch errors during the specific backend parsing logic above
             logger.error(f"[{self.request_id}] ❌ Unexpected error processing chunk content: {e}")
             logger.exception(e)
             chunk_text = f"[CHUNK_PROCESSING_ERROR: {e}]"
             processing_error = True
             # chunk_finish_reason = "error" # Mark as error?

        # Update the final finish reason if a new one is received
        if chunk_finish_reason:
            self.final_finish_reason = chunk_finish_reason

        # Accumulate text only if it's valid content
        if chunk_text and not processing_error:
            self.full_response_text += chunk_text

        return chunk_text, chunk_finish_reason, processing_error

    def _process_exception(self, item: Exception) -> Tuple[str, str]:
        """Processes an exception received from the generator thread."""
        logger.error(f"[{self.request_id}] ❌ Received exception from generator: {type(item).__name__} - {item}")
        error_type_name = type(item).__name__
        error_detail = str(item)
        finish_reason = "error" # OpenAI standard finish reason for errors

        # Add specific details for known error types
        if isinstance(item, (AzureAPIError, AzureAuthenticationError)):
            status_code = getattr(item, 'status_code', 500)
            body = getattr(item, 'body', {})
            error_detail = f"Status={status_code}, Detail={error_detail}, Body={body}"
        elif isinstance(item, gemini_generation_types.BlockedPromptException):
            error_detail = "Prompt blocked by safety settings."
            finish_reason = "content_filter" # More specific reason
        elif isinstance(item, gemini_generation_types.StopCandidateException):
             # This might indicate an issue but isn't always a fatal error itself.
             # The finish_reason should come from the candidate if available.
             # However, if it arrives here as an exception, treat it as an error.
            error_detail = "Generation stopped unexpectedly (check finish reason if available)."
            finish_reason = "error" # Or potentially map from candidate if possible

        error_content = f"[BACKEND_ERROR: {error_type_name} - {error_detail}]"
        self.sent_error_chunk = True # Mark that we've sent an error
        self.final_finish_reason = finish_reason # Ensure stream ends with error status

        return error_content, finish_reason

    async def _handle_final_parsing(self):
        """Handles parsing the full response if needed after the stream ends."""
        if not self.sent_error_chunk and self.final_finish_reason != "error" and self.parse_to_files and self.full_response_text:
            logger.info(f"[{self.request_id}] Attempting to parse full response for FilesToUpdate...")
            try:
                list_files_adapter = TypeAdapter(Response)
                # Use validate_json which handles potential JSON errors
                parsed_response = list_files_adapter.validate_json(self.full_response_text)
                if parsed_response.files_to_update:
                    parsed_count = len(parsed_response.files_to_update)
                    logger.info(f"[{self.request_id}] ✅ Successfully parsed stream into {parsed_count} FilesToUpdate.")
                    # Optionally yield a custom event here if needed by the client?
                    # yield f"event: files_parsed\ndata: {json.dumps([f.model_dump() for f in parsed_response.files_to_update])}\n\n"
                else:
                    logger.info(f"[{self.request_id}] ℹ️ Stream response parsed, but no 'files_to_update' found.")
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"[{self.request_id}] ⚠️ Failed to parse full stream response into FilesToUpdate model: {e}")
                # Do not yield an error chunk here; client-side parsing is separate.
                # Log is sufficient.

    async def process_stream(self) -> AsyncGenerator[str, None]:
        """
        Asynchronously processes items from the queue and yields SSE formatted strings.
        """
        # Start the generator thread
        threading.Thread(
            target=self._generator_thread_target,
            daemon=True,
            name=f"StreamGen-{self.request_id}"
        ).start()

        try:
            while True:
                item = await self.queue.get()

                if item is None:
                    logger.info(f"[{self.request_id}] 🏁 Reached end of stream signal (None received).")
                    break # End of stream

                content: Optional[str] = None
                finish_reason: Optional[str] = None
                is_error = False

                # --- Handle Exceptions from Queue ---
                if isinstance(item, Exception):
                    content, finish_reason = self._process_exception(item)
                    is_error = True
                    # Yield the error chunk immediately
                    yield _format_sse_payload(
                        self.request_id, self.requested_model, self.system_fingerprint,
                        self.is_chat_format, content, finish_reason, is_error=True
                    )
                    # Continue processing queue until None, but don't process further chunks normally
                    continue

                # --- Process Valid Chunk ---
                # Skip processing normal chunks if a fatal error was already sent
                if self.sent_error_chunk:
                    logger.debug(f"[{self.request_id}] Skipping chunk processing after error was sent.")
                    continue

                content, chunk_finish_reason, processing_error = self._process_chunk(item)

                # Only yield if there's content or a finish reason
                if content or chunk_finish_reason:
                    yield _format_sse_payload(
                        self.request_id, self.requested_model, self.system_fingerprint,
                        self.is_chat_format, content, chunk_finish_reason
                    )

            # --- Stream finished normally (processed None) ---
            await self._handle_final_parsing()

            # Ensure a final chunk with finish_reason is sent *if* the stream ended
            # without one AND no fatal error occurred.
            if self.final_finish_reason is None and not self.sent_error_chunk:
                self.final_finish_reason = 'stop' # Assume normal stop
                logger.warning(f"[{self.request_id}] ⚠️ No finish reason received from backend. Defaulting to 'stop'. Sending final chunk.")
                yield _format_sse_payload(
                    self.request_id, self.requested_model, self.system_fingerprint,
                    self.is_chat_format, content=None, finish_reason=self.final_finish_reason
                )

        except asyncio.CancelledError:
            logger.info(f"[{self.request_id}] 🚫 Stream cancelled by client.")
            # TODO: Consider signaling cancellation to the generator thread if possible/necessary.
            # This might involve setting a flag or using a more complex mechanism if
            # the underlying SDK supports cancellation. For now, the thread will exit
            # when the client disconnects and the yield fails.
        except Exception as e:
            logger.error(f"[{self.request_id}] ❌ Unexpected error during stream processing loop: {e}")
            logger.exception(e)
            # Attempt to yield a final error chunk if not already done
            if not self.sent_error_chunk:
                error_content = f"[STREAM_PROCESSING_ERROR: {e}]"
                finish_reason = "error"
                try:
                    yield _format_sse_payload(
                        self.request_id, self.requested_model, self.system_fingerprint,
                        self.is_chat_format, error_content, finish_reason, is_error=True
                    )
                except Exception as yield_err:
                    logger.error(f"[{self.request_id}] ❌ Failed to yield final error chunk after processing error: {yield_err}")
        finally:
            # Always yield [DONE] for SSE standard compliance
            yield SSE_DONE_MESSAGE
            logger.info(f"[{self.request_id}] ✅ Sent [DONE] message.")


# --- Public API Function ---

async def stream_response(
    backend_messages: List[Dict[str, Any]],
    generation_config_dict: Dict[str, Any],
    requested_model: str,
    request_id: str,
    is_chat_format: bool = True,
    parse_to_files: bool = False
) -> AsyncGenerator[str, None]:
    """~
    Handles streaming requests using the configured LLM client by delegating
    to the StreamProcessor class.

    Yields:
        Server-Sent Event (SSE) formatted strings.
    """
    llm_client = get_llm_client()
    processor = StreamProcessor(
        llm_client=llm_client,
        backend_messages=backend_messages,
        generation_config_dict=generation_config_dict,
        requested_model=requested_model,
        request_id=request_id,
        is_chat_format=is_chat_format,
        parse_to_files=parse_to_files
    )
    async for chunk in processor.process_stream():
        yield chunk
