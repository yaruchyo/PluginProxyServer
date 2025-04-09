import json
import time
import uuid
import asyncio
import threading
import traceback
from pydantic import TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator, Tuple, AsyncGenerator
import os
# Use relative imports
from ..utils.logger import logger
from ..domain_layer.file_responce import Response
from ..reporitory_layer.llm.llm_factory import LLMClient # Import base class/union type
from ..reporitory_layer.llm.gemini_llm import GeminiLLM
from ..reporitory_layer.llm.azure_llm import AzureLLM
# from ..reporitory_layer.llm.llm_factory import get_llm_client # REMOVE this import
from ..config import GEMINI_DEFAULT_MODEL, AZURE_OPENAI_DEPLOYMENT_NAME # Keep for potential defaults/info
from google.genai.types import GenerateContentResponse
# Import backend-specific types with aliases
from google.generativeai.types import generation_types as gemini_generation_types
from openai.types.chat import ChatCompletionChunk as AzureChatCompletionChunk
from openai import APIError as AzureAPIError, AuthenticationError as AzureAuthenticationError
from proxy_package.reporitory_layer.agents.tools import save_files_from_response
from proxy_package.service_layer.formating import _format_sse_payload
# Import the SSE constants Enum
from ..domain_layer.sse_domain import SSEConstants

# Define a union type for the possible stream chunk types from the backend generator
BackendStreamItem = Union[GenerateContentResponse, AzureChatCompletionChunk, Exception]

class StreamProcessor:
    """
    Manages the process of receiving chunks from an LLM backend in a separate
    thread, processing them, and formatting them as Server-Sent Events.
    """
    def __init__(
        self,
        llm_client: LLMClient, # Takes the specific client instance
        backend_messages: List[Dict[str, Any]],
        generation_config_dict: Dict[str, Any],
        requested_model: str,
        request_id: str,
        is_chat_format: bool = True,
        parse_to_files: bool = False
    ):
        self.llm_client = llm_client # Store the passed client
        self.backend_messages = backend_messages
        self.generation_config_dict = generation_config_dict
        self.requested_model = requested_model
        self.request_id = request_id
        self.is_chat_format = is_chat_format
        self.parse_to_files = parse_to_files

        self.loop = asyncio.get_event_loop()
        self.queue: asyncio.Queue[Optional[BackendStreamItem]] = asyncio.Queue()
        self.stream_iterator: Optional[Iterator[BackendStreamItem]] = None
        self.full_response_text = ""
        self.final_finish_reason: Optional[str] = None
        self.sent_error_chunk = False
        # Determine backend type from the passed client instance
        self.is_gemini = isinstance(self.llm_client, GeminiLLM)
        self.is_azure = isinstance(self.llm_client, AzureLLM)
        self.system_fingerprint = self._get_system_fingerprint() # Use the instance client

        logger.info(f"[{self.request_id}] Initialized StreamProcessor for {type(self.llm_client).__name__} backend.")

    def _get_system_fingerprint(self) -> Optional[str]:
        """Determines the system fingerprint based on the backend."""
        # Use self.llm_client
        if hasattr(self.llm_client, 'get_system_fingerprint'):
             return self.llm_client.get_system_fingerprint()
        else:
            backend_model_name = getattr(self.llm_client, 'model_name', 'unknown-model') # Or deployment_name for Azure
            if self.is_gemini:
                return f"gemini-{backend_model_name}"
            elif self.is_azure:
                 # Azure chunks might provide it, return None here initially
                 return None
            else:
                return "unknown-backend"

    def _generator_thread_target(self):
        """
        Runs the blocking stream iteration in a separate thread and puts
        items (chunks or exceptions) onto the asyncio queue.
        """
        try:
            # Use self.llm_client
            backend_name = type(self.llm_client).__name__
            logger.info(f"[{self.request_id}] Generator thread starting: Initiating stream from {backend_name}...")
            # Use self.llm_client
            self.stream_iterator = self.llm_client.generate_content_streaming(
                contents=self.backend_messages,
                generation_config_dict=self.generation_config_dict
            )

            for chunk in self.stream_iterator:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, chunk)

            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
            logger.info(f"[{self.request_id}] Generator thread finished normally.")

        # --- Exception Handling (remains largely the same, but uses self.llm_client context if needed) ---
        except (gemini_generation_types.BlockedPromptException, gemini_generation_types.StopCandidateException) as gemini_error:
            logger.error(f"[{self.request_id}] ❌ Gemini API Error in generator thread: {type(gemini_error).__name__} - {gemini_error}")
            self.loop.call_soon_threadsafe(self.queue.put_nowait, gemini_error)
        except (AzureAPIError, AzureAuthenticationError) as azure_error:
             logger.error(f"[{self.request_id}] ❌ Azure API Error in generator thread: {type(azure_error).__name__} - Status={getattr(azure_error, 'status_code', 'N/A')} Body={getattr(azure_error, 'body', 'N/A')}")
             self.loop.call_soon_threadsafe(self.queue.put_nowait, azure_error)
        except Exception as e:
            # Use self.llm_client to get backend name
            backend_name = type(self.llm_client).__name__
            logger.error(f"[{self.request_id}] ❌ Unexpected Error in generator thread ({backend_name}): {e}")
            logger.exception(e) # Log full traceback
            self.loop.call_soon_threadsafe(self.queue.put_nowait, e)

    # --- _parse_gemini_chunk and _parse_azure_chunk (Keep as is) ---
    # These are helper methods internal to the processor
    def _parse_gemini_chunk(self, item: gemini_generation_types.GenerateContentResponse) -> Tuple[Optional[str], Optional[str]]:
        # ... (implementation remains the same) ...
        chunk_text = None
        chunk_finish_reason = None
        try:
            if hasattr(item, 'text'):
                chunk_text = item.text
            elif hasattr(item, 'candidates') and item.candidates:
                candidate = item.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    chunk_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            if hasattr(item, 'candidates') and item.candidates:
                candidate = item.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason is not None:
                    finish_reason_gemini = candidate.finish_reason
                    finish_reason_key = finish_reason_gemini.name if hasattr(finish_reason_gemini, 'name') else str(finish_reason_gemini)
                    finish_reason_map = {
                        'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter',
                        'RECITATION': 'recitation', 'OTHER': 'unknown', 'UNKNOWN': 'unknown', 'UNSPECIFIED': 'unknown'
                    }
                    chunk_finish_reason = finish_reason_map.get(finish_reason_key.upper(), 'unknown')
        except Exception as e:
            logger.warning(f"[{self.request_id}] ⚠️ Error parsing Gemini chunk: {e}. Chunk: {item}")
            chunk_text = f"[GEMINI_CHUNK_PARSING_ERROR: {e}]"
        return chunk_text, chunk_finish_reason

    def _parse_azure_chunk(self, item: AzureChatCompletionChunk) -> Tuple[Optional[str], Optional[str]]:
        # ... (implementation remains the same) ...
        chunk_text = None
        chunk_finish_reason = None
        try:
            if item.choices:
                choice = item.choices[0]
                if choice.delta and choice.delta.content:
                    chunk_text = choice.delta.content
                if choice.finish_reason:
                    chunk_finish_reason = choice.finish_reason
        except Exception as e:
            logger.warning(f"[{self.request_id}] ⚠️ Error parsing Azure chunk: {e}. Chunk: {item}")
            chunk_text = f"[AZURE_CHUNK_PARSING_ERROR: {e}]"
        return chunk_text, chunk_finish_reason


    def _process_chunk(self, item: BackendStreamItem) -> Tuple[Optional[str], Optional[str], bool]:
        """Processes a valid chunk based on the backend type using polymorphism."""
        # llm_client = get_llm_client() # REMOVE THIS LINE
        chunk_text: Optional[str] = None
        chunk_finish_reason: Optional[str] = None
        processing_error = False

        try:
            chunk_text, chunk_finish_reason = self.llm_client.parse_chunks(item)

        except AttributeError:
             logger.error(f"[{self.request_id}] ❌ LLM client ({type(self.llm_client).__name__}) does not implement 'parse_chunks' method.")
             # Fallback to isinstance check if parse_chunks isn't implemented
             if self.is_gemini and isinstance(item, GenerateContentResponse):
                 chunk_text, chunk_finish_reason = self._parse_gemini_chunk(item)
             elif self.is_azure and isinstance(item, AzureChatCompletionChunk):
                 chunk_text, chunk_finish_reason = self._parse_azure_chunk(item)
             else:
                 logger.error(f"[{self.request_id}] ❌ Cannot parse chunk of type {type(item)} for backend {type(self.llm_client).__name__}.")
                 chunk_text = f"[CHUNK_PARSING_UNSUPPORTED_TYPE: {type(item)}]"
                 processing_error = True

        except Exception as e:
             logger.error(f"[{self.request_id}] ❌ Unexpected error processing chunk content: {e}")
             logger.exception(e)
             chunk_text = f"[CHUNK_PROCESSING_ERROR: {e}]"
             processing_error = True

        if chunk_finish_reason:
            self.final_finish_reason = chunk_finish_reason
        if chunk_text and not processing_error:
            self.full_response_text += chunk_text

        return chunk_text, chunk_finish_reason, processing_error

    # --- _process_exception (Keep as is) ---
    def _process_exception(self, item: Exception) -> Tuple[str, str]:
        # ... (implementation remains the same) ...
        logger.error(f"[{self.request_id}] ❌ Received exception from generator: {type(item).__name__} - {item}")
        error_type_name = type(item).__name__
        error_detail = str(item)
        finish_reason = "error"
        if isinstance(item, (AzureAPIError, AzureAuthenticationError)):
            status_code = getattr(item, 'status_code', 500)
            body = getattr(item, 'body', {})
            error_detail = f"Status={status_code}, Detail={error_detail}, Body={body}"
        elif isinstance(item, gemini_generation_types.BlockedPromptException):
            error_detail = "Prompt blocked by safety settings."
            finish_reason = "content_filter"
        elif isinstance(item, gemini_generation_types.StopCandidateException):
            error_detail = "Generation stopped unexpectedly (check finish reason if available)."
            finish_reason = "error"
        error_content = f"[BACKEND_ERROR: {error_type_name} - {error_detail}]"
        self.sent_error_chunk = True
        self.final_finish_reason = finish_reason
        return error_content, finish_reason


    # --- _handle_final_parsing (Keep as is) ---
    async def _handle_final_parsing(self):
        # ... (implementation remains the same) ...
        if not self.sent_error_chunk and self.final_finish_reason != "error" and self.parse_to_files and self.full_response_text:
            logger.info(f"[{self.request_id}] Attempting to parse full response for FilesToUpdate...")
            try:
                list_files_adapter = TypeAdapter(Response)
                parsed_response = list_files_adapter.validate_json(self.full_response_text)
                if parsed_response.files_to_update:
                    parsed_count = len(parsed_response.files_to_update)
                    logger.info(f"[{self.request_id}] ✅ Successfully parsed stream into {parsed_count} FilesToUpdate.")
                else:
                    logger.info(f"[{self.request_id}] ℹ️ Stream response parsed, but no 'files_to_update' found.")
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"[{self.request_id}] ⚠️ Failed to parse full stream response into FilesToUpdate model: {e}")


    # --- process_stream (Keep as is, relies on internal methods using self.llm_client) ---
    async def process_stream(self) -> AsyncGenerator[str, None]:
        # ... (implementation remains the same, uses self.llm_client implicitly via other methods) ...
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
                    break
                content: Optional[str] = None
                finish_reason: Optional[str] = None
                is_error = False
                if isinstance(item, Exception):
                    content, finish_reason = self._process_exception(item)
                    is_error = True
                    yield _format_sse_payload(
                        self.request_id, self.requested_model, self.system_fingerprint,
                        self.is_chat_format, content, finish_reason, is_error=True
                    )
                    continue
                if self.sent_error_chunk:
                    logger.debug(f"[{self.request_id}] Skipping chunk processing after error was sent.")
                    continue
                content, chunk_finish_reason, processing_error = self._process_chunk(item)
                if content or chunk_finish_reason:
                    yield _format_sse_payload(
                        self.request_id, self.requested_model, self.system_fingerprint,
                        self.is_chat_format, content, chunk_finish_reason
                    )
            await self._handle_final_parsing()
            if self.final_finish_reason is None and not self.sent_error_chunk:
                self.final_finish_reason = 'stop'
                logger.warning(f"[{self.request_id}] ⚠️ No finish reason received from backend. Defaulting to 'stop'. Sending final chunk.")
                yield _format_sse_payload(
                    self.request_id, self.requested_model, self.system_fingerprint,
                    self.is_chat_format, content=None, finish_reason=self.final_finish_reason
                )
        except asyncio.CancelledError:
            logger.info(f"[{self.request_id}] 🚫 Stream cancelled by client.")
        except Exception as e:
            logger.error(f"[{self.request_id}] ❌ Unexpected error during stream processing loop: {e}")
            logger.exception(e)
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
            # Use the Enum to get the DONE message
            yield SSEConstants.get_done_message()
            logger.info(f"[{self.request_id}] ✅ Sent [DONE] message.")


# --- Public API Function ---
async def stream_response(
    llm_client: LLMClient, # Takes the specific client instance
    backend_messages: List[Dict[str, Any]],
    generation_config_dict: Dict[str, Any],
    requested_model: str,
    request_id: str,
    is_chat_format: bool = True,
    parse_to_files: bool = False,

) -> AsyncGenerator[str, None]:
    """
    Handles streaming requests using the provided LLM client by delegating
    to the StreamProcessor class.

    Yields:
        Server-Sent Event (SSE) formatted strings.
    """
    # No need to get client here, it's passed in
    processor = StreamProcessor(
        llm_client=llm_client, # Pass the received client
        backend_messages=backend_messages,
        generation_config_dict=generation_config_dict,
        requested_model=requested_model,
        request_id=request_id,
        is_chat_format=is_chat_format,
        parse_to_files=parse_to_files
    )
    async for chunk in processor.process_stream():
        yield chunk


    if True: # TODO: This condition seems always true, review logic if needed
        structured_response = llm_client.generate_structured_content(processor.full_response_text)
        save_files_from_response(structured_response)
