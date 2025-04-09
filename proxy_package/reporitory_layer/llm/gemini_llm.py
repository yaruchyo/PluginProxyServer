# --- Import logger from the utility module using relative path ---
from proxy_package.utils.logger import logger
from google import genai
from google.genai import types # Import types
from google.generativeai.types import generation_types
from typing import Any, Optional, List, Dict, Union, Iterator, Tuple
from proxy_package.service_layer.formating import format_openai_to_gemini
from pydantic import BaseModel, TypeAdapter, ValidationError
# Removed Response import as it's domain layer, not directly used here
from ...domain_layer.file_responce import Response # Use relative import

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
            logger.error("❌ Error: GOOGLE_API_KEY is required for GeminiLLM.")
            raise ValueError("GOOGLE_API_KEY is required.")
        if not model_name:
            logger.error("❌ Error: GOOGLE_MODEL name is required for GeminiLLM.")
            raise ValueError("GOOGLE_MODEL is required.")

        self.model_name = model_name
        self.full_model_name = f"models/{model_name}" # Construct the full model path

        try:
            # Configure the client
            # TODO: Add support for multiple API keys if needed (round-robin, etc.)
            self.client = genai.Client(api_key=api_key)
            logger.info(f"✅ Configured Google Generative AI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Error configuring Gemini client: {e}")
            logger.exception(e) # Log traceback
            raise RuntimeError(f"Failed to configure Gemini client: {e}") from e

    def _create_config(self, generation_config_dict: Optional[Dict[str, Any]] = None) -> Optional[types.GenerateContentConfig]:
        """Creates a GenerateContentConfig object from a dictionary."""
        if not generation_config_dict:
            return None

        gemini_config_params = {}
        # Map common OpenAI params to Gemini params
        if generation_config_dict.get('max_tokens') is not None: # Map from OpenAI's standard key
            gemini_config_params['max_output_tokens'] = generation_config_dict['max_tokens']
        elif generation_config_dict.get('max_output_tokens') is not None: # Or use Gemini's key directly
            gemini_config_params['max_output_tokens'] = generation_config_dict['max_output_tokens']

        if generation_config_dict.get('temperature') is not None:
            gemini_config_params['temperature'] = generation_config_dict['temperature']
        if generation_config_dict.get('top_p') is not None:
            gemini_config_params['top_p'] = generation_config_dict['top_p']
        # Gemini uses 'stop_sequences'
        stop_val = generation_config_dict.get('stop') or generation_config_dict.get('stop_sequences')
        if stop_val:
            if isinstance(stop_val, str):
                gemini_config_params['stop_sequences'] = [stop_val]
            elif isinstance(stop_val, list):
                gemini_config_params['stop_sequences'] = stop_val

        # Handle JSON mode (mime type and schema)
        if generation_config_dict.get('response_mime_type'):
             gemini_config_params['response_mime_type'] = generation_config_dict['response_mime_type']
        if generation_config_dict.get('response_schema'):
             # Note: Gemini expects the Pydantic model class itself, not an instance
             gemini_config_params['response_schema'] = generation_config_dict['response_schema']
             # Ensure mime type is set if schema is used
             if 'response_mime_type' not in gemini_config_params:
                 gemini_config_params['response_mime_type'] = 'application/json'


        # Filter out None values before creating the config object
        filtered_config_dict = {k: v for k, v in gemini_config_params.items() if v is not None}

        if not filtered_config_dict:
            return None # Return None if dict becomes empty

        try:
            # logger.debug(f"Creating Gemini config with: {filtered_config_dict}") # Optional debug
            # Use ** to unpack the dictionary into keyword arguments
            return types.GenerateContentConfig(**filtered_config_dict)
        except Exception as e:
            logger.warning(f"⚠️ Could not create GenerateContentConfig from dict: {filtered_config_dict}. Error: {e}")
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

    def create_backend_messages(self, openai_messages) -> List[Dict[str, Any]]:
        return format_openai_to_gemini(openai_messages)

    def parse_chunks(self, item: types.GenerateContentResponse) -> Tuple[Optional[str], Optional[str]]:
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

    def generate_structured_content(self, full_answer) -> Response:
        prompt = f"extranct from the answer requred information\n\nANSWER:\n\n{full_answer}"


        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': Response,
            },
        )


        # Use instantiated objects.
        structured_response: Response = response.parsed
        return structured_response
