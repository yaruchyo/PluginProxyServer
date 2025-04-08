# --- Import logger from the utility module using relative path ---
from proxy_package.utils.logger import logger
from google import genai
from google.genai import types # Import types
from google.generativeai.types import generation_types
from typing import Any, Optional, List, Dict, Union, Iterator

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
