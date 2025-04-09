# --- Import logger from the utility module using relative path ---
from ...utils.logger import logger # Use relative import
from proxy_package.domain_layer.file_responce import Response
from openai import AzureOpenAI, APIError, AuthenticationError # Import AzureOpenAI and relevant errors
from openai.types.chat import ChatCompletion, ChatCompletionChunk # Import response types
from typing import Any, Optional, List, Dict, Union, Iterator, Tuple
from google.genai import types
from openai.types.chat import ChatCompletionChunk as AzureChatCompletionChunk
BackendStreamItem = Union[types.GenerateContentResponse, AzureChatCompletionChunk, Exception]

class AzureLLM:
    """Encapsulates Azure OpenAI API interactions."""
    def __init__(self, api_key: str, api_version: str, endpoint: str, deployment_name: str, max_retries: int = 3):
        """
        Initializes the AzureLLM client.

        Args:
            api_key: The Azure OpenAI API key.
            api_version: The Azure OpenAI API version (e.g., '2024-02-01').
            endpoint: The Azure OpenAI endpoint URL.
            deployment_name: The name of the deployed Azure OpenAI model.
            max_retries: Maximum number of retries for API calls.
        """
        if not api_key:
            logger.error("❌ Error: AZURE_OPENAI_KEY is required for AzureLLM.")
            raise ValueError("AZURE_OPENAI_KEY is required.")
        if not api_version:
            logger.error("❌ Error: AZURE_OPENAI_API_VERSION is required for AzureLLM.")
            raise ValueError("AZURE_OPENAI_API_VERSION is required.")
        if not endpoint:
            logger.error("❌ Error: AZURE_OPENAI_ENDPOINT is required for AzureLLM.")
            raise ValueError("AZURE_OPENAI_ENDPOINT is required.")
        if not deployment_name:
            logger.error("❌ Error: AZURE_OPENAI_DEPLOYMENT_NAME (model) is required for AzureLLM.")
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is required.")

        self.deployment_name = deployment_name
        self.model_name = deployment_name # Use deployment name as the identifier

        try:
            # Configure the client
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                max_retries=max_retries,
            )
            logger.info(f"✅ Configured Azure OpenAI client:")
            logger.info(f"   Endpoint: {endpoint}")
            logger.info(f"   Deployment: {self.deployment_name}")
            logger.info(f"   API Version: {api_version}")
        except Exception as e:
            logger.error(f"❌ Error configuring Azure OpenAI client: {e}")
            logger.exception(e) # Log traceback
            raise RuntimeError(f"Failed to configure Azure OpenAI client: {e}") from e

    def _prepare_params(self, generation_config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepares the parameter dictionary for the Azure API call."""
        if not generation_config_dict:
            return {}

        azure_params = {}
        # Map common OpenAI params to Azure params
        if generation_config_dict.get('max_output_tokens') is not None: # Map from Gemini's potential key
            azure_params['max_tokens'] = generation_config_dict['max_output_tokens']
        elif generation_config_dict.get('max_tokens') is not None: # Or use OpenAI's standard key
            azure_params['max_tokens'] = 16384

        if generation_config_dict.get('temperature') is not None:
            azure_params['temperature'] = generation_config_dict['temperature']
        if generation_config_dict.get('top_p') is not None:
            azure_params['top_p'] = generation_config_dict['top_p']
        if generation_config_dict.get('presence_penalty') is not None:
             azure_params['presence_penalty'] = generation_config_dict['presence_penalty']
        if generation_config_dict.get('frequency_penalty') is not None:
             azure_params['frequency_penalty'] = generation_config_dict['frequency_penalty']
        if generation_config_dict.get('logit_bias') is not None:
             azure_params['logit_bias'] = generation_config_dict['logit_bias']
        if generation_config_dict.get('user') is not None:
             azure_params['user'] = generation_config_dict['user']

        # Handle stop sequences (Gemini uses 'stop_sequences', OpenAI/Azure use 'stop')
        stop_val = generation_config_dict.get('stop_sequences') or generation_config_dict.get('stop')
        if stop_val:
            if isinstance(stop_val, str) or isinstance(stop_val, list):
                azure_params['stop'] = stop_val
            else:
                 logger.warning(f"⚠️ Invalid type for 'stop' parameter: {type(stop_val)}. Ignoring.")

        # Filter out None values before returning
        filtered_params = {k: v for k, v in azure_params.items() if v is not None}
        # logger.debug(f"Prepared Azure params: {filtered_params}") # Optional debug
        return filtered_params

    def generate_content(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> ChatCompletion:
        """
        Generates content using the Azure OpenAI API (non-streaming).

        Args:
            contents: The list of messages in standard OpenAI format.
            generation_config_dict: A dictionary containing generation parameters
                                     (e.g., max_tokens, temperature, stop).

        Returns:
            The raw ChatCompletion response object from the Azure OpenAI API.

        Raises:
            APIError: If the Azure API returns an error.
            AuthenticationError: If authentication fails.
            Exception: For other API or configuration errors.
        """
        azure_params = self._prepare_params(generation_config_dict)
        logger.info(f"⚙️ Calling Azure OpenAI (Non-Streaming) with params: {azure_params}")
        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.deployment_name, # Use the deployment name
                messages=contents,          # Expects OpenAI message format
                stream=False,
                **azure_params              # Pass filtered config parameters
            )
            logger.success("✅ Azure Raw Response (Non-Streaming) received.")
            # logger.debug(f"Azure Raw Response (Non-Streaming):\n{response.model_dump_json(indent=2)}") # Optional detailed logging
            return response
        except (APIError, AuthenticationError) as e:
             logger.error(f"❌ Azure API Error (Non-Streaming): {type(e).__name__} - Status={getattr(e, 'status_code', 'N/A')} Body={getattr(e, 'body', 'N/A')}")
             traceback.print_exc() # Log full traceback for API errors
             raise # Re-raise specific Azure exceptions
        except Exception as e:
            logger.error(f"❌ Unexpected Error during Azure non-streaming call: {e}")
            logger.exception(e)
            raise # Re-raise other exceptions

    def generate_structured_content_streaming(
            self,
            contents: List[Dict[str, Any]],
            generation_config_dict: Optional[Dict[str, Any]] = None,
            ) :
        """
        Generates content using the Azure OpenAI API (streaming).

        Args:
            contents: The list of messages in standard OpenAI format.
            generation_config_dict: A dictionary containing generation parameters.

        Returns:
            An iterator yielding ChatCompletionChunk objects from the Azure API.

        Raises:
            Exception: For API or configuration errors during stream initiation.
                     Errors during iteration are handled by the consumer.
        """
        azure_params = self._prepare_params(generation_config_dict)
        logger.info(f"⚙️ Calling Azure OpenAI (Streaming) with params: {azure_params}")
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=contents,
                response_format=Res,
                **azure_params
            )
            event = completion.choices[0].message.parsed

            return event

        except (APIError, AuthenticationError) as e:  # Catch errors during initiation
            logger.error(
                f"❌ Azure API Error (Streaming Init): {type(e).__name__} - Status={getattr(e, 'status_code', 'N/A')} Body={getattr(e, 'body', 'N/A')}")
            traceback.print_exc()
            raise  # Re-raise specific Azure exceptions
        except Exception as e:
            logger.error(f"❌ Error initiating Azure streaming call: {e}")
            logger.exception(e)
            raise  # Re-raise exceptions during stream initiation


    def generate_content_streaming(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> Iterator[ChatCompletionChunk]:
        """
        Generates content using the Azure OpenAI API (streaming).

        Args:
            contents: The list of messages in standard OpenAI format.
            generation_config_dict: A dictionary containing generation parameters.

        Returns:
            An iterator yielding ChatCompletionChunk objects from the Azure API.

        Raises:
            Exception: For API or configuration errors during stream initiation.
                     Errors during iteration are handled by the consumer.
        """
        azure_params = self._prepare_params(generation_config_dict)
        logger.info(f"⚙️ Calling Azure OpenAI (Streaming) with params: {azure_params}")
        try:
            stream: Iterator[ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=contents,
                stream=True,
                **azure_params
            )
            logger.success("✅ Azure Stream initiated.")
            return stream
        except (APIError, AuthenticationError) as e: # Catch errors during initiation
             logger.error(f"❌ Azure API Error (Streaming Init): {type(e).__name__} - Status={getattr(e, 'status_code', 'N/A')} Body={getattr(e, 'body', 'N/A')}")
             raise # Re-raise specific Azure exceptions
        except Exception as e:
            logger.error(f"❌ Error initiating Azure streaming call: {e}")
            logger.exception(e)
            raise # Re-raise exceptions during stream initiation


    def create_backend_messages(self, openai_messages) -> List[Dict[str, Any]]:
        return openai_messages

    def parse_chunks(self, item: BackendStreamItem) -> Tuple[Optional[str], Optional[str], bool]:
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