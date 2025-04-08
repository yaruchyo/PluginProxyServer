# --- Import logger from the utility module using relative path ---
from plugin_proxy_server.utils.logger import logger
from openai import AzureOpenAI, APIError, AuthenticationError # Import AzureOpenAI and relevant errors
from openai.types.chat import ChatCompletion, ChatCompletionChunk # Import response types
from typing import Any, Optional, List, Dict, Union, Iterator
import traceback

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
            logger.error("❌ Error: AZURE_OPENAI_KEY is required.")
            raise ValueError("AZURE_OPENAI_KEY is required.")
        if not api_version:
            logger.error("❌ Error: AZURE_OPENAI_API_VERSION is required.")
            raise ValueError("AZURE_OPENAI_API_VERSION is required.")
        if not endpoint:
            logger.error("❌ Error: AZURE_OPENAI_ENDPOINT is required.")
            raise ValueError("AZURE_OPENAI_ENDPOINT is required.")
        if not deployment_name:
            logger.error("❌ Error: AZURE_OPENAI_DEPLOYMENT_NAME is required.")
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
        # Filter out None values and potentially map keys if needed in the future
        # Note: Entrypoint layer already maps keys like max_tokens, temperature etc.
        filtered_params = {k: v for k, v in generation_config_dict.items() if v is not None}
        # Remove keys not directly supported by Azure chat completions create if necessary
        # Example: 'response_mime_type', 'response_schema' are Gemini specific
        filtered_params.pop('response_mime_type', None)
        filtered_params.pop('response_schema', None)
        # Azure uses 'stop' instead of 'stop_sequences'
        if 'stop_sequences' in filtered_params:
            filtered_params['stop'] = filtered_params.pop('stop_sequences')

        # logger.debug(f"Prepared Azure params: {filtered_params}") # Optional debug
        return filtered_params

    def generate_content(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> ChatCompletion:
        """
        Generates content using the Azure OpenAI API (non-streaming).

        Args:
            contents: The list of messages in OpenAI format.
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
             logger.error(f"❌ Azure API Error (Non-Streaming): {type(e).__name__} - {e}")
             traceback.print_exc() # Log full traceback for API errors
             raise # Re-raise specific Azure exceptions
        except Exception as e:
            logger.error(f"❌ Unexpected Error during Azure non-streaming call: {e}")
            logger.exception(e)
            raise # Re-raise other exceptions

    def generate_content_streaming(self, contents: List[Dict[str, Any]], generation_config_dict: Optional[Dict[str, Any]] = None) -> Iterator[ChatCompletionChunk]:
        """
        Generates content using the Azure OpenAI API (streaming).

        Args:
            contents: The list of messages in OpenAI format.
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
        except Exception as e:
            logger.error(f"❌ Error initiating Azure streaming call: {e}")
            logger.exception(e)
            raise # Re-raise exceptions during stream initiation
