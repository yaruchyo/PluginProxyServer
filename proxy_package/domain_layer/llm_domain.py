from typing import Optional

from pydantic import BaseModel, Field


class LLMParams(BaseModel):
    apiKey: Optional[str] = Field(None, description="The API key used for authentication with the LLM provider.")
    model: Optional[str] = Field(None, description="The specific model to be used for generating responses.")
    alternative_model: Optional[str] = Field(None, description="An alternative model to be used if the primary model is unavailable.")
    provider: Optional[str] = Field(None, description="The name of the LLM provider.")
    endpoint: Optional[str] = Field(None, description="The endpoint URL for accessing the LLM service.")
    version: Optional[str] = Field(None, description="The version of the LLM model or API.")
    max_retries: int = Field(3, description="The maximum number of retries for a failed request.")
class LLMResponseModel(BaseModel):
    llm_params: Optional[LLMParams] = Field(None, description="Parameters related to the LLM configuration.")

# Example usage
