from pydantic import BaseModel, Field
from typing import Optional

class LLMParams(BaseModel):
    apiKey: Optional[str] = Field(None, description="The API key used for authentication with the LLM provider.")
    model: Optional[str] = Field(None, description="The specific model to be used for generating responses.")
    provider: Optional[str] = Field(None, description="The name of the LLM provider.")
    endpoint: Optional[str] = Field(None, description="The name of the LLM provider.")
    version: Optional[str] = Field(None, description="The name of the LLM provider.")
    max_retries: int = 3
class LLMResponseModel(BaseModel):
    llm_params: Optional[LLMParams] = Field(None, description="Parameters related to the LLM configuration.")

# Example usage
