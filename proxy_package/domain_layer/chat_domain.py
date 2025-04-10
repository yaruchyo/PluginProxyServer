
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import (  # Added for potential request model
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
)

from ..config import DEFAULT_MODEL_NAME


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_NAME
    messages: List[ChatMessage]
    stream: bool = False
    # Include other potential OpenAI parameters with defaults or Optional
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
