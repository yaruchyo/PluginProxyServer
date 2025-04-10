from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class Files(BaseModel):
    filename: Optional[str] = Field(None, description="Provide the name for this file.")
    code: Optional[str] = Field(None, description="Provide the code content for this file.")
class Response(BaseModel):
    answer: str = Field(..., description="Provide the textual answer responding to the user's request.")
    files: Optional[List[Files]] = Field(None, description="Provide a list of files, including their names and Python code content, if required by the user's request.")