from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

class Files(BaseModel):
    filename: Optional[str]
    code: Optional[str]

class Response(BaseModel):
    answer: str
    files: Optional[List[Files]]