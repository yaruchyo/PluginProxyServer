from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Optional, List, Dict, Union, Iterator

class FilesToUpdate(BaseModel):
    filename: Optional[str]
    code_to_update: Optional[str]

class Response(BaseModel):
    answer: str
    files_to_update: Optional[List[FilesToUpdate]]