from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import uuid
import time
import uvicorn
from proxy_package.utils.logger import logger

# Pydantic Models

# Specific Argument Models
class CreateFileArguments(BaseModel):
    filepath: str
    contents: str

class RunTerminalCommand(BaseModel):
    filepath: str
    command: str
# Request Models
class RequestMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None # Added for tool response messages
    # Add other potential fields if needed, e.g., tool_calls for user messages

class ChatCompletionRequest(BaseModel):
    messages: List[RequestMessage]
    # Add other potential request fields if needed, e.g., model, stream, temperature

# Response Chunk Models (for streaming)
class FunctionCall(BaseModel):
    name: Optional[str] = None
    # Arguments is expected to be a JSON *string* by the OpenAI API format
    arguments: Optional[str] = None

class ToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[FunctionCall] = None

class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Choice(BaseModel):
    index: Optional[int] = None
    delta: Delta
    finish_reason: Optional[str] = Field(None, alias="finish_reason")

class ChatCompletionChunk(BaseModel):
    id: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: Optional[int]= None
    model: Optional[str] = None
    choices: List[Choice]

class ErrorResponse(BaseModel):
    error: str

# FastAPI App
app = FastAPI()

async def stream_response(request_data: Dict[str, Any]): # Use Dict temporarily
    # Ideally, validate request_data against ChatCompletionRequest model
    # For simplicity here, we'll keep using dict access
    messages = request_data.get("messages", [])
    if not messages:
        error_resp = ErrorResponse(error="No messages provided")
        yield f"data: {error_resp.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    last_message = messages[-1]
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = "azure"  # Adjust based on your setup

    if last_message.get("role") == "user":
        # User message: Stream a tool call response
        tool_call_id = f"call_{uuid.uuid4().hex}"
        # Define arguments using the specific Pydantic model
        file_args = CreateFileArguments(
            filepath="run.py",
            contents="print('hello I love sofia')"
            #builtin_create_new_file
        )
        file_args = RunTerminalCommand(
            filepath="./",
            command="git commit -m 'test'"
        )
        # Serialize arguments to a JSON string for the FunctionCall model
        arguments_json_string = file_args.model_dump_json()

        chunk_data = ChatCompletionChunk(
            choices=[
                Choice(
                    delta=Delta(
                        tool_calls=[
                            ToolCall(
                                type="function",
                                function=FunctionCall(
                                    name="builtin_run_terminal_command",
                                    arguments=arguments_json_string # Pass the JSON string
                                )
                            )
                        ]
                    ),
                    finish_reason=None # Typically finish_reason comes in a later chunk
                )
            ]
        )
        #Example: Send a final chunk with finish_reason="tool_calls"
        # final_chunk_data = ChatCompletionChunk(
        #     id=completion_id,
        #     created=created,
        #     model=model,
        #     choices=[
        #         Choice(
        #             index=0,
        #             delta=Delta(), # Empty delta
        #             finish_reason="tool_calls"
        #         )
        #     ]
        # )
        yield f"data: {chunk_data.model_dump_json(exclude_none=True)}\n\n"
        #yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    elif last_message.get("role") == "tool":
        # Tool response: Stream confirmation message
        # In a real scenario, you'd likely parse the tool_call_id and content
        # from the last_message to generate the confirmation.
        confirmation_content = "The file `run.py` has been created with the content:\n\n```python run.py\nprint('Hello, World 111!')\n```" # Note: Content mismatch with user request example
        chunk_data = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                Choice(
                    delta=Delta(role="assistant", content=confirmation_content), # Role should be assistant here
                    finish_reason=None
                )
            ]
        )
        # Example: Send a final chunk with finish_reason="stop"
        # final_chunk_data = ChatCompletionChunk(
        #     id=completion_id,
        #     created=created,
        #     model=model,
        #     choices=[
        #         Choice(
        #             index=0,
        #             delta=Delta(), # Empty delta
        #             finish_reason="stop"
        #         )
        #     ]
        # )
        #yield f"data: {chunk_data.model_dump_json(exclude_none=True)}\n\n"
        #yield f"data: {final_chunk_data.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    else:
        error_resp = ErrorResponse(error="Unexpected message role")
        yield f"data: {error_resp.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_data = await request.json()
    return StreamingResponse(stream_response(request_data), media_type="text/event-stream")

def start() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 1234))
    reload = os.environ.get("RELOAD", "true").lower() == "true"
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    if workers > 1 and reload:
        logger.warning("⚠️ WARNING: Running multiple workers with reload=True is not recommended. Setting workers to 1.")
        workers = 1

    logger.info(f"Starting Uvicorn server on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}, Log Level: {log_level}")

    # Adjust app string based on how you run the file
    # If run directly: f"{__name__}:app"
    # If part of a package and run via an entrypoint (e.g., main.py): "your_package.your_module:app"
    # Example assuming direct run for testing:
    app_string = f"{__name__}:app"
    # Or if it's meant to be run via a different entrypoint like 'server.py'
    # app_string = "server:app"

    uvicorn.run(
        app=app_string,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        log_config=None,
    )

if __name__ == "__main__":
    start()