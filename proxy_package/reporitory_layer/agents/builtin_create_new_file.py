from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import os
import uuid
import time
import uvicorn
from proxy_package.utils.logger import logger

app = FastAPI()

async def stream_response(request_data):
    messages = request_data.get("messages", [])
    if not messages:
        yield f"data: {json.dumps({'error': 'No messages provided'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    last_message = messages[-1]
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = "azure"  # Adjust based on your setup

    if last_message["role"] == "user":
        # User message: Stream a tool call response
        tool_call_id = f"call_{uuid.uuid4().hex}"
        chunks = [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "builtin_create_new_file",
                                        "arguments": '{"filepath":"run.py","contents":"print(\'Hello, World 111!\')"}'
                                    }
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            }
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    elif last_message["role"] == "tool":
        # Tool response: Stream confirmation message
        confirmation_content = "The file `run.py` has been created with the content:\n\n```python run.py\nprint('Hello, World 11111!')\n```"
        chunks = [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": confirmation_content},
                        "finish_reason": None
                    }
                ]
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    else:
        yield f"data: {json.dumps({'error': 'Unexpected message role'})}\n\n"
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
    logger.info(f"Proxy Backend configured via LLM_BACKEND env var.")

    uvicorn.run(
        app="server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        log_config=None,
    )

if __name__ == "__main__":
    start()