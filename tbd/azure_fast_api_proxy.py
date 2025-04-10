import asyncio
import json
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union

import loguru
import uvicorn

# --- Common Imports ---
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# --- Gemini Imports (Keep for non-streaming or if needed elsewhere) ---
# from google import genai
# from google.genai import types as gemini_types
# from google.generativeai.types import generation_types as gemini_generation_types
# --- Azure OpenAI Imports ---
from openai import (  # Import AzureOpenAI and relevant errors
    APIError,
    AuthenticationError,
    AzureOpenAI,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

logger = loguru.logger
# Load environment variables
load_dotenv()

# --- Gemini Configuration (Commented out or removed if fully switching) ---
# API_KEY = os.environ.get("GOOGLE_API_KEY", None)
# GEMINI_MODEL_NAME = os.environ.get("GOOGLE_MODEL", 'gemini-1.5-pro-latest')

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01") # Example version
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_MODEL") # Your deployment name
AZURE_OPENAI_MAX_RETRIES = int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", 3))

app = FastAPI(title="fast_api_proxy")

# --- Validate and configure Gemini API (Commented out or removed) ---
# if not API_KEY:
#     print("❌ Error: GOOGLE_API_KEY environment variable not set.")
#     # exit(1) # Don't exit if using Azure primarily
# else:
#     try:
#         # Configure the client as in the first code
#         gemini_client = genai.Client(api_key=API_KEY)
#         print(f"✅ Configured Google Generative AI client with model: {GEMINI_MODEL_NAME}")
#     except Exception as e:
#         print(f"❌ Error configuring Gemini client: {e}")
#         traceback.print_exc()
#         # exit(1)

# --- Validate and configure Azure OpenAI API ---
if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
    print("❌ Error: Missing one or more Azure OpenAI environment variables: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")
    exit(1)

try:
    azure_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        max_retries=AZURE_OPENAI_MAX_RETRIES,
    )
    print(f"✅ Configured Azure OpenAI client:")
    print(f"   Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"   Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    print(f"   API Version: {AZURE_OPENAI_API_VERSION}")
except Exception as e:
    print(f"❌ Error configuring Azure OpenAI client: {e}")
    traceback.print_exc()
    exit(1)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the Pydantic model as specified
class FilesToUpdate(BaseModel):
    filename: Optional[str]
    code_to_update: Optional[str]

class Response(BaseModel):
    answer: str
    files_to_update: Optional[List[FilesToUpdate]]

# --- Gemini Helper Functions (Keep if needed for non-streaming/comparison) ---
# def format_openai_to_gemini(...)
# def format_gemini_to_openai_chat(...)
# def format_gemini_to_openai_completion(...)
# async def handle_non_streaming_request(...) # Needs refactoring for Azure if used

# --- NEW: Async generator for Azure OpenAI streaming responses ---
async def stream_azure_openai_response(
    openai_messages: List[Dict[str, Any]],
    azure_params: Dict[str, Any],
    requested_model: str, # This will be the original requested model name
    azure_deployment_name: str, # Actual Azure deployment to use
    request_id: str,
    is_chat_format: bool = True,
    parse_to_files: bool = False
):
    """Streams responses from Azure OpenAI Chat Completions endpoint."""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    # Keep track of the full response for potential parsing
    full_response_text = ""
    final_finish_reason = None # Store the finish reason from the stream

    def generator_thread():
        try:
            print(f"⚙️ Calling Azure OpenAI Stream with params: {azure_params}")
            # Filter out None values from params, as Azure client might not like them
            filtered_params = {k: v for k, v in azure_params.items() if v is not None}

            stream = azure_client.chat.completions.create(
                model=azure_deployment_name, # Use the deployment name here
                messages=openai_messages,
                stream=True,
                **filtered_params # Pass other params like max_tokens, temperature, etc.
            )
            for chunk in stream:
                # print(f"Raw Azure chunk: {chunk.model_dump_json(indent=2)}") # Debugging
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None) # Signal end
        except APIError as e:
             print(f"❌ Azure API Error in generator thread: {e}")
             traceback.print_exc()
             loop.call_soon_threadsafe(queue.put_nowait, e) # Send exception
        except AuthenticationError as e:
             print(f"❌ Azure Authentication Error in generator thread: {e}")
             traceback.print_exc()
             loop.call_soon_threadsafe(queue.put_nowait, e) # Send exception
        except Exception as e:
            print(f"❌ Error in Azure generator thread: {e}")
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, e) # Send exception

    threading.Thread(target=generator_thread, daemon=True).start()

    try:
        while True:
            item = await queue.get()
            if item is None:
                print("🏁 Azure Stream finished.")
                break # End of stream

            if isinstance(item, Exception):
                print(f"❌ Received exception from Azure generator: {item}")
                error_content = f"[AZURE_ERROR: {item}]"
                finish_reason = "error"
                if is_chat_format:
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model, # Report the model requested by the user
                        "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": finish_reason}],
                        "system_fingerprint": f"azure-{azure_deployment_name}" # Indicate Azure backend
                    }
                else: # Mimic completion format
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0, "text": error_content, "finish_reason": finish_reason}]
                    }
                yield f"data: {json.dumps(payload)}\n\n"
                # Store the error finish reason and stop
                final_finish_reason = finish_reason
                break # Stop processing after error chunk

            # --- Process valid Azure chunk (openai.types.chat.ChatCompletionChunk) ---
            chunk_text = ""
            chunk_finish_reason = None
            try:
                if item.choices:
                    choice = item.choices[0]
                    if choice.delta and choice.delta.content:
                        chunk_text = choice.delta.content
                    if choice.finish_reason:
                        chunk_finish_reason = choice.finish_reason
                        final_finish_reason = chunk_finish_reason # Store the latest non-null finish reason

            except Exception as e:
                 print(f"⚠️ Error processing Azure chunk content: {e}")
                 chunk_text = f"[CHUNK_ERROR: {e}]"

            # Yield payload for this chunk
            if chunk_text or chunk_finish_reason: # Only yield if there's content or it's the final chunk marker
                if is_chat_format:
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": requested_model, # Report the model requested by the user
                        "choices": [{"index": 0,
                                     "delta": {"content": chunk_text} if chunk_text else {}, # Include delta only if content exists
                                     "finish_reason": chunk_finish_reason # Will be None until the last chunk
                                     }],
                        "system_fingerprint": f"azure-{azure_deployment_name}" # Indicate Azure backend
                        # Azure response might include usage in the last chunk, but it's not standard in OpenAI spec for chunks
                    }
                else: # Mimic completion format
                    payload = {
                        "id": request_id, "object": "text_completion", "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{"index": 0,
                                     "text": chunk_text,
                                     "finish_reason": chunk_finish_reason # Will be None until the last chunk
                                     }]
                    }

                if chunk_text: # Accumulate text only if present
                    full_response_text += chunk_text

                # print(f"Yielding Azure: {json.dumps(payload)}") # Debugging
                yield f"data: {json.dumps(payload)}\n\n"

        # --- Stream finished ---
        # If the stream ended normally (not via error chunk above) and parsing is needed, do it now.
        if final_finish_reason != "error" and parse_to_files and full_response_text:
            final_delta_content = ""
            try:
                # Attempt to parse the full response as JSON matching the Response model
                list_files_adapter = TypeAdapter(Response)
                parsed_response = list_files_adapter.validate_json(full_response_text)
                # If successful, structure the output or add metadata
                if parsed_response.files_to_update:
                    parsed_data = [file.model_dump() for file in parsed_response.files_to_update]
                    final_delta_content = f"\n[Parsed {len(parsed_data)} files to update]"
                    print(f"✅ Successfully parsed Azure response into FilesToUpdate model.")
                else:
                    print("ℹ️ Azure response parsed, but no 'files_to_update' found.")

            except (ValidationError, json.JSONDecodeError) as e:
                print(f"⚠️ Failed to parse full Azure response into FilesToUpdate model: {e}")
                final_delta_content = f"\n[WARNING: Failed to parse response JSON - {e}]"

            # If parsing generated content, yield one last chunk (optional, could also just rely on the finish_reason chunk)
            # This deviates slightly from strict OpenAI spec but might be useful for the client.
            # Let's skip this extra chunk and rely on the client seeing the full text + finish reason.
            # The standard way is that the final chunk ONLY contains the finish_reason.
            # If we needed to send parsed data, it would have to be non-standard or handled client-side.
            pass # Parsing happens after stream, no extra chunk needed here based on standard SSE

        # If the stream finished without ever sending a finish_reason (e.g., connection cut), set a default
        if final_finish_reason is None:
            final_finish_reason = 'stop' # Assume normal stop if nothing else indicated

        # Ensure a final chunk with finish_reason is sent if it wasn't already part of the last content chunk
        # (Azure usually sends it in the last chunk, but let's be safe)
        # This check might be redundant if Azure *always* includes finish_reason in the last chunk.
        # We'll rely on the loop above having yielded the final chunk from Azure.

    except asyncio.CancelledError:
         print("🚫 Azure Stream cancelled by client.")
    except Exception as e:
         print(f"❌ Unexpected error during Azure stream processing: {e}")
         traceback.print_exc()
         # Attempt to yield a final error chunk if not already done
         if final_finish_reason != "error":
             error_content = f"[STREAM_PROCESSING_ERROR: {e}]"
             finish_reason = "error"
             if is_chat_format:
                 payload = { "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": finish_reason}], "system_fingerprint": f"azure-{azure_deployment_name}" }
             else:
                 payload = { "id": request_id, "object": "text_completion", "created": int(time.time()), "model": requested_model, "choices": [{"index": 0, "text": error_content, "finish_reason": finish_reason}] }
             yield f"data: {json.dumps(payload)}\n\n"
    finally:
        # Always yield [DONE] for SSE standard compliance
        yield "data: [DONE]\n\n"
        print("🏁 Sent [DONE] for Azure stream")


# --- OLD Gemini Streaming Function (Commented out or remove) ---
# async def stream_gemini_response(...)


# Completions endpoint (OpenAI non-chat style) - MODIFIED FOR AZURE
@app.post("/v1/completions")
async def completions(request: Request):
    request_id = f"cmpl-{uuid.uuid4()}" # Standard prefix for completions
    print(f"\n--- Completion Request Received (Azure Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        print(f"➡️ Completion Request Body:\n{json.dumps(body, indent=2)}")

        requested_model = body.get('model', AZURE_OPENAI_DEPLOYMENT_NAME) # Use deployment as default if no model specified
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None)
        prompt = body.get('prompt')

        if not prompt:
             raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
        if not isinstance(prompt, str):
             raise HTTPException(status_code=400, detail="'prompt' must be a string.")

        # Convert prompt to OpenAI message format
        openai_messages = [{'role': 'user', 'content': prompt}]
        # Handle optional 'suffix' parameter (common in completions) by appending it
        suffix = body.get('suffix')
        if suffix:
            openai_messages[0]['content'] += suffix # Append suffix to the prompt

        # --- Build Azure OpenAI Parameters ---
        azure_params = {}
        if body.get('max_tokens') is not None:
            azure_params['max_tokens'] = body['max_tokens']
        if body.get('temperature') is not None:
            azure_params['temperature'] = body['temperature']
        if body.get('top_p') is not None:
            azure_params['top_p'] = body['top_p']
        # 'n' (number of choices) - Azure default is 1, can be changed but complicates streaming logic here.
        # 'logprobs' is not directly supported in the same way via basic chat completions create.
        # 'echo' needs manual implementation if required.
        # 'presence_penalty', 'frequency_penalty' are supported by Azure OpenAI.
        if body.get('presence_penalty') is not None:
             azure_params['presence_penalty'] = body['presence_penalty']
        if body.get('frequency_penalty') is not None:
             azure_params['frequency_penalty'] = body['frequency_penalty']

        stop_val = body.get('stop')
        if stop_val:
            # Azure expects 'stop' to be string or list of strings (up to 4)
            if isinstance(stop_val, str) or isinstance(stop_val, list):
                azure_params['stop'] = stop_val
            else:
                 print(f"⚠️ Invalid type for 'stop' parameter: {type(stop_val)}. Ignoring.")

        # Handle response_format for JSON mode if needed (Custom 'files_to_update')
        # Note: Standard OpenAI JSON mode ('json_object') isn't typically used with /v1/completions
        parse_to_files = False
        if response_format == "files_to_update":
             print("ℹ️ Custom JSON mode 'files_to_update' requested for /v1/completions.")
             # We can't directly enforce JSON output with a simple prompt easily in Azure ChatCompletion like with Gemini's schema.
             # We rely on the prompt instructing the model and attempt parsing client-side.
             # Add instruction to the prompt?
             openai_messages[0]['content'] += "\n\nPlease format your response as a JSON object matching the following structure: {'answer': '...', 'files_to_update': [{'filename': '...', 'code_to_update': '...'}]}"
             parse_to_files = True # Attempt parsing on client side

        # --- End Build Azure OpenAI Parameters ---

        if should_stream:
            print("🌊 Handling STREAMING completion request via Azure...")
            return StreamingResponse(
                stream_azure_openai_response(
                    openai_messages=openai_messages,
                    azure_params=azure_params,
                    requested_model=requested_model,
                    azure_deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME, # Use configured deployment
                    request_id=request_id,
                    is_chat_format=False, # Use completion format for streaming output
                    parse_to_files=parse_to_files
                ),
                media_type="text/event-stream"
            )
        else:
            print("📄 Handling NON-STREAMING completion request via Azure...")
            # Need a non-streaming Azure handler function (similar to handle_non_streaming_request for Gemini)
            # For now, raise error or implement it:
            # raise HTTPException(status_code=501, detail="Non-streaming /v1/completions via Azure not implemented yet.")
            # --- Basic Non-Streaming Implementation ---
            try:
                print(f"⚙️ Calling Azure OpenAI Non-Stream with params: {azure_params}")
                filtered_params = {k: v for k, v in azure_params.items() if v is not None}
                completion = azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=openai_messages,
                    stream=False,
                    **filtered_params
                )
                print(f"✅ Azure Raw Response (Non-Streaming):\n{completion.model_dump_json(indent=2)}")

                # Format into OpenAI /v1/completions format
                openai_response = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": completion.created,
                    "model": requested_model, # Report requested model
                    "choices": [],
                    "usage": completion.usage.model_dump() if completion.usage else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }

                if completion.choices:
                    choice = completion.choices[0]
                    openai_response["choices"].append({
                        "index": choice.index,
                        "text": choice.message.content if choice.message else "",
                        "logprobs": None, # Not directly available
                        "finish_reason": choice.finish_reason
                    })
                else:
                     openai_response["choices"].append({
                        "index": 0, "text": "", "logprobs": None, "finish_reason": "error" # Or handle appropriately
                    })

                # If JSON mode was requested, attempt parsing here for non-streaming
                if parse_to_files and openai_response.get("choices"):
                    full_text = openai_response["choices"][0].get("text", "")
                    try:
                        list_files_adapter = TypeAdapter(Response)
                        parsed_response_model = list_files_adapter.validate_json(full_text)
                        if parsed_response_model.files_to_update:
                            openai_response["parsed_data"] = [f.model_dump() for f in parsed_response_model.files_to_update]
                            print("✅ Successfully parsed non-streaming Azure response into FilesToUpdate model.")
                        else:
                            print("ℹ️ Non-streaming Azure response parsed, but no 'files_to_update' found.")
                    except (ValidationError, json.JSONDecodeError) as e:
                        print(f"⚠️ Failed to parse non-streaming Azure response into FilesToUpdate model: {e}")
                        openai_response["parsing_error"] = str(e) # Add error info

                return JSONResponse(content=openai_response)

            except APIError as e:
                 print(f"❌ Azure API Error (Non-Streaming): {e}")
                 raise HTTPException(status_code=e.status_code or 500, detail=str(e))
            except AuthenticationError as e:
                 print(f"❌ Azure Authentication Error (Non-Streaming): {e}")
                 raise HTTPException(status_code=401, detail=str(e))
            except Exception as e:
                 print(f"❌ Error in non-streaming Azure request: {e}")
                 traceback.print_exc()
                 raise HTTPException(status_code=500, detail=f"Azure API Error: {e}")
            # --- End Basic Non-Streaming Implementation ---


    except json.JSONDecodeError:
        print("❌ Error decoding JSON.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e: # Re-raise known HTTP exceptions
         raise e
    except Exception as e:
        print(f"❌ Unexpected error in /v1/completions (Azure): {e}")
        traceback.print_exc() # Add traceback for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Chat completions endpoint - MODIFIED FOR AZURE
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = f"chatcmpl-{uuid.uuid4()}" # Standard prefix for chat completions
    print(f"\n--- Chat Request Received (Azure Backend) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ID: {request_id} ---")

    try:
        body = await request.json()
        print(f"➡️ Chat Request Body:\n{json.dumps(body, indent=2)}")

        requested_model = body.get('model', AZURE_OPENAI_DEPLOYMENT_NAME) # Use deployment as default
        should_stream = body.get('stream', False)
        response_format = body.get('response_format', None) # Check for OpenAI v2 style response_format

        openai_messages = body.get('messages', [])
        if not openai_messages:
             raise HTTPException(status_code=400, detail="Missing 'messages' in request body.")

        # --- Build Azure OpenAI Parameters ---
        azure_params = {}
        # Standard Chat Completion Params
        if body.get('max_tokens') is not None:
            azure_params['max_tokens'] = body['max_tokens']
        if body.get('temperature') is not None:
            azure_params['temperature'] = body['temperature']
        if body.get('top_p') is not None:
            azure_params['top_p'] = body['top_p']
        if body.get('presence_penalty') is not None:
             azure_params['presence_penalty'] = body['presence_penalty']
        if body.get('frequency_penalty') is not None:
             azure_params['frequency_penalty'] = body['frequency_penalty']
        # 'n' is supported but complicates streaming logic here.
        # 'logit_bias' is supported.
        if body.get('logit_bias') is not None:
             azure_params['logit_bias'] = body['logit_bias']
        # 'user' field is informational for OpenAI, pass it if needed? Azure might ignore.
        if body.get('user') is not None:
             azure_params['user'] = body['user'] # Pass user identifier

        stop_val = body.get('stop')
        if stop_val:
            if isinstance(stop_val, str) or isinstance(stop_val, list):
                azure_params['stop'] = stop_val
            else:
                 print(f"⚠️ Invalid type for 'stop' parameter: {type(stop_val)}. Ignoring.")

        # Handle JSON mode requests
        parse_to_files = False
        json_mode_requested = False
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
             print("ℹ️ OpenAI JSON mode requested ('json_object').")
             # Add instruction to system prompt or last user message if no system prompt
             json_instruction = "\n\nYou MUST respond in JSON format."
             has_system_prompt = False
             for msg in openai_messages:
                 if msg.get('role') == 'system':
                     msg['content'] += json_instruction
                     has_system_prompt = True
                     break
             if not has_system_prompt:
                 # Find last user message or add a new one? Add to last user message.
                 for msg in reversed(openai_messages):
                     if msg.get('role') == 'user':
                         # Check content type (string or list)
                         if isinstance(msg.get('content'), str):
                              msg['content'] += json_instruction
                         elif isinstance(msg.get('content'), list):
                              # Add as a new text part
                              msg['content'].append({"type": "text", "text": json_instruction})
                         break
                 else: # No user message? This is odd, but handle it.
                      openai_messages.append({'role': 'user', 'content': json_instruction})

             # Azure OpenAI doesn't have a simple 'response_format' param like OpenAI v2 yet (as of common API versions).
             # Rely on prompt instructions for JSON mode.
             json_mode_requested = True # Flag that JSON was requested

        elif response_format == "files_to_update": # Custom format
             print("ℹ️ Custom JSON mode requested ('files_to_update').")
             # Add specific instruction for the custom format
             files_instruction = "\n\nPlease format your response as a JSON object matching the following structure: {'answer': '...', 'files_to_update': [{'filename': '...', 'code_to_update': '...'}]}"
             has_system_prompt = False
             for msg in openai_messages:
                 if msg.get('role') == 'system':
                     msg['content'] += files_instruction
                     has_system_prompt = True
                     break
             if not has_system_prompt:
                 for msg in reversed(openai_messages):
                     if msg.get('role') == 'user':
                         if isinstance(msg.get('content'), str):
                              msg['content'] += files_instruction
                         elif isinstance(msg.get('content'), list):
                              msg['content'].append({"type": "text", "text": files_instruction})
                         break
                 else:
                      openai_messages.append({'role': 'user', 'content': files_instruction})

             parse_to_files = True # Signal client-side parsing attempt
             json_mode_requested = True

        # --- End Build Azure OpenAI Parameters ---

        if should_stream:
            print("🌊 Handling STREAMING chat request via Azure...")
            return StreamingResponse(
                stream_azure_openai_response(
                    openai_messages=openai_messages,
                    azure_params=azure_params,
                    requested_model=requested_model,
                    azure_deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME, # Use configured deployment
                    request_id=request_id,
                    is_chat_format=True, # Use chat format for streaming output
                    parse_to_files=parse_to_files # Only relevant for 'files_to_update'
                ),
                media_type="text/event-stream"
            )
        else:
            print("📄 Handling NON-STREAMING chat request via Azure...")
            # --- Non-Streaming Implementation ---
            try:
                print(f"⚙️ Calling Azure OpenAI Non-Stream with params: {azure_params}")
                filtered_params = {k: v for k, v in azure_params.items() if v is not None}
                completion = azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=openai_messages,
                    stream=False,
                    **filtered_params
                )
                print(f"✅ Azure Raw Response (Non-Streaming):\n{completion.model_dump_json(indent=2)}")

                # Format into OpenAI /v1/chat/completions format
                openai_response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": completion.created,
                    "model": requested_model, # Report requested model
                    "choices": [],
                    "usage": completion.usage.model_dump() if completion.usage else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "system_fingerprint": f"azure-{AZURE_OPENAI_DEPLOYMENT_NAME}" # Indicate Azure backend
                }

                if completion.choices:
                    choice = completion.choices[0]
                    openai_response["choices"].append({
                        "index": choice.index,
                        "message": choice.message.model_dump() if choice.message else {"role": "assistant", "content": None},
                        "finish_reason": choice.finish_reason,
                        "logprobs": None, # Not directly available this way
                    })
                else:
                     openai_response["choices"].append({
                        "index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "error"
                    })

                # If JSON mode was requested ('files_to_update'), attempt parsing here for non-streaming
                if parse_to_files and openai_response.get("choices"):
                    full_text = openai_response["choices"][0].get("message", {}).get("content", "")
                    if full_text: # Only parse if content exists
                        try:
                            list_files_adapter = TypeAdapter(Response)
                            parsed_response_model = list_files_adapter.validate_json(full_text)
                            if parsed_response_model.files_to_update:
                                openai_response["parsed_data"] = [f.model_dump() for f in parsed_response_model.files_to_update]
                                print("✅ Successfully parsed non-streaming Azure response into FilesToUpdate model.")
                            else:
                                print("ℹ️ Non-streaming Azure response parsed, but no 'files_to_update' found.")
                        except (ValidationError, json.JSONDecodeError) as e:
                            print(f"⚠️ Failed to parse non-streaming Azure response into FilesToUpdate model: {e}")
                            openai_response["parsing_error"] = str(e) # Add error info
                    else:
                         print("ℹ️ Non-streaming Azure response has no content to parse for FilesToUpdate.")


                return JSONResponse(content=openai_response)

            except APIError as e:
                 print(f"❌ Azure API Error (Non-Streaming): {e}")
                 raise HTTPException(status_code=e.status_code or 500, detail=str(e))
            except AuthenticationError as e:
                 print(f"❌ Azure Authentication Error (Non-Streaming): {e}")
                 raise HTTPException(status_code=401, detail=str(e))
            except Exception as e:
                 print(f"❌ Error in non-streaming Azure request: {e}")
                 traceback.print_exc()
                 raise HTTPException(status_code=500, detail=f"Azure API Error: {e}")
            # --- End Non-Streaming Implementation ---


    except json.JSONDecodeError:
        print("❌ Error decoding JSON.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except HTTPException as e: # Re-raise known HTTP exceptions
         raise e
    except Exception as e:
        print(f"❌ Unexpected error in /v1/chat/completions (Azure): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

def start() -> None:
    logger.info("Starting FastAPI proxy (Azure Backend)...") # Updated log message
    # Use a reasonable number of workers, not necessarily CPU * 2 + 1 for dev
    workers = int(os.environ.get("WEB_CONCURRENCY", 1)) # Default to 1 for reload=True
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 1234))
    reload = os.environ.get("RELOAD", "true").lower() == "true"

    if workers > 1 and reload:
        print("⚠️ WARNING: Running multiple workers with reload=True is not recommended.")
        workers = 1
        print("ℹ️ Set workers to 1.")

    logger.info(f"Starting Uvicorn server on {host}:{port} with {workers} worker(s)")
    logger.info(f"Reloading enabled: {reload}")

    uvicorn.run(
        app="fast_api_proxy:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None, # Use loguru for logging
        workers=workers,
    )

# Run the server (for manual execution)
if __name__ == "__main__":
    start()