import json
import logging
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to type mismatches between streaming and non-streaming response types
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Configure httpx connection limits for LiteLLM
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
litellm.client_session = httpx.Client(limits=httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")

# litellm._turn_on_debug()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

if USE_LANGFUSE:
    litellm.success_callback = ["langfuse"]
else:
    litellm.success_callback = []

litellm.drop_params = True

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse, model: Optional[str] = None) -> float:
    """
    Get the cost of the response from the litellm completion.

    Args:
        response: The ModelResponse from litellm.
        model: The requested model name, with its provider prefix (e.g.
            "vertex_ai/claude-opus-4-5@20251101"). SASY patch: when given, it
            is passed to completion_cost instead of response.model, which some
            providers return without the prefix, so the price lookup fails.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        if model:
            cost = completion_cost(
                completion_response=response, model=_parse_ft_model_name(model)
            )
        else:
            cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def validate_message(message: Message) -> None:
    """
    Validate the message.
    """

    def has_text_content(message: Message) -> bool:
        """
        Check if the message has text content.
        """
        return message.content is not None and bool(message.content.strip())

    def has_content_or_tool_calls(message: ParticipantMessageBase) -> bool:
        """
        Check if the message has content or tool calls.
        """
        return message.has_content() or message.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: list[Message]) -> None:
    """
    Validate the message history.
    """
    for message in messages:
        validate_message(message)


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.

    Args:
        messages: List of litellm message dictionaries

    Returns:
        Modified message list with content split into lines for readability
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            # Split content on newlines for better readability
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    Behavior depends on the current log mode:
    - "all": Saves every LLM call
    - "latest": Only keeps the most recent call of each call_name type

    Args:
        request_data: Dictionary containing request information
        response_data: Dictionary containing response information
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message")
    """
    log_dir = llm_log_dir.get()

    if log_dir is None:
        # No log directory set, skip logging
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get current logging mode
    current_log_mode = llm_log_mode.get()

    # If mode is "latest" and call_name is provided, remove existing files with the same call_name
    if current_log_mode == "latest" and call_name:
        # Find and remove existing files with this call_name
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                # File might have been removed by another thread, ignore
                pass

    # Create a new file for this LLM call
    call_id = str(uuid.uuid4())[:8]  # Use short UUID for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

    # Include call_name in filename if provided
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    # Create complete JSON structure with both request and response
    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Write to file with indentation
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


# --- SASY patch: request deadlines and empty-response handling -------------
# See experiments/tau2-bench/PATCHES.md.
DEFAULT_REQUEST_TIMEOUT_S = 120
DEFAULT_NUM_RETRIES = 1
DEFAULT_EMPTY_RESPONSE_ATTEMPTS = (
    3  # SASY patch: retry empty responses before placeholder
)
EMPTY_RESPONSE_PLACEHOLDER = (
    "(empty model response; continuing — if you intended a tool call, please try again)"
)


def _env_int(name: str, default: int, minimum: int) -> int:
    """Read a positive integer override from the environment."""
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _is_empty_message(message: AssistantMessage) -> bool:
    """True when the message has neither text content nor tool calls."""
    content = message.content
    if content and (not isinstance(content, str) or content.strip()):
        return False
    if message.tool_calls:
        return False
    return True


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    SASY patch: some models occasionally return an assistant turn with
    neither content nor tool calls (for example a thinking-only turn, or
    max_tokens hit mid tool call). Such a message fails message validation
    later in the simulation, which makes the runner retry the whole
    simulation and finally record an infrastructure error. Instead, the
    completion is retried up to TAU2_EMPTY_RESPONSE_ATTEMPTS times (default
    1, i.e. no retry, because empty turns are usually sticky for a given
    conversation state), growing an integer max_tokens by 1.5x per retry.
    If every attempt is empty, the content is replaced by
    EMPTY_RESPONSE_PLACEHOLDER so the conversation can continue. The
    placeholder is not substituted when tool_choice="required": that
    caller accepts only a tool call, so placeholder text would not let
    the conversation continue; the empty message is returned unchanged
    and the caller's own check reports it.

    Each completion also gets a default per-request timeout
    (TAU2_REQUEST_TIMEOUT_S, default 120 s) and a default litellm
    num_retries (TAU2_NUM_RETRIES, default 1) so a hung provider
    connection fails fast instead of stalling the run. Explicit
    timeout= / num_retries= arguments take precedence.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        **kwargs: Additional arguments to pass to the model.

    Returns: The message produced by the model.
    """
    validate_message_history(messages)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = _env_int("TAU2_NUM_RETRIES", DEFAULT_NUM_RETRIES, 1)
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = _env_int(
            "TAU2_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S, 10
        )
    attempts = _env_int(
        "TAU2_EMPTY_RESPONSE_ATTEMPTS", DEFAULT_EMPTY_RESPONSE_ATTEMPTS, 1
    )
    message = None
    for attempt in range(1, attempts + 1):
        message = _generate_once(
            model,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            call_name=call_name,
            **kwargs,
        )
        if not _is_empty_message(message):
            return message
        logger.warning(
            f"Empty assistant response from {model} (attempt {attempt}/{attempts})"
        )
        max_tokens = kwargs.get("max_tokens")
        if isinstance(max_tokens, int):
            kwargs["max_tokens"] = int(max_tokens * 1.5)
    if tool_choice == "required":
        logger.error(
            f"Empty assistant response from {model} after {attempts} attempt(s); "
            "tool_choice='required', so no placeholder is substituted"
        )
        return message
    logger.error(
        f"Empty assistant response from {model} after {attempts} attempt(s); "
        "substituting placeholder content so the simulation can continue"
    )
    message.content = EMPTY_RESPONSE_PLACEHOLDER
    return message


def _generate_once(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Run a single completion. Upstream's generate() body; generate() wraps it.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        **kwargs: Additional arguments to pass to the model.

    Returns: The message produced by the model.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Prepare request data for logging
    formatted_messages = _format_messages_for_logging(litellm_messages)
    request_data = {
        "model": model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time
    cost = get_response_cost(response, model=model)
    usage = get_response_usage(response)

    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in raw_tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
        generation_time_seconds=generation_time_seconds,
    )

    # Log complete LLM call (request + response)
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    # Add timestamp to request data
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage


def extract_json_from_llm_response(response: str) -> str:
    """
    Extract JSON from an LLM response, handling markdown code blocks.
    """
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content between first { and last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    # Return original response as fallback
    return response
