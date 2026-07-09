#!/usr/bin/env python3
"""Codex-auth image generation with free local reference images.

This CLI calls the Codex Responses hosted-tool route through the OpenAI SDK by
default using the local Codex auth snapshot in `~/.codex/auth.json`. The Codex
Image API generation and edit endpoints remain available through
`--transport image-api`. Generated images are saved under
`~/.codex/generated_images_free_reference/` by default.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
from urllib import error, request
from uuid import uuid4

def _codex_home() -> Path | None:
    value = os.environ.get("CODEX_HOME")
    if not value:
        return None
    return Path(value).expanduser()


def _auth_path(auth_json: str | None) -> Path:
    if auth_json:
        auth_path = Path(auth_json).expanduser()
        if auth_path.exists():
            return auth_path
        _die(f"Codex auth file not found: {auth_path}")

    codex_home = _codex_home()
    candidates = []
    if codex_home:
        candidates.append(codex_home / "auth.json")
    candidates.append(Path.home() / ".codex" / "auth.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    _die(f"Codex auth file not found. Checked: {checked}")


def _output_root(auth_json: str | None) -> Path:
    if auth_json:
        return Path(auth_json).expanduser().parent
    codex_home = _codex_home()
    if codex_home:
        return codex_home
    return Path.home() / ".codex"


def _output_dir(auth_json: str | None) -> Path:
    output_dir = _output_root(auth_json) / "generated_images_free_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


CODEX_IMAGE_API_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_IMAGE_GENERATIONS_URL = f"{CODEX_IMAGE_API_BASE_URL}/images/generations"
CODEX_IMAGE_EDITS_URL = f"{CODEX_IMAGE_API_BASE_URL}/images/edits"
CODEX_RESPONSES_URL = f"{CODEX_IMAGE_API_BASE_URL}/responses"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_RESPONSES_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_BETA_HEADER = "responses=2025-06-21"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
HELP_FORMATTER_WIDTH = 120
FINAL_IMAGE_KEYS = {"result", "image", "b64_json"}
IMAGE_PAYLOAD_KEYS = FINAL_IMAGE_KEYS | {"partial_image_b64", "image_url"}
ALLOWED_ACTIONS = {"generate", "edit", "auto"}
ALLOWED_BACKGROUNDS = {"transparent", "opaque", "auto"}
ALLOWED_INPUT_FIDELITIES = {"high", "low"}
ALLOWED_MODERATIONS = {"auto", "low"}
ALLOWED_OUTPUT_FORMATS = {"png", "webp", "jpeg"}
ALLOWED_QUALITIES = {"low", "medium", "high", "auto"}
ALLOWED_TRANSPORTS = {"image-api", "responses", "responses-raw"}
RESPONSES_TRANSPORTS = {"responses", "responses-raw"}
DEPRECATED_TRANSPORTS = {"responses-raw"}
DEFAULT_TRANSPORT = "responses"
UNSUPPORTED_INPUT_FIDELITY_IMAGE_MODELS = {"gpt-image-1-mini"}
GPT_IMAGE_2_PREFIX = "gpt-image-2"
GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_EDGE = 3840
GPT_IMAGE_2_MAX_RATIO = 3.0
_CLI_LOG_FORMATS = {"responses-event", "image-json", "image-jsonl"}
_CLI_LOG_PATH: Path | None = None
_CLI_LOG_FORMAT: str | None = None
_CLI_LOG_MESSAGES: list[dict[str, str]] = []
_ACTIVE_CLI_LOG_HANDLE: Any = None


def _die(message: str) -> None:
    _log_cli_message("error", message)
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _info(message: str) -> None:
    _log_cli_message("info", message)
    print(message)


def _debug(message: str, *, verbose: bool) -> None:
    _log_cli_message("debug", message)
    if verbose:
        print(message)


def _parse_size(size: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", size)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_gpt_image_2_model(model: str | None) -> bool:
    return bool(model and model.startswith(GPT_IMAGE_2_PREFIX))


def _effective_image_model(args: argparse.Namespace) -> str:
    return args.image_model or args.model or DEFAULT_IMAGE_MODEL


def _effective_responses_model(args: argparse.Namespace) -> str:
    return args.model or DEFAULT_RESPONSES_MODEL


def _effective_timeout(args: argparse.Namespace) -> float:
    return args.timeout


def _uses_image_edit_endpoint(args: argparse.Namespace) -> bool:
    return bool(args.reference or args.mask or args.action == "edit")


def _uses_responses_transport(args: argparse.Namespace) -> bool:
    return args.transport in RESPONSES_TRANSPORTS


def _validate_choice(value: str | None, allowed: set[str], option: str) -> None:
    if value is not None and value not in allowed:
        values = ", ".join(sorted(allowed))
        _die(f"{option} must be one of {values}.")


def _validate_gpt_image_2_size(size: str) -> None:
    parsed = _parse_size(size)
    if parsed is None:
        _die("--size must be auto or WIDTHxHEIGHT, for example 1024x1024.")

    width, height = parsed
    max_edge = max(width, height)
    min_edge = min(width, height)
    total_pixels = width * height

    if max_edge > GPT_IMAGE_2_MAX_EDGE:
        _die("gpt-image-2 size maximum edge length must be less than or equal to 3840px.")
    if width % 16 != 0 or height % 16 != 0:
        _die("gpt-image-2 size width and height must be multiples of 16px.")
    if max_edge / min_edge > GPT_IMAGE_2_MAX_RATIO:
        _die("gpt-image-2 size long edge to short edge ratio must not exceed 3:1.")
    if total_pixels < GPT_IMAGE_2_MIN_PIXELS or total_pixels > GPT_IMAGE_2_MAX_PIXELS:
        _die(
            "gpt-image-2 size total pixels must be at least 655,360 and no more than 8,294,400."
        )


def _validate_size(size: str | None, image_model: str | None) -> None:
    if size is None:
        return
    if size == "auto":
        return
    if _is_gpt_image_2_model(image_model):
        _validate_gpt_image_2_size(size)
        return
    if _parse_size(size) is None:
        _die("--size must be auto or WIDTHxHEIGHT, for example 1024x1024.")


def _validate_tool_options(args: argparse.Namespace) -> None:
    _validate_choice(args.action, ALLOWED_ACTIONS, "--action")
    _validate_choice(args.background, ALLOWED_BACKGROUNDS, "--background")
    _validate_choice(args.input_fidelity, ALLOWED_INPUT_FIDELITIES, "--input-fidelity")
    _validate_choice(args.moderation, ALLOWED_MODERATIONS, "--moderation")
    _validate_choice(args.output_format, ALLOWED_OUTPUT_FORMATS, "--output-format")
    _validate_choice(args.quality, ALLOWED_QUALITIES, "--quality")
    _validate_choice(args.transport, ALLOWED_TRANSPORTS, "--transport")
    image_model = _effective_image_model(args) if args.transport == "image-api" else args.image_model
    _validate_size(args.size, image_model)

    if args.timeout <= 0:
        _die("--timeout must be greater than 0 seconds.")

    if args.output_compression is not None:
        if args.output_compression < 0 or args.output_compression > 100:
            _die("--output-compression must be between 0 and 100.")
        if args.output_format not in {"jpeg", "webp"}:
            _die("--output-compression requires --output-format jpeg or --output-format webp.")

    if args.partial_images is not None and (args.partial_images < 0 or args.partial_images > 3):
        _die("--partial-images must be between 0 and 3.")

    if args.background == "transparent":
        if args.output_format not in {"png", "webp"}:
            _die("--background transparent requires --output-format png or --output-format webp.")
        if _uses_responses_transport(args) and not args.image_model:
            _die("--background transparent requires an explicit --image-model with a Responses transport.")
        if _is_gpt_image_2_model(image_model):
            _die("transparent backgrounds are not supported by gpt-image-2 image models.")

    if args.mask and not args.reference:
        _die("--mask requires at least one --reference image; the first reference is the edit target.")

    if args.input_fidelity:
        if _uses_responses_transport(args) and not args.image_model:
            _die("--input-fidelity requires an explicit --image-model with a Responses transport.")
        if _is_gpt_image_2_model(image_model):
            _die("gpt-image-2 always uses high-fidelity image inputs; omit --input-fidelity.")
        if image_model in UNSUPPORTED_INPUT_FIDELITY_IMAGE_MODELS:
            _die(f"--input-fidelity is not supported by {image_model}.")

    if args.transport == "image-api":
        if args.instructions:
            _die("--instructions is only supported with --transport responses.")
        if args.action == "edit" and not args.reference:
            _die("--action edit requires at least one --reference image with --transport image-api.")


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt and prompt_file:
        _die("Use --prompt or --prompt-file, not both.")
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            _die(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
    elif prompt:
        text = prompt.strip()
    else:
        _die("Missing prompt. Use --prompt or --prompt-file.")
    if not text:
        _die("Prompt is empty.")
    return text


def _read_codex_auth(auth_json: str | None) -> tuple[str, str | None]:
    auth_path = _auth_path(auth_json)
    if not auth_path.exists():
        _die(f"Codex auth file not found: {auth_path}")
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"Could not parse Codex auth file: {exc}")
    tokens = auth.get("tokens") or {}
    token = tokens.get("access_token")
    if not token:
        _die("Codex access_token not found in auth.json.")
    account_id = auth.get("last_active_account_id") or tokens.get("account_id")
    return token, account_id


def _data_url(path: Path) -> str:
    if not path.exists():
        _die(f"Reference image not found: {path}")
    if not path.is_file():
        _die(f"Reference image is not a file: {path}")
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        _die(f"Reference file is not recognized as an image: {path}")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _slugify(value: str) -> str:
    keep = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
            last_dash = False
        elif not last_dash:
            keep.append("-")
            last_dash = True
    slug = "".join(keep).strip("-")
    return slug[:72] or "image"


def _output_path(name: str | None, output_format: str, auth_json: str | None) -> Path:
    suffix = output_format.lower().lstrip(".") or DEFAULT_OUTPUT_FORMAT
    stem = f"{uuid4()}-{_slugify(name or 'image')}"
    return _output_dir(auth_json) / f"{stem}.{suffix}"


def _log_path(final_path: Path) -> Path:
    return final_path.with_suffix(f"{final_path.suffix}.log")


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log_record_with_timestamp(value: Any) -> Any:
    plain = _to_plain_data(value)
    if isinstance(plain, dict):
        return {**plain, "logged_at": _utc_timestamp()}
    return {"value": plain, "logged_at": _utc_timestamp()}


def _configure_cli_log(
    log_path: Path | None,
    log_format: str | None,
    *,
    reset_messages: bool = False,
) -> None:
    global _CLI_LOG_PATH, _CLI_LOG_FORMAT, _CLI_LOG_MESSAGES
    if log_format is not None and log_format not in _CLI_LOG_FORMATS:
        raise ValueError(f"Unknown CLI log format: {log_format}")
    _CLI_LOG_PATH = log_path
    _CLI_LOG_FORMAT = log_format
    if reset_messages:
        _CLI_LOG_MESSAGES = []


def _set_active_cli_log_handle(log_handle: Any) -> None:
    global _ACTIVE_CLI_LOG_HANDLE
    _ACTIVE_CLI_LOG_HANDLE = log_handle


def _cli_log_message_record(level: str, message: str) -> dict[str, str]:
    return {
        "logged_at": _utc_timestamp(),
        "level": level,
        "message": message,
    }


def _append_image_json_cli_message(log_path: Path, record: dict[str, str]) -> None:
    try:
        current = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    except Exception:
        current = {}
    if not isinstance(current, dict):
        current = {}
    messages = current.get("messages")
    if not isinstance(messages, list):
        messages = []
    messages.append(_redact_image_api_log(record))
    current["messages"] = messages
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log_cli_message(level: str, message: str) -> None:
    record = _cli_log_message_record(level, message)
    _CLI_LOG_MESSAGES.append(record)
    if not _CLI_LOG_PATH or not _CLI_LOG_FORMAT:
        return
    try:
        _CLI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _CLI_LOG_FORMAT == "responses-event":
            if _ACTIVE_CLI_LOG_HANDLE is not None:
                _write_responses_log_event(
                    _ACTIVE_CLI_LOG_HANDLE,
                    f"codex_image_gen.{level}",
                    {"level": level, "message": message},
                )
                _ACTIVE_CLI_LOG_HANDLE.flush()
            else:
                with _CLI_LOG_PATH.open("a", encoding="utf-8") as log_handle:
                    _write_responses_log_event(
                        log_handle,
                        f"codex_image_gen.{level}",
                        {"level": level, "message": message},
                    )
        elif _CLI_LOG_FORMAT == "image-jsonl":
            if _ACTIVE_CLI_LOG_HANDLE is not None:
                _write_image_api_stream_log_item(
                    _ACTIVE_CLI_LOG_HANDLE,
                    {"log": {"level": level, "message": message}},
                )
                _ACTIVE_CLI_LOG_HANDLE.flush()
            else:
                with _CLI_LOG_PATH.open("a", encoding="utf-8") as log_handle:
                    _write_image_api_stream_log_item(
                        log_handle,
                        {"log": {"level": level, "message": message}},
                    )
        elif _CLI_LOG_FORMAT == "image-json":
            _append_image_json_cli_message(_CLI_LOG_PATH, record)
    except Exception:
        return


def _start_info(
    *,
    endpoint: str,
    transport: str,
    final_path: Path,
    request_payload: dict[str, Any],
    timeout_seconds: float,
    client: str | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "codex_image_gen.start",
        "started_at": _utc_timestamp(),
        "endpoint": endpoint,
        "transport": transport,
        "output": str(final_path),
        "timeout_seconds": timeout_seconds,
        "request": request_payload,
    }
    if client:
        info["client"] = client
    return info


def _partial_output_path(final_path: Path, index: int | str) -> Path:
    return final_path.with_name(f"{final_path.stem}-partial-{index}{final_path.suffix}")


def _copy_result(source: Path, destination: str, *, force: bool) -> Path:
    target = Path(destination)
    if target.exists() and target.is_dir():
        target = target / source.name
    if target.exists() and not force:
        _die(f"Copy target already exists: {target} (use --force to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _looks_like_image_base64(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 1000 and not value.startswith("data:")


def _scan_final_image_base64(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "partial_image_b64":
                continue
            if key in FINAL_IMAGE_KEYS and _looks_like_image_base64(nested):
                return nested
            found = _scan_final_image_base64(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _scan_final_image_base64(nested)
            if found:
                return found
    return None


def _partial_index(item: dict[str, Any], fallback: int) -> int | str:
    value = item.get("partial_image_index")
    if isinstance(value, int) and 0 <= value <= 3:
        return value
    return fallback


def _write_partial_image(
    item: dict[str, Any],
    final_path: Path,
    fallback_index: int,
    *,
    verbose: bool,
) -> tuple[Path, bytes] | None:
    image_b64 = item.get("partial_image_b64")
    if not _looks_like_image_base64(image_b64):
        return None
    image_bytes = base64.b64decode(image_b64)
    partial_path = _partial_output_path(final_path, _partial_index(item, fallback_index))
    partial_path.write_bytes(image_bytes)
    _info(f"Wrote partial {partial_path}")
    return partial_path, image_bytes


def _final_image_bytes(
    image_b64: str,
    last_partial: tuple[Path, bytes] | None,
    final_path: Path,
    *,
    verbose: bool,
) -> tuple[bytes, bool]:
    image_bytes = base64.b64decode(image_b64)
    if last_partial:
        partial_path, partial_bytes = last_partial
        if partial_bytes == image_bytes:
            partial_path.replace(final_path)
            _debug(
                f"Renamed final partial {partial_path} to {final_path}",
                verbose=verbose,
            )
            return image_bytes, True
    return image_bytes, False


def _build_image_tool(args: argparse.Namespace) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "image_generation", "output_format": args.output_format}
    optional_fields = {
        "action": args.action,
        "background": args.background,
        "input_fidelity": args.input_fidelity,
        "model": args.image_model,
        "moderation": args.moderation,
        "output_compression": args.output_compression,
        "partial_images": args.partial_images,
        "quality": args.quality,
        "size": args.size,
    }
    for key, value in optional_fields.items():
        if value is not None:
            tool[key] = value
    if args.mask:
        tool["input_image_mask"] = {"image_url": _data_url(Path(args.mask))}
    return tool


def _build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    content: list[dict[str, str]] = []
    for reference in args.reference:
        content.append({"type": "input_image", "image_url": _data_url(Path(reference))})
    content.append({"type": "input_text", "text": prompt})

    payload: dict[str, Any] = {
        "model": _effective_responses_model(args),
        "instructions": args.instructions or "",
        "input": [{"role": "user", "content": content}],
        "tools": [_build_image_tool(args)],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": True,
        "store": False,
        "include": [],
    }
    if args.reasoning_effort is not None:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def _build_image_api_options(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": _effective_image_model(args),
        "prompt": prompt,
        "output_format": args.output_format,
        "response_format": "b64_json",
    }
    optional_fields = {
        "background": args.background,
        "input_fidelity": args.input_fidelity if _uses_image_edit_endpoint(args) else None,
        "moderation": args.moderation,
        "output_compression": args.output_compression,
        "partial_images": args.partial_images,
        "quality": args.quality,
        "size": args.size,
    }
    for key, value in optional_fields.items():
        if value is not None:
            options[key] = value
    if args.partial_images is not None:
        options["stream"] = True
    return options


def _redact_image_api_preview(args: argparse.Namespace, options: dict[str, Any]) -> dict[str, Any]:
    preview = dict(options)
    if _uses_image_edit_endpoint(args):
        preview["images"] = [str(Path(path)) for path in args.reference]
        if args.mask:
            preview["mask"] = str(Path(args.mask))
    return preview


def _build_image_api_edit_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload = _build_image_api_options(args, prompt)
    payload["images"] = [{"image_url": _data_url(Path(reference))} for reference in args.reference]
    if args.mask:
        payload["mask"] = {"image_url": _data_url(Path(args.mask))}
    return payload


def _redact_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preview = dict(payload)
    preview["input"] = [
        {
            "role": "user",
            "content": [
                {
                    **item,
                    "image_url": "data:<redacted>" if item.get("type") == "input_image" else item.get("image_url"),
                }
                if item.get("type") == "input_image"
                else item
                for item in payload["input"][0]["content"]
            ],
        }
    ]
    preview_tools = []
    for tool in payload.get("tools", []):
        redacted_tool = dict(tool)
        if "input_image_mask" in redacted_tool:
            redacted_tool["input_image_mask"] = {"image_url": "data:<redacted>"}
        preview_tools.append(redacted_tool)
    preview["tools"] = preview_tools
    return preview


def _redact_responses_stream_item(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, nested in value.items():
            if key in IMAGE_PAYLOAD_KEYS and isinstance(nested, str):
                redacted[key] = f"<redacted {len(nested)} chars>"
            else:
                redacted[key] = _redact_responses_stream_item(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_responses_stream_item(item) for item in value]
    return value


def _compact_json(value: Any, *, limit: int = 2000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _format_stream_failure(
    message: str,
    *,
    log_path: Path | None = None,
    last_item: Any = None,
    output_item_done: Any = None,
    show_response_details: bool = True,
) -> str:
    details = [message]
    if log_path:
        details.append(f"Log: {log_path}")
    if show_response_details and output_item_done is not None:
        redacted = _redact_responses_stream_item(_to_plain_data(output_item_done))
        details.append(f"Output item done: {_compact_json(redacted)}")
    if show_response_details and last_item is not None:
        redacted = _redact_responses_stream_item(_to_plain_data(last_item))
        details.append(f"Last event: {_compact_json(redacted)}")
    return " ".join(details)


def _write_responses_failure_log(log_path: Path | None, event_type: str, details: dict[str, Any]) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        _write_responses_log_event(log_handle, event_type, details)


def _sdk_exception_details(exc: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {"error": str(exc)}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code
    request_id = getattr(exc, "request_id", None)
    if request_id:
        details["request_id"] = request_id
    body = getattr(exc, "body", None)
    if body is not None:
        details["body"] = _to_plain_data(body)
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status is not None and "status_code" not in details:
            details["status_code"] = response_status
        response_text = getattr(response, "text", None)
        if response_text:
            details["response_text"] = response_text[:4000]
    return details


def _responses_event_type(item: Any) -> str:
    if isinstance(item, dict):
        event_type = item.get("type")
        if isinstance(event_type, str) and event_type:
            return event_type
    return "response.sdk_event"


def _write_responses_log_event(log_handle: Any, event_type: str, data: Any) -> None:
    redacted = _redact_responses_stream_item(_to_plain_data(data))
    log_handle.write(f"event: {event_type}\n")
    log_handle.write(f"logged_at: {_utc_timestamp()}\n")
    log_handle.write("data: " + json.dumps(redacted, ensure_ascii=False) + "\n\n")


def _write_responses_sdk_log_item(log_handle: Any, item: Any) -> None:
    plain = _to_plain_data(item)
    _write_responses_log_event(log_handle, _responses_event_type(plain), plain)


def _stream_responses_raw(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path | None,
    final_path: Path,
    *,
    timeout_seconds: float,
    save_partials: bool,
    verbose: bool,
    show_response_details: bool,
) -> tuple[bytes, bool]:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            _write_responses_log_event(
                log_handle,
                "codex_image_gen.start",
                _start_info(
                    endpoint=CODEX_RESPONSES_URL,
                    transport="responses-raw",
                    final_path=final_path,
                    request_payload=payload,
                    timeout_seconds=timeout_seconds,
                ),
            )
    _info("--transport responses-raw is deprecated; use --transport responses.")

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": DEFAULT_BETA_HEADER,
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    req = request.Request(
        CODEX_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        response = request.urlopen(req, timeout=timeout_seconds)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        details = {"status": exc.code, "body": body[:2000], "endpoint": CODEX_RESPONSES_URL}
        _write_responses_failure_log(log_path, "response.request_failed", details)
        _die(
            _format_stream_failure(
                f"Codex Responses request failed with HTTP {exc.code}: {body[:2000]}",
                log_path=log_path,
                last_item=details,
                show_response_details=show_response_details,
            )
        )
    except error.URLError as exc:
        details = {"error": str(exc), "endpoint": CODEX_RESPONSES_URL}
        _write_responses_failure_log(log_path, "response.request_failed", details)
        _die(
            _format_stream_failure(
                f"Codex Responses request failed: {exc}",
                log_path=log_path,
                last_item=details,
                show_response_details=show_response_details,
            )
        )

    log_handle = None
    partial_count = 0
    last_partial: tuple[Path, bytes] | None = None
    last_item: Any = None
    last_output_item_done: Any = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            _set_active_cli_log_handle(log_handle)
        for raw in response:
            line = raw.decode("utf-8", "ignore").rstrip("\n")
            if not line.startswith("data:"):
                if log_handle and line:
                    _write_responses_log_event(
                        log_handle,
                        "response.raw_line",
                        {"line": line},
                    )
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                item = json.loads(data)
            except json.JSONDecodeError:
                if log_handle:
                    _write_responses_log_event(
                        log_handle,
                        "response.raw_data",
                        {"data": data},
                    )
                continue
            last_item = item
            if log_handle:
                _write_responses_log_event(log_handle, _responses_event_type(item), item)
            if isinstance(item, dict) and item.get("type") == "response.output_item.done":
                last_output_item_done = item
            if (
                save_partials
                and isinstance(item, dict)
                and item.get("type") == "response.image_generation_call.partial_image"
            ):
                partial_count += 1
                written_partial = _write_partial_image(
                    item,
                    final_path,
                    partial_count,
                    verbose=verbose,
                )
                if written_partial:
                    last_partial = written_partial
                continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                return _final_image_bytes(image_b64, last_partial, final_path, verbose=verbose)
    finally:
        _set_active_cli_log_handle(None)
        if log_handle:
            log_handle.close()
        response.close()
    _die(
        _format_stream_failure(
            "No generated image was found in the streamed response.",
            log_path=log_path,
            last_item=last_item,
            output_item_done=last_output_item_done,
            show_response_details=show_response_details,
        )
    )


def _create_codex_openai_client(token: str, account_id: str | None, timeout_seconds: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        _die("The openai package is not installed in the skill virtual environment.")

    headers = {}
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return OpenAI(
        api_key=token,
        base_url=CODEX_IMAGE_API_BASE_URL,
        default_headers=headers,
        timeout=timeout_seconds,
    )


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _stream_responses_sdk(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path | None,
    final_path: Path,
    *,
    timeout_seconds: float,
    save_partials: bool,
    verbose: bool,
    show_response_details: bool,
) -> tuple[bytes, bool]:
    client = _create_codex_openai_client(token, account_id, timeout_seconds)
    log_handle = None
    stream = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        _set_active_cli_log_handle(log_handle)
        _write_responses_log_event(
            log_handle,
            "codex_image_gen.start",
            _start_info(
                endpoint=CODEX_RESPONSES_URL,
                transport="responses",
                final_path=final_path,
                request_payload=payload,
                timeout_seconds=timeout_seconds,
                client="openai-sdk",
            ),
        )
        log_handle.flush()
    try:
        stream = client.responses.create(
            **payload,
            extra_headers={"OpenAI-Beta": DEFAULT_BETA_HEADER},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        details = {
            **_sdk_exception_details(exc),
            "endpoint": CODEX_RESPONSES_URL,
            "request": payload,
            "transport": "responses-sdk",
        }
        if log_handle:
            _write_responses_log_event(log_handle, "response.request_failed", details)
            log_handle.close()
            _set_active_cli_log_handle(None)
            log_handle = None
        else:
            _write_responses_failure_log(log_path, "response.request_failed", details)
        _die(
            _format_stream_failure(
                f"Codex Responses SDK request failed: {exc}",
                log_path=log_path,
                last_item=details,
                show_response_details=show_response_details,
            )
        )

    partial_count = 0
    last_partial: tuple[Path, bytes] | None = None
    last_item: Any = None
    last_output_item_done: Any = None
    try:
        for event in stream:
            item = _to_plain_data(event)
            last_item = item
            if log_handle:
                _write_responses_sdk_log_item(log_handle, item)
            if isinstance(item, dict) and item.get("type") == "response.output_item.done":
                last_output_item_done = item
            if (
                save_partials
                and isinstance(item, dict)
                and item.get("type") == "response.image_generation_call.partial_image"
            ):
                partial_count += 1
                written_partial = _write_partial_image(
                    item,
                    final_path,
                    partial_count,
                    verbose=verbose,
                )
                if written_partial:
                    last_partial = written_partial
                continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                return _final_image_bytes(image_b64, last_partial, final_path, verbose=verbose)
    except Exception as exc:
        details = {
            **_sdk_exception_details(exc),
            "endpoint": CODEX_RESPONSES_URL,
            "transport": "responses-sdk",
        }
        if log_handle:
            _write_responses_sdk_log_item(log_handle, {"type": "response.stream_error", **details})
            log_handle.close()
            _set_active_cli_log_handle(None)
            log_handle = None
        _die(
            _format_stream_failure(
                f"Codex Responses SDK stream failed: {exc}",
                log_path=log_path,
                last_item=last_item or details,
                output_item_done=last_output_item_done,
                show_response_details=show_response_details,
            )
        )
    finally:
        _set_active_cli_log_handle(None)
        if log_handle:
            log_handle.close()
        close = getattr(stream, "close", None) if stream is not None else None
        if close:
            close()
    _die(
        _format_stream_failure(
            "No generated image was found in the streamed response.",
            log_path=log_path,
            last_item=last_item,
            output_item_done=last_output_item_done,
            show_response_details=show_response_details,
        )
    )


def _image_response_bytes(
    response: Any,
    *,
    log_path: Path | None = None,
    show_response_details: bool = True,
) -> bytes:
    data = getattr(response, "data", None)
    if not data and isinstance(response, dict):
        data = response.get("data")
    if not data:
        _die(
            _format_stream_failure(
                "No image data was returned by the Codex Image API.",
                log_path=log_path,
                last_item=response,
                show_response_details=show_response_details,
            )
        )

    first = data[0]
    if isinstance(first, dict):
        image_b64 = first.get("b64_json")
    else:
        image_b64 = getattr(first, "b64_json", None)
    if not _looks_like_image_base64(image_b64):
        _die(
            _format_stream_failure(
                "Codex Image API response did not include b64_json image data.",
                log_path=log_path,
                last_item=first,
                show_response_details=show_response_details,
            )
        )
    return base64.b64decode(image_b64)


def _redact_image_api_log(value: Any) -> Any:
    plain = _to_plain_data(value)
    if isinstance(plain, dict):
        redacted = {}
        for key, nested in plain.items():
            if key in IMAGE_PAYLOAD_KEYS and isinstance(nested, str):
                redacted[key] = f"<redacted {len(nested)} chars>"
            else:
                redacted[key] = _redact_image_api_log(nested)
        return redacted
    if isinstance(plain, list):
        return [_redact_image_api_log(item) for item in plain]
    return plain


def _image_api_log_record(
    payload: dict[str, Any],
    response: Any,
    *,
    start_info: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"logged_at": _utc_timestamp()}
    if start_info is not None:
        record["start"] = _redact_image_api_log(start_info)
    if status is not None:
        record["status"] = status
    record["request"] = _redact_image_api_log(payload)
    record["response"] = _redact_image_api_log(response)
    if _CLI_LOG_MESSAGES:
        record["messages"] = _redact_image_api_log(_CLI_LOG_MESSAGES)
    return record


def _write_image_api_log(
    log_path: Path,
    payload: dict[str, Any],
    response: Any,
    *,
    start_info: dict[str, Any] | None = None,
    status: str | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            _image_api_log_record(
                payload,
                response,
                start_info=start_info,
                status=status,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_image_api_failure_log(
    log_path: Path,
    payload: dict[str, Any],
    *,
    start_info: dict[str, Any] | None = None,
    status_code: int | None = None,
    response_text: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    if status_code is not None:
        response["status_code"] = status_code
    if response_text is not None:
        response["text"] = response_text[:4000]
    if error_message is not None:
        response["error"] = error_message
    _write_image_api_log(log_path, payload, response, start_info=start_info, status="failed")
    return response


def _write_image_api_start_log(log_path: Path, payload: dict[str, Any], start_info: dict[str, Any]) -> None:
    _write_image_api_log(log_path, payload, None, start_info=start_info, status="started")


def _write_image_api_stream_start_log(log_path: Path, start_info: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        _write_image_api_stream_log_item(log_handle, {"start": start_info})


def _write_image_api_stream_log_item(log_handle: Any, item: Any) -> None:
    redacted = _redact_image_api_log(_log_record_with_timestamp(item))
    log_handle.write(json.dumps(redacted, ensure_ascii=False) + "\n")


def _stream_image_api_response(
    stream: Any,
    final_path: Path,
    log_path: Path,
    request_options: dict[str, Any],
    *,
    save_partials: bool,
    verbose: bool,
    show_response_details: bool,
) -> bytes:
    partial_count = 0
    last_item: Any = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        _set_active_cli_log_handle(log_handle)
        try:
            _write_image_api_stream_log_item(log_handle, {"request": request_options})
            for event in stream:
                item = _to_plain_data(event)
                last_item = item
                _write_image_api_stream_log_item(log_handle, {"event": item})
                if save_partials and isinstance(item, dict):
                    partial_b64 = item.get("partial_image_b64")
                    if not partial_b64:
                        partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
                    if _looks_like_image_base64(partial_b64):
                        partial_count += 1
                        partial_path = _partial_output_path(final_path, partial_count)
                        partial_path.write_bytes(base64.b64decode(partial_b64))
                        _info(f"Wrote partial {partial_path}")
                        continue
                image_b64 = _scan_final_image_base64(item)
                if image_b64:
                    return base64.b64decode(image_b64)
        finally:
            _set_active_cli_log_handle(None)
    _die(
        _format_stream_failure(
            "No generated image was found in the streamed Codex Image API response.",
            log_path=log_path,
            last_item=last_item,
            show_response_details=show_response_details,
        )
    )


def _stream_json_image_api_response(
    response: Any,
    final_path: Path,
    log_path: Path,
    request_options: dict[str, Any],
    *,
    save_partials: bool,
    verbose: bool,
    show_response_details: bool,
) -> bytes:
    partial_count = 0
    last_item: Any = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    _set_active_cli_log_handle(log_handle)
    try:
        _write_image_api_stream_log_item(log_handle, {"request": request_options})
        for raw in response:
            line = raw.decode("utf-8", "ignore").rstrip("\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                item = json.loads(data)
            except json.JSONDecodeError:
                continue
            last_item = item
            _write_image_api_stream_log_item(log_handle, {"event": item})
            if save_partials:
                partial_b64 = item.get("partial_image_b64") if isinstance(item, dict) else None
                if not partial_b64 and isinstance(item, dict):
                    partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
                if _looks_like_image_base64(partial_b64):
                    partial_count += 1
                    partial_path = _partial_output_path(final_path, partial_count)
                    partial_path.write_bytes(base64.b64decode(partial_b64))
                    _info(f"Wrote partial {partial_path}")
                    continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                return base64.b64decode(image_b64)
    finally:
        _set_active_cli_log_handle(None)
        log_handle.close()
        response.close()
    _die(
        _format_stream_failure(
            "No generated image was found in the streamed Codex Image API response.",
            log_path=log_path,
            last_item=last_item,
            show_response_details=show_response_details,
        )
    )


def _request_image_api_edit(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path,
    *,
    timeout_seconds: float,
    start_info: dict[str, Any],
    show_response_details: bool,
) -> bytes:
    if payload.get("stream"):
        _die("--partial-images with --transport image-api edit is not supported; use --transport responses.")
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "User-Agent": "codex-imagegen-free-reference",
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    try:
        import httpx

        response = httpx.post(
            CODEX_IMAGE_EDITS_URL,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        if response.status_code >= 400:
            failure = _write_image_api_failure_log(
                log_path,
                payload,
                start_info=start_info,
                status_code=response.status_code,
                response_text=response.text,
            )
            _die(
                _format_stream_failure(
                    f"Codex Image API edit request failed with HTTP {response.status_code}: "
                    f"{response.text[:2000]}",
                    log_path=log_path,
                    last_item=failure,
                    show_response_details=show_response_details,
                )
            )
        try:
            data = response.json()
        except Exception as exc:
            failure = _write_image_api_failure_log(
                log_path,
                payload,
                start_info=start_info,
                status_code=response.status_code,
                response_text=response.text,
                error_message=f"Could not parse JSON response: {exc}",
            )
            _die(
                _format_stream_failure(
                    f"Codex Image API edit response was not valid JSON: {exc}",
                    log_path=log_path,
                    last_item=failure,
                    show_response_details=show_response_details,
                )
            )
        _write_image_api_log(log_path, payload, data, start_info=start_info, status="completed")
    except Exception as exc:
        failure = _write_image_api_failure_log(
            log_path,
            payload,
            start_info=start_info,
            error_message=str(exc),
        )
        _die(
            _format_stream_failure(
                f"Codex Image API edit request failed: {exc}",
                log_path=log_path,
                last_item=failure,
                show_response_details=show_response_details,
            )
        )
    return _image_response_bytes(
        data,
        log_path=log_path,
        show_response_details=show_response_details,
    )


def _run_image_api(
    args: argparse.Namespace,
    prompt: str,
    token: str,
    account_id: str | None,
    final_path: Path,
    log_path: Path,
) -> bytes:
    timeout_seconds = _effective_timeout(args)
    if _uses_image_edit_endpoint(args):
        payload = _build_image_api_edit_payload(args, prompt)
        start_info = _start_info(
            endpoint=CODEX_IMAGE_EDITS_URL,
            transport="image-api",
            final_path=final_path,
            request_payload=payload,
            timeout_seconds=timeout_seconds,
            client="httpx-json",
        )
        _write_image_api_start_log(log_path, payload, start_info)
        return _request_image_api_edit(
            payload,
            token,
            account_id,
            log_path,
            timeout_seconds=timeout_seconds,
            start_info=start_info,
            show_response_details=not args.hide_response_details,
        )

    client = _create_codex_openai_client(token, account_id, timeout_seconds)
    options = _build_image_api_options(args, prompt)
    start_info = _start_info(
        endpoint=CODEX_IMAGE_GENERATIONS_URL,
        transport="image-api",
        final_path=final_path,
        request_payload=options,
        timeout_seconds=timeout_seconds,
        client="openai-sdk",
    )
    if options.get("stream"):
        _configure_cli_log(log_path, "image-jsonl")
        _write_image_api_stream_start_log(log_path, start_info)
    else:
        _configure_cli_log(log_path, "image-json")
        _write_image_api_start_log(log_path, options, start_info)
    try:
        response = client.images.generate(**options, timeout=timeout_seconds)
    except Exception as exc:
        failure = _write_image_api_failure_log(
            log_path,
            options,
            start_info=start_info,
            error_message=str(exc),
        )
        _die(
            _format_stream_failure(
                f"Codex Image API request failed: {exc}",
                log_path=log_path,
                last_item=failure,
                show_response_details=not args.hide_response_details,
            )
        )

    if options.get("stream"):
        return _stream_image_api_response(
            response,
            final_path,
            log_path,
            options,
            save_partials=bool(args.partial_images),
            verbose=args.verbose,
            show_response_details=not args.hide_response_details,
        )
    _write_image_api_log(log_path, options, response, start_info=start_info, status="completed")
    return _image_response_bytes(
        response,
        log_path=log_path,
        show_response_details=not args.hide_response_details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate images through Codex auth with optional local reference images.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=HELP_FORMATTER_WIDTH),
    )
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--reference", action="append", default=[], help="Local reference image path; repeat to attach multiple images.")
    parser.add_argument("--name", help="Human-readable filename suffix used after the UUID.")
    parser.add_argument("--copy-to", help="Optional project-local file or directory to receive a copied result.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the --copy-to target.")
    parser.add_argument(
        "--transport",
        choices=sorted(ALLOWED_TRANSPORTS),
        default=DEFAULT_TRANSPORT,
        help=(
            "Request path to use. responses (default) calls Codex /responses through "
            "the OpenAI SDK with the hosted image_generation tool; responses-raw is "
            "the deprecated raw SSE fallback; image-api calls Codex /images/generations "
            "or /images/edits."
        ),
    )
    parser.add_argument("--model", help="Model for the selected transport.")
    parser.add_argument(
        "--reasoning-effort",
        help=(
            "Optional Responses reasoning.effort value. Omit this option to use the "
            "model/server default. The default Responses model gpt-5.5 supports "
            "none, low, medium (default), high, and xhigh; other models can differ, "
            "and additional values may be supported. See "
            "https://developers.openai.com/api/docs/guides/reasoning."
        ),
    )
    parser.add_argument("--image-model", help="Optional GPT Image model; for image-api this overrides --model.")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--size", help="Optional image size, such as auto, 1024x1024, or 2048x1152.")
    parser.add_argument("--quality", help="Optional rendering quality: low, medium, high, or auto.")
    parser.add_argument("--background", help="Optional background behavior: transparent, opaque, or auto.")
    parser.add_argument("--output-compression", type=int, help="Compression level 0-100 for JPEG and WebP outputs.")
    parser.add_argument("--moderation", help="Optional image moderation level: auto or low.")
    parser.add_argument("--action", help="Optional image tool action: generate, edit, or auto.")
    parser.add_argument("--partial-images", type=int, help="Number of streamed partial images to request, 0-3.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"Network request timeout in seconds. Defaults to {DEFAULT_REQUEST_TIMEOUT_SECONDS}.",
    )
    parser.add_argument("--input-fidelity", help="Optional input fidelity for models that allow explicit selection: high or low.")
    parser.add_argument("--mask", help="Optional local mask image for inpainting; requires at least one --reference.")
    parser.add_argument("--instructions")
    parser.add_argument("--auth-json", help="Path to Codex auth.json. When provided, this exact file overrides automatic discovery.")
    parser.add_argument(
        "--hide-response-details",
        action="store_true",
        help="Omit last response event details from error messages; redacted logs are still written.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug details.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_tool_options(args)
    prompt = _read_prompt(args.prompt, args.prompt_file)
    out_path = _output_path(args.name or prompt, args.output_format, args.auth_json)

    if args.dry_run:
        if _uses_responses_transport(args):
            payload = _build_payload(args, prompt)
            preview = {
                "endpoint": CODEX_RESPONSES_URL,
                "output": str(out_path),
                "log": str(_log_path(out_path)),
                "transport": args.transport,
                "deprecated": args.transport in DEPRECATED_TRANSPORTS,
                "timeout_seconds": _effective_timeout(args),
                **_redact_preview_payload(payload),
            }
        else:
            options = _build_image_api_options(args, prompt)
            preview = {
                "endpoint": CODEX_IMAGE_EDITS_URL
                if _uses_image_edit_endpoint(args)
                else CODEX_IMAGE_GENERATIONS_URL,
                "output": str(out_path),
                "log": str(_log_path(out_path)),
                "timeout_seconds": _effective_timeout(args),
                **_redact_image_api_preview(args, options),
            }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    log_path = _log_path(out_path)
    if _uses_responses_transport(args):
        _configure_cli_log(log_path, "responses-event", reset_messages=True)
    else:
        _configure_cli_log(log_path, "image-json", reset_messages=True)
    token, account_id = _read_codex_auth(args.auth_json)
    started = time.time()
    final_written = False
    if _uses_responses_transport(args):
        payload = _build_payload(args, prompt)
        if args.transport in DEPRECATED_TRANSPORTS:
            image_bytes, final_written = _stream_responses_raw(
                payload,
                token,
                account_id,
                log_path,
                out_path,
                timeout_seconds=_effective_timeout(args),
                save_partials=bool(args.partial_images),
                verbose=args.verbose,
                show_response_details=not args.hide_response_details,
            )
        else:
            image_bytes, final_written = _stream_responses_sdk(
                payload,
                token,
                account_id,
                log_path,
                out_path,
                timeout_seconds=_effective_timeout(args),
                save_partials=bool(args.partial_images),
                verbose=args.verbose,
                show_response_details=not args.hide_response_details,
            )
    else:
        image_bytes = _run_image_api(
            args,
            prompt,
            token,
            account_id,
            out_path,
            log_path,
        )
    if not final_written:
        out_path.write_bytes(image_bytes)
    _info(f"Wrote {out_path}")
    if args.copy_to:
        copied = _copy_result(out_path, args.copy_to, force=args.force)
        _info(f"Copied {copied}")
    _info(f"Completed in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
