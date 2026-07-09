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
from dataclasses import dataclass, field
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
CLI_LOG_FORMATS = {"responses-event", "image-json", "image-jsonl"}


@dataclass
class RequestConfig:
    prompt: str | None
    prompt_file: str | None
    reference: list[str]
    name: str | None
    copy_to: str | None
    force: bool
    transport: str
    model: str | None
    reasoning_effort: str | None
    image_model: str | None
    output_format: str
    size: str | None
    quality: str | None
    background: str | None
    output_compression: int | None
    moderation: str | None
    action: str | None
    partial_images: int | None
    timeout: float
    input_fidelity: str | None
    mask: str | None
    instructions: str | None
    auth_json: str | None
    hide_response_details: bool
    verbose: bool
    dry_run: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "RequestConfig":
        return cls(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            reference=list(args.reference or []),
            name=args.name,
            copy_to=args.copy_to,
            force=args.force,
            transport=args.transport,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            image_model=args.image_model,
            output_format=args.output_format,
            size=args.size,
            quality=args.quality,
            background=args.background,
            output_compression=args.output_compression,
            moderation=args.moderation,
            action=args.action,
            partial_images=args.partial_images,
            timeout=args.timeout,
            input_fidelity=args.input_fidelity,
            mask=args.mask,
            instructions=args.instructions,
            auth_json=args.auth_json,
            hide_response_details=args.hide_response_details,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

    @property
    def effective_image_model(self) -> str:
        return self.image_model or self.model or DEFAULT_IMAGE_MODEL

    @property
    def effective_responses_model(self) -> str:
        return self.model or DEFAULT_RESPONSES_MODEL

    @property
    def timeout_seconds(self) -> float:
        return self.timeout

    @property
    def uses_image_edit_endpoint(self) -> bool:
        return bool(self.reference or self.mask or self.action == "edit")

    @property
    def uses_responses_transport(self) -> bool:
        return self.transport in RESPONSES_TRANSPORTS

    @property
    def save_partials(self) -> bool:
        return bool(self.partial_images)

    @property
    def show_response_details(self) -> bool:
        return not self.hide_response_details


@dataclass
class LogContext:
    path: Path | None = None
    format: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    active_handle: Any = None


@dataclass
class RunContext:
    output_path: Path
    log_path: Path
    token: str
    account_id: str | None
    timeout_seconds: float
    started: float


class Redaction:
    @staticmethod
    def to_plain_data(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return [Redaction.to_plain_data(item) for item in value]
        return value

    @staticmethod
    def log_record_with_timestamp(value: Any) -> Any:
        plain = Redaction.to_plain_data(value)
        if isinstance(plain, dict):
            return {**plain, "logged_at": Logging.utc_timestamp()}
        return {"value": plain, "logged_at": Logging.utc_timestamp()}

    @staticmethod
    def looks_like_image_base64(value: Any) -> bool:
        return isinstance(value, str) and len(value) > 1000 and not value.startswith("data:")

    @staticmethod
    def scan_final_image_base64(value: Any) -> str | None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "partial_image_b64":
                    continue
                if key in FINAL_IMAGE_KEYS and Redaction.looks_like_image_base64(nested):
                    return nested
                found = Redaction.scan_final_image_base64(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = Redaction.scan_final_image_base64(nested)
                if found:
                    return found
        return None

    @staticmethod
    def responses_stream_item(value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, nested in value.items():
                if key in IMAGE_PAYLOAD_KEYS and isinstance(nested, str):
                    redacted[key] = f"<redacted {len(nested)} chars>"
                else:
                    redacted[key] = Redaction.responses_stream_item(nested)
            return redacted
        if isinstance(value, list):
            return [Redaction.responses_stream_item(item) for item in value]
        return value

    @staticmethod
    def image_api_log(value: Any) -> Any:
        plain = Redaction.to_plain_data(value)
        if isinstance(plain, dict):
            redacted = {}
            for key, nested in plain.items():
                if key in IMAGE_PAYLOAD_KEYS and isinstance(nested, str):
                    redacted[key] = f"<redacted {len(nested)} chars>"
                else:
                    redacted[key] = Redaction.image_api_log(nested)
            return redacted
        if isinstance(plain, list):
            return [Redaction.image_api_log(item) for item in plain]
        return plain

    @staticmethod
    def responses_preview(payload: dict[str, Any]) -> dict[str, Any]:
        preview = dict(payload)
        preview["input"] = [
            {
                "role": "user",
                "content": [
                    {
                        **item,
                        "image_url": "data:<redacted>"
                        if item.get("type") == "input_image"
                        else item.get("image_url"),
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

    @staticmethod
    def image_api_preview(config: RequestConfig, options: dict[str, Any]) -> dict[str, Any]:
        preview = dict(options)
        if config.uses_image_edit_endpoint:
            preview["images"] = [str(Path(path)) for path in config.reference]
            if config.mask:
                preview["mask"] = str(Path(config.mask))
        return preview

    @staticmethod
    def compact_json(value: Any, *, limit: int = 2000) -> str:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    @staticmethod
    def format_stream_failure(
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
            redacted = Redaction.responses_stream_item(Redaction.to_plain_data(output_item_done))
            details.append(f"Output item done: {Redaction.compact_json(redacted)}")
        if show_response_details and last_item is not None:
            redacted = Redaction.responses_stream_item(Redaction.to_plain_data(last_item))
            details.append(f"Last event: {Redaction.compact_json(redacted)}")
        return " ".join(details)

    @staticmethod
    def sdk_exception_details(exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {"error": str(exc)}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code
        request_id = getattr(exc, "request_id", None)
        if request_id:
            details["request_id"] = request_id
        body = getattr(exc, "body", None)
        if body is not None:
            details["body"] = Redaction.to_plain_data(body)
        response = getattr(exc, "response", None)
        if response is not None:
            response_status = getattr(response, "status_code", None)
            if response_status is not None and "status_code" not in details:
                details["status_code"] = response_status
            response_text = getattr(response, "text", None)
            if response_text:
                details["response_text"] = response_text[:4000]
        return details


class Logging:
    def __init__(self, context: LogContext | None = None) -> None:
        self.context = context or LogContext()

    @staticmethod
    def utc_timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def responses_event_type(item: Any) -> str:
        if isinstance(item, dict):
            event_type = item.get("type")
            if isinstance(event_type, str) and event_type:
                return event_type
        return "response.sdk_event"

    @staticmethod
    def write_responses_log_event(log_handle: Any, event_type: str, data: Any) -> None:
        redacted = Redaction.responses_stream_item(Redaction.to_plain_data(data))
        log_handle.write(f"event: {event_type}\n")
        log_handle.write(f"logged_at: {Logging.utc_timestamp()}\n")
        log_handle.write("data: " + json.dumps(redacted, ensure_ascii=False) + "\n\n")

    @staticmethod
    def write_image_api_stream_log_item(log_handle: Any, item: Any) -> None:
        redacted = Redaction.image_api_log(Redaction.log_record_with_timestamp(item))
        log_handle.write(json.dumps(redacted, ensure_ascii=False) + "\n")

    def configure(
        self,
        log_path: Path | None,
        log_format: str | None,
        *,
        reset_messages: bool = False,
    ) -> None:
        if log_format is not None and log_format not in CLI_LOG_FORMATS:
            raise ValueError(f"Unknown CLI log format: {log_format}")
        self.context.path = log_path
        self.context.format = log_format
        if reset_messages:
            self.context.messages = []

    def set_active_handle(self, log_handle: Any) -> None:
        self.context.active_handle = log_handle

    def die(self, message: str) -> None:
        self.log_cli_message("error", message)
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1)

    def info(self, message: str) -> None:
        self.log_cli_message("info", message)
        print(message)

    def debug(self, message: str, *, verbose: bool) -> None:
        self.log_cli_message("debug", message)
        if verbose:
            print(message)

    def cli_log_message_record(self, level: str, message: str) -> dict[str, str]:
        return {
            "logged_at": self.utc_timestamp(),
            "level": level,
            "message": message,
        }

    def append_image_json_cli_message(self, log_path: Path, record: dict[str, str]) -> None:
        try:
            current = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
        except Exception:
            current = {}
        if not isinstance(current, dict):
            current = {}
        messages = current.get("messages")
        if not isinstance(messages, list):
            messages = []
        messages.append(Redaction.image_api_log(record))
        current["messages"] = messages
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def log_cli_message(self, level: str, message: str) -> None:
        record = self.cli_log_message_record(level, message)
        self.context.messages.append(record)
        if not self.context.path or not self.context.format:
            return
        try:
            self.context.path.parent.mkdir(parents=True, exist_ok=True)
            if self.context.format == "responses-event":
                if self.context.active_handle is not None:
                    self.write_responses_log_event(
                        self.context.active_handle,
                        f"codex_image_gen.{level}",
                        {"level": level, "message": message},
                    )
                    self.context.active_handle.flush()
                else:
                    with self.context.path.open("a", encoding="utf-8") as log_handle:
                        self.write_responses_log_event(
                            log_handle,
                            f"codex_image_gen.{level}",
                            {"level": level, "message": message},
                        )
            elif self.context.format == "image-jsonl":
                if self.context.active_handle is not None:
                    self.write_image_api_stream_log_item(
                        self.context.active_handle,
                        {"log": {"level": level, "message": message}},
                    )
                    self.context.active_handle.flush()
                else:
                    with self.context.path.open("a", encoding="utf-8") as log_handle:
                        self.write_image_api_stream_log_item(
                            log_handle,
                            {"log": {"level": level, "message": message}},
                        )
            elif self.context.format == "image-json":
                self.append_image_json_cli_message(self.context.path, record)
        except Exception:
            return

    def start_info(
        self,
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
            "started_at": self.utc_timestamp(),
            "endpoint": endpoint,
            "transport": transport,
            "output": str(final_path),
            "timeout_seconds": timeout_seconds,
            "request": request_payload,
        }
        if client:
            info["client"] = client
        return info

    def write_responses_sdk_log_item(self, log_handle: Any, item: Any) -> None:
        plain = Redaction.to_plain_data(item)
        self.write_responses_log_event(log_handle, self.responses_event_type(plain), plain)

    def write_responses_failure_log(
        self,
        log_path: Path | None,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        if not log_path:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            self.write_responses_log_event(log_handle, event_type, details)

    def image_api_log_record(
        self,
        payload: dict[str, Any],
        response: Any,
        *,
        start_info: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {"logged_at": self.utc_timestamp()}
        if start_info is not None:
            record["start"] = Redaction.image_api_log(start_info)
        if status is not None:
            record["status"] = status
        record["request"] = Redaction.image_api_log(payload)
        record["response"] = Redaction.image_api_log(response)
        if self.context.messages:
            record["messages"] = Redaction.image_api_log(self.context.messages)
        return record

    def write_image_api_log(
        self,
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
                self.image_api_log_record(
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

    def write_image_api_failure_log(
        self,
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
        self.write_image_api_log(log_path, payload, response, start_info=start_info, status="failed")
        return response

    def write_image_api_start_log(
        self,
        log_path: Path,
        payload: dict[str, Any],
        start_info: dict[str, Any],
    ) -> None:
        self.write_image_api_log(log_path, payload, None, start_info=start_info, status="started")

    def write_image_api_stream_start_log(self, log_path: Path, start_info: dict[str, Any]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            self.write_image_api_stream_log_item(log_handle, {"start": start_info})


class Paths:
    @staticmethod
    def codex_home() -> Path | None:
        value = os.environ.get("CODEX_HOME")
        if not value:
            return None
        return Path(value).expanduser()

    @staticmethod
    def auth_path(auth_json: str | None, logger: Logging) -> Path:
        if auth_json:
            auth_path = Path(auth_json).expanduser()
            if auth_path.exists():
                return auth_path
            logger.die(f"Codex auth file not found: {auth_path}")

        codex_home = Paths.codex_home()
        candidates = []
        if codex_home:
            candidates.append(codex_home / "auth.json")
        candidates.append(Path.home() / ".codex" / "auth.json")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        checked = ", ".join(str(candidate) for candidate in candidates)
        logger.die(f"Codex auth file not found. Checked: {checked}")

    @staticmethod
    def output_root(auth_json: str | None) -> Path:
        if auth_json:
            return Path(auth_json).expanduser().parent
        codex_home = Paths.codex_home()
        if codex_home:
            return codex_home
        return Path.home() / ".codex"

    @staticmethod
    def output_dir(auth_json: str | None) -> Path:
        output_dir = Paths.output_root(auth_json) / "generated_images_free_reference"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def slugify(value: str) -> str:
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

    @staticmethod
    def output_path(name: str | None, output_format: str, auth_json: str | None) -> Path:
        suffix = output_format.lower().lstrip(".") or DEFAULT_OUTPUT_FORMAT
        stem = f"{uuid4()}-{Paths.slugify(name or 'image')}"
        return Paths.output_dir(auth_json) / f"{stem}.{suffix}"

    @staticmethod
    def log_path(final_path: Path) -> Path:
        return final_path.with_suffix(f"{final_path.suffix}.log")

    @staticmethod
    def read_codex_auth(auth_json: str | None, logger: Logging) -> tuple[str, str | None]:
        auth_path = Paths.auth_path(auth_json, logger)
        if not auth_path.exists():
            logger.die(f"Codex auth file not found: {auth_path}")
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.die(f"Could not parse Codex auth file: {exc}")
        tokens = auth.get("tokens") or {}
        token = tokens.get("access_token")
        if not token:
            logger.die("Codex access_token not found in auth.json.")
        account_id = auth.get("last_active_account_id") or tokens.get("account_id")
        return token, account_id

    @staticmethod
    def data_url(path: Path, logger: Logging) -> str:
        if not path.exists():
            logger.die(f"Reference image not found: {path}")
        if not path.is_file():
            logger.die(f"Reference image is not a file: {path}")
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            logger.die(f"Reference file is not recognized as an image: {path}")
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class Validation:
    @staticmethod
    def parse_size(size: str) -> tuple[int, int] | None:
        match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", size)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def is_gpt_image_2_model(model: str | None) -> bool:
        return bool(model and model.startswith(GPT_IMAGE_2_PREFIX))

    @staticmethod
    def validate_choice(value: str | None, allowed: set[str], option: str, logger: Logging) -> None:
        if value is not None and value not in allowed:
            values = ", ".join(sorted(allowed))
            logger.die(f"{option} must be one of {values}.")

    @staticmethod
    def validate_gpt_image_2_size(size: str, logger: Logging) -> None:
        parsed = Validation.parse_size(size)
        if parsed is None:
            logger.die("--size must be auto or WIDTHxHEIGHT, for example 1024x1024.")

        width, height = parsed
        max_edge = max(width, height)
        min_edge = min(width, height)
        total_pixels = width * height

        if max_edge > GPT_IMAGE_2_MAX_EDGE:
            logger.die("gpt-image-2 size maximum edge length must be less than or equal to 3840px.")
        if width % 16 != 0 or height % 16 != 0:
            logger.die("gpt-image-2 size width and height must be multiples of 16px.")
        if max_edge / min_edge > GPT_IMAGE_2_MAX_RATIO:
            logger.die("gpt-image-2 size long edge to short edge ratio must not exceed 3:1.")
        if total_pixels < GPT_IMAGE_2_MIN_PIXELS or total_pixels > GPT_IMAGE_2_MAX_PIXELS:
            logger.die(
                "gpt-image-2 size total pixels must be at least 655,360 and no more than 8,294,400."
            )

    @staticmethod
    def validate_size(size: str | None, image_model: str | None, logger: Logging) -> None:
        if size is None:
            return
        if size == "auto":
            return
        if Validation.is_gpt_image_2_model(image_model):
            Validation.validate_gpt_image_2_size(size, logger)
            return
        if Validation.parse_size(size) is None:
            logger.die("--size must be auto or WIDTHxHEIGHT, for example 1024x1024.")

    @staticmethod
    def validate(config: RequestConfig, logger: Logging) -> None:
        Validation.validate_choice(config.action, ALLOWED_ACTIONS, "--action", logger)
        Validation.validate_choice(config.background, ALLOWED_BACKGROUNDS, "--background", logger)
        Validation.validate_choice(config.input_fidelity, ALLOWED_INPUT_FIDELITIES, "--input-fidelity", logger)
        Validation.validate_choice(config.moderation, ALLOWED_MODERATIONS, "--moderation", logger)
        Validation.validate_choice(config.output_format, ALLOWED_OUTPUT_FORMATS, "--output-format", logger)
        Validation.validate_choice(config.quality, ALLOWED_QUALITIES, "--quality", logger)
        Validation.validate_choice(config.transport, ALLOWED_TRANSPORTS, "--transport", logger)
        image_model = config.effective_image_model if config.transport == "image-api" else config.image_model
        Validation.validate_size(config.size, image_model, logger)

        if config.timeout <= 0:
            logger.die("--timeout must be greater than 0 seconds.")

        if config.output_compression is not None:
            if config.output_compression < 0 or config.output_compression > 100:
                logger.die("--output-compression must be between 0 and 100.")
            if config.output_format not in {"jpeg", "webp"}:
                logger.die("--output-compression requires --output-format jpeg or --output-format webp.")

        if config.partial_images is not None and (config.partial_images < 0 or config.partial_images > 3):
            logger.die("--partial-images must be between 0 and 3.")

        if config.background == "transparent":
            if config.output_format not in {"png", "webp"}:
                logger.die("--background transparent requires --output-format png or --output-format webp.")
            if config.uses_responses_transport and not config.image_model:
                logger.die("--background transparent requires an explicit --image-model with a Responses transport.")
            if Validation.is_gpt_image_2_model(image_model):
                logger.die("transparent backgrounds are not supported by gpt-image-2 image models.")

        if config.mask and not config.reference:
            logger.die("--mask requires at least one --reference image; the first reference is the edit target.")

        if config.input_fidelity:
            if config.uses_responses_transport and not config.image_model:
                logger.die("--input-fidelity requires an explicit --image-model with a Responses transport.")
            if Validation.is_gpt_image_2_model(image_model):
                logger.die("gpt-image-2 always uses high-fidelity image inputs; omit --input-fidelity.")
            if image_model in UNSUPPORTED_INPUT_FIDELITY_IMAGE_MODELS:
                logger.die(f"--input-fidelity is not supported by {image_model}.")

        if config.transport == "image-api":
            if config.instructions:
                logger.die("--instructions is only supported with --transport responses.")
            if config.action == "edit" and not config.reference:
                logger.die("--action edit requires at least one --reference image with --transport image-api.")


class Payloads:
    @staticmethod
    def build_image_tool(config: RequestConfig, logger: Logging) -> dict[str, Any]:
        tool: dict[str, Any] = {"type": "image_generation", "output_format": config.output_format}
        optional_fields = {
            "action": config.action,
            "background": config.background,
            "input_fidelity": config.input_fidelity,
            "model": config.image_model,
            "moderation": config.moderation,
            "output_compression": config.output_compression,
            "partial_images": config.partial_images,
            "quality": config.quality,
            "size": config.size,
        }
        for key, value in optional_fields.items():
            if value is not None:
                tool[key] = value
        if config.mask:
            tool["input_image_mask"] = {"image_url": Paths.data_url(Path(config.mask), logger)}
        return tool

    @staticmethod
    def build_responses_payload(config: RequestConfig, prompt: str, logger: Logging) -> dict[str, Any]:
        content: list[dict[str, str]] = []
        for reference in config.reference:
            content.append({"type": "input_image", "image_url": Paths.data_url(Path(reference), logger)})
        content.append({"type": "input_text", "text": prompt})

        payload: dict[str, Any] = {
            "model": config.effective_responses_model,
            "instructions": config.instructions or "",
            "input": [{"role": "user", "content": content}],
            "tools": [Payloads.build_image_tool(config, logger)],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
            "include": [],
        }
        if config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": config.reasoning_effort}
        return payload

    @staticmethod
    def build_image_api_options(config: RequestConfig, prompt: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "model": config.effective_image_model,
            "prompt": prompt,
            "output_format": config.output_format,
            "response_format": "b64_json",
        }
        optional_fields = {
            "background": config.background,
            "input_fidelity": config.input_fidelity if config.uses_image_edit_endpoint else None,
            "moderation": config.moderation,
            "output_compression": config.output_compression,
            "partial_images": config.partial_images,
            "quality": config.quality,
            "size": config.size,
        }
        for key, value in optional_fields.items():
            if value is not None:
                options[key] = value
        if config.partial_images is not None:
            options["stream"] = True
        return options

    @staticmethod
    def build_image_api_edit_payload(
        config: RequestConfig,
        prompt: str,
        logger: Logging,
    ) -> dict[str, Any]:
        payload = Payloads.build_image_api_options(config, prompt)
        payload["images"] = [{"image_url": Paths.data_url(Path(reference), logger)} for reference in config.reference]
        if config.mask:
            payload["mask"] = {"image_url": Paths.data_url(Path(config.mask), logger)}
        return payload


class Output:
    @staticmethod
    def partial_output_path(final_path: Path, index: int | str) -> Path:
        return final_path.with_name(f"{final_path.stem}-partial-{index}{final_path.suffix}")

    @staticmethod
    def copy_result(source: Path, destination: str, *, force: bool, logger: Logging) -> Path:
        target = Path(destination)
        if target.exists() and target.is_dir():
            target = target / source.name
        if target.exists() and not force:
            logger.die(f"Copy target already exists: {target} (use --force to overwrite)")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    @staticmethod
    def partial_index(item: dict[str, Any], fallback: int) -> int | str:
        value = item.get("partial_image_index")
        if isinstance(value, int) and 0 <= value <= 3:
            return value
        return fallback

    @staticmethod
    def write_partial_image(
        item: dict[str, Any],
        final_path: Path,
        fallback_index: int,
        *,
        logger: Logging,
        verbose: bool,
    ) -> tuple[Path, bytes] | None:
        image_b64 = item.get("partial_image_b64")
        if not Redaction.looks_like_image_base64(image_b64):
            return None
        image_bytes = base64.b64decode(image_b64)
        partial_path = Output.partial_output_path(final_path, Output.partial_index(item, fallback_index))
        partial_path.write_bytes(image_bytes)
        logger.info(f"Wrote partial {partial_path}")
        return partial_path, image_bytes

    @staticmethod
    def final_image_bytes(
        image_b64: str,
        last_partial: tuple[Path, bytes] | None,
        final_path: Path,
        *,
        logger: Logging,
        verbose: bool,
    ) -> tuple[bytes, bool]:
        image_bytes = base64.b64decode(image_b64)
        if last_partial:
            partial_path, partial_bytes = last_partial
            if partial_bytes == image_bytes:
                partial_path.replace(final_path)
                logger.debug(
                    f"Renamed final partial {partial_path} to {final_path}",
                    verbose=verbose,
                )
                return image_bytes, True
        return image_bytes, False

    @staticmethod
    def image_response_bytes(
        response: Any,
        *,
        logger: Logging,
        log_path: Path | None = None,
        show_response_details: bool = True,
    ) -> bytes:
        data = getattr(response, "data", None)
        if not data and isinstance(response, dict):
            data = response.get("data")
        if not data:
            logger.die(
                Redaction.format_stream_failure(
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
        if not Redaction.looks_like_image_base64(image_b64):
            logger.die(
                Redaction.format_stream_failure(
                    "Codex Image API response did not include b64_json image data.",
                    log_path=log_path,
                    last_item=first,
                    show_response_details=show_response_details,
                )
            )
        return base64.b64decode(image_b64)


class CodexClient:
    @staticmethod
    def create_openai(token: str, account_id: str | None, timeout_seconds: float, logger: Logging) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            logger.die("The openai package is not installed in the skill virtual environment.")

        headers = {}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return OpenAI(
            api_key=token,
            base_url=CODEX_IMAGE_API_BASE_URL,
            default_headers=headers,
            timeout=timeout_seconds,
        )


class ResponsesTransport:
    def __init__(self, logger: Logging) -> None:
        self.logger = logger

    def stream_raw(
        self,
        payload: dict[str, Any],
        run: RunContext,
        config: RequestConfig,
    ) -> tuple[bytes, bool]:
        if run.log_path:
            run.log_path.parent.mkdir(parents=True, exist_ok=True)
            with run.log_path.open("w", encoding="utf-8") as log_handle:
                self.logger.write_responses_log_event(
                    log_handle,
                    "codex_image_gen.start",
                    self.logger.start_info(
                        endpoint=CODEX_RESPONSES_URL,
                        transport="responses-raw",
                        final_path=run.output_path,
                        request_payload=payload,
                        timeout_seconds=run.timeout_seconds,
                    ),
                )
        self.logger.info("--transport responses-raw is deprecated; use --transport responses.")

        headers = {
            "Authorization": "Bearer " + run.token,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": DEFAULT_BETA_HEADER,
        }
        if run.account_id:
            headers["ChatGPT-Account-ID"] = run.account_id

        req = request.Request(
            CODEX_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            response = request.urlopen(req, timeout=run.timeout_seconds)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            details = {"status": exc.code, "body": body[:2000], "endpoint": CODEX_RESPONSES_URL}
            self.logger.write_responses_failure_log(run.log_path, "response.request_failed", details)
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Responses request failed with HTTP {exc.code}: {body[:2000]}",
                    log_path=run.log_path,
                    last_item=details,
                    show_response_details=config.show_response_details,
                )
            )
        except error.URLError as exc:
            details = {"error": str(exc), "endpoint": CODEX_RESPONSES_URL}
            self.logger.write_responses_failure_log(run.log_path, "response.request_failed", details)
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Responses request failed: {exc}",
                    log_path=run.log_path,
                    last_item=details,
                    show_response_details=config.show_response_details,
                )
            )

        log_handle = None
        partial_count = 0
        last_partial: tuple[Path, bytes] | None = None
        last_item: Any = None
        last_output_item_done: Any = None
        try:
            if run.log_path:
                run.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = run.log_path.open("a", encoding="utf-8")
                self.logger.set_active_handle(log_handle)
            for raw in response:
                line = raw.decode("utf-8", "ignore").rstrip("\n")
                if not line.startswith("data:"):
                    if log_handle and line:
                        self.logger.write_responses_log_event(
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
                        self.logger.write_responses_log_event(
                            log_handle,
                            "response.raw_data",
                            {"data": data},
                        )
                    continue
                last_item = item
                if log_handle:
                    self.logger.write_responses_log_event(log_handle, self.logger.responses_event_type(item), item)
                if isinstance(item, dict) and item.get("type") == "response.output_item.done":
                    last_output_item_done = item
                if (
                    config.save_partials
                    and isinstance(item, dict)
                    and item.get("type") == "response.image_generation_call.partial_image"
                ):
                    partial_count += 1
                    written_partial = Output.write_partial_image(
                        item,
                        run.output_path,
                        partial_count,
                        logger=self.logger,
                        verbose=config.verbose,
                    )
                    if written_partial:
                        last_partial = written_partial
                    continue
                image_b64 = Redaction.scan_final_image_base64(item)
                if image_b64:
                    return Output.final_image_bytes(
                        image_b64,
                        last_partial,
                        run.output_path,
                        logger=self.logger,
                        verbose=config.verbose,
                    )
        finally:
            self.logger.set_active_handle(None)
            if log_handle:
                log_handle.close()
            response.close()
        self.logger.die(
            Redaction.format_stream_failure(
                "No generated image was found in the streamed response.",
                log_path=run.log_path,
                last_item=last_item,
                output_item_done=last_output_item_done,
                show_response_details=config.show_response_details,
            )
        )

    def stream_sdk(
        self,
        payload: dict[str, Any],
        run: RunContext,
        config: RequestConfig,
    ) -> tuple[bytes, bool]:
        client = CodexClient.create_openai(run.token, run.account_id, run.timeout_seconds, self.logger)
        log_handle = None
        stream = None
        if run.log_path:
            run.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = run.log_path.open("w", encoding="utf-8")
            self.logger.set_active_handle(log_handle)
            self.logger.write_responses_log_event(
                log_handle,
                "codex_image_gen.start",
                self.logger.start_info(
                    endpoint=CODEX_RESPONSES_URL,
                    transport="responses",
                    final_path=run.output_path,
                    request_payload=payload,
                    timeout_seconds=run.timeout_seconds,
                    client="openai-sdk",
                ),
            )
            log_handle.flush()
        try:
            stream = client.responses.create(
                **payload,
                extra_headers={"OpenAI-Beta": DEFAULT_BETA_HEADER},
                timeout=run.timeout_seconds,
            )
        except Exception as exc:
            details = {
                **Redaction.sdk_exception_details(exc),
                "endpoint": CODEX_RESPONSES_URL,
                "request": payload,
                "transport": "responses-sdk",
            }
            if log_handle:
                self.logger.write_responses_log_event(log_handle, "response.request_failed", details)
                log_handle.close()
                self.logger.set_active_handle(None)
                log_handle = None
            else:
                self.logger.write_responses_failure_log(run.log_path, "response.request_failed", details)
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Responses SDK request failed: {exc}",
                    log_path=run.log_path,
                    last_item=details,
                    show_response_details=config.show_response_details,
                )
            )

        partial_count = 0
        last_partial: tuple[Path, bytes] | None = None
        last_item: Any = None
        last_output_item_done: Any = None
        try:
            for event in stream:
                item = Redaction.to_plain_data(event)
                last_item = item
                if log_handle:
                    self.logger.write_responses_sdk_log_item(log_handle, item)
                if isinstance(item, dict) and item.get("type") == "response.output_item.done":
                    last_output_item_done = item
                if (
                    config.save_partials
                    and isinstance(item, dict)
                    and item.get("type") == "response.image_generation_call.partial_image"
                ):
                    partial_count += 1
                    written_partial = Output.write_partial_image(
                        item,
                        run.output_path,
                        partial_count,
                        logger=self.logger,
                        verbose=config.verbose,
                    )
                    if written_partial:
                        last_partial = written_partial
                    continue
                image_b64 = Redaction.scan_final_image_base64(item)
                if image_b64:
                    return Output.final_image_bytes(
                        image_b64,
                        last_partial,
                        run.output_path,
                        logger=self.logger,
                        verbose=config.verbose,
                    )
        except Exception as exc:
            details = {
                **Redaction.sdk_exception_details(exc),
                "endpoint": CODEX_RESPONSES_URL,
                "transport": "responses-sdk",
            }
            if log_handle:
                self.logger.write_responses_sdk_log_item(log_handle, {"type": "response.stream_error", **details})
                log_handle.close()
                self.logger.set_active_handle(None)
                log_handle = None
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Responses SDK stream failed: {exc}",
                    log_path=run.log_path,
                    last_item=last_item or details,
                    output_item_done=last_output_item_done,
                    show_response_details=config.show_response_details,
                )
            )
        finally:
            self.logger.set_active_handle(None)
            if log_handle:
                log_handle.close()
            close = getattr(stream, "close", None) if stream is not None else None
            if close:
                close()
        self.logger.die(
            Redaction.format_stream_failure(
                "No generated image was found in the streamed response.",
                log_path=run.log_path,
                last_item=last_item,
                output_item_done=last_output_item_done,
                show_response_details=config.show_response_details,
            )
        )


class ImageApiTransport:
    def __init__(self, logger: Logging) -> None:
        self.logger = logger

    def request_edit(
        self,
        payload: dict[str, Any],
        run: RunContext,
        start_info: dict[str, Any],
        config: RequestConfig,
    ) -> bytes:
        if payload.get("stream"):
            self.logger.die("--partial-images with --transport image-api edit is not supported; use --transport responses.")
        headers = {
            "Authorization": "Bearer " + run.token,
            "Content-Type": "application/json",
            "User-Agent": "codex-imagegen-free-reference",
        }
        if run.account_id:
            headers["ChatGPT-Account-ID"] = run.account_id
        try:
            import httpx

            response = httpx.post(
                CODEX_IMAGE_EDITS_URL,
                headers=headers,
                json=payload,
                timeout=run.timeout_seconds,
            )
            if response.status_code >= 400:
                failure = self.logger.write_image_api_failure_log(
                    run.log_path,
                    payload,
                    start_info=start_info,
                    status_code=response.status_code,
                    response_text=response.text,
                )
                self.logger.die(
                    Redaction.format_stream_failure(
                        f"Codex Image API edit request failed with HTTP {response.status_code}: "
                        f"{response.text[:2000]}",
                        log_path=run.log_path,
                        last_item=failure,
                        show_response_details=config.show_response_details,
                    )
                )
            try:
                data = response.json()
            except Exception as exc:
                failure = self.logger.write_image_api_failure_log(
                    run.log_path,
                    payload,
                    start_info=start_info,
                    status_code=response.status_code,
                    response_text=response.text,
                    error_message=f"Could not parse JSON response: {exc}",
                )
                self.logger.die(
                    Redaction.format_stream_failure(
                        f"Codex Image API edit response was not valid JSON: {exc}",
                        log_path=run.log_path,
                        last_item=failure,
                        show_response_details=config.show_response_details,
                    )
                )
            self.logger.write_image_api_log(run.log_path, payload, data, start_info=start_info, status="completed")
        except Exception as exc:
            failure = self.logger.write_image_api_failure_log(
                run.log_path,
                payload,
                start_info=start_info,
                error_message=str(exc),
            )
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Image API edit request failed: {exc}",
                    log_path=run.log_path,
                    last_item=failure,
                    show_response_details=config.show_response_details,
                )
            )
        return Output.image_response_bytes(
            data,
            logger=self.logger,
            log_path=run.log_path,
            show_response_details=config.show_response_details,
        )

    def stream_response(
        self,
        stream: Any,
        run: RunContext,
        request_options: dict[str, Any],
        config: RequestConfig,
    ) -> bytes:
        partial_count = 0
        last_item: Any = None
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        with run.log_path.open("a", encoding="utf-8") as log_handle:
            self.logger.set_active_handle(log_handle)
            try:
                self.logger.write_image_api_stream_log_item(log_handle, {"request": request_options})
                for event in stream:
                    item = Redaction.to_plain_data(event)
                    last_item = item
                    self.logger.write_image_api_stream_log_item(log_handle, {"event": item})
                    if config.save_partials and isinstance(item, dict):
                        partial_b64 = item.get("partial_image_b64")
                        if not partial_b64:
                            partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
                        if Redaction.looks_like_image_base64(partial_b64):
                            partial_count += 1
                            partial_path = Output.partial_output_path(run.output_path, partial_count)
                            partial_path.write_bytes(base64.b64decode(partial_b64))
                            self.logger.info(f"Wrote partial {partial_path}")
                            continue
                    image_b64 = Redaction.scan_final_image_base64(item)
                    if image_b64:
                        return base64.b64decode(image_b64)
            finally:
                self.logger.set_active_handle(None)
        self.logger.die(
            Redaction.format_stream_failure(
                "No generated image was found in the streamed Codex Image API response.",
                log_path=run.log_path,
                last_item=last_item,
                show_response_details=config.show_response_details,
            )
        )

    def stream_json_response(
        self,
        response: Any,
        run: RunContext,
        request_options: dict[str, Any],
        config: RequestConfig,
    ) -> bytes:
        partial_count = 0
        last_item: Any = None
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = run.log_path.open("a", encoding="utf-8")
        self.logger.set_active_handle(log_handle)
        try:
            self.logger.write_image_api_stream_log_item(log_handle, {"request": request_options})
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
                self.logger.write_image_api_stream_log_item(log_handle, {"event": item})
                if config.save_partials:
                    partial_b64 = item.get("partial_image_b64") if isinstance(item, dict) else None
                    if not partial_b64 and isinstance(item, dict):
                        partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
                    if Redaction.looks_like_image_base64(partial_b64):
                        partial_count += 1
                        partial_path = Output.partial_output_path(run.output_path, partial_count)
                        partial_path.write_bytes(base64.b64decode(partial_b64))
                        self.logger.info(f"Wrote partial {partial_path}")
                        continue
                image_b64 = Redaction.scan_final_image_base64(item)
                if image_b64:
                    return base64.b64decode(image_b64)
        finally:
            self.logger.set_active_handle(None)
            log_handle.close()
            response.close()
        self.logger.die(
            Redaction.format_stream_failure(
                "No generated image was found in the streamed Codex Image API response.",
                log_path=run.log_path,
                last_item=last_item,
                show_response_details=config.show_response_details,
            )
        )

    def run(self, config: RequestConfig, prompt: str, run: RunContext) -> bytes:
        if config.uses_image_edit_endpoint:
            payload = Payloads.build_image_api_edit_payload(config, prompt, self.logger)
            start_info = self.logger.start_info(
                endpoint=CODEX_IMAGE_EDITS_URL,
                transport="image-api",
                final_path=run.output_path,
                request_payload=payload,
                timeout_seconds=run.timeout_seconds,
                client="httpx-json",
            )
            self.logger.write_image_api_start_log(run.log_path, payload, start_info)
            return self.request_edit(payload, run, start_info, config)

        client = CodexClient.create_openai(run.token, run.account_id, run.timeout_seconds, self.logger)
        options = Payloads.build_image_api_options(config, prompt)
        start_info = self.logger.start_info(
            endpoint=CODEX_IMAGE_GENERATIONS_URL,
            transport="image-api",
            final_path=run.output_path,
            request_payload=options,
            timeout_seconds=run.timeout_seconds,
            client="openai-sdk",
        )
        if options.get("stream"):
            self.logger.configure(run.log_path, "image-jsonl")
            self.logger.write_image_api_stream_start_log(run.log_path, start_info)
        else:
            self.logger.configure(run.log_path, "image-json")
            self.logger.write_image_api_start_log(run.log_path, options, start_info)
        try:
            response = client.images.generate(**options, timeout=run.timeout_seconds)
        except Exception as exc:
            failure = self.logger.write_image_api_failure_log(
                run.log_path,
                options,
                start_info=start_info,
                error_message=str(exc),
            )
            self.logger.die(
                Redaction.format_stream_failure(
                    f"Codex Image API request failed: {exc}",
                    log_path=run.log_path,
                    last_item=failure,
                    show_response_details=config.show_response_details,
                )
            )

        if options.get("stream"):
            return self.stream_response(response, run, options, config)
        self.logger.write_image_api_log(run.log_path, options, response, start_info=start_info, status="completed")
        return Output.image_response_bytes(
            response,
            logger=self.logger,
            log_path=run.log_path,
            show_response_details=config.show_response_details,
        )


class Cli:
    def __init__(self, logger: Logging | None = None) -> None:
        self.logger = logger or Logging()

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
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
        return parser

    def parse_config(self, argv: list[str] | None = None) -> RequestConfig:
        args = self.build_parser().parse_args(argv)
        return RequestConfig.from_namespace(args)

    def read_prompt(self, config: RequestConfig) -> str:
        if config.prompt and config.prompt_file:
            self.logger.die("Use --prompt or --prompt-file, not both.")
        if config.prompt_file:
            path = Path(config.prompt_file)
            if not path.exists():
                self.logger.die(f"Prompt file not found: {path}")
            text = path.read_text(encoding="utf-8").strip()
        elif config.prompt:
            text = config.prompt.strip()
        else:
            self.logger.die("Missing prompt. Use --prompt or --prompt-file.")
        if not text:
            self.logger.die("Prompt is empty.")
        return text

    def dry_run_preview(self, config: RequestConfig, prompt: str, out_path: Path) -> dict[str, Any]:
        if config.uses_responses_transport:
            payload = Payloads.build_responses_payload(config, prompt, self.logger)
            return {
                "endpoint": CODEX_RESPONSES_URL,
                "output": str(out_path),
                "log": str(Paths.log_path(out_path)),
                "transport": config.transport,
                "deprecated": config.transport in DEPRECATED_TRANSPORTS,
                "timeout_seconds": config.timeout_seconds,
                **Redaction.responses_preview(payload),
            }
        options = Payloads.build_image_api_options(config, prompt)
        return {
            "endpoint": CODEX_IMAGE_EDITS_URL
            if config.uses_image_edit_endpoint
            else CODEX_IMAGE_GENERATIONS_URL,
            "output": str(out_path),
            "log": str(Paths.log_path(out_path)),
            "timeout_seconds": config.timeout_seconds,
            **Redaction.image_api_preview(config, options),
        }

    def execute(self, config: RequestConfig, prompt: str, out_path: Path) -> int:
        log_path = Paths.log_path(out_path)
        if config.uses_responses_transport:
            self.logger.configure(log_path, "responses-event", reset_messages=True)
        else:
            self.logger.configure(log_path, "image-json", reset_messages=True)
        token, account_id = Paths.read_codex_auth(config.auth_json, self.logger)
        run = RunContext(
            output_path=out_path,
            log_path=log_path,
            token=token,
            account_id=account_id,
            timeout_seconds=config.timeout_seconds,
            started=time.time(),
        )

        final_written = False
        if config.uses_responses_transport:
            payload = Payloads.build_responses_payload(config, prompt, self.logger)
            responses = ResponsesTransport(self.logger)
            if config.transport in DEPRECATED_TRANSPORTS:
                image_bytes, final_written = responses.stream_raw(payload, run, config)
            else:
                image_bytes, final_written = responses.stream_sdk(payload, run, config)
        else:
            image_bytes = ImageApiTransport(self.logger).run(config, prompt, run)

        if not final_written:
            out_path.write_bytes(image_bytes)
        self.logger.info(f"Wrote {out_path}")
        if config.copy_to:
            copied = Output.copy_result(out_path, config.copy_to, force=config.force, logger=self.logger)
            self.logger.info(f"Copied {copied}")
        self.logger.info(f"Completed in {time.time() - run.started:.1f}s")
        return 0

    def main(self, argv: list[str] | None = None) -> int:
        config = self.parse_config(argv)
        Validation.validate(config, self.logger)
        prompt = self.read_prompt(config)
        out_path = Paths.output_path(config.name or prompt, config.output_format, config.auth_json)

        if config.dry_run:
            print(json.dumps(self.dry_run_preview(config, prompt, out_path), ensure_ascii=False, indent=2))
            return 0

        return self.execute(config, prompt, out_path)


def main() -> int:
    return Cli().main()


if __name__ == "__main__":
    raise SystemExit(main())
