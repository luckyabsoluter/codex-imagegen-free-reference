#!/usr/bin/env python3
"""Codex-auth image generation with free local reference images.

This CLI calls the Codex Image API endpoints by default using the local Codex
auth snapshot in `~/.codex/auth.json`. Prompt-only generation uses the OpenAI
SDK with the Codex base URL. Local reference images are sent to the edit
endpoint as JSON `image_url` inputs. The Codex Responses hosted-tool route
remains available through `--transport responses`. Generated images are saved
under `~/.codex/generated_images_free_reference/` by default.
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
FINAL_IMAGE_KEYS = {"result", "image", "b64_json"}
ALLOWED_ACTIONS = {"generate", "edit", "auto"}
ALLOWED_BACKGROUNDS = {"transparent", "opaque", "auto"}
ALLOWED_INPUT_FIDELITIES = {"high", "low"}
ALLOWED_MODERATIONS = {"auto", "low"}
ALLOWED_OUTPUT_FORMATS = {"png", "webp", "jpeg"}
ALLOWED_QUALITIES = {"low", "medium", "high", "auto"}
ALLOWED_TRANSPORTS = {"image-api", "responses"}
UNSUPPORTED_INPUT_FIDELITY_IMAGE_MODELS = {"gpt-image-1-mini"}
GPT_IMAGE_2_PREFIX = "gpt-image-2"
GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_EDGE = 3840
GPT_IMAGE_2_MAX_RATIO = 3.0


def _die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def _uses_image_edit_endpoint(args: argparse.Namespace) -> bool:
    return bool(args.reference or args.mask or args.action == "edit")


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
        if args.transport == "responses" and not args.image_model:
            _die("--background transparent requires an explicit --image-model with --transport responses.")
        if _is_gpt_image_2_model(image_model):
            _die("transparent backgrounds are not supported by gpt-image-2 image models.")

    if args.mask and not args.reference:
        _die("--mask requires at least one --reference image; the first reference is the edit target.")

    if args.input_fidelity:
        if args.transport == "responses" and not args.image_model:
            _die("--input-fidelity requires an explicit --image-model with --transport responses.")
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
) -> tuple[Path, bytes] | None:
    image_b64 = item.get("partial_image_b64")
    if not _looks_like_image_base64(image_b64):
        return None
    image_bytes = base64.b64decode(image_b64)
    partial_path = _partial_output_path(final_path, _partial_index(item, fallback_index))
    partial_path.write_bytes(image_bytes)
    print(f"Wrote partial {partial_path}")
    return partial_path, image_bytes


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

    return {
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
            if key in {"result", "image", "b64_json", "partial_image_b64", "image_url"} and isinstance(nested, str):
                redacted[key] = f"<redacted {len(nested)} chars>"
            else:
                redacted[key] = _redact_responses_stream_item(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_responses_stream_item(item) for item in value]
    return value


def _redact_responses_log_line(line: str) -> str:
    if not line.startswith("data:"):
        return line
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return line
    try:
        item = json.loads(data)
    except json.JSONDecodeError:
        return line
    return "data: " + json.dumps(_redact_responses_stream_item(item), ensure_ascii=False)


def _stream_image(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path | None,
    final_path: Path,
    *,
    save_partials: bool,
) -> tuple[bytes, bool]:
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
        response = request.urlopen(req, timeout=600)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        _die(f"Codex Responses request failed with HTTP {exc.code}: {body[:2000]}")
    except error.URLError as exc:
        _die(f"Codex Responses request failed: {exc}")

    log_handle = None
    partial_count = 0
    last_partial: tuple[Path, bytes] | None = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
        for raw in response:
            line = raw.decode("utf-8", "ignore").rstrip("\n")
            if log_handle:
                log_handle.write(_redact_responses_log_line(line) + "\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                item = json.loads(data)
            except json.JSONDecodeError:
                continue
            if (
                save_partials
                and isinstance(item, dict)
                and item.get("type") == "response.image_generation_call.partial_image"
            ):
                partial_count += 1
                written_partial = _write_partial_image(item, final_path, partial_count)
                if written_partial:
                    last_partial = written_partial
                continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
                if last_partial:
                    partial_path, partial_bytes = last_partial
                    if partial_bytes == image_bytes:
                        partial_path.replace(final_path)
                        print(f"Renamed final partial {partial_path} to {final_path}")
                        return image_bytes, True
                return image_bytes, False
    finally:
        if log_handle:
            log_handle.close()
        response.close()
    _die("No generated image was found in the streamed response.")


def _create_codex_openai_client(token: str, account_id: str | None) -> Any:
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


def _image_response_bytes(response: Any) -> bytes:
    data = getattr(response, "data", None)
    if not data and isinstance(response, dict):
        data = response.get("data")
    if not data:
        _die("No image data was returned by the Codex Image API.")

    first = data[0]
    if isinstance(first, dict):
        image_b64 = first.get("b64_json")
    else:
        image_b64 = getattr(first, "b64_json", None)
    if not _looks_like_image_base64(image_b64):
        _die("Codex Image API response did not include b64_json image data.")
    return base64.b64decode(image_b64)


def _redact_image_api_log(value: Any) -> Any:
    plain = _to_plain_data(value)
    if isinstance(plain, dict):
        redacted = {}
        for key, nested in plain.items():
            if key in {"b64_json", "image_url"} and isinstance(nested, str):
                redacted[key] = f"<redacted {len(nested)} chars>"
            else:
                redacted[key] = _redact_image_api_log(nested)
        return redacted
    if isinstance(plain, list):
        return [_redact_image_api_log(item) for item in plain]
    return plain


def _write_image_api_log(log_path: Path, payload: dict[str, Any], response: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "request": _redact_image_api_log(payload),
                "response": _redact_image_api_log(response),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _stream_image_api_response(stream: Any, final_path: Path, *, save_partials: bool) -> bytes:
    partial_count = 0
    for event in stream:
        item = _to_plain_data(event)
        if save_partials and isinstance(item, dict):
            partial_b64 = item.get("partial_image_b64")
            if not partial_b64:
                partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
            if _looks_like_image_base64(partial_b64):
                partial_count += 1
                partial_path = _partial_output_path(final_path, partial_count)
                partial_path.write_bytes(base64.b64decode(partial_b64))
                print(f"Wrote partial {partial_path}")
                continue
        image_b64 = _scan_final_image_base64(item)
        if image_b64:
            return base64.b64decode(image_b64)
    _die("No generated image was found in the streamed Codex Image API response.")


def _stream_json_image_api_response(response: Any, final_path: Path, *, save_partials: bool) -> bytes:
    partial_count = 0
    try:
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
            if save_partials:
                partial_b64 = item.get("partial_image_b64") if isinstance(item, dict) else None
                if not partial_b64 and isinstance(item, dict):
                    partial_b64 = item.get("b64_json") if "partial" in str(item.get("type", "")) else None
                if _looks_like_image_base64(partial_b64):
                    partial_count += 1
                    partial_path = _partial_output_path(final_path, partial_count)
                    partial_path.write_bytes(base64.b64decode(partial_b64))
                    print(f"Wrote partial {partial_path}")
                    continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                return base64.b64decode(image_b64)
    finally:
        response.close()
    _die("No generated image was found in the streamed Codex Image API response.")


def _request_image_api_edit(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path,
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
            timeout=600,
        )
        if response.status_code >= 400:
            _die(
                f"Codex Image API edit request failed with HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )
        data = response.json()
        _write_image_api_log(log_path, payload, data)
    except Exception as exc:
        _die(f"Codex Image API edit request failed: {exc}")
    return _image_response_bytes(data)


def _run_image_api(
    args: argparse.Namespace,
    prompt: str,
    token: str,
    account_id: str | None,
    final_path: Path,
    log_path: Path,
) -> bytes:
    if _uses_image_edit_endpoint(args):
        payload = _build_image_api_edit_payload(args, prompt)
        return _request_image_api_edit(
            payload,
            token,
            account_id,
            log_path,
        )

    client = _create_codex_openai_client(token, account_id)
    options = _build_image_api_options(args, prompt)
    try:
        response = client.images.generate(**options)
    except Exception as exc:
        _die(f"Codex Image API request failed: {exc}")

    if options.get("stream"):
        return _stream_image_api_response(response, final_path, save_partials=bool(args.partial_images))
    _write_image_api_log(log_path, options, response)
    return _image_response_bytes(response)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate images through Codex auth with optional local reference images."
    )
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--reference", action="append", default=[], help="Local reference image path; repeat to attach multiple images.")
    parser.add_argument("--name", help="Human-readable filename suffix used after the UUID.")
    parser.add_argument("--copy-to", help="Optional project-local file or directory to receive a copied result.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the --copy-to target.")
    parser.add_argument(
        "--transport",
        default="image-api",
        help="Transport to use: image-api (default) or responses.",
    )
    parser.add_argument("--model", help="Model for the selected transport.")
    parser.add_argument("--image-model", help="Optional GPT Image model; for image-api this overrides --model.")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--size", help="Optional image size, such as auto, 1024x1024, or 2048x1152.")
    parser.add_argument("--quality", help="Optional rendering quality: low, medium, high, or auto.")
    parser.add_argument("--background", help="Optional background behavior: transparent, opaque, or auto.")
    parser.add_argument("--output-compression", type=int, help="Compression level 0-100 for JPEG and WebP outputs.")
    parser.add_argument("--moderation", help="Optional image moderation level: auto or low.")
    parser.add_argument("--action", help="Optional image tool action: generate, edit, or auto.")
    parser.add_argument("--partial-images", type=int, help="Number of streamed partial images to request, 0-3.")
    parser.add_argument("--input-fidelity", help="Optional input fidelity for models that allow explicit selection: high or low.")
    parser.add_argument("--mask", help="Optional local mask image for inpainting; requires at least one --reference.")
    parser.add_argument("--instructions")
    parser.add_argument("--auth-json", help="Path to Codex auth.json. When provided, this exact file overrides automatic discovery.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_tool_options(args)
    prompt = _read_prompt(args.prompt, args.prompt_file)
    out_path = _output_path(args.name or prompt, args.output_format, args.auth_json)

    if args.dry_run:
        if args.transport == "responses":
            payload = _build_payload(args, prompt)
            preview = {
                "endpoint": CODEX_RESPONSES_URL,
                "output": str(out_path),
                "log": str(_log_path(out_path)),
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
                **_redact_image_api_preview(args, options),
            }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    token, account_id = _read_codex_auth(args.auth_json)
    started = time.time()
    final_written = False
    if args.transport == "responses":
        payload = _build_payload(args, prompt)
        image_bytes, final_written = _stream_image(
            payload,
            token,
            account_id,
            _log_path(out_path),
            out_path,
            save_partials=bool(args.partial_images),
        )
    else:
        image_bytes = _run_image_api(
            args,
            prompt,
            token,
            account_id,
            out_path,
            _log_path(out_path),
        )
    if not final_written:
        out_path.write_bytes(image_bytes)
    print(f"Wrote {out_path}")
    if args.copy_to:
        copied = _copy_result(out_path, args.copy_to, force=args.force)
        print(f"Copied {copied}")
    print(f"Completed in {time.time() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
