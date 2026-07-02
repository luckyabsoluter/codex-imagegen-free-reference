#!/usr/bin/env python3
"""Codex-auth image generation with free local reference images.

This CLI calls the Codex Responses endpoint directly with the built-in
`image_generation` tool. It uses the local Codex auth snapshot in
`~/.codex/auth.json`, accepts local reference images as `input_image` data URLs,
and saves generated images under `~/.codex/generated_images_free_reference/` by
default.
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


CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_BETA_HEADER = "responses=2025-06-21"
FINAL_IMAGE_KEYS = {"result", "image", "b64_json"}
ALLOWED_ACTIONS = {"generate", "edit", "auto"}
ALLOWED_BACKGROUNDS = {"transparent", "opaque", "auto"}
ALLOWED_INPUT_FIDELITIES = {"high", "low"}
ALLOWED_MODERATIONS = {"auto", "low"}
ALLOWED_OUTPUT_FORMATS = {"png", "webp", "jpeg"}
ALLOWED_QUALITIES = {"low", "medium", "high", "auto"}
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
    _validate_size(args.size, args.image_model)

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
        if not args.image_model:
            _die("--background transparent requires an explicit --image-model that supports transparency.")
        if _is_gpt_image_2_model(args.image_model):
            _die("transparent backgrounds are not supported by gpt-image-2 image models.")

    if args.mask and not args.reference:
        _die("--mask requires at least one --reference image; the first reference is the edit target.")

    if args.input_fidelity:
        if not args.image_model:
            _die("--input-fidelity requires an explicit --image-model.")
        if (
            args.image_model in UNSUPPORTED_INPUT_FIDELITY_IMAGE_MODELS
            or _is_gpt_image_2_model(args.image_model)
        ):
            _die(f"--input-fidelity is not supported by {args.image_model}.")


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
    output_dir = _output_root(auth_json) / "generated_images_free_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_format.lower().lstrip(".") or DEFAULT_OUTPUT_FORMAT
    stem = f"{uuid4()}-{_slugify(name or 'image')}"
    return output_dir / f"{stem}.{suffix}"


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


def _write_partial_image(item: dict[str, Any], final_path: Path, fallback_index: int) -> bool:
    image_b64 = item.get("partial_image_b64")
    if not _looks_like_image_base64(image_b64):
        return False
    partial_path = _partial_output_path(final_path, _partial_index(item, fallback_index))
    partial_path.write_bytes(base64.b64decode(image_b64))
    print(f"Wrote partial {partial_path}")
    return True


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
        "model": args.model,
        "instructions": args.instructions or "",
        "input": [{"role": "user", "content": content}],
        "tools": [_build_image_tool(args)],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": True,
        "store": False,
        "include": [],
    }


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


def _stream_image(
    payload: dict[str, Any],
    token: str,
    account_id: str | None,
    log_path: Path | None,
    final_path: Path,
    *,
    save_partials: bool,
) -> bytes:
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
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
        for raw in response:
            line = raw.decode("utf-8", "ignore").rstrip("\n")
            if log_handle:
                log_handle.write(line + "\n")
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
                _write_partial_image(item, final_path, partial_count)
                continue
            image_b64 = _scan_final_image_base64(item)
            if image_b64:
                return base64.b64decode(image_b64)
    finally:
        if log_handle:
            log_handle.close()
        response.close()
    _die("No generated image was found in the streamed response.")


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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-model", help="Optional GPT Image model for the image_generation tool.")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--size", help="Optional image size, such as auto, 1024x1024, or 2048x1152.")
    parser.add_argument("--quality", help="Optional rendering quality: low, medium, high, or auto.")
    parser.add_argument("--background", help="Optional background behavior: transparent, opaque, or auto.")
    parser.add_argument("--output-compression", type=int, help="Compression level 0-100 for JPEG and WebP outputs.")
    parser.add_argument("--moderation", help="Optional image moderation level: auto or low.")
    parser.add_argument("--action", help="Optional image tool action: generate, edit, or auto.")
    parser.add_argument("--partial-images", type=int, help="Number of streamed partial images to request, 0-3.")
    parser.add_argument("--input-fidelity", help="Optional input fidelity for supported explicit image models: high or low.")
    parser.add_argument("--mask", help="Optional local mask image for inpainting; requires at least one --reference.")
    parser.add_argument("--instructions")
    parser.add_argument("--auth-json", help="Path to Codex auth.json. When provided, this exact file overrides automatic discovery.")
    parser.add_argument("--log", help="Optional path for the raw SSE log. Do not commit logs unless reviewed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_tool_options(args)
    prompt = _read_prompt(args.prompt, args.prompt_file)
    out_path = _output_path(args.name or prompt, args.output_format, args.auth_json)
    payload = _build_payload(args, prompt)

    if args.dry_run:
        preview = _redact_preview_payload(payload)
        print(json.dumps({"endpoint": CODEX_RESPONSES_URL, "output": str(out_path), **preview}, ensure_ascii=False, indent=2))
        return 0

    token, account_id = _read_codex_auth(args.auth_json)
    started = time.time()
    image_bytes = _stream_image(
        payload,
        token,
        account_id,
        Path(args.log) if args.log else None,
        out_path,
        save_partials=bool(args.partial_images),
    )
    out_path.write_bytes(image_bytes)
    print(f"Wrote {out_path}")
    if args.copy_to:
        copied = _copy_result(out_path, args.copy_to, force=args.force)
        print(f"Copied {copied}")
    print(f"Completed in {time.time() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
