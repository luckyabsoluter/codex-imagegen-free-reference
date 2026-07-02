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
IMAGE_KEYS = {"result", "image", "b64_json", "partial_image_b64"}


def _die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def _copy_result(source: Path, destination: str, *, force: bool) -> Path:
    target = Path(destination)
    if target.exists() and target.is_dir():
        target = target / source.name
    if target.exists() and not force:
        _die(f"Copy target already exists: {target} (use --force to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _scan_image_base64(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in IMAGE_KEYS and isinstance(nested, str) and len(nested) > 1000:
                return nested
            found = _scan_image_base64(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _scan_image_base64(nested)
            if found:
                return found
    return None


def _build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    content: list[dict[str, str]] = []
    for reference in args.reference:
        content.append({"type": "input_image", "image_url": _data_url(Path(reference))})
    content.append({"type": "input_text", "text": prompt})

    return {
        "model": args.model,
        "instructions": args.instructions or "",
        "input": [{"role": "user", "content": content}],
        "tools": [{"type": "image_generation", "output_format": args.output_format}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": True,
        "store": False,
        "include": [],
    }


def _stream_image(payload: dict[str, Any], token: str, account_id: str | None, log_path: Path | None) -> bytes:
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
            image_b64 = _scan_image_base64(item)
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
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT, choices=["png", "webp", "jpeg"])
    parser.add_argument("--instructions")
    parser.add_argument("--auth-json", help="Path to Codex auth.json. When provided, this exact file overrides automatic discovery.")
    parser.add_argument("--log", help="Optional path for the raw SSE log. Do not commit logs unless reviewed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = _read_prompt(args.prompt, args.prompt_file)
    out_path = _output_path(args.name or prompt, args.output_format, args.auth_json)
    payload = _build_payload(args, prompt)

    if args.dry_run:
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
        print(json.dumps({"endpoint": CODEX_RESPONSES_URL, "output": str(out_path), **preview}, ensure_ascii=False, indent=2))
        return 0

    token, account_id = _read_codex_auth(args.auth_json)
    started = time.time()
    image_bytes = _stream_image(payload, token, account_id, Path(args.log) if args.log else None)
    out_path.write_bytes(image_bytes)
    print(f"Wrote {out_path}")
    if args.copy_to:
        copied = _copy_result(out_path, args.copy_to, force=args.force)
        print(f"Copied {copied}")
    print(f"Completed in {time.time() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
