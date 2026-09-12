# Codex ImageGen Free Reference

This project is an extension of the built-in `imagegen` skill in Codex. It introduces a Codex-auth direct path for handling local reference images, giving you explicit control over the generation process.

## Features

- **Reference Image Selection:** This tool gives you full control to explicitly choose your reference images, unlike the built-in tool which automatically manages them without manual selection.
- **Codex API Direct Integration:** This tool routes requests through the Codex base URL to ensure Codex-auth image models like `gpt-image-2` function correctly. The original fallback CLI relied on the standard OpenAI API, which is incompatible with these workflows.
- **Transport Selection:** This tool allows you to select the request path—either Codex Responses hosted `image_generation` or the Codex Image API generation/edit route—using the `--transport` flag. In contrast, the built-in tool is restricted to the Responses hosted-tool flow.
- **Optional Low-memory Mode:** Add `--low-memory` to use Responses streaming without loading the OpenAI SDK. This mode releases uploaded reference data before reading results and saves images in small decoded chunks. The default Responses path continues to use the SDK.
- **Model and Reasoning Selection:** This tool allows you to customize the Responses model, image-generation model, and `reasoning.effort` using `--model`, `--image-model`, and `--reasoning-effort`. When `--model` is omitted for Responses, the direct CLI follows the selected Codex home's current model cache instead of pinning a model in this project. Codex's documented default Power setting is currently `gpt-5.6-sol` with medium reasoning, and it is subject to change.
- **Output Timezone Selection:** This tool accepts fixed UTC offsets with `--timezone`. Supported examples include `1:30`, `01:00`, `1`, `+01:00`, and `-01:00`, while omitting the option preserves the runtime-local date.
- **Partial JPEG Previews:** Save streamed partial images as JPEG with independent compression quality and a white, custom-color, or checkerboard transparency background.

> **Note:** Direct-mode original images and append-only redacted request/response logs are stored under date directories at `~/.codex/generated_images_free_reference/<YYYY-MM-DD>/`, based on `--timezone` when provided or the runtime's local date otherwise. If a date directory cannot be created, the files are stored directly under `~/.codex/generated_images_free_reference/`. Outputs are copied from this directory tree to your project, which means saved project assets are intentionally duplicated.

## Install

```bash
cd ~/.codex/skills
git clone <repo-url> codex-imagegen-free-reference
```

Restart Codex after installation.

## Usage

To generate or edit an image for your current project, simply use the following command in Codex:

```text
Use $codex-imagegen-free-reference to make or edit an image for this project.
```

Alternatively, you can use the direct CLI tool provided at:
`scripts/codex_image_gen.py`

To enable low-memory mode, add `--low-memory` to your usual Responses command:

```bash
.venv/bin/python scripts/codex_image_gen.py --prompt "A mountain landscape" --low-memory
```

On Windows, use `.venv\Scripts\python.exe` as the interpreter.

### Partial JPEG previews

Use `--partial-output-format jpg` with `--partial-images 1..3` to encode previews locally as `.jpg` files. This requires Pillow in the CLI's Python environment (`python -m pip install Pillow`).

```bash
.venv/bin/python scripts/codex_image_gen.py --prompt "A mountain landscape" --partial-images 3 --partial-output-format jpg --partial-output-compression 85 --partial-background checkerboard
```

- `--partial-output-compression 0..100` controls JPEG quality independently of the API's `--output-compression`. The default is `90`; higher values produce better quality and generally larger files.
- `--partial-background white` is the default. Use a color name such as `navy`, a quoted HEX value such as `"#e8eef5"`, or `checkerboard` for white and gray squares. The background fills transparent areas; opaque pixels keep their original colors.
- The API's `--output-format`, `--output-compression`, and `--background` still control image generation. Partial JPEG options apply only to local previews, including in `--low-memory` mode. Image API edits do not support partial streaming.
- Previews are saved beside the original as `<final-stem>-partial-<index>.jpg`. The completed image retains the API's original bytes, and `--copy-to` copies only that completed image.
