# Codex ImageGen Free Reference

This project is an extension of the built-in `imagegen` skill in Codex. It introduces a Codex-auth direct path for handling local reference images, giving you explicit control over the generation process.

## Features

- **Reference Image Selection:** This tool gives you full control to explicitly choose your reference images, unlike the built-in tool which automatically manages them without manual selection.
- **Codex API Direct Integration:** This tool routes requests through the Codex base URL to ensure Codex-auth image models like `gpt-image-2` function correctly. The original fallback CLI relied on the standard OpenAI API, which is incompatible with these workflows.
- **Transport Selection:** This tool allows you to select the request path—either Codex Responses hosted `image_generation` or the Codex Image API generation/edit route—using the `--transport` flag. In contrast, the built-in tool is restricted to the Responses hosted-tool flow.
- **Model and Reasoning Selection:** This tool allows you to customize the Responses model, image-generation model, and `reasoning.effort` using `--model`, `--image-model`, and `--reasoning-effort`. When `--model` is omitted for Responses, the direct CLI follows the selected Codex home's current model cache instead of pinning a model in this project. Codex's documented default Power setting is currently `gpt-5.6-sol` with medium reasoning, and it is subject to change.
- **Output Timezone Selection:** This tool accepts fixed UTC offsets with `--timezone`. Supported examples include `1:30`, `01:00`, `1`, `+01:00`, and `-01:00`, while omitting the option preserves the runtime-local date.

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
