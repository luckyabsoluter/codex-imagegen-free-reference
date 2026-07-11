# Codex ImageGen Free Reference

This project is an extension of the built-in `imagegen` skill in Codex. It introduces a Codex-auth direct path for handling local reference images, giving you explicit control over the generation process.

## Features

- **Reference Image Selection:** This tool gives you full control to explicitly choose your reference images, unlike the built-in tool which automatically manages them without manual selection.
- **Codex API Direct Integration:** This tool routes requests through the Codex base URL to ensure Codex-auth image models like `gpt-image-2` function correctly. The original fallback CLI relied on the standard OpenAI API, which is incompatible with these workflows.
- **Transport Selection:** This tool allows you to select the request path—either Codex Responses hosted `image_generation` or the Codex Image API generation/edit route—using the `--transport` flag. In contrast, the built-in tool is restricted to the Responses hosted-tool flow.
- **Model and Reasoning Selection:** This tool allows you to customize the image model and `reasoning.effort` (for Responses requests) using the `--model` and `--reasoning-effort` flags. The built-in tool relies on Codex's default settings (currently `gpt-5.5` with medium reasoning), which are subject to change.

> **Note:** Direct-mode original images and append-only redacted request/response logs are stored under `~/.codex/generated_images_free_reference/`. Outputs are copied from this directory to your project, which means saved project assets are intentionally duplicated.

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
