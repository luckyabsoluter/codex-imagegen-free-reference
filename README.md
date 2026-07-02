# Codex ImageGen Free Reference

This project is an extension of the built-in `imagegen` skill in Codex. It introduces a Codex-auth direct path for handling local reference images, giving you explicit control over the generation process.

## Features

- **Reference Image Selection:** This tool gives you full control to explicitly choose your reference image, unlike the built-in tool which automatically manages them without manual selection.
- **Codex API Direct Integration:** The original fallback CLI tool relied on the standard OpenAI API. However, for models like `gpt-image-2` to function correctly, requests must be routed through Codex. This tool achieves this by sending requests to the Codex base URL instead of the OpenAI API base URL, ensuring the model operates precisely as intended.

> **Note:** Direct-mode original images are stored under `~/.codex/generated_images_free_reference/`. Outputs are copied from this directory to your project, which means saved project assets are intentionally duplicated.

## Install

```bash
cd ~/.codex
git clone <repo-url> skills/codex-imagegen-free-reference
```

Restart Codex after installation.

## Usage

To generate or edit an image for your current project, simply use the following command in Codex:

```text
Use $codex-imagegen-free-reference to make or edit an image for this project.
```

Alternatively, you can use the direct CLI tool provided at:
`scripts/codex_image_gen.py`
