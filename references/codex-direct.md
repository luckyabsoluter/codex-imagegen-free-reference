# Codex API direct image generation path

Use this reference when the normal built-in `image_gen` tool is unavailable or when local reference-image files must be attached without exposing the agent to one-off Python snippets.

## Summary

The practical Codex-auth path defaults to the Codex Responses hosted-tool route through the OpenAI SDK:

```text
https://chatgpt.com/backend-api/codex/responses
```

It uses the same hosted-tool shape used by Codex itself: `ToolSpec::ImageGeneration` serializes to `{"type":"image_generation","output_format":"png"}` and is included in the Responses request. The payload looks like:

```json
{
  "stream": true,
  "tools": [{ "type": "image_generation", "output_format": "png" }]
}
```

The Codex Image API namespace remains available when explicitly requested with `--transport image-api`. Prompt-only generation goes to:

```text
https://chatgpt.com/backend-api/codex/images/generations
```

Requests with `--reference`, `--mask`, or `--action edit` go to:

```text
https://chatgpt.com/backend-api/codex/images/edits
```

For the default Responses transport and prompt-only Image API generation, the CLI uses the OpenAI SDK with:

```text
base_url="https://chatgpt.com/backend-api/codex"
```

and the Codex access token from `auth.json`. For image edits, the SDK's public multipart helper does not match this Codex endpoint, so the CLI sends the Codex JSON schema directly: `images: [{ "image_url": "data:image/..." }]`.

```text
python scripts/codex_image_gen.py --transport image-api ...
```

The legacy raw SSE Responses caller remains available only as a deprecated fallback:

```text
python scripts/codex_image_gen.py --transport responses-raw ...
```

The tool object can also carry optional image-generation controls:

```json
{
  "type": "image_generation",
  "output_format": "webp",
  "quality": "high",
  "size": "2048x1152",
  "background": "opaque",
  "output_compression": 80,
  "moderation": "low",
  "action": "generate",
  "partial_images": 2,
  "model": "gpt-image-2"
}
```

Important observations:

- The default Responses transport uses the SDK streaming interface; partial image previews can arrive before the final completed image. Save only the completed `image_generation_call.result` as the final artifact.
- The default Responses transport rejects non-streaming requests with `Stream must be set to true`.
- Local reference images are attached to the default Responses transport as `input_image` items using `data:image/...;base64,...` URLs.
- `--transport responses-raw` is deprecated and exists only to keep the previous direct SSE caller available for debugging.
- With `--transport image-api`, prompt-only generation uses the SDK image generation method, while edits use the direct Codex JSON schema.
- With `--transport image-api`, local reference images are attached to the edit endpoint as JSON `image_url` objects.
- Direct mode supports local mask images. The first `--reference` is the edit target when a mask is used.
- This path uses Codex auth automatically. If `--auth-json /path/to/auth.json` is provided, that exact auth file is used. Otherwise it checks `$CODEX_HOME/auth.json` first, then `~/.codex/auth.json`.
- If none of those auth files is available, the CLI exits with a configuration error. It does not require `OPENAI_API_KEY`.

## Why this path is recommended here

The default built-in image path is convenient when the harness exposes the `image_gen` tool, but it has two practical limits for this skill package:

1. It is a harness tool, not a stable script interface that this package can invoke directly.
2. Local filesystem reference images need to be made visible to that harness context first, which is not always available from reusable CLI documentation.

The fallback public OpenAI Image API CLI remains useful for explicit API-key workflows, but it is not the right shape for Codex auth:

- It calls the public `/v1/images/generations` or `/v1/images/edits` path through the public OpenAI base URL.
- It requires `OPENAI_API_KEY`.
- Passing a Codex `access_token` as `OPENAI_API_KEY` is not the native path and can fail, especially for multipart image edit uploads.

Therefore, for Codex-auth generation with local references, prefer:

```text
scripts/codex_image_gen.py
```

## Output policy

The Codex API direct CLI saves generated originals under the selected Codex home:

```text
<selected-codex-home>/generated_images_free_reference/
```

Selection order:

1. The parent directory of an explicit `--auth-json` path
2. `$CODEX_HOME`
3. `~/.codex`

File names use:

```text
<uuid>-<human-readable-name>.<ext>
```

Default log names append `.log` to the generated original name. Logs begin with start metadata for endpoint, transport, output path, and request payload. Image API logs are redacted JSON request/response records; Responses logs keep event structure while redacting image payloads:

```text
<uuid>-<human-readable-name>.<ext>.log
```

Examples:

```text
3d4a3d8c-9b8c-41fd-a79f-9866f5eb2e31-product-packshot.png
8e5f5fb7-3f61-42a4-a90c-358512e3b6c2-landing-page-hero.png
8e5f5fb7-3f61-42a4-a90c-358512e3b6c2-landing-page-hero.png.log
```

Project-local placement should be done by copying the generated original:

```text
$CODEX_HOME/generated_images_free_reference/<uuid>-<name>.png
→ ./output/imagegen/<name>.png
```

Use `--copy-to` for this. Do not make the Codex home output directory the project source of truth.

## Direct tool options

The Codex direct CLI exposes Image API controls without using `OPENAI_API_KEY`.

- `--model <gpt-image-model>`
- `--quality low|medium|high|auto`
- `--size auto|WIDTHxHEIGHT`
- `--output-format png|webp|jpeg`
- `--output-compression 0..100`, only with `webp` or `jpeg`
- `--background transparent|opaque|auto`
- `--moderation auto|low`
- `--action generate|edit|auto`
- `--partial-images 0..3`
- `--image-model <gpt-image-model>`, mapped to the tool-level `model` field for the default Responses transport and overriding `--model` for `--transport image-api`
- `--input-fidelity high|low`, for models that allow explicit input-fidelity selection
- `--mask <local-image>`, requiring at least one `--reference`
- `--hide-response-details`, suppressing last response event details in error messages while preserving redacted logs
- `--verbose`, printing debug details

Validation notes:
- The CLI accepts `auto` or `WIDTHxHEIGHT` for `--size` by default and lets the server decide model-specific support.
- When `--image-model` starts with `gpt-image-2`, the CLI also enforces the documented `gpt-image-2` hard constraints: max edge `<= 3840px`, both edges multiples of `16px`, long-to-short ratio `<= 3:1`, and total pixels between `655,360` and `8,294,400`.
- For other explicit image models, the CLI performs local syntax validation only.
- `--background transparent` requires `png` or `webp`, a transparency-capable image model, and not `gpt-image-2*`.
- `--input-fidelity` is rejected for `gpt-image-1-mini` and `gpt-image-2*`. For `gpt-image-2`, omit the flag because the model already processes every image input at high fidelity and the API does not allow changing it.
- `--partial-images` writes preview files next to the Codex-home original as `<final-stem>-partial-<index>.<ext>` when the selected transport streams previews. If the last partial image is byte-identical to the completed image, the CLI renames that partial file to the final output path instead of writing a duplicate; `--copy-to` copies only the completed final image.
- `--hide-response-details` prevents `Last event` and `Output item done` JSON from being printed into the caller context on failures; inspect the redacted log file when those details are needed.
- `--verbose` shows debug details; default output still reports generated originals, partial previews, and copy targets.
- The CLI writes `<final-path>.log` next to the Codex-home original. Logs start with endpoint, transport, output path, and request metadata. Image API logs redact base64 image payloads; Responses logs keep event structure while redacting image payloads.

## CLI examples

Use ordinary `python`. Do not document environment-manager-specific wrappers here; Python installation and environment selection are user-local concerns.

Generate without references:

```bash
python scripts/codex_image_gen.py \
  --prompt "A clean product packshot on a neutral studio background" \
  --quality high \
  --size 2048x1152 \
  --name product-packshot \
  --copy-to output/imagegen/product-packshot.png
```

Fast draft:

```bash
python scripts/codex_image_gen.py \
  --prompt "A clean product thumbnail on a neutral studio background" \
  --quality low \
  --size 1024x1024 \
  --name product-packshot-draft \
  --copy-to output/imagegen/product-packshot-draft.png
```

Compressed WebP output:

```bash
python scripts/codex_image_gen.py \
  --prompt "A polished landing-page hero image of a matte ceramic mug on a stone surface" \
  --quality high \
  --size 2048x1152 \
  --output-format webp \
  --output-compression 82 \
  --name mug-hero \
  --copy-to output/imagegen/mug-hero.webp
```

Generate with one local reference image:

```bash
python scripts/codex_image_gen.py \
  --reference ./references/product-photo.png \
  --prompt "Image A is the product reference. Create a clean studio packshot with soft lighting." \
  --quality high \
  --name product-packshot-reference \
  --copy-to output/imagegen/product-packshot-reference.png
```

Attach multiple references by repeating `--reference` in the exact order described by the prompt:

```bash
python scripts/codex_image_gen.py \
  --reference ./references/product-photo.png \
  --reference ./references/brand-style.png \
  --prompt "Image A is the product reference. Image B is only for visual style and color direction." \
  --name product-brand-style \
  --copy-to output/imagegen/product-brand-style.png
```

Masked edit with an explicit image model:

```bash
python scripts/codex_image_gen.py \
  --reference ./references/product-photo.png \
  --mask ./references/product-mask.png \
  --image-model gpt-image-1.5 \
  --action edit \
  --input-fidelity high \
  --prompt "Change only the masked background area to a clean warm studio backdrop. Keep the product unchanged." \
  --name product-masked-edit \
  --copy-to output/imagegen/product-masked-edit.png
```

Stream partial previews:

```bash
python scripts/codex_image_gen.py \
  --prompt "A detailed architectural visualization at golden hour" \
  --quality high \
  --size 2048x1152 \
  --partial-images 2 \
  --name architecture-preview \
  --copy-to output/imagegen/architecture-preview.png
```

Use the optional Image API transport only when the generation/edit namespace path is required:

```bash
python scripts/codex_image_gen.py \
  --transport image-api \
  --prompt "A detailed architectural visualization at golden hour" \
  --quality high \
  --size 2048x1152 \
  --partial-images 2 \
  --name architecture-preview \
  --copy-to output/imagegen/architecture-preview.png
```

Dry-run without network or auth validation:

```bash
python scripts/codex_image_gen.py \
  --reference ./references/product-photo.png \
  --prompt "Image A is the product reference." \
  --quality high \
  --size 2048x1152 \
  --name dry-run-example \
  --dry-run
```

## Agent guidance

- Prefer this CLI over ad-hoc embedded Python when using Codex auth and local references.
- Keep prompts in `--prompt` or `--prompt-file`; do not generate temporary runner scripts for routine jobs.
- Do not print auth token values.
- Do not commit generated images or generation logs unless the user explicitly asks.
- Copy selected project-bound outputs into the project path with `--copy-to`.
