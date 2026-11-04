# Known issues

## RESP-001: Responses reasoning effort causes image complexity errors

Status: Open.

On `/codex/responses`, `gpt-6-luna` accepted explicit reasoning settings but image generation failed with `invalid_value`, parameter `complexity`: `Complexity '<value>' is not supported for this image model.` The returned image tool model was `gpt-image-2-codex`. The internal forwarding mechanism is unconfirmed; other routes and models are outside this finding.

| Request | Observed result |
| --- | --- |
| reasoning omitted | Final image + 3 partials; returned default effort: medium |
| effort none, low, medium, high, xhigh, max | Complexity error for every explicit value |
| JSON effort null | Final image + 2 partials; returned default effort: medium |
| effort none + tool complexity high | Complexity high echoed, then rejected |

**Workaround:** omit `--reasoning-effort`. Keep the option for explicit caller requests and rechecks; do not silently drop or replace supplied values. Explicit medium is not equivalent to omission. JSON null was a diagnostic case, not the CLI string `null`. Project CLI generation, original saving, and copying succeeded with omission.

### Recheck procedure

After backend, model, or SDK changes, use one fixed prompt with `--transport responses --model gpt-6-luna --partial-images 3`. Compare omission against each explicit effort above; require completed final images, not just previews. Fewer than 3 partials can be normal.

Append the route/model/SDK versions, sent settings, returned tool model/effort, partial count, final status, and error code/parameter to the history. Keep redacted logs. An unrelated baseline failure is inconclusive. Close the issue and update linked guidance only after repeat checks verify supported explicit efforts; record partial recovery without discarding earlier results.

### Verification history

| Outcome | Follow-up |
| --- | --- |
| Omission/null succeeded; six explicit efforts failed | Keep option and omission workaround; recheck after relevant changes |
