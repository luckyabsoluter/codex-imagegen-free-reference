from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import codex_image_gen as image_gen


class ExecutionMetadataTests(unittest.TestCase):
    def parse(self, argv: list[str]) -> image_gen.RequestConfig:
        return image_gen.Cli().parse_config(argv)

    def test_empty_inputs_and_complete_argv(self) -> None:
        argv = ["--prompt", 'A prompt with "quotes" and spaces']
        invocation, inputs = image_gen.ExecutionMetadata.build(argv, self.parse(argv))
        expected_argv = [sys.executable, str((ROOT / "scripts" / "codex_image_gen.py").resolve()), *argv]

        self.assertEqual(invocation["cwd"], str(Path.cwd().resolve()))
        self.assertEqual(invocation["argv"], expected_argv)
        self.assertEqual(inputs, {"references": [], "mask": None})
        expected_command = subprocess.list2cmdline(expected_argv) if os.name == "nt" else shlex.join(expected_argv)
        self.assertEqual(invocation["command"], expected_command)

    def test_reference_order_and_resolved_paths(self) -> None:
        argv = [
            "--prompt",
            "Use the attached images",
            "--reference",
            "assets/imagegen.png",
            "--reference",
            "reference images/style sample.png",
            "--mask",
            "masks/edit area.png",
        ]
        _, inputs = image_gen.ExecutionMetadata.build(argv, self.parse(argv))
        cwd = Path.cwd().resolve()

        self.assertEqual(
            inputs["references"],
            [
                {
                    "index": 1,
                    "path": "assets/imagegen.png",
                    "resolved_path": str((cwd / "assets/imagegen.png").resolve()),
                },
                {
                    "index": 2,
                    "path": "reference images/style sample.png",
                    "resolved_path": str((cwd / "reference images/style sample.png").resolve()),
                },
            ],
        )
        self.assertEqual(
            inputs["mask"],
            {
                "path": "masks/edit area.png",
                "resolved_path": str((cwd / "masks/edit area.png").resolve()),
            },
        )


class StartLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = image_gen.Logging()
        self.invocation = {"cwd": "workspace", "command": "python script.py", "argv": ["python", "script.py"]}
        self.inputs = {
            "references": [{"index": 1, "path": "source.png", "resolved_path": "workspace/source.png"}],
            "mask": None,
        }
        self.payload = {
            "input": [{"content": [{"type": "input_image", "image_url": "data:image/png;base64,secret"}]}],
            "model": "gpt-5.5",
        }

    def start_info(self, transport: str) -> dict[str, object]:
        return self.logger.start_info(
            endpoint="https://example.invalid/responses",
            transport=transport,
            final_path=Path("output.png"),
            invocation=self.invocation,
            inputs=self.inputs,
            request_payload=self.payload,
            timeout_seconds=600,
        )

    def test_start_metadata_is_shared_across_transports(self) -> None:
        for transport in ("responses", "responses-raw", "image-api"):
            with self.subTest(transport=transport):
                start = self.start_info(transport)
                self.assertEqual(start["invocation"], self.invocation)
                self.assertEqual(start["inputs"], self.inputs)
                self.assertIs(start["request"], self.payload)

    def test_responses_log_keeps_request_and_redacts_image_data(self) -> None:
        handle = StringIO()
        self.logger.write_responses_log_event(handle, "codex_image_gen.start", self.start_info("responses"))
        data_line = next(line for line in handle.getvalue().splitlines() if line.startswith("data: "))
        record = json.loads(data_line.removeprefix("data: "))

        self.assertEqual(record["invocation"], self.invocation)
        self.assertEqual(record["inputs"], self.inputs)
        self.assertEqual(record["request"]["model"], "gpt-5.5")
        self.assertTrue(record["request"]["input"][0]["content"][0]["image_url"].startswith("<redacted "))
        self.assertNotIn("access_token", json.dumps(record))

    def test_image_api_log_keeps_request_and_redacts_image_data(self) -> None:
        handle = StringIO()
        self.logger.write_image_api_log_event(handle, "codex_image_gen.start", self.start_info("image-api"))
        record = json.loads(handle.getvalue())

        self.assertEqual(record["event"], "codex_image_gen.start")
        self.assertEqual(record["data"]["invocation"], self.invocation)
        self.assertEqual(record["data"]["inputs"], self.inputs)
        self.assertEqual(record["data"]["request"]["model"], "gpt-5.5")
        self.assertTrue(record["data"]["request"]["input"][0]["content"][0]["image_url"].startswith("<redacted "))
        self.assertNotIn("access_token", json.dumps(record))

    def test_image_api_log_appends_start_response_and_cli_records(self) -> None:
        log_path = ROOT / f"codex-image-gen-{uuid.uuid4()}.test.log"
        self.addCleanup(log_path.unlink, missing_ok=True)
        existing_record = {"logged_at": "earlier", "event": "existing", "data": {"preserved": True}}
        with log_path.open("x", encoding="utf-8") as log_handle:
            log_handle.write(json.dumps(existing_record) + "\n")

        self.logger.configure(log_path, "image-jsonl")
        self.logger.write_image_api_start_log(log_path, self.start_info("image-api"))
        self.logger.log_cli_message("info", "Request completed")
        self.logger.write_image_api_response_log(
            log_path,
            {"data": [{"b64_json": "secret" * 500}]},
            status="completed",
        )

        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0], existing_record)
        self.assertEqual(
            [record["event"] for record in records[1:]],
            ["codex_image_gen.start", "codex_image_gen.info", "image_api.response"],
        )
        self.assertEqual(records[3]["data"]["status"], "completed")
        self.assertTrue(records[3]["data"]["response"]["data"][0]["b64_json"].startswith("<redacted "))


class CliTests(unittest.TestCase):
    def test_dry_run_remains_network_free(self) -> None:
        argv = ["--prompt", "A dry test", "--dry-run"]
        cli = image_gen.Cli()
        stdout = StringIO()

        with (
            mock.patch.object(image_gen.Paths, "output_path", return_value=Path("generated.png")),
            redirect_stdout(stdout),
        ):
            result = cli.main(argv)

        preview = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(preview["transport"], "responses")
        self.assertEqual(preview["output"], "generated.png")

    def test_main_preserves_explicit_argv_before_parsing(self) -> None:
        argv = ["--prompt", "A dry test", "--name", "dry-test"]
        cli = image_gen.Cli()

        with (
            mock.patch.object(image_gen.Paths, "output_path", return_value=Path("generated.png")),
            mock.patch.object(cli, "execute", return_value=0) as execute,
        ):
            result = cli.main(argv)

        self.assertEqual(result, 0)
        self.assertEqual(execute.call_args.args[3], argv)
        self.assertIsNot(execute.call_args.args[3], argv)


if __name__ == "__main__":
    unittest.main()
