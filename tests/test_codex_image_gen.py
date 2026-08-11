from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import shutil
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


class TimezoneOffsetTests(unittest.TestCase):
    def test_supported_timezone_formats(self) -> None:
        expected_offsets = {
            "1:30": timedelta(hours=1, minutes=30),
            "01:00": timedelta(hours=1),
            "1": timedelta(hours=1),
            "+01:00": timedelta(hours=1),
            "-01:00": timedelta(hours=-1),
        }
        
        for value, expected in expected_offsets.items():
            with self.subTest(value=value):
                self.assertEqual(image_gen.Paths.parse_timezone_offset(value), expected)
    
    def test_invalid_timezone_formats_are_rejected(self) -> None:
        for value in ("1.5", "1:60", "14:01", "-12:01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_gen.Paths.parse_timezone_offset(value)


class OutputPathTests(unittest.TestCase):
    DATE_FOLDER = "2026-08-10"
    
    def setUp(self) -> None:
        self.root = ROOT / f"codex-output-path-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    
    def patch_date(self) -> None:
        date_patch = mock.patch.object(image_gen, "date")
        mocked_date = date_patch.start()
        mocked_date.today.return_value.isoformat.return_value = self.DATE_FOLDER
        self.addCleanup(date_patch.stop)
    
    def test_output_path_uses_local_date_directory(self) -> None:
        self.patch_date()
        with mock.patch.object(image_gen.Paths, "output_root", return_value=self.root):
            output_path = image_gen.Paths.output_path("Dated image", "png", None)
        
        expected_dir = self.root / "generated_images_free_reference" / self.DATE_FOLDER
        self.assertEqual(output_path.parent, expected_dir)
        self.assertTrue(expected_dir.is_dir())
        self.assertEqual(image_gen.Paths.log_path(output_path).parent, expected_dir)
        self.assertEqual(image_gen.Output.partial_output_path(output_path, 1).parent, expected_dir)
    
    def test_output_path_preserves_name_case(self) -> None:
        self.patch_date()
        with mock.patch.object(image_gen.Paths, "output_root", return_value=self.root):
            output_path = image_gen.Paths.output_path("BrandLogo", "png", None)
        
        self.assertTrue(output_path.name.endswith("-BrandLogo.png"))
    
    def test_output_path_uses_configured_timezone(self) -> None:
        with (
            mock.patch.object(image_gen.Paths, "output_root", return_value=self.root),
            mock.patch.object(image_gen, "datetime") as mocked_datetime,
        ):
            mocked_datetime.now.return_value.date.return_value.isoformat.return_value = self.DATE_FOLDER
            output_path = image_gen.Paths.output_path("Dated image", "png", None, "1:30")
        
        self.assertEqual(
            output_path.parent,
            self.root / "generated_images_free_reference" / self.DATE_FOLDER,
        )
        selected_timezone = mocked_datetime.now.call_args.args[0]
        self.assertEqual(selected_timezone.utcoffset(None), timedelta(hours=1, minutes=30))
    
    def test_output_name_prefers_explicit_name(self) -> None:
        output_name = image_gen.Paths.resolve_output_name(
            "explicit-name",
            "output/copy-target.png",
            "Prompt fallback",
        )
        
        self.assertEqual(output_name, "explicit-name")
    
    def test_output_name_uses_copy_target_stem(self) -> None:
        output_name = image_gen.Paths.resolve_output_name(
            None,
            "output/copy-target.png",
            "Prompt fallback",
        )
        
        self.assertEqual(output_name, "copy-target")
    
    def test_output_name_falls_back_to_prompt_without_file_copy_target(self) -> None:
        copy_directory = self.root / "copy-output"
        copy_directory.mkdir()
        
        self.assertEqual(
            image_gen.Paths.resolve_output_name(None, None, "Prompt fallback"),
            "Prompt fallback",
        )
        self.assertEqual(
            image_gen.Paths.resolve_output_name(None, str(copy_directory), "Prompt fallback"),
            "Prompt fallback",
        )
    
    def test_output_directory_falls_back_when_date_path_is_a_file(self) -> None:
        self.patch_date()
        base_output_dir = self.root / "generated_images_free_reference"
        base_output_dir.mkdir()
        (base_output_dir / self.DATE_FOLDER).write_text("occupied", encoding="utf-8")
        
        with mock.patch.object(image_gen.Paths, "output_root", return_value=self.root):
            output_dir = image_gen.Paths.output_dir(None)
        
        self.assertEqual(output_dir, base_output_dir)
    
    def test_output_directory_falls_back_when_date_directory_creation_fails(self) -> None:
        self.patch_date()
        base_output_dir = self.root / "generated_images_free_reference"
        dated_output_dir = base_output_dir / self.DATE_FOLDER
        original_mkdir = Path.mkdir
        
        def mkdir(path: Path, *args: object, **kwargs: object) -> None:
            if path == dated_output_dir:
                raise PermissionError("date directory is unavailable")
            original_mkdir(path, *args, **kwargs)
        
        with (
            mock.patch.object(image_gen.Paths, "output_root", return_value=self.root),
            mock.patch.object(Path, "mkdir", autospec=True, side_effect=mkdir),
        ):
            output_dir = image_gen.Paths.output_dir(None)
        
        self.assertEqual(output_dir, base_output_dir)


class CodexModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = image_gen.Logging()
        self.root = ROOT / f"codex-model-resolution-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    
    def write_cache(self, models: list[object]) -> None:
        (self.root / "models_cache.json").write_text(
            json.dumps({"models": models}),
            encoding="utf-8",
        )
    
    def test_explicit_model_does_not_read_codex_cache(self) -> None:
        with mock.patch.object(image_gen.Paths, "output_root", side_effect=AssertionError("cache read")):
            model = image_gen.CodexModels.resolve("gpt-explicit", None, self.logger)
        
        self.assertEqual(model, "gpt-explicit")
    
    def test_omitted_model_uses_highest_priority_visible_cache_entry(self) -> None:
        self.write_cache(
            [
                {"slug": "gpt-hidden", "priority": 0, "visibility": "hide"},
                {"slug": "gpt-secondary", "priority": 2, "visibility": "list"},
                {"slug": "gpt-primary", "priority": 1, "visibility": "list"},
            ]
        )
        
        with mock.patch.object(image_gen.Paths, "output_root", return_value=self.root):
            model = image_gen.CodexModels.resolve(None, None, self.logger)
        
        self.assertEqual(model, "gpt-primary")
    
    def test_omitted_model_requires_cache_or_explicit_override(self) -> None:
        stderr = StringIO()
        with (
            mock.patch.object(image_gen.Paths, "output_root", return_value=self.root),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            image_gen.CodexModels.resolve(None, None, self.logger)
        
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Pass --model", stderr.getvalue())


class ResponsesPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = image_gen.Logging()
    
    def test_omitted_model_is_resolved_before_building_request(self) -> None:
        config = image_gen.Cli(self.logger).parse_config(["--prompt", "A test"])
        with mock.patch.object(image_gen.CodexModels, "resolve", return_value="gpt-cached") as resolve:
            payload = image_gen.Payloads.build_responses_payload(config, "A test", self.logger)
        
        self.assertEqual(payload["model"], "gpt-cached")
        self.assertNotIn("reasoning", payload)
        resolve.assert_called_once_with(None, None, self.logger)


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
            "model": "test-model",
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
        self.assertEqual(record["request"]["model"], "test-model")
        self.assertTrue(record["request"]["input"][0]["content"][0]["image_url"].startswith("<redacted "))
        self.assertNotIn("access_token", json.dumps(record))

    def test_image_api_log_keeps_request_and_redacts_image_data(self) -> None:
        handle = StringIO()
        self.logger.write_image_api_log_event(handle, "codex_image_gen.start", self.start_info("image-api"))
        record = json.loads(handle.getvalue())

        self.assertEqual(record["event"], "codex_image_gen.start")
        self.assertEqual(record["data"]["invocation"], self.invocation)
        self.assertEqual(record["data"]["inputs"], self.inputs)
        self.assertEqual(record["data"]["request"]["model"], "test-model")
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
        argv = ["--prompt", "A dry test", "--timezone", "1:30", "--dry-run"]
        cli = image_gen.Cli()
        stdout = StringIO()

        with (
            mock.patch.object(image_gen.Paths, "output_path", return_value=Path("generated.png")),
            mock.patch.object(image_gen.CodexModels, "resolve", return_value="gpt-cached"),
            redirect_stdout(stdout),
        ):
            result = cli.main(argv)

        preview = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(preview["transport"], "responses")
        self.assertEqual(preview["output"], "generated.png")
        self.assertEqual(preview["model"], "gpt-cached")
        self.assertEqual(preview["timezone"], "1:30")

    def test_supported_timezone_cli_formats_are_preserved(self) -> None:
        cli = image_gen.Cli()
        
        for value in ("1:30", "01:00", "1", "+01:00", "-01:00"):
            with self.subTest(value=value):
                config = cli.parse_config(["--prompt", "A test", "--timezone", value])
                self.assertEqual(config.timezone, value)

    def test_invalid_timezone_is_rejected_before_output_path_selection(self) -> None:
        argv = ["--prompt", "A dry test", "--timezone", "1.5", "--dry-run"]
        cli = image_gen.Cli()
        stderr = StringIO()
        
        with (
            mock.patch.object(image_gen.Paths, "output_path") as output_path,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main(argv)
        
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("--timezone must be a UTC offset from -12:00 through +14:00", stderr.getvalue())
        output_path.assert_not_called()

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

    def test_main_passes_timezone_to_output_path(self) -> None:
        argv = ["--prompt", "A dry test", "--timezone", "+01:00"]
        cli = image_gen.Cli()
        
        with (
            mock.patch.object(image_gen.Paths, "output_path", return_value=Path("generated.png")) as output_path,
            mock.patch.object(cli, "execute", return_value=0),
        ):
            result = cli.main(argv)
        
        self.assertEqual(result, 0)
        output_path.assert_called_once_with("A dry test", "png", None, "+01:00")
    
    def test_main_uses_copy_target_stem_when_name_is_omitted(self) -> None:
        argv = ["--prompt", "A dry test", "--copy-to", "output/copy-target.png"]
        cli = image_gen.Cli()
        
        with (
            mock.patch.object(image_gen.Paths, "output_path", return_value=Path("generated.png")) as output_path,
            mock.patch.object(cli, "execute", return_value=0),
        ):
            result = cli.main(argv)
        
        self.assertEqual(result, 0)
        output_path.assert_called_once_with("copy-target", "png", None, None)


if __name__ == "__main__":
    unittest.main()
