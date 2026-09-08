from __future__ import annotations

import base64
import binascii
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import sys
import time
import tracemalloc
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import codex_image_gen as image_gen


class ResponsesModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / f"responses-modes-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.final = self.root / "image.png"
        self.image = b"completed-image" * 1000
        self.encoded = base64.b64encode(self.image).decode()
        self.final_event = {
            "type": "response.output_item.done",
            "item": {"type": "image_generation_call", "result": self.encoded},
        }
    
    def execute(self, *options: str) -> int:
        argv = ["--prompt", "test", "--model", "test-model", *options]
        cli = image_gen.Cli()
        config = cli.parse_config(argv)
        with (
            mock.patch.object(image_gen.Paths, "read_codex_auth", return_value=("offline-test-token", None)),
            redirect_stdout(StringIO()),
        ):
            return cli.execute(config, "test", self.final, argv)
    
    def test_default_mode_uses_sdk_and_saves_completed_image(self) -> None:
        stream = mock.MagicMock()
        stream.__iter__.return_value = iter([self.final_event])
        client = mock.Mock()
        client.responses.create.return_value = stream
        with (
            mock.patch.object(image_gen.CodexClient, "create_openai", return_value=client) as create_client,
            mock.patch.object(image_gen.request, "urlopen", side_effect=AssertionError("Unexpected direct HTTP request")),
        ):
            result = self.execute()
        self.assertEqual(result, 0)
        create_client.assert_called_once()
        self.assertEqual(client.responses.create.call_args.kwargs["model"], "test-model")
        self.assertTrue(client.responses.create.call_args.kwargs["stream"])
        self.assertNotIn("low_memory", client.responses.create.call_args.kwargs)
        stream.close.assert_called_once()
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertIn('"client": "openai-sdk"', image_gen.Paths.log_path(self.final).read_text(encoding="utf-8"))
    
    def test_low_memory_mode_uses_direct_http_without_sdk(self) -> None:
        response = BytesIO(b"data: " + json.dumps(self.final_event).encode() + b"\n\n")
        with (
            mock.patch.object(image_gen.CodexClient, "create_openai", side_effect=AssertionError("Unexpected SDK load")),
            mock.patch.object(image_gen.request, "urlopen", return_value=response) as urlopen,
        ):
            result = self.execute("--low-memory")
        self.assertEqual(result, 0)
        urlopen.assert_called_once()
        self.assertTrue(response.closed)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertIn('"client": "urllib"', image_gen.Paths.log_path(self.final).read_text(encoding="utf-8"))
    
    def test_legacy_raw_mode_remains_available_without_low_memory(self) -> None:
        response = BytesIO(b"data: " + json.dumps(self.final_event).encode() + b"\n\n")
        with (
            mock.patch.object(image_gen.CodexClient, "create_openai", side_effect=AssertionError("Unexpected SDK load")),
            mock.patch.object(image_gen.request, "urlopen", return_value=response),
        ):
            result = self.execute("--transport", "responses-raw")
        self.assertEqual(result, 0)
        self.assertTrue(response.closed)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertIn('"transport": "responses-raw"', image_gen.Paths.log_path(self.final).read_text(encoding="utf-8"))
    
    def test_dry_run_reports_opt_in_mode_without_sending_it_to_api(self) -> None:
        for options, expected in (([], False), (["--low-memory"], True)):
            with self.subTest(low_memory=expected):
                cli = image_gen.Cli()
                config = cli.parse_config(["--prompt", "test", "--model", "test-model", *options])
                preview = cli.dry_run_preview(config, "test", self.final)
                self.assertEqual(config.low_memory, expected)
                self.assertEqual(preview["low_memory"], expected)
                payload = image_gen.Payloads.build_responses_payload(config, "test", cli.logger)
                self.assertNotIn("low_memory", payload)
    
    def test_image_api_rejects_low_memory_before_auth_or_output_creation(self) -> None:
        with (
            mock.patch.object(image_gen.Paths, "read_codex_auth") as read_auth,
            mock.patch.object(image_gen.Paths, "output_path") as output_path,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit),
        ):
            image_gen.Cli().main(["--prompt", "test", "--transport", "image-api", "--low-memory"])
        read_auth.assert_not_called()
        output_path.assert_not_called()


class LowMemoryResponsesEventTests(unittest.TestCase):
    def test_multiline_events_comments_unicode_and_done(self) -> None:
        response = BytesIO(
            b': heartbeat\r\nevent: response.created\r\nid: 1\r\n'
            b'data: {"response":\r\ndata: {"id":"example"}}\r\n\r\n'
            + 'data:{"type":"response.output_text.delta","delta":"caf\u00e9"}\n\n'.encode()
            + b'data: [DONE]\n\ndata: {"ignored":true}\n\n'
        )
        self.assertEqual(list(image_gen.LowMemoryResponsesTransport.events(response)), [
            {"type": "response.created", "response": {"id": "example"}},
            {"type": "response.output_text.delta", "delta": "caf\u00e9"},
        ])
    
    def test_final_event_without_trailing_blank_line(self) -> None:
        response = BytesIO(b'data: {"type":"response.completed"}')
        self.assertEqual(list(image_gen.LowMemoryResponsesTransport.events(response)), [{"type": "response.completed"}])
    
    def test_malformed_event_is_rejected(self) -> None:
        for data in (b'data: {invalid}\n\n', b'data: []\n\n'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(image_gen.LowMemoryResponsesTransport.events(BytesIO(data)))


class LowMemoryImageOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / f"responses-output-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.final = self.root / "image.png"
        self.logger = image_gen.Logging()
    
    def test_chunk_boundaries_and_base64_padding_preserve_bytes(self) -> None:
        for size in (49151, 49152, 49153, 98305):
            with self.subTest(size=size):
                expected = (bytes(range(256)) * (size // 256 + 1))[:size]
                image_gen.LowMemoryOutput.write_image(base64.b64encode(expected).decode(), self.final)
                self.assertEqual(self.final.read_bytes(), expected)
    
    def test_large_image_save_uses_less_than_one_mib_of_decode_memory(self) -> None:
        encoded = "QUJD" * (2 * 1024 * 1024)
        tracemalloc.start()
        try:
            image_gen.LowMemoryOutput.write_image(encoded, self.final)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1024 * 1024)
        self.assertEqual(self.final.stat().st_size, 6 * 1024 * 1024)
    
    def test_invalid_base64_does_not_leave_incomplete_final_or_temporary_file(self) -> None:
        encoded = "QUJD" * 16384 + "A"
        with self.assertRaises(binascii.Error):
            image_gen.LowMemoryOutput.write_final_image(
                encoded, None, self.final, logger=self.logger, verbose=False,
            )
        self.assertEqual(list(self.root.iterdir()), [])
    
    def test_invalid_base64_preserves_existing_final_and_preview(self) -> None:
        self.final.write_bytes(b"existing final")
        partial = self.root / "partial.png"
        partial.write_bytes(b"existing preview")
        encoded = "QUJD" * 16384 + "A"
        with self.assertRaises(binascii.Error):
            image_gen.LowMemoryOutput.write_final_image(
                encoded, partial, self.final, logger=self.logger, verbose=False,
            )
        self.assertEqual(self.final.read_bytes(), b"existing final")
        self.assertEqual(partial.read_bytes(), b"existing preview")
        self.assertEqual(set(self.root.iterdir()), {self.final, partial})
    
    def test_write_failure_preserves_target_and_removes_temporary_file(self) -> None:
        self.final.write_bytes(b"existing final")
        create_temporary = image_gen.tempfile.NamedTemporaryFile
        
        def failing_temporary(*args, **kwargs):
            temporary = create_temporary(*args, **kwargs)
            write = temporary.write
            
            def fail_after_writing(data):
                write(data[:8])
                raise OSError("disk full")
            
            temporary.write = fail_after_writing
            return temporary
        
        with (
            mock.patch.object(image_gen.tempfile, "NamedTemporaryFile", side_effect=failing_temporary),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            image_gen.LowMemoryOutput.write_image("QUJD" * 16384, self.final)
        self.assertEqual(self.final.read_bytes(), b"existing final")
        self.assertEqual(list(self.root.iterdir()), [self.final])
    
    def test_replace_failure_preserves_target_and_removes_temporary_file(self) -> None:
        self.final.write_bytes(b"existing final")
        with (
            mock.patch.object(Path, "replace", side_effect=PermissionError("target is locked")),
            self.assertRaisesRegex(PermissionError, "target is locked"),
        ):
            image_gen.LowMemoryOutput.write_image("QUJD" * 16384, self.final)
        self.assertEqual(self.final.read_bytes(), b"existing final")
        self.assertEqual(list(self.root.iterdir()), [self.final])
    
    def test_identical_partial_is_renamed_and_different_partial_is_preserved(self) -> None:
        image = b"complete-image" * 10000
        encoded = base64.b64encode(image).decode()
        for partial_bytes in (image, image + b"extra", image[:-1], b"different"):
            with self.subTest(partial_size=len(partial_bytes)):
                partial = self.root / "partial.png"
                partial.write_bytes(partial_bytes)
                image_gen.LowMemoryOutput.write_final_image(
                    encoded, partial, self.final, logger=self.logger, verbose=False,
                )
                self.assertEqual(self.final.read_bytes(), image)
                self.assertEqual(partial.exists(), partial_bytes != image)
                if partial.exists():
                    self.assertEqual(partial.read_bytes(), partial_bytes)


class LowMemoryResponsesTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / f"responses-transport-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.final = self.root / "image.png"
        self.logger = image_gen.Logging()
        self.run = image_gen.RunContext(
            output_path=self.final, log_path=self.root / "image.png.log",
            token="offline-test-token", account_id="offline-account", timeout_seconds=17,
            started=time.time(), invocation={"argv": ["--prompt", "test"]}, inputs={},
        )
        self.logger.configure(self.run.log_path, "responses-event")
        self.config = image_gen.Cli().parse_config(["--prompt", "test", "--model", "test-model", "--low-memory"])
        self.image = b"image-content" * 5000
        self.encoded = base64.b64encode(self.image).decode()
    
    def response(self, *items: dict) -> BytesIO:
        return BytesIO(b"".join(b"data: " + json.dumps(item).encode() + b"\n\n" for item in items))
    
    def final_event(self) -> dict:
        return {"type": "response.output_item.done", "item": {"type": "image_generation_call", "result": self.encoded}}
    
    def run_response(self, response: BytesIO) -> mock.Mock:
        with (
            mock.patch.object(image_gen.request, "urlopen", return_value=response) as urlopen,
            redirect_stdout(StringIO()),
        ):
            image_gen.LowMemoryResponsesTransport(self.logger).run(self.config, "test", self.run)
        return urlopen
    
    def test_request_preserves_controls_references_auth_and_redacted_logs(self) -> None:
        reference = self.root / "reference.png"
        reference.write_bytes(self.image)
        self.config.reference = [str(reference)]
        self.config.mask = str(reference)
        self.config.reasoning_effort = "high"
        self.config.partial_images = 2
        self.config.size = "1024x1024"
        self.config.image_model = "test-image-model"
        response = self.response(self.final_event())
        sent = {}
        
        def open_request(req, timeout):
            sent.update(payload=json.loads(req.data), headers=dict(req.header_items()), timeout=timeout, req=req)
            return response
        
        with mock.patch.object(image_gen.request, "urlopen", side_effect=open_request), redirect_stdout(StringIO()):
            image_gen.LowMemoryResponsesTransport(self.logger).run(self.config, "test", self.run)
        
        self.assertEqual(sent["payload"]["model"], "test-model")
        self.assertEqual(sent["payload"]["reasoning"], {"effort": "high"})
        self.assertTrue(sent["payload"]["stream"])
        tool = sent["payload"]["tools"][0]
        self.assertEqual(tool["partial_images"], 2)
        self.assertEqual(tool["size"], "1024x1024")
        self.assertEqual(tool["model"], "test-image-model")
        image_url = "data:image/png;base64," + self.encoded
        self.assertEqual(tool["input_image_mask"]["image_url"], image_url)
        self.assertEqual(sent["payload"]["input"][0]["content"][0]["image_url"], image_url)
        self.assertEqual(sent["headers"]["Authorization"], "Bearer offline-test-token")
        self.assertEqual(sent["headers"]["Chatgpt-account-id"], "offline-account")
        self.assertEqual(sent["timeout"], 17)
        self.assertIsNone(sent["req"].data)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertTrue(response.closed)
        self.assertIsNone(self.logger.context.active_handle)
        log = self.run.log_path.read_text(encoding="utf-8")
        self.assertNotIn(self.encoded, log)
        self.assertNotIn(self.run.token, log)
        self.assertIn(f"<redacted {len(self.encoded)} chars>", log)
    
    def test_final_after_partial_promotes_only_completed_image(self) -> None:
        self.config.partial_images = 1
        self.run_response(self.response(
            {"type": "response.image_generation_call.partial_image", "partial_image_index": 0, "partial_image_b64": self.encoded},
            self.final_event(),
        ))
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertFalse(image_gen.Output.partial_output_path(self.final, 0).exists())
    
    def test_partial_only_stream_never_produces_final_image(self) -> None:
        self.config.partial_images = 1
        response = self.response({"type": "response.image_generation_call.partial_image", "partial_image_b64": self.encoded})
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.run_response(response)
        self.assertFalse(self.final.exists())
        self.assertEqual(image_gen.Output.partial_output_path(self.final, 1).read_bytes(), self.image)
        self.assertTrue(response.closed)
    
    def test_disabled_previews_are_not_saved(self) -> None:
        self.run_response(self.response(
            {"type": "response.image_generation_call.partial_image", "partial_image_b64": self.encoded},
            self.final_event(),
        ))
        self.assertEqual(list(self.root.glob("*-partial-*")), [])
        self.assertEqual(self.final.read_bytes(), self.image)
    
    def test_failure_details_are_logged_and_can_be_hidden(self) -> None:
        self.config.hide_response_details = True
        response = self.response({"type": "response.failed", "response": {"error": {"message": "example failure"}}})
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.run_response(response)
        self.assertNotIn("example failure", stderr.getvalue())
        self.assertIn("example failure", self.run.log_path.read_text(encoding="utf-8"))
        self.assertFalse(self.final.exists())
        self.assertTrue(response.closed)
        self.assertIsNone(self.logger.context.active_handle)
    
    def test_malformed_stream_closes_response_and_logs_failure(self) -> None:
        response = BytesIO(b"data: {invalid}\n\n")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.run_response(response)
        self.assertFalse(self.final.exists())
        self.assertTrue(response.closed)
        self.assertIn("response.stream_failed", self.run.log_path.read_text(encoding="utf-8"))
    
    def test_low_memory_with_legacy_transport_generates_image(self) -> None:
        self.config.transport = "responses-raw"
        self.run_response(self.response(self.final_event()))
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertTrue(self.run.log_path.read_text(encoding="utf-8").startswith("event: codex_image_gen.start\n"))
    
    def test_cli_copies_completed_output(self) -> None:
        self.config.copy_to = str(self.root / "project.png")
        with (
            mock.patch.object(image_gen.Paths, "read_codex_auth", return_value=("offline-test-token", None)),
            mock.patch.object(image_gen.request, "urlopen", return_value=self.response(self.final_event())),
            redirect_stdout(StringIO()),
        ):
            result = image_gen.Cli().execute(self.config, "test", self.final, ["--prompt", "test"])
        self.assertEqual(result, 0)
        self.assertEqual(Path(self.config.copy_to).read_bytes(), self.image)
        self.assertEqual(self.final.read_bytes(), self.image)
    
    def test_interrupted_stream_is_closed_without_retry(self) -> None:
        class InterruptedResponse(BytesIO):
            def __next__(self):
                raise ConnectionResetError("stream interrupted")
        
        response = InterruptedResponse()
        with (
            mock.patch.object(image_gen.request, "urlopen", return_value=response) as urlopen,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit),
        ):
            image_gen.LowMemoryResponsesTransport(self.logger).run(self.config, "test", self.run)
        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(self.final.exists())
        self.assertTrue(response.closed)
    
    def test_http_failure_body_is_bounded_and_response_closed(self) -> None:
        body = BytesIO(b"failure-details" * 1000)
        failure = HTTPError("https://example.invalid", 400, "bad request", {}, body)
        with (
            mock.patch.object(image_gen.request, "urlopen", side_effect=failure),
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit),
        ):
            image_gen.LowMemoryResponsesTransport(self.logger).run(self.config, "test", self.run)
        self.assertTrue(body.closed)
        records = [
            json.loads(line[6:]) for line in self.run.log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("data: ")
        ]
        failure_record = next(record for record in records if record.get("status") == 400)
        self.assertEqual(len(failure_record["body"]), 4000)


class LowMemoryResponsesRetryTests(unittest.TestCase):
    def test_transient_failures_retry_twice_and_close_error_responses(self) -> None:
        body = BytesIO(b"retry later")
        failure = HTTPError("https://example.invalid", 429, "rate limited", {"Retry-After": "2"}, body)
        response = BytesIO()
        req = Request("https://example.invalid", data=b"{}")
        with (
            mock.patch.object(image_gen.request, "urlopen", side_effect=[failure, URLError("connection"), response]) as open_request,
            mock.patch.object(image_gen.time, "sleep") as sleep,
        ):
            result = image_gen.LowMemoryResponsesTransport.open_request(req, 17)
        self.assertIs(result, response)
        self.assertEqual(open_request.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(2), mock.call(1)])
        self.assertTrue(body.closed)
        self.assertEqual(req.data, b"{}")
    
    def test_non_retryable_http_error_is_not_retried(self) -> None:
        failure = HTTPError("https://example.invalid", 400, "bad request", {}, BytesIO())
        with (
            mock.patch.object(image_gen.request, "urlopen", side_effect=failure) as open_request,
            mock.patch.object(image_gen.time, "sleep") as sleep,
            self.assertRaises(HTTPError),
        ):
            image_gen.LowMemoryResponsesTransport.open_request(Request("https://example.invalid"), 17)
        self.assertEqual(open_request.call_count, 1)
        sleep.assert_not_called()
        failure.close()
    
    def test_connection_failure_stops_after_three_attempts(self) -> None:
        with (
            mock.patch.object(image_gen.request, "urlopen", side_effect=TimeoutError("timed out")) as open_request,
            mock.patch.object(image_gen.time, "sleep"),
            self.assertRaises(TimeoutError),
        ):
            image_gen.LowMemoryResponsesTransport.open_request(Request("https://example.invalid"), 17)
        self.assertEqual(open_request.call_count, 3)


if __name__ == "__main__":
    unittest.main()
