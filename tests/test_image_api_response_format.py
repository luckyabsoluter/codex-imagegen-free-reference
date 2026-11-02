from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import uuid

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import codex_image_gen as image_gen


class ImageApiResponseFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "tmp" / "image-api-tests" / str(uuid.uuid4())
        self.root.mkdir(parents=True)
        self.image = b"final-image" * 1000
        self.encoded = base64.b64encode(self.image).decode()
        self.final = self.root / "original.png"
        self.copy = self.root / "copied.png"
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.response = None
        self.request = None
    
    def execute(self, body: bytes, content_type: str, *, status: int = 200, partials: bool = True, model: bool = True) -> int:
        def respond(request: httpx.Request) -> httpx.Response:
            self.request = json.loads(request.content)
            self.response = httpx.Response(status, headers={"Content-Type": content_type}, content=body)
            return self.response
        
        argv = ["--transport", "image-api", "--prompt", "test", "--copy-to", str(self.copy)]
        if model:
            argv += ["--image-model", "gpt-image-2"]
        if partials:
            argv += ["--partial-images", "3"]
        cli = image_gen.Cli()
        config = cli.parse_config(argv)
        client = OpenAI(api_key="offline-test-token", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            with (
                mock.patch.object(image_gen.Paths, "read_codex_auth", return_value=("offline-test-token", None)),
                mock.patch.object(image_gen.CodexClient, "create_openai", return_value=client),
                redirect_stdout(self.stdout),
                redirect_stderr(self.stderr),
            ):
                return cli.execute(config, "test", self.final, argv)
        finally:
            if self.response is not None:
                self.assertTrue(self.response.is_closed)
            client.close()
    
    def json_image(self) -> bytes:
        return json.dumps({"created": 1, "data": [{"b64_json": self.encoded}]}).encode()
    
    def log(self) -> str:
        return image_gen.Paths.log_path(self.final).read_text(encoding="utf-8")
    
    def test_json_image_under_streaming_request_saves_original_and_copy(self) -> None:
        self.assertEqual(self.execute(self.json_image(), "application/json; charset=utf-8"), 0)
        self.assertTrue(self.request["stream"])
        self.assertEqual(self.request["partial_images"], 3)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertEqual(self.copy.read_bytes(), self.image)
        self.assertEqual(list(self.root.glob("*-partial-*")), [])
        self.assertIn("instead of streamed partial previews", self.stdout.getvalue())
        self.assertIn("image_api.response", self.log())
        self.assertNotIn(self.encoded, self.log())
    
    def test_nonstreaming_json_image_remains_supported(self) -> None:
        self.assertEqual(self.execute(self.json_image(), "application/json", partials=False), 0)
        self.assertNotIn("stream", self.request)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertNotIn("instead of streamed", self.stdout.getvalue())
    
    def test_omitted_model_is_absent_from_sdk_request(self) -> None:
        self.assertEqual(self.execute(self.json_image(), "application/json", model=False), 0)
        self.assertNotIn("model", self.request)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertEqual(self.copy.read_bytes(), self.image)
    
    def test_explicit_image_api_model_precedence(self) -> None:
        cases = [([], None), (["--model", "image-base"], "image-base"),
                 (["--model", "image-base", "--image-model", "image-override"], "image-override")]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                config = image_gen.Cli().parse_config(["--transport", "image-api", "--prompt", "test", *arguments])
                options = image_gen.Payloads.build_image_api_options(config, "test")
                edit = image_gen.Payloads.build_image_api_edit_payload(config, "test", image_gen.Logging())
                for payload in (options, edit):
                    if expected is None:
                        self.assertNotIn("model", payload)
                    else:
                        self.assertEqual(payload["model"], expected)
    
    def test_sse_partial_and_final_images_remain_supported(self) -> None:
        partial = base64.b64encode(b"partial-image" * 1000).decode()
        events = [
            {"type": "image_generation.partial_image", "partial_image_index": 0, "b64_json": partial},
            {"type": "image_generation.completed", "b64_json": self.encoded},
        ]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
        self.assertEqual(self.execute(body, "text/event-stream; charset=utf-8"), 0)
        self.assertEqual(self.final.read_bytes(), self.image)
        self.assertEqual(self.copy.read_bytes(), self.image)
        self.assertEqual(image_gen.Output.partial_output_path(self.final, 1).read_bytes(), base64.b64decode(partial))
        self.assertNotIn(partial, self.log())
        self.assertNotIn(self.encoded, self.log())
    
    def test_json_error_details_are_preserved_without_creating_images(self) -> None:
        body = json.dumps({"error": {"message": "Unsupported model", "code": "invalid_value"}}).encode()
        with self.assertRaises(SystemExit):
            self.execute(body, "application/json")
        self.assertIn("Unsupported model", self.log())
        self.assertFalse(self.final.exists())
        self.assertFalse(self.copy.exists())
    
    def test_invalid_json_body_is_logged(self) -> None:
        with self.assertRaises(SystemExit):
            self.execute(b"upstream returned HTML", "text/html")
        self.assertIn("upstream returned HTML", self.log())
        self.assertIn("not valid JSON", self.stderr.getvalue())
        self.assertFalse(self.final.exists())
    
    def test_empty_response_is_logged(self) -> None:
        with self.assertRaises(SystemExit):
            self.execute(b"", "application/json")
        self.assertIn("Could not parse image JSON response", self.log())
        self.assertFalse(self.final.exists())
    
    def test_http_error_is_logged(self) -> None:
        body = json.dumps({"error": {"message": "Unsupported model", "type": "invalid_request_error"}}).encode()
        with self.assertRaises(SystemExit):
            self.execute(body, "application/json", status=400)
        self.assertIn("Unsupported model", self.log())
        self.assertFalse(self.final.exists())


if __name__ == "__main__":
    unittest.main()
