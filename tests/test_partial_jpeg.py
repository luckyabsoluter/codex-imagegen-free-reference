from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock
import uuid

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import codex_image_gen as image_gen


class PartialJpegTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / f"partial-jpeg-{uuid.uuid4()}.test"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.cli = image_gen.Cli()
        self.final = self.root / "image.png"
        self.config = self.cli.parse_config([
            "--prompt", "test", "--model", "test-model", "--partial-images", "1", "--partial-output-format", "jpg",
        ])
        self.source = Image.new("RGBA", (128, 128), (0, 0, 255, 0))
        self.addCleanup(self.source.close)
        self.source.paste((255, 0, 0, 255), (64, 0, 96, 128))
        self.source.paste((255, 0, 0, 128), (96, 0, 128, 128))
        self.original = self.encode(self.source)
        self.encoded = base64.b64encode(self.original).decode()
    
    @staticmethod
    def encode(image: Image.Image, output_format: str = "PNG") -> bytes:
        with BytesIO() as buffer:
            image.save(buffer, format=output_format, compress_level=0)
            return buffer.getvalue()
    
    def write_preview(self, *, low_memory: bool = False, encoded: str | None = None) -> Path:
        writer = image_gen.LowMemoryOutput if low_memory else image_gen.Output
        with redirect_stdout(StringIO()):
            result = writer.write_partial_image(
                {"partial_image_b64": encoded or self.encoded, "partial_image_index": 0}, self.final, 1,
                logger=self.cli.logger, verbose=False, config=self.config,
            )
        suffix = ".jpg" if self.config.partial_output_format else self.final.suffix
        path = self.final.with_name(f"{self.final.stem}-partial-0{suffix}")
        if self.config.partial_output_format:
            self.assertIsNone(result)
        elif low_memory:
            self.assertEqual(result, path)
        else:
            self.assertEqual(result, (path, base64.b64decode(encoded or self.encoded)))
        return path
    
    def assert_color(self, actual: tuple, expected: tuple) -> None:
        for value, target in zip(actual, expected):
            self.assertLessEqual(abs(value - target), 4)
    
    def test_jpeg_backgrounds_composite_transparent_and_opaque_pixels(self) -> None:
        for low_memory in (False, True):
            for background, color in ((None, (255, 255, 255)), ("navy", (0, 0, 128)), ("#20a040", (32, 160, 64))):
                with self.subTest(low_memory=low_memory, background=background):
                    self.config.partial_background = background
                    path = self.write_preview(low_memory=low_memory)
                    self.assertEqual(path.name, "image-partial-0.jpg")
                    with Image.open(path) as preview:
                        self.assertEqual(preview.format, "JPEG")
                        self.assertEqual(preview.mode, "RGB")
                        self.assertEqual(preview.size, self.source.size)
                        self.assert_color(preview.getpixel((8, 8)), color)
                        self.assert_color(preview.getpixel((80, 8)), (255, 0, 0))
                        blended = tuple(round(fg * 128 / 255 + bg * 127 / 255) for fg, bg in zip((255, 0, 0), color))
                        self.assert_color(preview.getpixel((112, 8)), blended)
    
    def test_checkerboard_alternates_in_both_directions(self) -> None:
        self.config.partial_background = "checkerboard"
        with Image.open(self.write_preview()) as preview:
            for point, color in (((8, 8), 255), ((24, 8), 204), ((8, 24), 204), ((24, 24), 255)):
                self.assert_color(preview.getpixel(point), (color,) * 3)
    
    def test_palette_transparency_and_rgb_inputs(self) -> None:
        with Image.new("P", (128, 128), 0) as palette:
            palette.putpalette([0, 0, 255] + [0] * 765)
            palette.info["transparency"] = 0
            encoded = base64.b64encode(self.encode(palette)).decode()
            with Image.open(self.write_preview(encoded=encoded)) as preview:
                self.assert_color(preview.getpixel((8, 8)), (255, 255, 255))
        with self.source.convert("RGB") as rgb:
            encoded = base64.b64encode(self.encode(rgb)).decode()
            with Image.open(self.write_preview(encoded=encoded)) as preview:
                self.assert_color(preview.getpixel((8, 8)), (0, 0, 255))
    
    def test_quality_controls_jpeg_quantization_including_zero(self) -> None:
        quantization = []
        for quality in (0, 35, None, 100):
            self.config.partial_output_compression = quality
            with Image.open(self.write_preview()) as preview:
                quantization.append(sum(preview.quantization[0]))
        self.assertTrue(all(a > b for a, b in zip(quantization, quantization[1:])))
    
    def test_unconfigured_partial_keeps_original_bytes(self) -> None:
        self.config.partial_output_format = None
        for low_memory in (False, True):
            with self.subTest(low_memory=low_memory):
                path = self.write_preview(low_memory=low_memory)
                self.assertEqual(path.suffix, ".png")
                self.assertEqual(path.read_bytes(), self.original)
    
    def test_failed_jpeg_write_preserves_existing_preview_and_cleans_temporary_files(self) -> None:
        path = self.root / "image-partial-0.jpg"
        path.write_bytes(b"existing preview")
        for low_memory in (False, True):
            with self.subTest(low_memory=low_memory):
                with mock.patch.object(Image.Image, "save", side_effect=OSError("disk full")), self.assertRaises(OSError):
                    self.write_preview(low_memory=low_memory)
                self.assertEqual(path.read_bytes(), b"existing preview")
                self.assertEqual(set(self.root.iterdir()), {path})
    
    def test_local_options_do_not_change_api_controls(self) -> None:
        for transport in ("responses", "responses-raw", "image-api"):
            with self.subTest(transport=transport):
                self.config.transport = transport
                self.config.output_format = "webp"
                self.config.output_compression = 72
                self.config.partial_output_compression = 15
                self.config.partial_background = "checkerboard"
                image_gen.Validation.validate(self.config, self.cli.logger)
                for payload in (
                    image_gen.Payloads.build_image_tool(self.config, self.cli.logger),
                    image_gen.Payloads.build_image_api_options(self.config, "test"),
                ):
                    self.assertEqual(payload["output_format"], "webp")
                    self.assertEqual(payload["output_compression"], 72)
                    self.assertFalse(any(key.startswith("partial_") and key != "partial_images" for key in payload))
                preview = self.cli.dry_run_preview(self.config, "test", self.final)
                self.assertEqual(preview["partial_output"], {"format": "jpg", "compression": 15, "background": "checkerboard"})
    
    def test_invalid_options_fail_before_auth_and_output_creation(self) -> None:
        invalid = [
            ["--partial-output-format", "jpg"],
            ["--partial-images", "0", "--partial-output-format", "jpg"],
            ["--partial-images", "1", "--partial-background", "white"],
            ["--partial-images", "1", "--partial-output-compression", "50"],
        ]
        base = ["--partial-images", "1", "--partial-output-format", "jpg"]
        invalid.extend(base + ["--partial-output-compression", value] for value in ("-1", "101"))
        invalid.extend(base + ["--partial-background", value] for value in ("invalid-color", "#ffffff00"))
        for options in invalid:
            with (
                self.subTest(options=options),
                mock.patch.object(image_gen.Paths, "read_codex_auth") as auth,
                mock.patch.object(image_gen.Paths, "output_path") as output,
                redirect_stderr(StringIO()), self.assertRaises(SystemExit),
            ):
                self.cli.main(["--prompt", "test", *options])
            auth.assert_not_called()
            output.assert_not_called()
    
    def test_pillow_is_required_only_for_local_jpeg_output(self) -> None:
        with mock.patch.dict(sys.modules, {"PIL": None}), redirect_stderr(StringIO()) as errors:
            with self.assertRaises(SystemExit):
                image_gen.Validation.validate(self.config, self.cli.logger)
            self.assertIn("Pillow", errors.getvalue())
            self.config.partial_output_format = None
            image_gen.Validation.validate(self.config, self.cli.logger)
    
    def test_completed_bytes_and_jpeg_previews_are_preserved_across_transports(self) -> None:
        modes = ([], ["--low-memory"], ["--transport", "responses-raw"], ["--transport", "image-api"])
        for output_format in ("png", "jpeg"):
            with self.source.convert("RGB") as rgb:
                original = self.encode(rgb, output_format.upper())
            encoded = base64.b64encode(original).decode()
            for index, mode in enumerate(modes):
                with self.subTest(output_format=output_format, mode=mode):
                    final = self.root / f"result-{index}.{output_format}"
                    copied = self.root / f"copy-{index}.{output_format}"
                    argv = [
                        "--prompt", "test", "--model", "test-model", "--partial-images", "1",
                        "--partial-output-format", "jpeg", "--output-format", output_format,
                        "--copy-to", str(copied), *mode,
                    ]
                    config = self.cli.parse_config(argv)
                    if config.transport == "image-api":
                        events = [
                            {"type": "image_generation.partial_image", "b64_json": encoded},
                            {"type": "image_generation.completed", "b64_json": encoded},
                        ]
                    else:
                        events = [
                            {"type": "response.image_generation_call.partial_image", "partial_image_b64": encoded},
                            {"type": "response.output_item.done", "item": {"type": "image_generation_call", "result": encoded}},
                        ]
                    stream = mock.MagicMock()
                    stream.__iter__.return_value = iter(events)
                    client = mock.Mock()
                    client.responses.create.return_value = stream
                    client.images.generate.return_value = stream
                    response = BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events))
                    with (
                        mock.patch.object(image_gen.Paths, "read_codex_auth", return_value=("offline-test-token", None)),
                        mock.patch.object(image_gen.CodexClient, "create_openai", return_value=client),
                        mock.patch.object(image_gen.request, "urlopen", return_value=response),
                        redirect_stdout(StringIO()),
                    ):
                        self.assertEqual(self.cli.execute(config, "test", final, argv), 0)
                    self.assertEqual(final.read_bytes(), original)
                    self.assertEqual(copied.read_bytes(), original)
                    with Image.open(final.with_name(f"{final.stem}-partial-1.jpg")) as preview:
                        self.assertEqual(preview.format, "JPEG")
    
    def test_image_api_json_stream_uses_local_preview_settings(self) -> None:
        self.config.transport = "image-api"
        self.config.partial_background = "navy"
        run = image_gen.RunContext(
            output_path=self.final, log_path=self.root / "image.log", token="offline-test-token",
            account_id=None, timeout_seconds=10, started=0, invocation={}, inputs={},
        )
        events = [
            {"type": "image_generation.partial_image", "partial_image_b64": self.encoded},
            {"type": "image_generation.completed", "b64_json": self.encoded},
        ]
        response = BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n" for event in events))
        with redirect_stdout(StringIO()):
            result = image_gen.ImageApiTransport(self.cli.logger).stream_json_response(response, run, self.config)
        self.assertEqual(result, self.original)
        with Image.open(self.root / "image-partial-1.jpg") as preview:
            self.assert_color(preview.getpixel((8, 8)), (0, 0, 128))


if __name__ == "__main__":
    unittest.main()
