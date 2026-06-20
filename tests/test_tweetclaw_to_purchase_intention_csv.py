"""Tests for TweetClaw purchase-intention export conversion."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "tweetclaw_to_purchase_intention_csv.py"
)
SPEC = importlib.util.spec_from_file_location("tweetclaw_converter", MODULE_PATH)
assert SPEC is not None
tweetclaw_converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tweetclaw_converter)


class TestTweetClawToPurchaseIntentionCsv(unittest.TestCase):
    def test_jsonl_export_converts_to_project_schema(self) -> None:
        source_path = self._write_file(
            "\n".join(
                [
                    json.dumps({"text": "I will buy the new phone tomorrow."}),
                    json.dumps({"full_text": "Not buying this model yet."}),
                ]
            ),
            ".jsonl",
        )

        rows = list(
            tweetclaw_converter.convert_rows(tweetclaw_converter.read_export(source_path), "yes")
        )

        self.assertEqual(rows[0]["class"], "yes")
        self.assertEqual(rows[0]["text"], "I will buy the new phone tomorrow.")
        self.assertEqual(rows[1]["text"], "Not buying this model yet.")

    def test_csv_export_writes_header(self) -> None:
        source_path = self._write_file("tweet_text\nNeed to buy this phone\n", ".csv")
        output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)

        tweetclaw_converter.write_csv(
            tweetclaw_converter.convert_rows(tweetclaw_converter.read_export(source_path), "no"),
            output_path,
        )

        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["class"], "no")
        self.assertEqual(rows[0]["text"], "Need to buy this phone")

    def test_rows_without_text_are_skipped(self) -> None:
        rows = list(tweetclaw_converter.convert_rows([{"created_at": "2026-06-20"}], "yes"))

        self.assertEqual(rows, [])

    def _write_file(self, content: str, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix, encoding="utf-8")
        with handle:
            handle.write(content)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
