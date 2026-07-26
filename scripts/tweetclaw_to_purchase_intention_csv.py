"""Convert reviewed TweetClaw exports into purchase-intention CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

TEXT_FIELDS = ("text", "tweet_text", "tweet", "full_text", "content", "body")
FIELDNAMES = ("class", "text")
CLASS_LABELS = ("yes", "no")


def read_export(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    content = source.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(content)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_json_lines(content)
    if content[0] in "[{":
        return _rows_from_payload(json.loads(content))
    return _read_json_lines(content)


def convert_rows(
    rows: Iterable[dict[str, object]],
    class_label: str,
) -> Iterable[dict[str, str]]:
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        yield {
            "class": class_label,
            "text": normalize_text(text),
        }


def write_csv(rows: Iterable[dict[str, str]], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _read_csv_rows(content: str) -> list[dict[str, object]]:
    first_line = content.splitlines()[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    return [dict(row) for row in csv.DictReader(content.splitlines(), delimiter=delimiter)]


def _read_json_lines(content: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rows_from_payload(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "tweets", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _first_text(row: dict[str, object]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert reviewed TweetClaw exports into class,text CSV rows."
    )
    parser.add_argument("input", help="TweetClaw JSON, JSONL, NDJSON, or CSV export.")
    parser.add_argument("output", help="Output CSV path for the model pipeline.")
    parser.add_argument(
        "--class-label",
        required=True,
        choices=CLASS_LABELS,
        help="Reviewed purchase-intention label to apply to every exported row.",
    )
    args = parser.parse_args()

    write_csv(convert_rows(read_export(args.input), args.class_label), args.output)


if __name__ == "__main__":
    main()
