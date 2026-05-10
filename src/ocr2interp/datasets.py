from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ocr2interp.io import read_jsonl


@dataclass(frozen=True)
class Example:
    id: str
    image_path: str
    target_markdown_path: str | None = None
    spans: list[dict] | None = None


def load_jsonl_dataset(path) -> list[Example]:
    examples = []
    for row in read_jsonl(Path(path)):
        examples.append(
            Example(
                id=row["id"],
                image_path=row["image_path"],
                target_markdown_path=row.get("target_markdown_path"),
                spans=row.get("spans"),
            )
        )
    return examples
