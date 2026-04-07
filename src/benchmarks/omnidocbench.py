"""Utilities for bulk inference on OmniDocBench-formatted data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from PIL import Image, ImageOps


@dataclass(frozen=True)
class OmniDocBenchSample:
    """One OmniDocBench page entry resolved to a local image path."""

    sample_id: str
    image_path: Path
    image_name: str
    page_no: Optional[int]
    width: Optional[int]
    height: Optional[int]
    page_attributes: Dict[str, Any]
    raw_entry: Dict[str, Any]

    @property
    def output_name(self) -> str:
        """Default markdown filename expected by the benchmark."""
        return f"{Path(self.image_name).stem}.md"


class OmniDocBenchDataset(Sequence[OmniDocBenchSample]):
    """Loader for the official OmniDocBench JSON manifest format."""

    def __init__(self, samples: Iterable[OmniDocBenchSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> OmniDocBenchSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[OmniDocBenchSample]:
        return iter(self.samples)

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | Path,
        *,
        manifest_name: str = "OmniDocBench.json",
        filters: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> "OmniDocBenchDataset":
        """Load samples from a standard OmniDocBench directory."""
        dataset_root = Path(dataset_root)
        manifest_path = dataset_root / manifest_name
        return cls.from_manifest(
            manifest_path,
            image_root=dataset_root,
            filters=filters,
            limit=limit,
            offset=offset,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        image_root: Optional[str | Path] = None,
        filters: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> "OmniDocBenchDataset":
        """Load samples from the benchmark JSON file described in OmniDocBench docs."""
        manifest_path = Path(manifest_path)
        image_root_path = Path(image_root) if image_root is not None else manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = cls._normalize_entries(payload)

        filtered_entries = cls._apply_offset_limit(
            [entry for entry in entries if cls._matches_filters(entry, filters)],
            offset=offset,
            limit=limit,
        )

        samples = [
            cls._entry_to_sample(entry, manifest_path=manifest_path, image_root=image_root_path)
            for entry in filtered_entries
        ]
        cls._ensure_unique_output_names(samples)
        return cls(samples)

    @staticmethod
    def _normalize_entries(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "entries", "pages", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError("Unsupported OmniDocBench manifest format.")

    @staticmethod
    def _apply_offset_limit(entries: List[Dict[str, Any]], *, offset: int, limit: Optional[int]) -> List[Dict[str, Any]]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        sliced = entries[offset:]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be >= 0")
            sliced = sliced[:limit]
        return sliced

    @classmethod
    def _entry_to_sample(
        cls,
        entry: Dict[str, Any],
        *,
        manifest_path: Path,
        image_root: Path,
    ) -> OmniDocBenchSample:
        page_info = entry.get("page_info", {})
        image_name = page_info.get("image_path")
        if not image_name:
            raise ValueError("OmniDocBench entry is missing page_info.image_path")

        image_path = cls._resolve_image_path(
            image_name,
            image_root=image_root,
            manifest_path=manifest_path,
        )

        sample_id = str(page_info.get("page_no", Path(image_name).stem))
        sample_id = f"{Path(image_name).stem}:{sample_id}"
        return OmniDocBenchSample(
            sample_id=sample_id,
            image_path=image_path,
            image_name=Path(image_name).name,
            page_no=page_info.get("page_no"),
            width=page_info.get("width"),
            height=page_info.get("height"),
            page_attributes=dict(page_info.get("page_attribute", {})),
            raw_entry=entry,
        )

    @staticmethod
    def _resolve_image_path(image_name: str, *, image_root: Path, manifest_path: Path) -> Path:
        image_name_path = Path(image_name)
        candidates = [
            image_name_path,
            image_root / image_name_path,
            manifest_path.parent / image_name_path,
            image_root / "images" / image_name_path.name,
            manifest_path.parent / "images" / image_name_path.name,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not resolve image path '{image_name}' from manifest '{manifest_path}'."
        )

    @staticmethod
    def _matches_filters(entry: Mapping[str, Any], filters: Optional[Mapping[str, Any]]) -> bool:
        if not filters:
            return True

        page_info = entry.get("page_info", {})
        page_attributes = page_info.get("page_attribute", {})

        for key, expected in filters.items():
            if key in page_attributes:
                actual = page_attributes.get(key)
            else:
                actual = page_info.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _ensure_unique_output_names(samples: Sequence[OmniDocBenchSample]) -> None:
        seen: Dict[str, str] = {}
        for sample in samples:
            if sample.output_name in seen and seen[sample.output_name] != sample.sample_id:
                raise ValueError(
                    "Duplicate output filename detected for OmniDocBench samples: "
                    f"{sample.output_name}. Use a disambiguated export strategy first."
                )
            seen[sample.output_name] = sample.sample_id


class OmniDocBenchRunner:
    """Run bulk OCR inference on OmniDocBench samples and save markdown outputs."""

    def __init__(self, dataset: OmniDocBenchDataset):
        self.dataset = dataset

    def run(
        self,
        predictor: Callable[..., str],
        output_dir: str | Path,
        *,
        prompt: str,
        overwrite: bool = False,
        dry_run: bool = False,
        metadata_name: str = "run_manifest.jsonl",
    ) -> List[Dict[str, Any]]:
        """Write one markdown file per page using benchmark-compatible filenames."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for index, sample in enumerate(self.dataset, start=1):
            output_path = output_dir / sample.output_name
            status = "skipped" if output_path.exists() and not overwrite else "written"

            if dry_run:
                status = "planned"
            elif status == "written":
                with Image.open(sample.image_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    prediction = self._predict(predictor, image=image, prompt=prompt)
                output_path.write_text(prediction, encoding="utf-8")

            records.append(
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "output_path": str(output_path),
                    "status": status,
                    "page_no": sample.page_no,
                    "page_attributes": sample.page_attributes,
                }
            )

        metadata_path = output_dir / metadata_name
        metadata_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        return records

    @staticmethod
    def _predict(predictor: Callable[..., str], *, image: Image.Image, prompt: str) -> str:
        try:
            return predictor(image, prompt=prompt)
        except TypeError:
            return predictor(image)


def sample_to_dict(sample: OmniDocBenchSample) -> Dict[str, Any]:
    """Serialize a sample for debugging or dry-run inspection."""
    data = asdict(sample)
    data["image_path"] = str(sample.image_path)
    return data
