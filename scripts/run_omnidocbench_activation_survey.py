from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ocr2interp.infer import run_model_infer
from ocr2interp.loading import ModelConfig, load_ocr2
from ocr2interp.utils import ensure_dir, save_config, seed_everything


DEFAULT_MODULES = [
    "model.sam_model.neck",
    "model.qwen2_model.model.model.layers.0",
    "model.qwen2_model.model.model.layers.11",
    "model.qwen2_model.model.model.layers.23",
    "model.projector",
    "model.layers.0",
    "model.layers.11",
]


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from iter_tensors(item)


class ActivationStatsRecorder:
    def __init__(self) -> None:
        self.records = defaultdict(list)

    def clear(self) -> None:
        self.records.clear()

    def save(self, name: str, output) -> None:
        tensors = list(iter_tensors(output))
        summary = []
        for tensor in tensors:
            detached = tensor.detach()
            values = detached.float()
            summary.append(
                {
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype).replace("torch.", ""),
                    "mean": float(values.mean().item()),
                    "std": float(values.std().item()) if values.numel() > 1 else 0.0,
                    "abs_mean": float(values.abs().mean().item()),
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "numel": int(values.numel()),
                }
            )
        self.records[name].append(summary)

    def summarize(self) -> dict[str, dict]:
        out = {}
        for module_name, calls in self.records.items():
            flat = [item for call in calls for item in call]
            out[module_name] = {
                "calls": len(calls),
                "tensor_records": len(flat),
                "shapes": sorted({json.dumps(item["shape"]) for item in flat}),
                "dtypes": sorted({item["dtype"] for item in flat}),
                "mean_avg": avg(item["mean"] for item in flat),
                "std_avg": avg(item["std"] for item in flat),
                "abs_mean_avg": avg(item["abs_mean"] for item in flat),
                "min": min((item["min"] for item in flat), default=None),
                "max": max((item["max"] for item in flat), default=None),
            }
        return out


class HookManager:
    def __init__(self) -> None:
        self.handles = []

    def add(self, model, module_name: str, recorder: ActivationStatsRecorder) -> None:
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"Unknown module: {module_name}")

        def hook(_module, _inputs, output):
            recorder.save(module_name, output)

        self.handles.append(modules[module_name].register_forward_hook(hook))

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()


def avg(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def completed_images(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                done.add(row["image"])
    return done


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR2 activation summaries over OmniDocBench.")
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--image-dir", default="data/raw/OmniDocBench/images")
    parser.add_argument("--output-dir", default="outputs/runs/run_2_omnidocbench_activation_survey")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--module", action="append", dest="modules")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.infer.save_results = False
    cfg.infer.eval_mode = True
    cfg.output.dir = args.output_dir
    output_dir = ensure_dir(args.output_dir)
    log_dir = ensure_dir(output_dir / "log")
    transcript_dir = ensure_dir(log_dir / "transcripts")
    save_config(cfg, log_dir / "config.yaml")

    image_dir = Path(args.image_dir)
    image_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in image_dir.glob(pattern)
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    image_paths = image_paths[args.shard_index :: args.num_shards]

    records_name = "activation_summary.jsonl"
    if args.num_shards > 1:
        records_name = f"activation_summary_shard-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    records_path = log_dir / records_name
    done = completed_images(records_path) if args.resume else set()
    modules = args.modules or DEFAULT_MODULES

    seed_everything(cfg.seed)
    model, tokenizer = load_ocr2(ModelConfig(**cfg.model))
    recorder = ActivationStatsRecorder()
    manager = HookManager()
    try:
        for module_name in modules:
            manager.add(model, module_name, recorder)

        for image_path in tqdm(image_paths, desc="OmniDocBench activation survey"):
            if str(image_path) in done:
                continue
            recorder.clear()
            page_output_dir = ensure_dir(log_dir / "pages" / image_path.stem)
            transcript_path = transcript_dir / f"{image_path.stem}.txt"
            started = time.time()
            row = {"image": str(image_path), "ok": False, "modules": modules}
            try:
                with transcript_path.open("w", encoding="utf-8") as transcript:
                    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                        result = run_model_infer(
                            model,
                            tokenizer,
                            cfg,
                            image_file=image_path,
                            output_dir=page_output_dir,
                        )
                row.update(
                    {
                        "ok": True,
                        "elapsed_sec": time.time() - started,
                        "result_repr_len": len(repr(result)),
                        "activation_summary": recorder.summarize(),
                        "transcript": str(transcript_path),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "elapsed_sec": time.time() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "activation_summary": recorder.summarize(),
                        "transcript": str(transcript_path),
                    }
                )
            append_jsonl(records_path, row)
    finally:
        manager.close()

    print(f"Wrote activation summaries to {records_path}")


if __name__ == "__main__":
    main()
