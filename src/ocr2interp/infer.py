from __future__ import annotations

from pathlib import Path


def run_model_infer(model, tokenizer, cfg, image_file=None, output_dir=None):
    image_file = Path(image_file or cfg.input.image_file)
    output_dir = Path(output_dir or cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return model.infer(
        tokenizer,
        prompt=cfg.infer.prompt,
        image_file=str(image_file),
        output_path=str(output_dir),
        base_size=cfg.infer.base_size,
        image_size=cfg.infer.image_size,
        crop_mode=cfg.infer.crop_mode,
        save_results=cfg.infer.save_results,
        eval_mode=cfg.infer.get("eval_mode", False),
    )


def save_result_repr(result, output_dir) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "result_repr.txt").write_text(repr(result), encoding="utf-8")
