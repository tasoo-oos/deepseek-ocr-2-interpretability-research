from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ocr2interp.infer import run_model_infer, save_result_repr
from ocr2interp.loading import ModelConfig, load_ocr2
from ocr2interp.utils import save_config, seed_everything


def main() -> None:
    cfg = OmegaConf.load("configs/infer.yaml")
    output_dir = Path(cfg.output.dir)
    seed_everything(cfg.seed)
    save_config(cfg, output_dir / "config.yaml")
    model, tokenizer = load_ocr2(ModelConfig(**cfg.model))
    result = run_model_infer(model, tokenizer, cfg)
    save_result_repr(result, output_dir)


if __name__ == "__main__":
    main()
