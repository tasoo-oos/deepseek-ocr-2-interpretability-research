from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path

from ocr2interp.infer import run_model_infer, save_result_repr
from ocr2interp.loading import ModelConfig, load_ocr2
from ocr2interp.utils import save_config, seed_everything


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg) -> None:
    output_dir = Path(to_absolute_path(cfg.output.dir))
    image_file = Path(to_absolute_path(cfg.input.image_file))
    seed_everything(cfg.seed)
    save_config(cfg, output_dir / "config.yaml")
    model, tokenizer = load_ocr2(ModelConfig(**cfg.model))
    result = run_model_infer(model, tokenizer, cfg, image_file=image_file, output_dir=output_dir)
    save_result_repr(result, output_dir)


if __name__ == "__main__":
    main()
