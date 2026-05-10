from __future__ import annotations

import hydra

from ocr2interp.loading import ModelConfig, load_ocr2
from ocr2interp.utils import seed_everything


@hydra.main(config_path="../configs", config_name="infer", version_base=None)
def main(cfg) -> None:
    seed_everything(cfg.seed)
    model, _tokenizer = load_ocr2(ModelConfig(**cfg.model))
    for name, module in model.named_modules():
        print(f"{name}\t{module.__class__.__name__}")


if __name__ == "__main__":
    main()
