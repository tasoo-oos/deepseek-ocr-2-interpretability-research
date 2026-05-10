from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path

from ocr2interp.hooks import ActivationStore, HookManager
from ocr2interp.infer import run_model_infer, save_result_repr
from ocr2interp.io import write_json
from ocr2interp.loading import ModelConfig, load_ocr2
from ocr2interp.utils import save_config, seed_everything


@hydra.main(config_path="../configs", config_name="capture", version_base=None)
def main(cfg) -> None:
    output_dir = Path(to_absolute_path(cfg.output.dir))
    image_file = Path(to_absolute_path(cfg.input.image_file))
    seed_everything(cfg.seed)
    save_config(cfg, output_dir / "config.yaml")

    model, tokenizer = load_ocr2(ModelConfig(**cfg.model))
    store = ActivationStore()
    manager = HookManager()
    try:
        for module_name in cfg.capture.modules:
            manager.add_activation_hook(model, module_name, store)
        result = run_model_infer(model, tokenizer, cfg, image_file=image_file, output_dir=output_dir)
    finally:
        manager.close()

    save_result_repr(result, output_dir)
    torch.save(store.activations, output_dir / "activations.pt")
    write_json(store.shapes(), output_dir / "activation_shapes.json")
    for name, shape in store.shapes().items():
        print(f"{name}: {shape}")


if __name__ == "__main__":
    main()
