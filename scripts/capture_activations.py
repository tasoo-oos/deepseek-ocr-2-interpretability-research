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


def capture_module_state(model, module_names) -> dict[str, dict[str, object]]:
    modules = dict(model.named_modules())
    captured = {}
    for module_name in module_names:
        if module_name not in modules:
            continue
        module = modules[module_name]
        state = {}
        for name, value in module.state_dict().items():
            state[name] = value.detach().cpu()
        if state:
            captured[module_name] = state
    return captured


@hydra.main(config_path="../configs", config_name="capture", version_base=None)
def main(cfg) -> None:
    output_dir = Path(to_absolute_path(cfg.output.dir))
    log_dir = output_dir / "log"
    image_file = Path(to_absolute_path(cfg.input.image_file))
    seed_everything(cfg.seed)
    save_config(cfg, log_dir / "config.yaml")

    model, tokenizer = load_ocr2(ModelConfig(**cfg.model))
    store = ActivationStore()
    manager = HookManager()
    try:
        for module_name in cfg.capture.modules:
            manager.add_activation_hook(model, module_name, store)
        result = run_model_infer(model, tokenizer, cfg, image_file=image_file, output_dir=log_dir)
    finally:
        manager.close()

    save_result_repr(result, log_dir)
    torch.save(store.activations, log_dir / "activations.pt")
    state_modules = cfg.capture.get("state_modules", [])
    torch.save(capture_module_state(model, state_modules), log_dir / "captured_module_state.pt")
    write_json(store.shapes(), log_dir / "activation_shapes.json")
    write_json(store.metadata, log_dir / "activation_metadata.json")
    write_json(store.call_counts(), log_dir / "activation_call_counts.json")
    for name, count in store.call_counts().items():
        print(f"{name}: {count} calls")


if __name__ == "__main__":
    main()
