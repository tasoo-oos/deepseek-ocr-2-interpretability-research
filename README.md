# DeepSeek-OCR2 Interpretability

Tools for running DeepSeek-OCR2 inference, inspecting modules, capturing activations, and producing experiment artifacts.

## Setup

```bash
uv sync
```

This project uses a `src/` layout and keeps upstream DeepSeek code under `upstream-ocr-2/` unmodified.

## Smoke Test

Put a test image at:

```text
data/raw/example.png
```

Run:

```bash
make smoke
```

The smoke run writes to `outputs/runs/smoke/`, including `config.yaml`, `result_repr.txt`, and any files emitted by `model.infer`.

## Config-Driven Inference

```bash
uv run python scripts/infer_one.py input.image_file=data/raw/example.png output.dir=outputs/runs/example
```

Default inference config lives at `configs/infer.yaml`.

## Module Inspection

```bash
make inspect
```

This writes the module tree to `outputs/module_tree.txt`. Use it to identify hook targets such as vision encoder layers, decoder blocks, attention layers, MLPs, and any exposed routing modules.

## Activation Capture

After identifying module names, run:

```bash
uv run python scripts/capture_activations.py capture.modules='["MODULE_NAME_HERE"]'
```

The capture run writes `config.yaml`, `result_repr.txt`, `activations.pt`, and `activation_shapes.json` under `outputs/runs/capture_test/` by default.

## Artifact Rules

Do not commit model weights, raw datasets, activation dumps, or run outputs. Large artifacts are ignored by `.gitignore`.
