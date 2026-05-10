# PLAN.md — DeepSeek-OCR2 Interpretability Repo Bootstrap

This repo is for running DeepSeek-OCR2 inference, capturing internals, building probes, and generating paper artifacts.

Keep it simple first:

- use `uv` for Python/package management
- keep upstream DeepSeek code unmodified
- start with reproducible inference
- then add hooks, captures, ablations, and patching
- generate figures/tables from saved experiment outputs

---

## 0. Target repo shape

```text
deepseek-ocr2-interp/
  README.md
  PLAN.md
  pyproject.toml
  uv.lock
  Makefile
  .gitignore
  .python-version

  configs/
    infer.yaml
    capture.yaml
    patching.yaml

  src/
    ocr2interp/
      __init__.py
      loading.py
      infer.py
      hooks.py
      capture.py
      patching.py
      datasets.py
      metrics.py
      io.py
      utils.py

  scripts/
    smoke_test.py
    infer_one.py
    infer_batch.py
    inspect_modules.py
    capture_activations.py
    run_patching.py
    make_figures.py
    export_tables.py

  data/
    raw/
    processed/
    synthetic/

  outputs/
    runs/
    figures/
    tables/

  paper/
    main.tex
    refs.bib
    macros.tex
    sections/
    figs/
    tables/
```

---

## 1. Initialize the repo

```bash
mkdir deepseek-ocr2-interp
cd deepseek-ocr2-interp

git init
uv init --python 3.12
rm -f main.py

mkdir -p \
  configs \
  src/ocr2interp \
  scripts \
  data/raw data/processed data/synthetic \
  outputs/runs outputs/figures outputs/tables \
  paper/sections paper/figs paper/tables

touch src/ocr2interp/__init__.py
```

Commit the empty skeleton early:

```bash
git add .
git commit -m "Initialize repo skeleton"
```

---

## 2. Add dependencies with uv

Start with the interpretability/debug environment first.

```bash
uv add torch torchvision torchaudio --torch-backend=auto

uv add \
  transformers \
  accelerate \
  safetensors \
  pillow \
  numpy \
  pandas \
  tqdm \
  matplotlib \
  omegaconf \
  hydra-core \
  rich \
  editdistance \
  opencv-python \
  einops

uv add --dev \
  ruff \
  pytest \
  ipykernel \
  pre-commit
```

Use `--torch-backend=auto` initially. If exact CUDA control is needed later, switch to an explicit backend such as `cu118`, `cu121`, `cu124`, etc.

For the first interpretability pass, avoid `flash-attn`. Use eager attention.

Optional fast-path dependency later:

```bash
uv pip install flash-attn==2.7.3 --no-build-isolation
```

Do not make `flash-attn` mandatory until basic eager-mode probing works.

---

## 3. Configure pyproject.toml

Make sure the package uses the `src/` layout.

Add or verify:

```toml
[project]
name = "ocr2interp"
version = "0.1.0"
requires-python = ">=3.12"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

Then sync:

```bash
uv sync
```

Run commands through uv:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Add .gitignore

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/

# uv / env
.venv/
.env

# data and large artifacts
data/raw/
data/processed/
outputs/runs/
outputs/tmp/
*.pt
*.pth
*.safetensors
*.ckpt

# paper build files
paper/*.aux
paper/*.bbl
paper/*.blg
paper/*.fdb_latexmk
paper/*.fls
paper/*.log
paper/*.out
paper/*.toc
paper/*.synctex.gz

# local editor
.vscode/
.idea/
```

Commit lockfile and config files. Do not commit model weights, activation dumps, or raw datasets.

---

## 5. Create the first configs

`configs/infer.yaml`:

```yaml
seed: 0

model:
  name: deepseek-ai/DeepSeek-OCR-2
  device: cuda
  dtype: bfloat16
  attn_implementation: eager
  trust_remote_code: true
  use_safetensors: true

infer:
  prompt: "<image>\n<|grounding|>Convert the document to markdown."
  base_size: 1024
  image_size: 768
  crop_mode: true
  save_results: true

input:
  image_file: data/raw/example.png

output:
  dir: outputs/runs/smoke
```

`configs/capture.yaml`:

```yaml
seed: 0

model:
  name: deepseek-ai/DeepSeek-OCR-2
  device: cuda
  dtype: bfloat16
  attn_implementation: eager
  trust_remote_code: true
  use_safetensors: true

infer:
  prompt: "<image>\n<|grounding|>Convert the document to markdown."
  base_size: 1024
  image_size: 768
  crop_mode: true
  save_results: false

input:
  image_file: data/raw/example.png

output:
  dir: outputs/runs/capture_test

capture:
  modules: []
```

Keep `flash_attention_2` out of the default config. Add it only for fast inference sweeps.

---

## 6. Implement core modules

### `src/ocr2interp/loading.py`

Purpose:

- load tokenizer
- load model
- centralize dtype/device/attention settings
- avoid duplicated `AutoModel.from_pretrained(...)` calls

Required API:

```python
@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    use_safetensors: bool


def load_ocr2(cfg: ModelConfig):
    ...
```

### `src/ocr2interp/hooks.py`

Purpose:

- register forward hooks
- capture activations
- later support modifying activations

Required API:

```python
class ActivationStore:
    def clear(self) -> None: ...
    def save(self, name: str, tensor) -> None: ...

class HookManager:
    def add_activation_hook(self, model, module_name, store) -> None: ...
    def close(self) -> None: ...
```

### `src/ocr2interp/utils.py`

Purpose:

- seeding
- config saving
- small helpers

Required API:

```python
def seed_everything(seed: int) -> None: ...
def save_config(cfg, path) -> None: ...
```

---

## 7. Implement first scripts

### `scripts/smoke_test.py`

Goal:

- prove the model runs on one image
- save output to `outputs/runs/smoke/`

Use the official high-level call:

```python
model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_dir,
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True,
)
```

### `scripts/infer_one.py`

Goal:

- same as smoke test, but config-driven through Hydra

Run:

```bash
uv run python scripts/infer_one.py input.image_file=data/raw/example.png output.dir=outputs/runs/example
```

### `scripts/inspect_modules.py`

Goal:

- print `model.named_modules()`
- save the module tree
- identify hook targets

Run:

```bash
uv run python scripts/inspect_modules.py > outputs/module_tree.txt
```

### `scripts/capture_activations.py`

Goal:

- load configured module names
- register hooks
- run inference
- save `activations.pt`

Run:

```bash
uv run python scripts/capture_activations.py capture.modules='["MODULE_NAME_HERE"]'
```

---

## 8. Add Makefile shortcuts

```makefile
.PHONY: sync smoke infer inspect capture lint test clean

sync:
	uv sync

smoke:
	CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test.py

infer:
	CUDA_VISIBLE_DEVICES=0 uv run python scripts/infer_one.py

inspect:
	CUDA_VISIBLE_DEVICES=0 uv run python scripts/inspect_modules.py > outputs/module_tree.txt

capture:
	CUDA_VISIBLE_DEVICES=0 uv run python scripts/capture_activations.py

lint:
	uv run ruff check src scripts

test:
	uv run pytest

clean:
	rm -rf outputs/runs/tmp* .pytest_cache .ruff_cache
```

---

## 9. First milestone: reproducible inference

Definition of done:

- `uv sync` works on a clean clone
- `make smoke` runs on one image
- output directory contains:

```text
outputs/runs/smoke/
  config.yaml
  result_repr.txt
  generated output files
```

Commit:

```bash
git add .
git commit -m "Add reproducible OCR2 inference"
```

---

## 10. Second milestone: module discovery

Definition of done:

- `make inspect` writes `outputs/module_tree.txt`
- candidate encoder/decoder module names are manually identified
- notes are added to `README.md`

Look for modules related to:

- image preprocessing
- visual tokenizer
- DeepEncoder / vision encoder
- causal-flow tokens
- decoder blocks
- attention layers
- MLP layers
- MoE routing, if exposed

Commit:

```bash
git add scripts src configs README.md
 git commit -m "Add module inspection workflow"
```

---

## 11. Third milestone: activation capture

Definition of done:

- capture config lists real module names
- `make capture` saves `activations.pt`
- tensor names and shapes are logged
- no massive accidental Git commits

Output shape target:

```text
outputs/runs/capture_test/
  config.yaml
  result_repr.txt
  activations.pt
  activation_shapes.json
```

Commit:

```bash
git add scripts src configs
 git commit -m "Add activation capture hooks"
```

---

## 12. Fourth milestone: first intervention

Start with simple ablation before full causal patching.

Implement:

- zero ablation
- mean ablation
- token-index ablation
- module-output replacement hook

First target:

```text
selected hidden state tensor
  batch dimension
  token dimension
  hidden dimension
```

Minimal experiment:

```text
1. run clean inference
2. ablate one token group
3. compare output edit distance
4. save metrics.json
```

Do not start with full clean/corrupt patching until simple ablation works.

---

## 13. Fifth milestone: clean/corrupt patching

Only after activation capture and ablation are stable.

Required pieces:

- clean image
- corrupted image
- clean activation cache
- corrupt forward pass
- patch hook
- span-level metric

Output:

```text
outputs/runs/patching_*/
  config.yaml
  clean_output.md
  corrupt_output.md
  patched_output.md
  metrics.json
```

Start with synthetic documents where text spans and coordinates are known.

---

## 14. Dataset format

Use JSONL for examples.

```json
{
  "id": "synthetic_000001",
  "image_path": "data/synthetic/synthetic_000001.png",
  "target_markdown_path": "data/synthetic/synthetic_000001.md",
  "spans": [
    {
      "span_id": "line_000",
      "text": "DeepSeek-OCR2 converts document images to Markdown.",
      "bbox": [64, 96, 900, 130],
      "kind": "line",
      "reading_order": 0
    }
  ]
}
```

Keep early synthetic data small and easy to inspect manually.

---

## 15. Metrics to implement first

In `src/ocr2interp/metrics.py`:

- exact match
- normalized edit distance
- line-level recovery
- span-level recovery

Later:

- reading-order edit distance
- table-cell F1
- formula exact match
- Markdown structural validity

Do not overbuild metrics before the first intervention works.

---

## 16. Paper integration

Paper artifacts should be generated, not manually pasted.

Scripts:

```text
scripts/make_figures.py
scripts/export_tables.py
```

Generated files:

```text
outputs/figures/*.pdf
outputs/tables/*.csv
paper/figs/*.pdf
paper/tables/*.tex
```

Rule:

```text
raw run outputs -> aggregate CSV/JSON -> paper figure/table
```

---

## 17. Fast mode vs interpretability mode

Use two configs later:

```text
configs/infer_fast.yaml
configs/infer_interp.yaml
```

Fast mode:

```yaml
attn_implementation: flash_attention_2
```

Interpretability mode:

```yaml
attn_implementation: eager
```

Default to interpretability mode while building hooks.

---

## 18. Initial README checklist

README should contain:

```text
1. Project goal
2. Setup commands
3. Smoke test command
4. Expected input image path
5. Output directory convention
6. How to inspect modules
7. How to run activation capture
8. Warning: do not commit model weights or activations
```

Keep README practical, not essay-like.

---

## 19. Bootstrap order

Do this in order:

```text
1. repo skeleton
2. uv environment
3. smoke test
4. config-driven inference
5. module inspection
6. activation capture
7. simple ablation
8. clean/corrupt patching
9. synthetic dataset probes
10. figure/table generation
11. paper draft
```

Do not skip from smoke test directly to full mechanistic claims.

---

## 20. Definition of a usable v0 repo

The repo is usable when this works from a clean clone:

```bash
uv sync
make smoke
make inspect
make capture
```

And the repo contains:

```text
- central model loader
- config-driven inference
- module inspection script
- activation capture hook system
- run directory convention
- no large artifacts committed
```

That is enough for v0. Everything else can grow from there.
