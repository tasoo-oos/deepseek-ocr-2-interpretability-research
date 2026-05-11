# AGENTS.md

Guidance for agents working in this repository.

## Project Scope

This repo is a DeepSeek-OCR2 interpretability workspace. The current v0 scope is:

- reproducible DeepSeek-OCR2 inference
- config-driven runs with Hydra
- module inspection
- activation capture hooks
- local OmniDocBench download workflow
- paper/output directory conventions

Later work can add ablations, clean/corrupt patching, synthetic probes, figures, and tables, but do not jump to mechanistic claims before validating capture and intervention workflows.

## Core Rules

- Use `uv` for Python dependency and command execution.
- Keep upstream DeepSeek code under `upstream-ocr-2/` unmodified unless explicitly asked.
- Do not commit raw datasets, model weights, activation dumps, generated run outputs, or Hydra run logs.
- Keep `data/raw/OmniDocBench/` local-only. Recreate it with `make download-omnidocbench`.
- Prefer small, direct changes over broad rewrites.
- Do not add compatibility layers unless there is a concrete need.
- Keep default configs in interpretability mode with eager attention unless a task explicitly needs fast inference.

## Common Commands

Install/sync dependencies:

```bash
uv sync
```

Download OmniDocBench locally:

```bash
make download-omnidocbench
```

Run one configured inference:

```bash
uv run python scripts/infer_one.py input.image_file=data/raw/example.png output.dir=outputs/runs/example
```

Run the known OmniDocBench smoke sample:

```bash
uv run python scripts/infer_one.py input.image_file=data/raw/OmniDocBench/images/PPT_1001115_eng_page_003.png output.dir=outputs/runs/omnidocbench_smoke
```

Inspect modules:

```bash
make inspect
```

Capture activations after choosing real module names:

```bash
uv run python scripts/capture_activations.py capture.modules='["MODULE_NAME_HERE"]'
```

Run the OmniDocBench activation survey in a detached background job:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_omnidocbench_activation_survey.py --resume --output-dir outputs/runs/omnidocbench_activation_survey > logs/omnidocbench_activation_survey.out 2>&1 &
```

Inspect the survey and write a timestamped Markdown log:

```bash
uv run python scripts/inspect_omnidocbench_activation_survey.py --output-dir outputs/runs/omnidocbench_activation_survey --name omnidocbench_activation_survey
```

Verify code changes:

```bash
uv run ruff check src scripts
PYTHONPATH=src uv run python -m compileall src scripts
```

## Repository Layout

- `configs/`: Hydra configs for inference, capture, and future patching.
- `src/ocr2interp/`: project package code.
- `scripts/`: runnable workflows.
- `data/README.md`: dataset policy and download notes.
- `data/raw/`: ignored local datasets.
- `outputs/runs/`: ignored run artifacts.
- `paper/`: paper skeleton and generated artifact destinations.
- `upstream-ocr-2/`: upstream DeepSeek-OCR2 reference code.

## Dependency Notes

DeepSeek-OCR2 remote code currently expects the upstream-compatible stack pinned in `pyproject.toml`, including:

- `transformers==4.46.3`
- `tokenizers==0.20.3`
- `addict`
- `requests`
- `easydict`
- `PyMuPDF`
- `img2pdf`

Avoid upgrading `transformers` casually; newer versions can break remote-code imports such as `LlamaFlashAttention2`.

## Long-Running Runs

- Run long jobs detached with `tmux` or another background-job mechanism, and write stdout/stderr under `logs/`.
- Store PID files under `logs/` when launching detached jobs so status can be checked later.
- The high-level DeepSeek `model.infer()` path is single-image and mostly serial; full OmniDocBench activation surveys can become CPU/Python-bound even on an RTX 4090.
- Observed behavior on this machine: one survey process used about one CPU core, about 9.7 GB GPU memory, and only about 25-30% GPU utilization on dense pages.
- If throughput matters, prefer sharding over trying to batch through `model.infer()`.
- The survey script supports `--num-shards N` and `--shard-index I`; use separate output JSONL names per shard.
- On a single RTX 4090, two concurrent shard processes may better saturate the GPU because each process leaves VRAM headroom, but three or more can OOM when KV/cache usage spikes.
- For multi-GPU machines, run one shard per GPU with separate `CUDA_VISIBLE_DEVICES` values.
- Do not commit survey outputs, transcripts, activation summaries, or generated logs unless the user explicitly asks for a small Markdown report to be tracked.

Example two-shard single-GPU launch, only if VRAM headroom is confirmed:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_omnidocbench_activation_survey.py --resume --num-shards 2 --shard-index 0 --output-dir outputs/runs/omnidocbench_activation_survey_sharded > logs/omnidocbench_activation_survey_shard0.out 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_omnidocbench_activation_survey.py --resume --num-shards 2 --shard-index 1 --output-dir outputs/runs/omnidocbench_activation_survey_sharded > logs/omnidocbench_activation_survey_shard1.out 2>&1 &
```

Check status with:

```bash
nvidia-smi
ps -p PID -o pid,ppid,stat,pcpu,pmem,etime,cmd
```

## Code Conventions

- Use small, direct implementations and avoid broad rewrites.
- Prefer config-driven scripts and explicit output directories.
- Do not store full activation tensors for broad surveys unless the task explicitly requires them; use JSONL summaries for dataset-wide diagnostics.
- Keep comments rare and focused on non-obvious behavior.
- Verify Python changes with `uv run ruff check src scripts` and `PYTHONPATH=src uv run python -m compileall src scripts`.

## Documentation And Logs

- Keep `README.md` practical and user-facing.
- Keep `AGENTS.md` operational and agent-facing.
- Use `logs/journel/YYYY-MM-DD_session.md` for human session journals.
- Use `logs/<timestamp>_<name>.md` for generated experiment summaries.
- Use `logs/*.out` for detached command stdout/stderr and `logs/*.pid` for process IDs; these are ignored by git.
- Markdown logs can be committed when they are small and useful; raw outputs, `.out` logs, PID files, activation dumps, and datasets should not be committed.
- When documenting an experiment, include the command, output directory, dataset path, model/config notes, verification status, and findings.
- Update `README.md` for user-facing workflows, `AGENTS.md` for future-agent operating rules, and `data/README.md` for dataset policy changes.

## Git Hygiene

Before committing, check:

```bash
git status --short
git diff --cached --stat
```

Do not stage:

- `data/raw/`
- `outputs/runs/`
- `outputs/YYYY-MM-DD/`
- `.venv/`
- `*.pt`, `*.pth`, `*.safetensors`, `*.ckpt`
- `__pycache__/` or `*.pyc`

## Current Status

The v0 bootstrap from the old `PLAN.md` is applied for the usable repo target:

- central model loader exists
- config-driven inference exists
- module inspection script exists
- activation capture hooks exist
- OmniDocBench download workflow exists
- raw datasets and run artifacts are ignored

Future milestones from the old plan, such as robust ablations, clean/corrupt patching, synthetic dataset probes, and paper figure/table generation, are intentionally not fully implemented yet.
