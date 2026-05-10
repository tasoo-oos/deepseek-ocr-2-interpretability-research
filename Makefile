.PHONY: sync download-omnidocbench smoke infer inspect capture lint test clean

sync:
	uv sync

download-omnidocbench:
	uv run python scripts/download_omnidocbench.py

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
