# TODO

## Model Code

- Validate `src.models.deepseek_ocr_v1.DeepseekOCRV1Model.from_pretrained(...)`
  against the upstream `deepseek-ai/DeepSeek-OCR` remote-code model on a small
  image batch.
- Decide whether OCR v1 should expose language-model token merging locally, or
  whether full OCR v1 text generation should remain delegated to
  `src.inference.deepseek_ocr_v1.DeepSeekOCRV1Pipeline`.
- Add a small integration test that compares OCR v1 local vision features
  against the upstream model modules when weights are available.

## Experiments

- Add unit tests for `src/experiments/query_trace_mask_ablation.py` geometry
  helpers and aggregation logic.
- Add unit tests for `src/experiments/real_doc_ordering.py` page selection,
  target construction, and ablation summaries.
- Move any remaining reusable code from large CLI scripts into `src/` modules;
  keep `scripts/` as command wrappers only.

## Outputs And Docs

- Treat `output/query_trace_mask_ablation_v2_full/` as the canonical full-sweep
  artifact unless a newer rerun supersedes it.
- If canonical outputs are too large for local development, archive them outside
  the repo and keep only summaries needed by `docs/RESEARCH_AUDIT.md`.
- Update `docs/API.md` with OCR v1 model and inference APIs.
- Add a short OCR v1 usage section to `docs/OMNIDOCBENCH.md`.

## Verification

- Run the full CPU test suite before merging.
- Run GPU-gated inference tests for both OCR v2 and OCR v1 on a CUDA machine.
- Re-run the v1 OmniDocBench smoke test after any changes to
  `src/inference/deepseek_ocr_v1.py`.
