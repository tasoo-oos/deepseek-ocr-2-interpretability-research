# OmniDocBench Bulk Inference

## Benchmark Format

OmniDocBench's official documentation describes a local dataset layout like this:

```text
OmniDocBench/
|- images/
|- pdfs/
`- OmniDocBench.json
```

The `OmniDocBench.json` file is a list of page-level entries. The fields most relevant for bulk inference are:

- `page_info.image_path`: relative image filename for the page
- `page_info.page_no`: page number
- `page_info.width` / `page_info.height`: page size
- `page_info.page_attribute`: page-level attributes like `language`, `layout`, `data_source`, etc.
- `layout_dets`: block/span annotations used by the benchmark evaluator

For end-to-end evaluation, OmniDocBench expects model predictions as markdown files whose names match the image filename stem, for example:

- `images/page_001.jpg` -> `page_001.md`

## Bulk Runner

This repo now includes a dataset-aware bulk runner:

- Loader: `src/benchmarks/omnidocbench.py`
- CLI: `scripts/run_omnidocbench.py`

### Dry Run

Validate the dataset mapping and planned outputs without loading the model:

```bash
.venv/bin/python scripts/run_omnidocbench.py \
  --dataset_root /path/to/OmniDocBench \
  --output_dir output/omnidocbench \
  --limit 20 \
  --dry_run
```

This writes `output/omnidocbench/run_manifest.jsonl` and prints a short summary.

### Real Bulk Inference

Run sequential inference over many benchmark pages:

```bash
.venv/bin/python scripts/run_omnidocbench.py \
  --dataset_root /path/to/OmniDocBench \
  --output_dir output/omnidocbench \
  --model_path deepseek-ai/DeepSeek-OCR-2 \
  --limit 100
```

The runner now loads the real upstream Hugging Face model path through `DeepseekOCRPipeline.from_pretrained(...)`, so it works with `deepseek-ai/DeepSeek-OCR-2` on GPU without requiring `AutoModelForCausalLM` support for the custom config.

### Filtering

You can filter by page-level attributes from `page_info.page_attribute`:

```bash
.venv/bin/python scripts/run_omnidocbench.py \
  --dataset_root /path/to/OmniDocBench \
  --output_dir output/omnidocbench_english \
  --filter language=english \
  --filter layout=single_column \
  --limit 50 \
  --dry_run
```

## Output Files

The runner writes:

- one `.md` file per benchmark page
- `run_manifest.jsonl` with sample id, source image path, output path, page attributes, and status

This output layout is designed to be easy to inspect before sending the markdown directory into OmniDocBench's own evaluator.

## Output Validation

This repo also includes a lightweight checker for runner outputs:

```bash
.venv/bin/python scripts/check_omnidocbench_outputs.py \
  --output_dir output/omnidocbench \
  --verbose
```

The checker validates that:

- every `written` or `skipped` manifest entry points to an existing markdown file
- written markdown files are non-empty
- aggregate counts and per-page stats are sane

If you have benchmark reference markdown files, you can also compare predictions against them:

```bash
.venv/bin/python scripts/check_omnidocbench_outputs.py \
  --output_dir output/omnidocbench \
  --gt_dir /path/to/reference_markdown \
  --json
```

The optional metrics include:

- character-level normalized edit-distance ratio
- simple word-level BLEU-4
- exact-match rate
