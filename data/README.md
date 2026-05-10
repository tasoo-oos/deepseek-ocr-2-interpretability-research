# Data

Raw benchmark data is local-only and must not be committed.

Download OmniDocBench with:

```bash
make download-omnidocbench
```

This writes to `data/raw/OmniDocBench/`, which is ignored by git.

Use `HF_TOKEN` in the environment if Hugging Face rate limits anonymous downloads.
