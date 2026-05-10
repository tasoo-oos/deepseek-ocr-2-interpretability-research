from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OmniDocBench into ignored local data.")
    parser.add_argument("--repo-id", default="opendatalab/OmniDocBench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="data/raw/OmniDocBench")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output_dir,
    )
    print(f"Downloaded {args.repo_id}@{args.revision} to {path}")


if __name__ == "__main__":
    main()
