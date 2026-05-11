#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
paper_dir="$repo_root/paper"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required to build paper/main.pdf" >&2
  exit 1
fi

cd "$paper_dir"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

if command -v bibtex >/dev/null 2>&1 && grep -q "\\bibdata" main.aux; then
  bibtex main >/dev/null || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
fi

test -f main.pdf
