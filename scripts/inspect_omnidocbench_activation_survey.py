from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_rows(output_dir: Path) -> list[dict]:
    rows = []
    paths = list(output_dir.glob("activation_summary*.jsonl"))
    paths.extend((output_dir / "log").glob("activation_summary*.jsonl"))
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source_file"] = str(path)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OmniDocBench activation survey summaries.")
    parser.add_argument("--output-dir", default="outputs/runs/run_2_omnidocbench_activation_survey")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--name", default="omnidocbench_activation_survey")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "log"
    rows = load_rows(output_dir)
    ok_rows = [row for row in rows if row.get("ok")]
    error_rows = [row for row in rows if not row.get("ok")]
    modules = sorted({module for row in ok_rows for module in row.get("activation_summary", {})})

    elapsed = [float(row.get("elapsed_sec", 0.0)) for row in ok_rows]
    by_module = defaultdict(lambda: defaultdict(list))
    shape_counts = defaultdict(Counter)
    call_counts = defaultdict(Counter)

    for row in ok_rows:
        for module, summary in row.get("activation_summary", {}).items():
            for key in ("mean_avg", "std_avg", "abs_mean_avg"):
                value = summary.get(key)
                if value is not None:
                    by_module[module][key].append(float(value))
            for shape in summary.get("shapes", []):
                shape_counts[module][shape] += 1
            call_counts[module][summary.get("calls", 0)] += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}_{args.name}.md"

    lines = [
        "# OmniDocBench Activation Survey",
        "",
        f"Output directory: `{output_dir}`",
        f"Records read: {len(rows)}",
        f"Successful pages: {len(ok_rows)}",
        f"Failed pages: {len(error_rows)}",
        f"Mean page time: {fmt(avg(elapsed))} sec",
        f"Min page time: {fmt(min(elapsed) if elapsed else None)} sec",
        f"Max page time: {fmt(max(elapsed) if elapsed else None)} sec",
        "",
        "## Hooked Modules",
        "",
    ]

    for module in modules:
        lines.extend(
            [
                f"### `{module}`",
                "",
                f"Call-count distribution: `{dict(sorted(call_counts[module].items()))}`",
                f"Mean activation mean: {fmt(avg(by_module[module]['mean_avg']))}",
                f"Mean activation std: {fmt(avg(by_module[module]['std_avg']))}",
                f"Mean absolute activation: {fmt(avg(by_module[module]['abs_mean_avg']))}",
                "Most common shapes:",
            ]
        )
        for shape, count in shape_counts[module].most_common(8):
            lines.append(f"- `{shape}`: {count} pages")
        lines.append("")

    if error_rows:
        lines.extend(["## Failures", ""])
        for row in error_rows[:50]:
            lines.append(f"- `{row.get('image')}`: {row.get('error_type')} {row.get('error')}")
        if len(error_rows) > 50:
            lines.append(f"- Additional failures omitted: {len(error_rows) - 50}")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "This survey stores activation summary statistics, not full activation tensors.",
            "The fixed run sets explicit generation token IDs and uses eval mode to avoid streaming output noise.",
            "Findings are descriptive only; causal claims require controlled ablations or patching.",
            "",
        ]
    )

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(log_path)


if __name__ == "__main__":
    main()
