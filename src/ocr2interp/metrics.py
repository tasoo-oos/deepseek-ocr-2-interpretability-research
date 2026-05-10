from __future__ import annotations

import editdistance


def exact_match(prediction: str, target: str) -> bool:
    return prediction == target


def normalized_edit_distance(prediction: str, target: str) -> float:
    denom = max(len(prediction), len(target), 1)
    return editdistance.eval(prediction, target) / denom


def line_level_recovery(prediction: str, target: str) -> float:
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    if not target_lines:
        return 1.0
    prediction_lines = {line.strip() for line in prediction.splitlines() if line.strip()}
    return sum(line in prediction_lines for line in target_lines) / len(target_lines)


def span_level_recovery(prediction: str, spans: list[dict]) -> float:
    if not spans:
        return 1.0
    return sum(span.get("text", "") in prediction for span in spans) / len(spans)
