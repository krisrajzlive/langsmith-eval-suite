"""Judge calibration statistics: how closely an LLM judge agrees with human labels."""

from __future__ import annotations


def confusion(human: list[int], judge: list[int]) -> dict[str, int]:
    pairs = list(zip(human, judge))
    return {
        "tp": sum(1 for h, j in pairs if h == 1 and j == 1),
        "tn": sum(1 for h, j in pairs if h == 0 and j == 0),
        "fp": sum(1 for h, j in pairs if h == 0 and j == 1),
        "fn": sum(1 for h, j in pairs if h == 1 and j == 0),
    }


def cohen_kappa(human: list[int], judge: list[int]) -> float:
    """Chance-corrected agreement for two binary raters. 1.0 is perfect, 0.0 is chance level."""
    n = len(human)
    if n == 0:
        return float("nan")
    c = confusion(human, judge)
    observed = (c["tp"] + c["tn"]) / n
    p_h1, p_j1 = (c["tp"] + c["fn"]) / n, (c["tp"] + c["fp"]) / n
    expected = p_h1 * p_j1 + (1 - p_h1) * (1 - p_j1)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def calibration_summary(outputs: list[dict], reference_outputs: list[dict]) -> list[dict]:
    """Summary evaluator over judge verdicts (outputs) and human labels (reference_outputs)."""
    pairs = [(int(r["human_label"]), int(o["verdict"])) for o, r in zip(outputs, reference_outputs) if o and "verdict" in o]
    if not pairs:
        return [{"key": "judge_accuracy", "score": None, "comment": "no verdicts"}]
    human, judge = [p[0] for p in pairs], [p[1] for p in pairs]
    c = confusion(human, judge)
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    return [
        {"key": "judge_accuracy", "score": (c["tp"] + c["tn"]) / len(pairs)},
        {"key": "judge_kappa", "score": cohen_kappa(human, judge)},
        {"key": "judge_true_positive_rate", "score": c["tp"] / pos if pos else None},
        {"key": "judge_true_negative_rate", "score": c["tn"] / neg if neg else None},
    ]
