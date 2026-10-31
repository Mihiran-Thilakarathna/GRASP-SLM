"""Metrics that need no model: schema validity, citation validity and numeric accuracy (Table 4)."""
from __future__ import annotations

import json
import re

from .common import schema_errors

NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")


def parse(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def schema_valid(obj: dict | None) -> bool:
    return obj is not None and not schema_errors(obj)


def numbers(text: str) -> set[float]:
    """Numeric values in a string. '1,024' and '1024' are equal; '92.7%' is 92.7."""
    out = set()
    for m in NUMBER.finditer(text or ""):
        try:
            out.add(round(float(m.group().replace(",", "")), 6))
        except ValueError:
            pass
    return out


def metric_numbers(obj: dict | None) -> set[float]:
    if not obj:
        return set()
    return set().union(*[numbers(m.get("value", "")) for m in obj.get("key_metrics", []) if isinstance(m, dict)])


def numeric_accuracy(pred: dict | None, reference: dict, source_text: str) -> dict:
    """
    grounded_precision: share of numbers in the predicted metric values that occur in the paper
                        (a number not in the paper is a hallucinated value)
    reference_recall:   share of numbers in the verified metric values that the prediction reports
    f1:                 harmonic mean of the two
    """
    p, r, src = metric_numbers(pred), metric_numbers(reference), numbers(source_text)
    precision = len(p & src) / len(p) if p else (1.0 if not r else 0.0)
    recall = len(p & r) / len(r) if r else (1.0 if not p else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"grounded_precision": precision, "reference_recall": recall, "numeric_f1": f1,
            "n_pred_numbers": len(p), "n_ref_numbers": len(r)}


def citation_validity(obj: dict | None, context_ids: list[str]) -> float | None:
    """Share of citations that point at a chunk the model was actually given."""
    if not obj:
        return None
    cited = []
    for field in ("core_problem", "proposed_method"):
        cited += (obj.get(field) or {}).get("citations", []) if isinstance(obj.get(field), dict) else []
    for item in (obj.get("key_metrics") or []) + (obj.get("limitations") or []):
        if isinstance(item, dict):
            cited += item.get("citations", [])
    if not cited:
        return 0.0
    allowed = set(context_ids)
    return sum(c in allowed for c in cited) / len(cited)


def field_texts(obj: dict | None) -> dict[str, str]:
    """Plain text per field, used for BERTScore against the verified reference."""
    obj = obj or {}

    def text(c):
        return c.get("text", "") if isinstance(c, dict) else ""

    return {
        "core_problem": text(obj.get("core_problem")),
        "proposed_method": text(obj.get("proposed_method")),
        "key_metrics": "; ".join(f"{m.get('metric', '')}: {m.get('value', '')} ({m.get('dataset', '')})"
                                 for m in obj.get("key_metrics") or [] if isinstance(m, dict)),
        "limitations": "; ".join(text(c) for c in obj.get("limitations") or []),
    }


def as_answer(obj: dict | None) -> str:
    """The analysis as prose, used as the Ragas 'response'."""
    t = field_texts(obj)
    return (f"Core problem: {t['core_problem']}\nProposed method: {t['proposed_method']}\n"
            f"Key metrics: {t['key_metrics'] or 'none reported'}\nLimitations: {t['limitations'] or 'none stated'}")
