"""Rejected answers for DPO (proposal section 6.5): copies of a verified answer with one defect."""
from __future__ import annotations

import copy
import json
import random
import re

from .common import citations, to_target

NUMBER = re.compile(r"\d+(?:\.\d+)?")
KINDS = ("altered_number", "invented_metric", "missing_citation", "malformed_json")
METRIC_NAMES = ("Accuracy", "F1", "BLEU", "AUC", "Precision", "Recall", "Top-1 accuracy", "mAP", "RMSE")
DATASETS = ("ImageNet", "GLUE", "SQuAD", "CIFAR-10", "MS MARCO", "COCO", "WMT14 En-De")


def _change_number(text: str, rng: random.Random) -> str | None:
    matches = list(NUMBER.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    old = m.group()
    if "." in old:
        whole, frac = old.split(".")
        digits = list(frac)
        i = rng.randrange(len(digits))
        digits[i] = str((int(digits[i]) + rng.randint(1, 9)) % 10)
        new = whole + "." + "".join(digits)
    else:
        new = str(int(old) + rng.choice([-1, 1]) * rng.randint(1, max(2, int(old) // 10 + 1)))
        new = new.lstrip("-") or "0"
    if new == old:
        return None
    return text[:m.start()] + new + text[m.end():]


def altered_number(obj: dict, rng: random.Random) -> dict | None:
    out = copy.deepcopy(obj)
    for m in rng.sample(out["key_metrics"], len(out["key_metrics"])):
        changed = _change_number(m["value"], rng)
        if changed:
            m["value"] = changed
            return out
    claims = [out["core_problem"], out["proposed_method"]] + out["limitations"]
    for c in rng.sample(claims, len(claims)):
        changed = _change_number(c["text"], rng)
        if changed:
            c["text"] = changed
            return out
    return None


def invented_metric(obj: dict, rng: random.Random) -> dict | None:
    out = copy.deepcopy(obj)
    cited = citations(obj) or ["c0"]
    existing = [m["dataset"] for m in obj["key_metrics"] if m["dataset"]]
    fake = {"metric": rng.choice(METRIC_NAMES),
            "value": f"{rng.uniform(55, 99):.1f}%",
            "dataset": rng.choice(existing) if existing and rng.random() < 0.5 else rng.choice(DATASETS),
            "citations": [rng.choice(cited)]}
    if len(out["key_metrics"]) >= 8:
        out["key_metrics"][rng.randrange(8)] = fake
    else:
        out["key_metrics"].insert(rng.randint(0, len(out["key_metrics"])), fake)
    return out


def missing_citation(obj: dict, rng: random.Random) -> dict | None:
    out = copy.deepcopy(obj)
    claims = [out["core_problem"], out["proposed_method"]] + out["key_metrics"] + out["limitations"]
    rng.choice(claims)["citations"] = []
    return out


def malformed_json(obj: dict, rng: random.Random) -> str:
    s = to_target(obj)
    choice = rng.randrange(3)
    if choice == 0:
        return s[: rng.randint(len(s) // 2, len(s) - 2)]  # truncated
    if choice == 1:
        return s.replace('"citations": [', '"citations": ', 1)  # unbalanced bracket
    return s.replace('"text": "', "'text': '", 1)  # wrong quotes


def make_rejected(obj: dict, kind: str, rng: random.Random) -> str | None:
    """The rejected response string for one perturbation kind, or None if it does not apply."""
    if kind == "malformed_json":
        return malformed_json(obj, rng)
    fn = {"altered_number": altered_number, "invented_metric": invented_metric,
          "missing_citation": missing_citation}[kind]
    out = fn(obj, rng)
    if out is None:
        return None
    s = json.dumps(out, ensure_ascii=False)
    return None if s == to_target(obj) else s
