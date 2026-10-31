"""Paths, the shared JSON Schema and small helpers used by every script."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "shared"
DATA = Path(os.environ.get("GRASP_DATA_DIR", ROOT / "data"))

FIELDS = ("core_problem", "proposed_method", "key_metrics", "limitations")
CLAIM_KEYS = ("text", "citations")
METRIC_KEYS = ("metric", "value", "dataset", "citations")


def load_schema() -> dict:
    return json.loads((SHARED / "analysis.schema.json").read_text(encoding="utf-8"))


SCHEMA = load_schema()
_VALIDATOR = Draft202012Validator(SCHEMA)


def schema_errors(obj: Any) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in _VALIDATOR.iter_errors(obj)]


def citations(obj: dict) -> list[str]:
    """Every cited chunk id in an analysis, in order of appearance."""
    out: list[str] = []
    for field in ("core_problem", "proposed_method"):
        out += list(obj.get(field, {}).get("citations", []))
    for item in obj.get("key_metrics", []) + obj.get("limitations", []):
        out += list(item.get("citations", []))
    return out


def unknown_citations(obj: dict, context_ids: Iterable[str]) -> list[str]:
    allowed = set(context_ids)
    return sorted({c for c in citations(obj) if c not in allowed})


def canonical(obj: dict) -> dict:
    """Key order of the JSON Schema. llama.cpp's schema grammar emits properties in this order,
    so training targets must use it too."""

    def claim(c: dict) -> dict:
        return {"text": c.get("text", ""), "citations": list(c.get("citations", []))}

    return {
        "core_problem": claim(obj.get("core_problem", {})),
        "proposed_method": claim(obj.get("proposed_method", {})),
        "key_metrics": [{k: (list(m.get(k, [])) if k == "citations" else m.get(k, "")) for k in METRIC_KEYS}
                        for m in obj.get("key_metrics", [])],
        "limitations": [claim(c) for c in obj.get("limitations", [])],
    }


def to_target(obj: dict) -> str:
    """The exact assistant string the model is trained to produce."""
    return json.dumps(canonical(obj), ensure_ascii=False)


def chatml(system: str, user: str) -> str:
    """Same prompt string as the Java ChatMl class and Qwen2.5's chat template."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def hf_token() -> str | None:
    """HF token from the environment, Kaggle Secrets or Colab userdata (in that order)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get("HF_TOKEN")
    except Exception:
        return None
