"""Draft target analyses with a teacher model (proposal section 6.3).

The teacher sees exactly the messages the Java exporter produced (same chunks, same prompt),
plus the JSON Schema. Any OpenAI-compatible endpoint works, for example:

  Groq (free tier):   --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY
  Gemini (free tier): --base-url https://generativelanguage.googleapis.com/v1beta/openai --model gemini-2.5-flash --api-key-env GEMINI_API_KEY
  Ollama (local):     --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct-q4_K_M

    python -m grasp_train.draft_labels --base-url ... --model ... --api-key-env ...

Drafts are saved one file per paper in data/drafts/, so the script can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .common import DATA, SCHEMA, canonical, read_jsonl, schema_errors, unknown_citations, write_json

TEACHER_NOTE = ("\n\nYour JSON must follow this JSON Schema exactly (property names, nesting and citation format "
                "\"c<number>\"):\n")


def call(base_url: str, api_key: str, model: str, messages: list[dict], response_format: str,
         timeout: int = 300) -> str:
    body: dict = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 2048}
    if response_format == "json_schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "analysis", "schema": SCHEMA, "strict": True}}
    elif response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    for attempt in range(8):
        r = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code in (429, 500, 502, 503):
            wait = float(r.headers.get("retry-after", 0) or 0) or min(120, 10 * 2 ** attempt)
            print(f"    {r.status_code} from teacher; waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Teacher returned {r.status_code}: {r.text[:500]}")
        return r.json()["choices"][0]["message"]["content"] or ""
    raise RuntimeError("Teacher kept failing; try again later")


def parse(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def draft_one(record: dict, args, api_key: str) -> dict:
    system, user = record["messages"][0]["content"], record["messages"][1]["content"]
    messages = [{"role": "system", "content": system + TEACHER_NOTE + json.dumps(SCHEMA)},
                {"role": "user", "content": user}]
    context_ids = record["context_chunk_ids"]
    raw, obj, errors, unknown = "", None, ["no attempt"], []
    for attempt in range(args.max_attempts):
        raw = call(args.base_url, api_key, args.model, messages, args.response_format)
        obj = parse(raw)
        if obj is None:
            errors, unknown = ["output is not a JSON object"], []
        else:
            obj = canonical(obj) if not schema_errors(obj) else obj
            errors, unknown = schema_errors(obj), unknown_citations(obj, context_ids)
        if obj is not None and not errors and not unknown:
            break
        problems = errors + [f"citation {c} is not one of the context chunk ids {context_ids}" for c in unknown]
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "Your answer has problems:\n- " + "\n- ".join(problems)
                      + "\nReturn the corrected JSON object only."}]
        time.sleep(60 / args.rpm)
    return {
        "paper_id": record["paper_id"],
        "teacher": {"base_url": args.base_url, "model": args.model},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid": obj is not None and not errors and not unknown,
        "errors": errors,
        "unknown_citations": unknown,
        "analysis": obj,
        "raw": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=str(DATA / "export.jsonl"))
    ap.add_argument("--out", default=str(DATA / "drafts"))
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="TEACHER_API_KEY",
                    help="name of the environment variable holding the API key (not the key itself)")
    ap.add_argument("--response-format", default="json_object", choices=["json_schema", "json_object", "none"])
    ap.add_argument("--rpm", type=float, default=20, help="max requests per minute (respect free-tier limits)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new drafts (0 = all)")
    ap.add_argument("--redo-invalid", action="store_true", help="re-draft papers whose previous draft was invalid")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "none")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = n_valid = 0
    for record in read_jsonl(Path(args.export)):
        path = out / f"{record['paper_id']}.json"
        if path.exists():
            if not args.redo_invalid or json.loads(path.read_text(encoding="utf-8")).get("valid"):
                continue
        print(f"draft {record['paper_id']}")
        try:
            result = draft_one(record, args, api_key)
        except RuntimeError as e:
            print(f"  failed: {e}")
            continue
        write_json(path, result)
        done += 1
        n_valid += result["valid"]
        print(f"  {'valid' if result['valid'] else 'INVALID: ' + '; '.join(result['errors'] + result['unknown_citations'])}")
        if args.limit and done >= args.limit:
            break
        time.sleep(60 / args.rpm)
    print(f"Drafted {done} papers ({n_valid} valid) into {out}")


if __name__ == "__main__":
    main()
