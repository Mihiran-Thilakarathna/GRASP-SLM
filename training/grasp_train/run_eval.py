"""Send every test paper to the Java application and record the answers (proposal section 8).

    python -m grasp_train.run_eval --config A --api http://localhost:7860

The Java app decides which model answers (set by GRASP_LLM_* when it starts); this script sets
the retrieval options for the configuration and records the analysis, the retrieved chunks,
validation results and timings. Results are appended to results/<config>.jsonl, so an
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from .common import DATA, ROOT, append_jsonl, read_jsonl

CONFIGS = {
    "A": {"description": "Base model + dense-only RAG", "retrieval": "dense", "rerank": False},
    "B": {"description": "Base model + hybrid RAG", "retrieval": "hybrid", "rerank": True},
    "C": {"description": "SFT model + hybrid RAG", "retrieval": "hybrid", "rerank": True},
    "D": {"description": "SFT + DPO model + hybrid RAG", "retrieval": "hybrid", "rerank": True},
    "E": {"description": "Configuration D quantised to 4-bit GGUF", "retrieval": "hybrid", "rerank": True},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--api", default="http://localhost:7860")
    ap.add_argument("--test", default=str(DATA / "datasets" / "test.jsonl"))
    ap.add_argument("--pdf-dir", default=str(DATA / "pdfs"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--expect-model", default="", help="warn if the server's model description lacks this text")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    health = requests.get(f"{args.api}/api/health", timeout=30).json()
    print(f"Config {args.config}: {cfg['description']}\nServer model: {health['model']}")
    if args.expect_model and args.expect_model not in health["model"]:
        raise SystemExit(f"Server model does not contain '{args.expect_model}'. Restart the app with the right model.")

    out = Path(args.out_dir) / f"{args.config}.jsonl"
    done = {r["paper_id"] for r in read_jsonl(out)}
    tests = list(read_jsonl(Path(args.test)))
    for i, t in enumerate(tests, 1):
        if t["paper_id"] in done:
            continue
        pdf = Path(args.pdf_dir) / t["source_file"]
        start = time.time()
        with open(pdf, "rb") as f:
            r = requests.post(f"{args.api}/api/analyze", timeout=args.timeout,
                              files={"file": (pdf.name, f, "application/pdf")},
                              data={"retrieval": cfg["retrieval"], "rerank": str(cfg["rerank"]).lower()})
        wall = time.time() - start
        row = {"config": args.config, "paper_id": t["paper_id"], "status": r.status_code, "wall_seconds": wall}
        if r.ok:
            row["response"] = r.json()
            v = "valid" if row["response"]["schemaValid"] else "INVALID"
            print(f"[{i}/{len(tests)}] {t['paper_id']}: {v}, {wall:.0f}s")
        else:
            row["error"] = r.text[:2000]
            print(f"[{i}/{len(tests)}] {t['paper_id']}: HTTP {r.status_code} {r.text[:200]}")
        append_jsonl(out, row)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
