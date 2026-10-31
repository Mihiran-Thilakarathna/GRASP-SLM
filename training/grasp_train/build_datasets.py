"""Build the SFT, DPO and test datasets (proposal sections 6.3-6.5).

    python -m grasp_train.build_datasets

Labels: a human-verified analysis when one exists, otherwise a valid teacher draft. The split is
80/10/10 at paper level, and the test set contains only verified papers.

Outputs in data/datasets/:
  sft_train.jsonl, sft_val.jsonl   {"paper_id", "messages": [system, user, assistant]}
  dpo_train.jsonl, dpo_val.jsonl   {"paper_id", "prompt": [system, user], "chosen", "rejected", "perturbation"}
  test.jsonl                       {"paper_id", "source_file", "title", "reference", "context_chunk_ids", "full_text"}
  test_pdfs/                       the test papers' PDFs, for the evaluation notebook
  split.json, README.md            split ids and a dataset card
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from .common import (DATA, canonical, read_json, read_jsonl, schema_errors, to_target, unknown_citations,
                     write_json, write_jsonl)
from .perturb import KINDS, make_rejected


def labels(export: list[dict], drafts: Path, verified: Path) -> dict[str, tuple[dict, bool]]:
    """paper_id -> (analysis, is_verified)."""
    out: dict[str, tuple[dict, bool]] = {}
    for rec in export:
        pid = rec["paper_id"]
        v, d = verified / f"{pid}.json", drafts / f"{pid}.json"
        if v.exists():
            review = read_json(v)
            if review["status"] == "verified":
                out[pid] = (canonical(review["analysis"]), True)
            continue  # rejected papers are excluded entirely
        if d.exists():
            draft = read_json(d)
            a = draft.get("analysis")
            if draft.get("valid") and a and not schema_errors(a) and not unknown_citations(a, rec["context_chunk_ids"]):
                out[pid] = (canonical(a), False)
    return out


def split(ids_verified: list[str], ids_draft: list[str], rng: random.Random,
          val_frac: float, test_frac: float) -> dict[str, list[str]]:
    n = len(ids_verified) + len(ids_draft)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    verified = sorted(ids_verified)
    rng.shuffle(verified)
    if len(verified) < n_test:
        print(f"WARNING: only {len(verified)} verified papers; the test set should have {n_test}. "
              f"Verify more papers to keep the test set at 10%.")
    test = verified[:n_test]
    rest = verified[n_test:] + sorted(ids_draft)
    rng.shuffle(rest)
    return {"test": sorted(test), "val": sorted(rest[:n_val]), "train": sorted(rest[n_val:])}


def dpo_rows(rec: dict, analysis: dict, rng: random.Random, per_paper: int) -> list[dict]:
    rows = []
    kinds = list(KINDS)
    rng.shuffle(kinds)
    for kind in kinds:
        if len(rows) >= per_paper:
            break
        rejected = make_rejected(analysis, kind, rng)
        if rejected:
            rows.append({"paper_id": rec["paper_id"], "prompt": rec["messages"][:2], "chosen": to_target(analysis),
                         "rejected": rejected, "perturbation": kind})
    return rows


CARD = """---
license: cc-by-4.0
language: [en]
task_categories: [summarization, text-generation]
tags: [scientific-papers, rag, dpo, structured-output]
---
# GRASP-SLM paper analysis dataset

Structured analyses (core problem, proposed method, key metrics, limitations) of openly licensed
arXiv papers, built for fine-tuning Qwen2.5-1.5B-Instruct with QLoRA SFT and DPO.

Each prompt contains the chunks selected by the GRASP-SLM hybrid retriever (BM25 + bge-small-en-v1.5,
Reciprocal Rank Fusion, bge-reranker-base). Targets were drafted by a teacher model; the test split
contains only human-verified targets. DPO rejected answers are automatic perturbations of the chosen
answer (altered number, invented metric, missing citation, malformed JSON).

| Split | Papers |
|---|---|
{rows}

Licences of the source papers are recorded in the GRASP-SLM repository (data/metadata.jsonl);
only CC BY, CC BY-SA and CC0 papers are used unless stated otherwise.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=str(DATA / "export.jsonl"))
    ap.add_argument("--drafts", default=str(DATA / "drafts"))
    ap.add_argument("--verified", default=str(DATA / "verified"))
    ap.add_argument("--out", default=str(DATA / "datasets"))
    ap.add_argument("--pdf-dir", default=str(DATA / "pdfs"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--dpo-per-paper", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    export = [r for r in read_jsonl(Path(args.export))
              if r["num_pages"] <= args.max_pages and len(r["context_chunk_ids"]) >= 3]
    by_id = {r["paper_id"]: r for r in export}
    lab = labels(export, Path(args.drafts), Path(args.verified))
    ids_v = [p for p, (_, v) in lab.items() if v]
    ids_d = [p for p, (_, v) in lab.items() if not v]
    print(f"{len(lab)} labelled papers: {len(ids_v)} verified, {len(ids_d)} teacher drafts")
    if len(lab) < 10:
        raise SystemExit("Need at least 10 labelled papers to build datasets.")

    parts = split(ids_v, ids_d, rng, args.val_frac, args.test_frac)
    out = Path(args.out)

    def sft(ids):
        for pid in ids:
            rec, (analysis, _) = by_id[pid], lab[pid]
            yield {"paper_id": pid, "messages": rec["messages"][:2] + [
                {"role": "assistant", "content": to_target(analysis)}]}

    counts = {
        "sft_train": write_jsonl(out / "sft_train.jsonl", sft(parts["train"])),
        "sft_val": write_jsonl(out / "sft_val.jsonl", sft(parts["val"])),
    }
    kinds: Counter = Counter()
    for name, ids in (("dpo_train", parts["train"]), ("dpo_val", parts["val"])):
        rows = [row for pid in ids for row in dpo_rows(by_id[pid], lab[pid][0], rng, args.dpo_per_paper)]
        kinds.update(r["perturbation"] for r in rows)
        counts[name] = write_jsonl(out / f"{name}.jsonl", rows)
    counts["test"] = write_jsonl(out / "test.jsonl", (
        {"paper_id": pid, "source_file": by_id[pid]["source_file"], "title": by_id[pid]["title"],
         "reference": lab[pid][0], "context_chunk_ids": by_id[pid]["context_chunk_ids"],
         "full_text": "\n\n".join(c["text"] for c in by_id[pid]["chunks"])}
        for pid in parts["test"]))

    # Test PDFs travel with the dataset so the GPU evaluation notebook can send them to the Java app.
    pdf_dir, test_pdfs = Path(args.pdf_dir), out / "test_pdfs"
    test_pdfs.mkdir(parents=True, exist_ok=True)
    for pid in parts["test"]:
        src = pdf_dir / by_id[pid]["source_file"]
        if src.exists():
            shutil.copyfile(src, test_pdfs / src.name)

    write_json(out / "split.json", {"seed": args.seed, **parts, "counts": counts, "dpo_perturbations": dict(kinds)})
    rows = "\n".join(f"| {k} | {len(v)} |" for k, v in parts.items())
    (out / "README.md").write_text(CARD.format(rows=rows), encoding="utf-8")
    for k, v in counts.items():
        print(f"  {k}: {v} rows")
    print(f"DPO perturbations: {dict(kinds)}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
