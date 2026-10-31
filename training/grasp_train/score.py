"""Score the recorded runs and build the ablation table (proposal Tables 3 and 4).

    python -m grasp_train.score                       # schema, numeric, citation, latency, BERTScore
    python -m grasp_train.score --ragas \\
        --judge-base-url https://api.groq.com/openai/v1 --judge-model llama-3.3-70b-versatile \\
        --judge-api-key-env GROQ_API_KEY              # adds the four Ragas metrics

Writes results/per_paper.csv, results/summary.csv and results/summary.md. Ragas scores are
cached in results/ragas_<config>.csv so the paid-per-call judge runs only once per answer.
Use one fixed judge model and keep it for every configuration (proposal, Table 6).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .common import DATA, ROOT, read_jsonl
from .metrics import as_answer, citation_validity, field_texts, numeric_accuracy, parse, schema_valid
from .run_eval import CONFIGS

RAGAS_QUESTION = ("What problem does this paper address, what method does it propose, what are its key "
                  "quantitative results, and what limitations do the authors state?")


def basic_rows(config: str, runs: Path, tests: dict[str, dict]) -> list[dict]:
    rows = []
    for r in read_jsonl(runs):
        t = tests.get(r["paper_id"])
        if t is None:
            continue
        resp = r.get("response") or {}
        pred = parse(resp.get("rawOutput"))
        ctx_ids = [c["id"] for c in resp.get("context", [])]
        timings = resp.get("timingsMs") or {}
        rows.append({
            "config": config, "paper_id": r["paper_id"], "http_ok": r["status"] == 200,
            "schema_valid": schema_valid(pred),
            "citation_validity": citation_validity(pred, ctx_ids),
            **numeric_accuracy(pred, t["reference"], t["full_text"]),
            "latency_s": timings.get("total", float("nan")) / 1000,
            "generate_s": timings.get("generate", float("nan")) / 1000,
            "retrieve_s": timings.get("retrieve", float("nan")) / 1000,
            "completion_tokens": (resp.get("tokens") or {}).get("completion"),
            "peak_rss_mb": (resp.get("peakRssBytes") or float("nan")) / 1e6,
            "_pred": pred, "_contexts": [c["text"] for c in resp.get("context", [])],
        })
    return rows


def add_bertscore(rows: list[dict], tests: dict[str, dict], model: str) -> None:
    from bert_score import score as bert_score

    cands, refs, index = [], [], []
    for i, row in enumerate(rows):
        p, ref = field_texts(row["_pred"]), field_texts(tests[row["paper_id"]]["reference"])
        for field in p:
            if ref[field]:
                cands.append(p[field] or "none")
                refs.append(ref[field])
                index.append(i)
    if not cands:
        return
    _, _, f1 = bert_score(cands, refs, model_type=model, lang="en", rescale_with_baseline=True, verbose=False)
    sums: dict[int, list[float]] = {}
    for i, v in zip(index, f1.tolist()):
        sums.setdefault(i, []).append(v)
    for i, vals in sums.items():
        rows[i]["bertscore_f1"] = sum(vals) / len(vals)


def add_ragas(rows: list[dict], tests: dict[str, dict], config: str, out_dir: Path, args) -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    cache = out_dir / f"ragas_{config}.csv"
    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["paper_id"])
    todo = [r for r in rows if r["paper_id"] not in set(cached["paper_id"]) and r["_pred"] is not None]
    if todo:
        judge = LangchainLLMWrapper(ChatOpenAI(base_url=args.judge_base_url, model=args.judge_model, temperature=0,
                                               api_key=os.environ.get(args.judge_api_key_env, "none")))
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
        samples = [SingleTurnSample(user_input=RAGAS_QUESTION, retrieved_contexts=r["_contexts"],
                                    response=as_answer(r["_pred"]),
                                    reference=as_answer(tests[r["paper_id"]]["reference"])) for r in todo]
        result = evaluate(EvaluationDataset(samples=samples),
                          metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                                   LLMContextRecall()],
                          llm=judge, embeddings=emb,
                          run_config=RunConfig(max_workers=args.judge_workers, timeout=300, max_retries=6))
        df = result.to_pandas()
        df = df.rename(columns={"answer_relevancy": "ragas_answer_relevancy", "faithfulness": "ragas_faithfulness",
                                "llm_context_precision_with_reference": "ragas_context_precision",
                                "context_recall": "ragas_context_recall"})
        df["paper_id"] = [r["paper_id"] for r in todo]
        keep = ["paper_id"] + [c for c in df.columns if c.startswith("ragas_")]
        cached = pd.concat([cached, df[keep]], ignore_index=True)
        cached.to_csv(cache, index=False)
    by_id = cached.set_index("paper_id").to_dict("index")
    for r in rows:
        r.update(by_id.get(r["paper_id"], {}))


SUMMARY_COLUMNS = {
    "schema_valid": "Schema validity",
    "citation_validity": "Citation validity",
    "grounded_precision": "Numeric grounded precision",
    "reference_recall": "Numeric recall",
    "numeric_f1": "Numeric F1",
    "bertscore_f1": "BERTScore F1",
    "ragas_faithfulness": "Ragas faithfulness",
    "ragas_answer_relevancy": "Ragas answer relevancy",
    "ragas_context_precision": "Ragas context precision",
    "ragas_context_recall": "Ragas context recall",
    "latency_s": "Latency (s)",
    "peak_rss_mb": "Peak RSS (MB)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--test", default=str(DATA / "datasets" / "test.jsonl"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--judge-base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--judge-api-key-env", default="GROQ_API_KEY")
    ap.add_argument("--judge-workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.results)
    tests = {t["paper_id"]: t for t in read_jsonl(Path(args.test))}
    all_rows = []
    for cfg in args.configs:
        runs = out / f"{cfg}.jsonl"
        if not runs.exists():
            print(f"skip {cfg}: {runs} not found")
            continue
        rows = basic_rows(cfg, runs, tests)
        if not args.no_bertscore:
            add_bertscore(rows, tests, args.bertscore_model)
        if args.ragas:
            add_ragas(rows, tests, cfg, out, args)
        print(f"{cfg}: {len(rows)} papers scored")
        all_rows += rows
    if not all_rows:
        raise SystemExit("Nothing to score. Run grasp_train.run_eval first.")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows])
    df.to_csv(out / "per_paper.csv", index=False)
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    summary = df.groupby("config")[cols].mean(numeric_only=True)
    summary.insert(0, "papers", df.groupby("config").size())
    summary.insert(0, "configuration", [CONFIGS[c]["description"] for c in summary.index])
    summary.to_csv(out / "summary.csv")
    md = summary.rename(columns=SUMMARY_COLUMNS).to_markdown(floatfmt=".3f")
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
