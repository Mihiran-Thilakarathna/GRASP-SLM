"""Download the model files the Java application needs into ./models.

    python scripts/download_models.py                 # base Qwen2.5-1.5B GGUF (baseline)
    python scripts/download_models.py --llm-repo <you>/grasp-slm-gguf --llm-file grasp-slm-dpo-q4_k_m.gguf

Layout created (paths match app/src/main/resources/application.yml):
    models/bge-small-en-v1.5/{model.onnx,tokenizer.json}
    models/bge-reranker-base/{model.onnx,tokenizer.json}
    models/llm/model.gguf
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def fetch(repo: str, filename: str, dest: Path, token: str | None = None) -> None:
    if dest.exists():
        print(f"ok      {dest.relative_to(ROOT)}")
        return
    print(f"get     {repo}/{filename}")
    cached = hf_hub_download(repo_id=repo, filename=filename, token=token)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    print(f"saved   {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--embedding-repo", default="BAAI/bge-small-en-v1.5",
                    help="repo with onnx/model.onnx and tokenizer.json (your Optimum export or the official one)")
    ap.add_argument("--reranker-repo", default="BAAI/bge-reranker-base")
    ap.add_argument("--onnx-file", default="onnx/model.onnx",
                    help="path of the ONNX file inside the embedding/re-ranker repos")
    ap.add_argument("--onnx-repo", default=None,
                    help="repo holding bge-small-en-v1.5/ and bge-reranker-base/ folders (overrides the two above)")
    ap.add_argument("--llm-repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--llm-file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--force-llm", action="store_true", help="replace an existing models/llm/model.gguf")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--token", default=None, help="Hugging Face token, only needed for private repos")
    args = ap.parse_args()

    models = Path(args.models_dir)
    for name, repo in (("bge-small-en-v1.5", args.embedding_repo), ("bge-reranker-base", args.reranker_repo)):
        if args.onnx_repo:  # your own Optimum export, uploaded by `python -m grasp_train.hub onnx`
            fetch(args.onnx_repo, f"{name}/model.onnx", models / name / "model.onnx", args.token)
            fetch(args.onnx_repo, f"{name}/tokenizer.json", models / name / "tokenizer.json", args.token)
        else:
            fetch(repo, args.onnx_file, models / name / "model.onnx", args.token)
            fetch(repo, "tokenizer.json", models / name / "tokenizer.json", args.token)
    if not args.skip_llm:
        gguf = models / "llm" / "model.gguf"
        if args.force_llm and gguf.exists():
            gguf.unlink()
        fetch(args.llm_repo, args.llm_file, gguf, args.token)
        (models / "llm" / "SOURCE.txt").write_text(f"{args.llm_repo}/{args.llm_file}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
