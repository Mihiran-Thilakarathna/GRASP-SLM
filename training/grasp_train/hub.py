"""Upload project artefacts to the Hugging Face Hub (needs HF_TOKEN with write access).

    python -m grasp_train.hub dataset <you>/grasp-slm-papers      # data/datasets/ -> dataset repo
    python -m grasp_train.hub onnx    <you>/grasp-slm-onnx        # models/bge-*/ -> model repo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from .common import DATA, ROOT, hf_token

ONNX_CARD = """---
license: mit
tags: [onnx, grasp-slm]
---
# GRASP-SLM ONNX retrieval models

ONNX exports (Hugging Face Optimum) of BAAI/bge-small-en-v1.5 (embeddings) and BAAI/bge-reranker-base
(cross-encoder re-ranker), used by the GRASP-SLM Java application through ONNX Runtime. Both upstream
models are MIT licensed.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["dataset", "onnx"])
    ap.add_argument("repo_id")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=hf_token())
    if args.what == "dataset":
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=DATA / "datasets",
                          commit_message="Upload GRASP-SLM datasets")
    else:
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        models = ROOT / "models"
        for name in ("bge-small-en-v1.5", "bge-reranker-base"):
            for f in ("model.onnx", "tokenizer.json"):
                api.upload_file(repo_id=args.repo_id, path_or_fileobj=models / name / f, path_in_repo=f"{name}/{f}")
        api.upload_file(repo_id=args.repo_id, path_or_fileobj=ONNX_CARD.encode(), path_in_repo="README.md")
    kind = "datasets/" if args.what == "dataset" else ""
    print(f"Uploaded to https://huggingface.co/{kind}{args.repo_id}")


if __name__ == "__main__":
    main()
