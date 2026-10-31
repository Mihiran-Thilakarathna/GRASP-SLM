"""Upload the application to a Hugging Face Space (Docker runtime). The Space builds the Dockerfile.

    python scripts/deploy_space.py <you>/grasp-slm \
        --llm-repo <you>/grasp-slm-gguf --llm-file grasp-slm-dpo-q4_k_m.gguf [--onnx-repo <you>/grasp-slm-onnx]

Needs HF_TOKEN (write access) in the environment. Only the files the image needs are uploaded.
"""
import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: GRASP-SLM
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Structured, cited analysis of research papers on CPU
---
# GRASP-SLM

Upload an English research paper (PDF) and get a structured, citation-backed analysis: core problem,
proposed method, key metrics and limitations. A QLoRA + DPO fine-tuned Qwen2.5-1.5B (4-bit GGUF) answers
from chunks found by hybrid retrieval (Lucene BM25 + bge-small-en-v1.5 vectors, Reciprocal Rank Fusion,
bge-reranker-base). Built with Spring Boot, LangChain4j, Apache Lucene, ONNX Runtime and java-llama.cpp.

REST API: `POST /api/analyze` with a multipart `file`. Analysis takes one to three minutes on the free CPU.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space_id")
    ap.add_argument("--llm-repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--llm-file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--onnx-repo", default="")
    args = ap.parse_args()

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for name, value in (("LLM_REPO", args.llm_repo), ("LLM_FILE", args.llm_file), ("ONNX_REPO", args.onnx_repo)):
        dockerfile = re.sub(rf"^ARG {name}=.*$", f"ARG {name}={value}", dockerfile, flags=re.M)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=ROOT,
                      allow_patterns=["app/src/**", "app/pom.xml", "app/mvnw", "app/.mvn/**", "shared/**",
                                      "scripts/download_models.py", ".dockerignore"],
                      commit_message="Deploy GRASP-SLM")
    api.upload_file(repo_id=args.space_id, repo_type="space", path_or_fileobj=dockerfile.encode(),
                    path_in_repo="Dockerfile")
    api.upload_file(repo_id=args.space_id, repo_type="space", path_or_fileobj=SPACE_README.encode(),
                    path_in_repo="README.md")
    print(f"Uploaded. The Space is building: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
