"""Export the embedding and re-ranking models to ONNX with Hugging Face Optimum (proposal 6.8).

    python -m grasp_train.export_onnx --out models --quantize

Creates models/bge-small-en-v1.5/ and models/bge-reranker-base/, each with model.onnx and
tokenizer.json, which is the layout the Java app reads. --quantize replaces model.onnx with a
dynamic int8 version (about 4x smaller and faster on CPU); keep the fp32 file for comparison as
model_fp32.onnx.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODELS = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "feature-extraction"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "text-classification"),
}


def export(name: str, repo: str, task: str, out: Path, quantize: bool) -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    target = out / name
    work = target / "_export"
    cls = ORTModelForFeatureExtraction if task == "feature-extraction" else ORTModelForSequenceClassification
    print(f"Exporting {repo} -> {target}")
    model = cls.from_pretrained(repo, export=True)
    model.save_pretrained(work)
    AutoTokenizer.from_pretrained(repo).save_pretrained(work)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "tokenizer.json", target / "tokenizer.json")
    shutil.copyfile(work / "model.onnx", target / "model_fp32.onnx")
    if quantize:
        quantizer = ORTQuantizer.from_pretrained(work)
        qdir = work / "int8"
        quantizer.quantize(save_dir=qdir, quantization_config=AutoQuantizationConfig.avx2(is_static=False,
                                                                                        per_channel=False))
        shutil.copyfile(next(qdir.glob("*.onnx")), target / "model.onnx")
    else:
        shutil.copyfile(work / "model.onnx", target / "model.onnx")
    shutil.rmtree(work)
    print(f"  model.onnx: {(target / 'model.onnx').stat().st_size / 1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--only", choices=list(MODELS))
    args = ap.parse_args()
    for name, (repo, task) in MODELS.items():
        if args.only in (None, name):
            export(name, repo, task, Path(args.out), args.quantize)


if __name__ == "__main__":
    main()
