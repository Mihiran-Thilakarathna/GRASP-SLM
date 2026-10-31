# %% [markdown]
# # GRASP-SLM 3/4: Convert to GGUF and quantise to Q4_K_M
#
# Uses the official llama.cpp tools (proposal section 6.8):
# * `convert_hf_to_gguf.py` turns the merged SFT and SFT+DPO models into **F16 GGUF** files
#   (used for ablation configurations C and D, served on GPU), and
# * `llama-quantize` makes the **Q4_K_M** file used for configuration E and the deployed app.
#
# **This notebook needs no GPU**: run it on a Kaggle CPU session (Accelerator: None) to save GPU quota.
# Internet on, `HF_TOKEN` secret, set `HF_USER`.

# %%
HF_USER = "your-hf-username"          # <-- change this
SFT_MERGED_REPO = f"{HF_USER}/grasp-slm-sft-merged"
DPO_MERGED_REPO = f"{HF_USER}/grasp-slm-dpo-merged"
GGUF_REPO = f"{HF_USER}/grasp-slm-gguf"
LLAMA_CPP_TAG = "b6000"   # pinned llama.cpp release; any recent tag that supports Qwen2 works

# %%
import os, subprocess
from pathlib import Path


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")


HF_TOKEN = hf_token()
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
OUT = WORK / "gguf"
OUT.mkdir(exist_ok=True)


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

# %%
sh(f"git clone --depth 1 --branch {LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
sh(f"pip install -q -r {WORK}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
sh(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF")
sh(f"cmake --build {WORK}/llama.cpp/build --target llama-quantize -j 4")

# %%
from huggingface_hub import snapshot_download

for name, repo in (("sft", SFT_MERGED_REPO), ("dpo", DPO_MERGED_REPO)):
    local = snapshot_download(repo, token=HF_TOKEN, local_dir=str(WORK / f"{name}-merged"))
    sh(f"python {WORK}/llama.cpp/convert_hf_to_gguf.py {local} --outtype f16 "
       f"--outfile {OUT}/grasp-slm-{name}-f16.gguf")

sh(f"{WORK}/llama.cpp/build/bin/llama-quantize {OUT}/grasp-slm-dpo-f16.gguf {OUT}/grasp-slm-dpo-q4_k_m.gguf Q4_K_M")
for f in sorted(OUT.glob("*.gguf")):
    print(f"{f.name}: {f.stat().st_size / 1e9:.2f} GB")

# %%
from huggingface_hub import HfApi

CARD = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags: [gguf, qlora, dpo, rag, scientific-papers, grasp-slm]
language: [en]
---
# GRASP-SLM (GGUF)

Qwen2.5-1.5B-Instruct fine-tuned with QLoRA SFT and then DPO to write structured, citation-backed
analyses of research papers (core problem, proposed method, key metrics, limitations) from retrieved
context chunks. Built for the GRASP-SLM Java application (java-llama.cpp, CPU).

| File | Use |
|---|---|
| grasp-slm-dpo-q4_k_m.gguf | deployment (ablation configuration E) |
| grasp-slm-dpo-f16.gguf | SFT + DPO, unquantised (configuration D) |
| grasp-slm-sft-f16.gguf | SFT only (configuration C) |

LoRA adapters: [{HF_USER}/grasp-slm-sft-lora](https://huggingface.co/{HF_USER}/grasp-slm-sft-lora),
[{HF_USER}/grasp-slm-dpo-lora](https://huggingface.co/{HF_USER}/grasp-slm-dpo-lora).

The model must be prompted with the GRASP-SLM prompt and context format and answers only from the
supplied chunks. Output is constrained by the JSON Schema in the GRASP-SLM repository.
"""

api = HfApi(token=HF_TOKEN)
api.create_repo(GGUF_REPO, exist_ok=True)
for f in sorted(OUT.glob("*.gguf")):
    api.upload_file(repo_id=GGUF_REPO, path_or_fileobj=str(f), path_in_repo=f.name)
api.upload_file(repo_id=GGUF_REPO, path_or_fileobj=CARD.encode(), path_in_repo="README.md")
print(f"https://huggingface.co/{GGUF_REPO}")
