# %% [markdown]
# # GRASP-SLM 4/4: Ablation evaluation, configurations A-D (GPU)
#
# Proposal section 8: every test paper is sent to the **Java application's REST API**, which retrieves
# the chunks and asks the language model. For A-D the model is an **unquantised F16 GGUF** served on the
# T4 GPU by a llama.cpp server; the Java app calls it over HTTP. All configurations use the same
# inference engine, prompt and JSON-schema grammar, so only the component under test changes:
#
# | ID | Model | Retrieval |
# |---|---|---|
# | A | base Qwen2.5-1.5B-Instruct | dense only |
# | B | base | hybrid + re-ranker |
# | C | SFT | hybrid + re-ranker |
# | D | SFT + DPO | hybrid + re-ranker |
#
# Configuration **E** (Q4_K_M on CPU, the deployed set-up) is run on a CPU machine: see the project guide.
#
# Before running: GPU T4, Internet on, `HF_TOKEN` secret, set `HF_USER` and `REPO_URL`.

# %%
HF_USER = "your-hf-username"                               # <-- change this
REPO_URL = "https://github.com/your-github-user/GRASP-SLM"  # <-- change this
DATASET_REPO = f"{HF_USER}/grasp-slm-papers"
GGUF_REPO = f"{HF_USER}/grasp-slm-gguf"
MODELS = {
    "base": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-fp16.gguf"),
    "sft": (GGUF_REPO, "grasp-slm-sft-f16.gguf"),
    "dpo": (GGUF_REPO, "grasp-slm-dpo-f16.gguf"),
}
RUNS = [("A", "base"), ("B", "base"), ("C", "sft"), ("D", "dpo")]

# %%
import os, subprocess, time
from pathlib import Path

import requests


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
os.environ["HF_TOKEN"] = HF_TOKEN
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
REPO = WORK / "GRASP-SLM"


def sh(cmd, cwd=None):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

# %%
# Project code, Java 21 and the Java application.
if not REPO.exists():
    sh(f"git clone --depth 1 {REPO_URL} {REPO}")
JDK = WORK / "jdk"
if not JDK.exists():
    sh(f"curl -sL -o {WORK}/jdk.tgz 'https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse'")
    sh(f"mkdir -p {JDK} && tar -xzf {WORK}/jdk.tgz -C {JDK} --strip-components=1")
os.environ["JAVA_HOME"] = str(JDK)
os.environ["PATH"] = f"{JDK}/bin:" + os.environ["PATH"]
sh("chmod +x mvnw && ./mvnw -q -B -DskipTests package", cwd=REPO / "app")
sh("pip install -q requests jsonschema pandas tabulate huggingface_hub")
sh("python scripts/download_models.py --skip-llm", cwd=REPO)

# %%
# GPU llama.cpp server with an OpenAI-compatible API (prebuilt CUDA 12.4 wheel).
sh('pip install -q "llama-cpp-python[server]==0.3.19" --only-binary llama-cpp-python '
   '--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124')
# If that fails (no wheel for this Python version), build from source instead (about 20 minutes):
# sh('CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python[server]"')

# %%
from huggingface_hub import hf_hub_download, snapshot_download

data = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=HF_TOKEN,
                              allow_patterns=["test.jsonl", "test_pdfs/*"]))
gguf = {k: hf_hub_download(repo, f, token=HF_TOKEN) for k, (repo, f) in MODELS.items()}
print(gguf)


def wait(url, seconds=600):
    for _ in range(seconds // 5):
        try:
            if requests.get(url, timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"{url} did not come up")


def run(config, model_key):
    llm = subprocess.Popen(["python", "-m", "llama_cpp.server", "--model", gguf[model_key], "--n_gpu_layers", "-1",
                            "--n_ctx", "8192", "--host", "127.0.0.1", "--port", "8081", "--model_alias", model_key],
                           stdout=open(WORK / f"llm_{config}.log", "w"), stderr=subprocess.STDOUT)
    env = dict(os.environ, GRASP_LLM_BACKEND="openai", GRASP_LLM_BASE_URL="http://127.0.0.1:8081/v1",
               GRASP_LLM_MODEL_NAME=model_key, GRASP_LLM_RESPONSE_FORMAT="json_object", PORT="7860")
    app = subprocess.Popen(["java", "-jar", "app/target/grasp-slm.jar"], cwd=REPO, env=env,
                           stdout=open(WORK / f"app_{config}.log", "w"), stderr=subprocess.STDOUT)
    try:
        wait("http://127.0.0.1:8081/v1/models")
        wait("http://127.0.0.1:7860/api/health")
        sh(f"python -m grasp_train.run_eval --config {config} --api http://127.0.0.1:7860 "
           f"--test {data}/test.jsonl --pdf-dir {data}/test_pdfs --out-dir {WORK}/results --expect-model {model_key}",
           cwd=REPO / "training")
    finally:
        app.terminate(); llm.terminate()
        app.wait(); llm.wait()

# %%
for config, model_key in RUNS:
    print(f"===== configuration {config} ({model_key}) =====")
    run(config, model_key)

# %%
# Save the raw runs next to the dataset so they can be scored anywhere.
from huggingface_hub import HfApi

HfApi(token=HF_TOKEN).upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(WORK / "results"),
                                    path_in_repo="results", commit_message="Ablation runs A-D")
print(f"Uploaded to https://huggingface.co/datasets/{DATASET_REPO}/tree/main/results")

# %%
# Optional: schema, numeric and BERTScore metrics here on the GPU (Ragas needs a judge API key; run it locally).
sh("pip install -q bert-score==0.3.13")
sh(f"python -m grasp_train.score --results {WORK}/results --test {data}/test.jsonl --configs A B C D",
   cwd=REPO / "training")
