# GRASP-SLM: what YOU need to do

Claude built the code. These steps need your accounts, your judgement, or a free GPU, so only you can do them.
Run commands from the project folder `E:\my_projects\GRASP-SLM` in Git Bash.

## Already done and tested on your PC
- Java app: 29 JUnit tests pass. A real end-to-end run with the downloaded models gives schema-valid JSON
  (sample paper: 33 s; real 18-page arXiv paper: 93 s on your CPU, of which 22 s is retrieval and 70 s is generation).
- Python pipeline: 12 pytest tests pass. The arXiv collector downloaded 2 real CC BY papers into `data/`.
- The models are already in `models/` (base Qwen2.5-1.5B Q4_K_M, bge-small, bge-reranker-base).
- To try the web UI now: `java -jar app/target/grasp-slm.jar`, then open http://localhost:7860.
- To repeat the real-model test: `cd app && ./mvnw test -Pmodels`.

What the baseline test showed (good material for your report):
- The **untuned base model** gives vague metrics ("reduced" instead of numbers) and once cited a chunk it
  was not given (`c1`). The validator flags this. SFT and DPO are meant to fix exactly this (RQ2, RQ3).
- In the 18-page paper, the "4.3 Results" chunks were not selected for key_metrics. During your pilot
  (step 3), try improving the wording in `shared/field_queries.json`; Java and Python both read that file.
- About 22 s of the time is the fp32 re-ranker on CPU. For the free Space (2 CPUs), consider the int8 export:
  `python -m grasp_train.export_onnx --quantize` (needs `requirements-eval.txt`), then upload it with `grasp_train.hub onnx`.
- Docker Desktop is not running on your PC, so the Docker image was not built locally. Start Docker
  Desktop to try `docker build -t grasp-slm .`; GitHub Actions also builds it on every push.

## 0. One-time setup (accounts)
1. Create free accounts: **Hugging Face**, **GitHub**, **Kaggle** (verify your phone on Kaggle to unlock GPUs).
2. Hugging Face → Settings → Access Tokens → create a **Write** token.
3. Get a free teacher/judge API key: **Groq** (console.groq.com) or **Google AI Studio** (Gemini).
4. Put the keys in your shell (never in code):
   ```bash
   export HF_TOKEN=hf_xxx
   export GROQ_API_KEY=gsk_xxx
   ```

## 1. Push the code to GitHub
```bash
git add . && git commit -m "GRASP-SLM initial implementation"
git remote add origin https://github.com/<you>/GRASP-SLM.git
git push -u origin main
```
GitHub Actions then runs the Java and Python tests automatically.

## 2. Collect papers (about 2-3 hours, runs by itself)
```bash
cd training && ../.venv/Scripts/python -m pip install -r requirements.txt
../.venv/Scripts/python -m grasp_train.collect_arxiv --per-field 100
```
4 fields x 100 = 400 papers (the proposal says 300-800).

## 3. Chunk the papers with the Java app
```bash
cd .. && (cd app && ./mvnw -q -DskipTests package)
.venv/Scripts/python scripts/download_models.py
java -jar app/target/grasp-slm.jar export --input data/pdfs --output data/export.jsonl
```
Pilot check (proposal risk table): open `data/export.jsonl` for about 10 papers and check the sections and text look right.

## 4. Draft labels with the teacher model
```bash
cd training
../.venv/Scripts/python -m grasp_train.draft_labels --base-url https://api.groq.com/openai/v1 \
  --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY --rpm 20
```
If the free quota runs out, run it again tomorrow: it continues where it stopped.

## 5. Verify at least 100 drafts by hand (the most important human step)
```bash
../.venv/Scripts/python -m grasp_train.review_app     # open http://127.0.0.1:7861
```
Check every number against the chunks, fix mistakes, then press **Save as verified**. The test set uses only verified papers.

## 6. Build the datasets and upload them
```bash
../.venv/Scripts/python -m grasp_train.build_datasets
../.venv/Scripts/python -m grasp_train.hub dataset <you>/grasp-slm-papers
```

## 7. Train on Kaggle (free T4 GPU)
Upload each notebook from `training/notebooks/` to Kaggle. In each one: set `HF_USER`, add the secret `HF_TOKEN`, then Run All.
1. `01_sft_qlora.ipynb`: GPU, about 1 hour
2. `02_dpo.ipynb`: GPU, about 1 hour
3. `03_export_gguf.ipynb`: **CPU only** (saves GPU quota)
4. `04_ablation_eval.ipynb`: GPU, also set `REPO_URL`; runs configurations A-D

If a session stops, run the same notebook again: it resumes from the last checkpoint.

## 8. Configuration E (4-bit model on your CPU) and scoring
```bash
.venv/Scripts/python scripts/download_models.py --llm-repo <you>/grasp-slm-gguf --llm-file grasp-slm-dpo-q4_k_m.gguf --force-llm
java -jar app/target/grasp-slm.jar          # open http://localhost:7860
cd training && ../.venv/Scripts/python -m grasp_train.run_eval --config E
# download results/A-D.jsonl from your HF dataset repo into results/, then:
../.venv/Scripts/python -m pip install -r requirements-eval.txt
../.venv/Scripts/python -m grasp_train.score --ragas
```
The ablation table appears in `results/summary.md`; use it to answer RQ1-RQ4.

## 9. Deploy the demo (Hugging Face Space)
```bash
.venv/Scripts/python scripts/deploy_space.py <you>/grasp-slm --llm-repo <you>/grasp-slm-gguf --llm-file grasp-slm-dpo-q4_k_m.gguf
```
Then record a short demo video and write the final report.

## Note for your report
Configurations A-D run the unquantised **F16 GGUF** on GPU through the same llama.cpp engine and JSON grammar as E.
Only the component under test changes, which makes the ablation a fair comparison.
