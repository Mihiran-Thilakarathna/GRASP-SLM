# %% [markdown]
# # GRASP-SLM 1/4: QLoRA supervised fine-tuning
#
# Fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit NF4, LoRA rank 16 on attention and MLP projections) to turn
# *instruction + retrieved chunks* into the GRASP JSON analysis (proposal section 6.4).
#
# **Before running**
# 1. Kaggle: *Settings → Accelerator → GPU T4 x2* (or Colab: *Runtime → T4 GPU*), *Internet on*.
# 2. Add your Hugging Face **write** token as a secret named `HF_TOKEN`
#    (Kaggle: *Add-ons → Secrets*; Colab: the key icon on the left). Tick "notebook access".
# 3. Set `HF_USER` below to your Hugging Face username.
#
# Checkpoints are pushed to the Hub every `SAVE_STEPS`, so if the free session stops you can run the
# notebook again and it resumes from the last checkpoint (proposal, Table 6).

# %%
HF_USER = "your-hf-username"          # <-- change this
DATASET_REPO = f"{HF_USER}/grasp-slm-papers"
SFT_LORA_REPO = f"{HF_USER}/grasp-slm-sft-lora"
SFT_MERGED_REPO = f"{HF_USER}/grasp-slm-sft-merged"
CKPT_REPO = f"{HF_USER}/grasp-slm-sft-checkpoints"

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ = 6144
LORA_RANK = 16
EPOCHS = 3
LR = 2e-4
BATCH, GRAD_ACC = 1, 8
SAVE_STEPS = 25
SEED = 3407

# %%
# Installs Unsloth with a matching TRL / PEFT / bitsandbytes. Takes a few minutes.
!pip install -q unsloth

# %%
import inspect, json, os
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
os.environ["HF_TOKEN"] = HF_TOKEN
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
OUT = WORK / "sft"

# %%
from datasets import load_dataset
from huggingface_hub import hf_hub_download

files = {s: hf_hub_download(DATASET_REPO, f"sft_{s}.jsonl", repo_type="dataset", token=HF_TOKEN)
         for s in ("train", "val")}
ds = load_dataset("json", data_files=files)
print(ds)

# %%
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

# %%
# The chat template turns [system, user, assistant] into the same ChatML string the Java app sends.
def to_text(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}


ds = ds.map(to_text)
lengths = [len(tokenizer(t).input_ids) for t in ds["train"]["text"]]
print(f"train tokens: max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}")
too_long = sum(l > MAX_SEQ for l in lengths)
if too_long:
    print(f"dropping {too_long} examples longer than {MAX_SEQ} tokens")
    ds = ds.filter(lambda ex: len(tokenizer(ex["text"]).input_ids) <= MAX_SEQ)

# %%
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only


def make_config(cls, **kw):
    """TRL renamed some arguments between versions; map them to whatever this version accepts."""
    params = inspect.signature(cls.__init__).parameters
    if "max_seq_length" in kw and "max_seq_length" not in params:
        kw["max_length"] = kw.pop("max_seq_length")
    unknown = [k for k in kw if k not in params]
    if unknown:
        print("ignoring arguments not supported by this TRL version:", unknown)
    return cls(**{k: v for k, v in kw.items() if k in params})


args = make_config(
    SFTConfig, output_dir=str(OUT), dataset_text_field="text", max_seq_length=MAX_SEQ, packing=False,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACC, num_train_epochs=EPOCHS,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01, optim="adamw_8bit",
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
    eval_strategy="steps", eval_steps=SAVE_STEPS, save_strategy="steps", save_steps=SAVE_STEPS,
    save_total_limit=2, per_device_eval_batch_size=1, report_to="none", seed=SEED,
    push_to_hub=True, hub_model_id=CKPT_REPO, hub_strategy="checkpoint", hub_private_repo=True,
    hub_token=HF_TOKEN)

trainer_kw = {"processing_class": tokenizer} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters \
    else {"tokenizer": tokenizer}
trainer = SFTTrainer(model=model, train_dataset=ds["train"], eval_dataset=ds["val"], args=args, **trainer_kw)
# Loss only on the assistant JSON, not on the long prompt.
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")

# %%
# Resume from the last checkpoint on the Hub if an earlier session was interrupted.
from huggingface_hub import HfApi, snapshot_download

resume = None
try:
    if any(f.startswith("last-checkpoint/") for f in HfApi().list_repo_files(CKPT_REPO, token=HF_TOKEN)):
        resume = str(Path(snapshot_download(CKPT_REPO, allow_patterns=["last-checkpoint/*"], token=HF_TOKEN))
                     / "last-checkpoint")
        print("resuming from", resume)
except Exception as e:
    print("no checkpoint to resume from:", type(e).__name__)

stats = trainer.train(resume_from_checkpoint=resume)
print(stats)
print(trainer.evaluate())

# %%
# Quick check on one validation paper: does the fine-tuned model produce schema-shaped JSON?
FastLanguageModel.for_inference(model)
ex = ds["val"][0]
prompt = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)
try:
    parsed = json.loads(answer)
    print("\nparsed OK, fields:", list(parsed))
except json.JSONDecodeError as e:
    print("\nnot valid JSON yet:", e)

# %%
# Deliverables: the LoRA adapter, and a merged 16-bit model that the DPO notebook starts from.
model.push_to_hub(SFT_LORA_REPO, token=HF_TOKEN, private=False)
tokenizer.push_to_hub(SFT_LORA_REPO, token=HF_TOKEN, private=False)
model.push_to_hub_merged(SFT_MERGED_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN, private=True)
print(f"https://huggingface.co/{SFT_LORA_REPO}\nhttps://huggingface.co/{SFT_MERGED_REPO}")
