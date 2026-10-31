# %% [markdown]
# # GRASP-SLM 2/4: Direct Preference Optimization
#
# Starts from the merged SFT model and trains a fresh LoRA adapter with DPO on pairs of
# *grounded* (chosen) and *perturbed* (rejected) answers: altered number, invented metric,
# missing citation, malformed JSON (proposal section 6.5).
#
# The reference policy is the SFT model itself (the base weights with the new adapter switched off),
# which is the standard DPO set-up.
#
# Same set-up as notebook 1: GPU T4, Internet on, `HF_TOKEN` secret, set `HF_USER`.

# %%
HF_USER = "your-hf-username"          # <-- change this
DATASET_REPO = f"{HF_USER}/grasp-slm-papers"
SFT_MERGED_REPO = f"{HF_USER}/grasp-slm-sft-merged"
DPO_LORA_REPO = f"{HF_USER}/grasp-slm-dpo-lora"
DPO_MERGED_REPO = f"{HF_USER}/grasp-slm-dpo-merged"
CKPT_REPO = f"{HF_USER}/grasp-slm-dpo-checkpoints"

MAX_SEQ = 6144
MAX_PROMPT = 5120
LORA_RANK = 16
BETA = 0.1
EPOCHS = 1
LR = 5e-6
BATCH, GRAD_ACC = 1, 8
SAVE_STEPS = 25
SEED = 3407

# %%
!pip install -q unsloth

# %%
import inspect, os
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

# %%
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

PatchDPOTrainer()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_MERGED_REPO, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None, token=HF_TOKEN)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

# %%
from datasets import load_dataset
from huggingface_hub import hf_hub_download

files = {s: hf_hub_download(DATASET_REPO, f"dpo_{s}.jsonl", repo_type="dataset", token=HF_TOKEN)
         for s in ("train", "val")}
ds = load_dataset("json", data_files=files)


# TRL appends the EOS token (<|im_end|> for Qwen2.5-Instruct) to chosen and rejected.
def to_strings(ex):
    return {"prompt": tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True),
            "chosen": ex["chosen"], "rejected": ex["rejected"]}


ds = ds.map(to_strings, remove_columns=[c for c in ds["train"].column_names if c not in ("prompt", "chosen", "rejected")])
print(ds)
print("perturbations:", load_dataset("json", data_files=files["train"])["train"].to_pandas()["perturbation"].value_counts().to_dict())

# %%
from trl import DPOConfig, DPOTrainer


def make_config(cls, **kw):
    params = inspect.signature(cls.__init__).parameters
    unknown = [k for k in kw if k not in params]
    if unknown:
        print("ignoring arguments not supported by this TRL version:", unknown)
    return cls(**{k: v for k, v in kw.items() if k in params})


args = make_config(
    DPOConfig, output_dir=str(WORK / "dpo"), beta=BETA, max_length=MAX_SEQ, max_prompt_length=MAX_PROMPT,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACC, num_train_epochs=EPOCHS,
    learning_rate=LR, lr_scheduler_type="linear", warmup_ratio=0.1, optim="adamw_8bit",
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
    eval_strategy="steps", eval_steps=SAVE_STEPS, save_strategy="steps", save_steps=SAVE_STEPS,
    save_total_limit=2, per_device_eval_batch_size=1, report_to="none", seed=SEED,
    push_to_hub=True, hub_model_id=CKPT_REPO, hub_strategy="checkpoint", hub_private_repo=True,
    hub_token=HF_TOKEN)

trainer_kw = {"processing_class": tokenizer} if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters \
    else {"tokenizer": tokenizer}
trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                     **trainer_kw)

# %%
from huggingface_hub import HfApi, snapshot_download

resume = None
try:
    if any(f.startswith("last-checkpoint/") for f in HfApi().list_repo_files(CKPT_REPO, token=HF_TOKEN)):
        resume = str(Path(snapshot_download(CKPT_REPO, allow_patterns=["last-checkpoint/*"], token=HF_TOKEN))
                     / "last-checkpoint")
        print("resuming from", resume)
except Exception as e:
    print("no checkpoint to resume from:", type(e).__name__)

trainer.train(resume_from_checkpoint=resume)
# rewards/accuracies near 1.0 means the model now prefers grounded answers on held-out pairs.
print(trainer.evaluate())

# %%
model.push_to_hub(DPO_LORA_REPO, token=HF_TOKEN, private=False)
tokenizer.push_to_hub(DPO_LORA_REPO, token=HF_TOKEN, private=False)
model.push_to_hub_merged(DPO_MERGED_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN, private=True)
print(f"https://huggingface.co/{DPO_LORA_REPO}\nhttps://huggingface.co/{DPO_MERGED_REPO}")
