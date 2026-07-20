#!/usr/bin/env python3
"""Generate the three model-specific fine-tuning notebooks from one template.

Why generated: the three notebooks must stay cell-for-cell comparable so that a
result difference is attributable to the checkpoint, not to drift between
hand-edited notebooks. Model-specific facts (chat response template, LoRA
target-module names, batch geometry) live in MODELS below; everything else is
shared. Re-run this script after editing to regenerate all three.
"""
import nbformat as nbf
import copy, json, os

MODELS = {
    "qwen_aegis": dict(
        pretty="Qwen2.5-Coder-7B-Instruct",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        hub_repo="Daveonyango254/Qwen25-Coder7B-Aegis-Text2SQL",
        response_template="<|im_start|>assistant\n",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_r=32, lora_alpha=64, lr=1e-4, per_device_bs=2, grad_accum=8,
        rationale=(
"**Why this model.** Qwen2.5-Coder-7B-Instruct is the strongest open 7B prior for SQL: it is "
"the base checkpoint of the current best small text-to-SQL systems (CSC-SQL, SLM-SQL, SQL-R1 "
"all build on it), ships a 32K context, and carries an Apache-2.0 license. It is the primary "
"candidate for the AEGIS local path and the reference point the other two notebooks are "
"compared against. If the request's 'Qwen 7B' meant the general Qwen2.5-7B-Instruct: the Coder "
"variant strictly dominates it on SQL benchmarks at identical size and cost, so we fine-tune "
"the Coder variant.")),
    "llama31_aegis": dict(
        pretty="Llama-3.1-8B-Instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        hub_repo="Daveonyango254/Llama31-8B-Aegis-Text2SQL",
        response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_r=32, lora_alpha=64, lr=1e-4, per_device_bs=2, grad_accum=8,
        rationale=(
"**Why this model.** Llama-3.1-8B-Instruct is the continuity baseline: the predecessor agentic "
"system's local generator was Llama-3.1-8B, so this run answers 'what does the same family gain "
"from domain SFT?' directly. It brings a 128K context (headroom for very wide enterprise "
"schemas), the largest tooling ecosystem of any open model, and strong general reasoning. It is "
"not expected to beat the Coder-7B on raw EX at equal size, and that comparison is exactly the "
"point. Note: the repo is license-gated — accept the license on the model page with the same HF "
"account as your token before running.")),
    "phi4_aegis": dict(
        pretty="Phi-4 (14B)",
        model_id="microsoft/phi-4",
        hub_repo="Daveonyango254/Phi4-14B-Aegis-Text2SQL",
        response_template="<|im_start|>assistant<|im_sep|>",
        target_modules=["qkv_proj","o_proj","gate_up_proj","down_proj"],
        lora_r=16, lora_alpha=32, lr=8e-5, per_device_bs=1, grad_accum=16,
        rationale=(
"**Why this model.** Phi-4 (14B, MIT license) has the best reasoning-per-parameter of current "
"open SLMs and is the candidate most likely to *outperform* the 7B models after SFT, "
"particularly on BIRD's moderate/challenging tiers where multi-step reasoning, not SQL syntax, "
"is the bottleneck. Trade-offs accepted knowingly: ~2x the training and inference cost of a 7B, "
"a 16K context (ample — our longest prompt is ~3.7K tokens), and packed attention modules "
"(qkv_proj / gate_up_proj), which is why its LoRA target list differs from the Llama-style "
"models. LoRA rank is halved (r=16) to keep the adapter lean at 14B.")),
}

SHARED_PARAMS = dict(
    data_dir="data", train_file="train.jsonl", val_file="validation.jsonl",
    test_file="test.jsonl", max_seq_len=4096, epochs=3, warmup_ratio=0.03,
    weight_decay=0.0, lora_dropout=0.05, seed=42, output_root="outputs",
    push_to_hub=True, merge_adapter=False, resume=True, quick_eval_n=50,
    logging_steps=10,
)

def build(name, cfg):
    nb = nbf.v4.new_notebook()
    # Pin a kernelspec so papermill can launch the notebook non-interactively
    # without needing an explicit `-k`/`--kernel` override; without this,
    # papermill raises "No kernel name found in notebook and no override provided."
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    C, M = [], nb.cells
    def md(s):   M.append(nbf.v4.new_markdown_cell(s))
    def code(s, tags=None):
        c = nbf.v4.new_code_cell(s)
        if tags: c.metadata["tags"] = tags
        M.append(c)

    md(f"""# {cfg['pretty']} — AEGIS Text-to-SQL Fine-Tuning

Fine-tunes **`{cfg['model_id']}`** with QLoRA on the merged BIRD+Spider corpus (1,000 examples),
evaluates a quick proxy on the held-out test split, and pushes the result to
**`{cfg['hub_repo']}`**.

{cfg['rationale']}

**Run non-interactively** (recommended on RunPod, inside tmux):
```bash
papermill notebooks/{name}.ipynb outputs/{name}_output.ipynb -f configs/{name.replace('_aegis','')}.yaml
```
All knobs below are papermill parameters; see `DETAILED_GUIDE.md` for what each one does,
why this value was chosen, and how changing it moves accuracy/cost.""")

    params = dict(SHARED_PARAMS)
    params.update(model_id=cfg["model_id"], hub_repo=cfg["hub_repo"],
                  response_template=cfg["response_template"],
                  target_modules=cfg["target_modules"], lora_r=cfg["lora_r"],
                  lora_alpha=cfg["lora_alpha"], learning_rate=cfg["lr"],
                  per_device_bs=cfg["per_device_bs"], grad_accum=cfg["grad_accum"],
                  run_name=name)
    lines = ["# Parameters (papermill overrides land here)"]
    for k, v in params.items():
        # repr() emits valid Python literals (True/False/None, quoted strings);
        # json.dumps() would emit JSON's true/false/null, which are NameErrors
        # when the parameters cell is executed.
        lines.append(f"{k} = {repr(v)}")
    code("\n".join(lines), tags=["parameters"])

    code('''# 1) Environment ----------------------------------------------------------
# RunPod PyTorch images ship torch+CUDA. Everything else is pinned in requirements.txt:
#   pip install -r requirements.txt
import os, sys, json, time, random, re
import torch
from dotenv import load_dotenv

load_dotenv()                                   # reads HF_TOKEN from .env
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN not found — copy .env.example to .env and fill it in"
from huggingface_hub import login
login(token=HF_TOKEN)

assert torch.cuda.is_available(), "No CUDA device visible — check RunPod GPU + drivers"
print("GPU:", torch.cuda.get_device_name(0),
      f"| VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
import transformers, peft, trl, bitsandbytes, datasets as hf_datasets
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| bitsandbytes", bitsandbytes.__version__)

random.seed(seed); torch.manual_seed(seed)
os.makedirs(f"{output_root}/{run_name}", exist_ok=True)''')

    code('''# 2) Data -----------------------------------------------------------------
# Records are chat triples {system,user,assistant}; we render each with the model's own
# chat template into a single "text" field. Loss is masked to the assistant span only
# (completion-only) so the model is never trained to regenerate schemas.
from datasets import load_dataset

data_files = {"train": f"{data_dir}/{train_file}", "validation": f"{data_dir}/{val_file}"}
ds = load_dataset("json", data_files=data_files)
print(ds)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # required for batching (Llama, Phi)

def render(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                                  add_generation_prompt=False)}
ds = ds.map(render, remove_columns=["messages"])
print(ds["train"][0]["text"][:600], "…")

lens = [len(tokenizer(t).input_ids) for t in ds["train"]["text"][:200]]
print(f"token lengths (first 200): p50={sorted(lens)[100]} max={max(lens)} "
      f"(max_seq_len={max_seq_len} — raise it if max approaches the cap)")''')

    code('''# 3) Model — 4-bit NF4 QLoRA base ------------------------------------------
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

attn_impl = "sdpa"
try:                                             # flash-attn is optional; sdpa is a safe fallback
    import flash_attn                            # noqa: F401
    attn_impl = "flash_attention_2"
except ImportError:
    pass
print("attention implementation:", attn_impl)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl, device_map="auto", trust_remote_code=True)
model.config.use_cache = False                   # incompatible with gradient checkpointing''')

    code('''# 4) LoRA ------------------------------------------------------------------
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                  bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
model = get_peft_model(model, lora)
model.print_trainable_parameters()''')

    code('''# 5) Trainer ---------------------------------------------------------------
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(response_template=response_template,
                                           tokenizer=tokenizer)
args = SFTConfig(
    output_dir=f"{output_root}/{run_name}", run_name=run_name,
    num_train_epochs=epochs, learning_rate=learning_rate,
    per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=grad_accum,
    per_device_eval_batch_size=per_device_bs,
    lr_scheduler_type="cosine", warmup_ratio=warmup_ratio, weight_decay=weight_decay,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=max_seq_len, dataset_text_field="text", packing=False,
    logging_steps=logging_steps, eval_strategy="epoch",
    save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    seed=seed, report_to="none")

trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                     train_dataset=ds["train"], eval_dataset=ds["validation"],
                     data_collator=collator)''')

    code('''# 6) Train (auto-resumes from the last checkpoint if one exists) -----------
import glob
last_ckpt = None
if resume:
    ckpts = sorted(glob.glob(f"{output_root}/{run_name}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    last_ckpt = ckpts[-1] if ckpts else None
    print("resuming from:", last_ckpt or "scratch")
t0 = time.time()
trainer.train(resume_from_checkpoint=last_ckpt)
print(f"training wall-clock: {(time.time()-t0)/60:.1f} min")
print(trainer.state.log_history[-3:])''')

    code('''# 7) Quick proxy evaluation on the test split ------------------------------
# sqlglot parse rate (floor sanity) + normalized exact match (conservative EX proxy).
# Full BIRD Execution Accuracy needs the BIRD databases + official evaluator — see
# DETAILED_GUIDE.md "Full BIRD evaluation".
import sqlglot
test = [json.loads(l) for l in open(f"{data_dir}/{test_file}")][:quick_eval_n]

def norm(s):
    s = re.sub(r"\\s+", " ", s.strip().rstrip(";")).strip()
    return re.sub(r"\\s*([(),=<>])\\s*", r"\\1", s).upper()

model.eval(); model.config.use_cache = True
parse_ok = em = 0; samples = []
for i, rec in enumerate(test):
    prompt = tokenizer.apply_chat_template(rec["messages"][:2], tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    sql = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    sql = sql.split("```")[-2].replace("sql", "", 1).strip() if "```" in sql else sql
    gold = rec["messages"][2]["content"]
    try:
        sqlglot.parse_one(re.sub(r"`[^`]*`", "x", sql), read="sqlite"); parse_ok += 1
    except Exception:
        pass
    em += norm(sql) == norm(gold)
    if i < 3: samples.append((rec["messages"][1]["content"].split("Question:")[-1][:120], sql, gold))

n = len(test)
metrics = {"n": n, "parse_rate": round(parse_ok/n, 4), "normalized_em": round(em/n, 4)}
print(metrics)
json.dump(metrics, open(f"{output_root}/{run_name}/quick_eval.json", "w"), indent=1)
for q, p, g in samples:
    print("\\nQ:", q.strip(), "\\nPRED:", p, "\\nGOLD:", g)
model.config.use_cache = False''')

    code('''# 8) Save adapter (and optionally a merged fp16 model) ----------------------
adapter_dir = f"{output_root}/{run_name}/adapter"
trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
json.dump({k: v for k, v in vars().items()
           if k in ("model_id","learning_rate","epochs","lora_r","lora_alpha",
                    "lora_dropout","target_modules","per_device_bs","grad_accum",
                    "max_seq_len","warmup_ratio","seed")},
          open(f"{adapter_dir}/training_config.json", "w"), indent=1)
print("adapter saved:", adapter_dir)

if merge_adapter:                                # fp16 merged weights for vLLM serving
    merged = trainer.model.merge_and_unload()
    merged_dir = f"{output_root}/{run_name}/merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print("merged model saved:", merged_dir)''')

    code(f'''# 9) Model card + push to the Hub ------------------------------------------
card = f"""---
license: other
base_model: {{model_id}}
tags: [text-to-sql, bird, spider, qlora, aegis-sql]
---
# {{hub_repo.split('/')[-1]}}

QLoRA fine-tune of `{{model_id}}` for Text-to-SQL, trained as the local-path
generator of the AEGIS-SQL hybrid system (three-axis constrained NL2SQL:
accuracy / cost / privacy).

- **Data**: 1,000 curated examples — 500 BIRD-train + 500 diversity-stratified
  Spider-train (set ops, HAVING, nested, multi-join rebalanced); splits 798/101/101.
- **Recipe**: 4-bit NF4 QLoRA, r={{lora_r}} alpha={{lora_alpha}}, lr={{learning_rate}},
  {{epochs}} epochs, completion-only loss, max_seq_len={{max_seq_len}}, seed={{seed}}.
- **Prompt format**: chat messages; user = `Database:` + schema (Tables + Foreign
  Keys) + `Question:` + `Return only SQL.`; assistant = bare SQL.
- **Quick proxy on held-out test**: see `quick_eval.json` in this repo.
  Full BIRD EX requires the official evaluator + databases.

Trained with the text2sql_aegis_experiment package (notebooks/{name}.ipynb).
"""
if push_to_hub:
    from huggingface_hub import HfApi
    open(f"{{adapter_dir}}/README.md", "w").write(card)
    api = HfApi(token=HF_TOKEN)
    api.create_repo(hub_repo, exist_ok=True)
    api.upload_folder(folder_path=adapter_dir, repo_id=hub_repo)
    api.upload_file(path_or_fileobj=f"{{output_root}}/{{run_name}}/quick_eval.json",
                    path_in_repo="quick_eval.json", repo_id=hub_repo)
    print("pushed:", f"https://huggingface.co/{{hub_repo}}")
else:
    print("push_to_hub=False — skipped")''')

    code('''# 10) Inference example ----------------------------------------------------
def generate_sql(question: str, db_id: str, schema_text: str,
                 max_new_tokens: int = 256) -> str:
    """schema_text: the 'Tables ... Foreign Keys ...' block exactly as in training."""
    user = (f"Database: {db_id}\\n\\nSchema:\\n\\n{schema_text}\\n\\n"
            f"Question:\\n{question}\\n\\nReturn only SQL.")
    msgs = [{"role": "system", "content":
             "You are an expert SQL generator. Return only executable SQL with no explanation."},
            {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.config.use_cache = True
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    model.config.use_cache = False
    return tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

demo_schema = "Tables\\n\\nemployees(\\n    id,\\n    name,\\n    hired\\n)"
print(generate_sql("List names of employees hired after 2020.", "demo_db", demo_schema))''')

    md("""## Plugging the model into the AEGIS agentic system

The AEGIS local path calls one function: `generate_sql(question, schema_text, evidence)`.
The adapter below is a drop-in `LocalGenerator` for the graph pipeline (Query Planner →
router → **local SLM** → Reviewer). Candidate sampling feeds the pipeline's
execution-guided self-consistency lever — the Reviewer executes the candidates and votes
over result sets, so `n_candidates > 1` is where the harness earns its accuracy.""")

    code('''# 11) AEGIS local-path adapter ---------------------------------------------
class LocalGenerator:
    """Drop-in local SLM for the AEGIS graph pipeline.

    Contract (matches the Query Planner Agent's output):
      generate(question, db_id, schema_text, evidence=None, n_candidates=4)
        -> list[str] SQL candidates (greedy first, then temperature samples),
      which the Reviewer executes and votes over (result-set self-consistency).
    """
    def __init__(self, model, tokenizer, system_prompt=None):
        self.m, self.t = model, tokenizer
        self.sys = system_prompt or ("You are an expert SQL generator. "
                                     "Return only executable SQL with no explanation.")

    def _prompt(self, question, db_id, schema_text, evidence):
        ev = f"\\n\\nEvidence:\\n{evidence}" if evidence else ""
        user = (f"Database: {db_id}\\n\\nSchema:\\n\\n{schema_text}{ev}\\n\\n"
                f"Question:\\n{question}\\n\\nReturn only SQL.")
        msgs = [{"role": "system", "content": self.sys},
                {"role": "user", "content": user}]
        return self.t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question, db_id, schema_text, evidence=None, n_candidates=4):
        ids = self.t(self._prompt(question, db_id, schema_text, evidence),
                     return_tensors="pt").to(self.m.device)
        outs, self.m.config.use_cache = [], True
        with torch.no_grad():
            g = self.m.generate(**ids, max_new_tokens=256, do_sample=False,
                                pad_token_id=self.t.pad_token_id)
            outs.append(g[0][ids.input_ids.shape[1]:])
            if n_candidates > 1:
                s = self.m.generate(**ids, max_new_tokens=256, do_sample=True,
                                    temperature=0.8, top_p=0.95,
                                    num_return_sequences=n_candidates - 1,
                                    pad_token_id=self.t.pad_token_id)
                outs += [row[ids.input_ids.shape[1]:] for row in s]
        self.m.config.use_cache = False
        return [self.t.decode(o, skip_special_tokens=True).strip() for o in outs]

gen = LocalGenerator(model, tokenizer)
print(gen.generate("How many employees were hired after 2020?", "demo_db",
                   demo_schema, n_candidates=2))''')

    md(f"""### Serving for the agent at scale (vLLM)

For pipeline-scale evaluation, serve the **merged** model (set `merge_adapter = True`,
re-run cell 8) behind an OpenAI-compatible endpoint and point the AEGIS local path at it:

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model outputs/{name}/merged \\
    --served-model-name {name} \\
    --max-model-len {SHARED_PARAMS['max_seq_len']} --gpu-memory-utilization 0.90
```

Troubleshooting (OOM, HF auth, bitsandbytes, flash-attn, resume): see
`DETAILED_GUIDE.md` — every failure mode listed there includes its exact fix.""")

    return nb

def main():
    os.makedirs("notebooks", exist_ok=True)
    for name, cfg in MODELS.items():
        nb = build(name, cfg)
        path = f"notebooks/{name}.ipynb"
        nbf.write(nb, path)
        print("wrote", path, f"({len(nb.cells)} cells)")

if __name__ == "__main__":
    main()
