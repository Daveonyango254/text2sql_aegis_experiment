# DETAILED_GUIDE.md

Everything this package does, why each decision was made, and how to change it.

- [1. Project architecture](#1-project-architecture)
- [2. Model selection rationale](#2-model-selection-rationale)
- [3. Dataset construction](#3-dataset-construction)
- [4. Every configuration parameter](#4-every-configuration-parameter)
- [5. RunPod deployment](#5-runpod-deployment)
- [6. tmux session management](#6-tmux-session-management)
- [7. Papermill execution](#7-papermill-execution)
- [8. Evaluation](#8-evaluation)
- [9. Path to >70% EX](#9-path-to-70-ex)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. Project architecture

### Data flow

```
data/raw/bird_train_500_original.jsonl ──clean_bird.py──────► data/bird_train.jsonl   (500)
data/raw/spider_train_slim.json      ─┐
data/raw/spider_tables.json          ─┴build_spider_subset.py► data/spider_train.jsonl (500)
                                                    │
                                      merge_and_split.py
                                                    ▼
              data/merged_train.jsonl (1,000) ─► train 798 / validation 101 / test 101
                                                    │
                                          (+ line-aligned *_manifest.jsonl)
```

Every record is a chat triple `{system, user, assistant}` where `user` carries
`Database:` + `Schema:` (Tables + Foreign Keys) + `Question:` + `Return only SQL.`
and `assistant` is bare SQL. The template is byte-identical across both sources —
that consistency is what lets the model treat schema-reading as one skill.

### Training flow (identical in all three notebooks)

```
chat records ─apply_chat_template─► text
              │
              ├─ 4-bit NF4 quantized base model (bitsandbytes)
              ├─ LoRA adapters on attention + MLP projections (peft)
              └─ SFTTrainer + DataCollatorForCompletionOnlyLM
                       │  loss masked to the assistant span only
                       ▼
              cosine LR schedule, 3 epochs, eval+save each epoch
                       ▼
              best-by-eval_loss adapter ─► outputs/<run>/adapter/ ─► Hugging Face Hub
```

**Why completion-only loss.** With full-sequence loss, ~90% of the tokens are schema
text, so most of the gradient teaches the model to *regenerate schemas* — a skill it
never needs at inference. Masking to the assistant span puts the entire optimization
budget on SQL generation. This is the single most important recipe choice in the
package for a schema-heavy corpus.

### Evaluation flow

```
test.jsonl ─► greedy generation ─► parse rate (sqlglot) + normalized exact match
                                            │
                                    outputs/<run>/quick_eval.json ─► pushed to Hub
```

The in-notebook metric is a **proxy** (see §8). True BIRD Execution Accuracy needs
the BIRD databases and the official evaluator, and is run through the AEGIS pipeline,
not the bare model.

### Deployment flow

```
adapter ──(merge_adapter=True)──► merged fp16 ──► vLLM OpenAI-compatible server
                                                        │
                                       AEGIS graph pipeline local path
                     Query Planner → content-independent router → LOCAL SLM → Reviewer
```

The notebooks' final `LocalGenerator` class is the integration contract: it returns
*a list of candidates* (greedy + temperature samples) because the AEGIS Reviewer
executes them and votes over result sets — the execution-guided self-consistency
lever. A generator returning one string cannot feed that lever.

---

## 2. Model selection rationale

You asked for Qwen 7B plus two SLMs likely to beat it after domain fine-tuning,
traded off across fine-tuning efficiency, inference cost, SQL reasoning, and context.

### Qwen2.5-Coder-7B-Instruct — the reference run

The strongest open 7B prior for SQL by a wide margin: it is the base checkpoint of
essentially every leading small text-to-SQL system (CSC-SQL, SLM-SQL, SQL-R1), it is
code-pretrained (SQL is in-distribution rather than incidental), it ships a 32K
context, and it is Apache-2.0. It is also the family your paper's current local
generator comes from (`CSC-SQL-Merge-Qwen2.5-Coder-7B`), which keeps this study
commensurable with the results already in the paper.

> **Note on "Qwen 7B".** If you meant the general `Qwen2.5-7B-Instruct`, the Coder
> variant dominates it on SQL at identical parameter count, VRAM, and latency — same
> cost, better SQL. Swap `model_id` in `configs/qwen.yaml` if you specifically want
> the general variant for a controlled comparison.

### Llama-3.1-8B-Instruct — continuity baseline

Your predecessor system's local generator was Llama-3.1-8B, so this run answers a
question no other checkpoint can: *what does the same family gain from domain SFT?*
It adds a 128K context (headroom for very wide enterprise schemas well beyond BIRD's)
and the largest tooling/quantization ecosystem of any open model. It is **not**
expected to beat the Coder-7B on raw EX — and that measured gap is a real result for
the multi-SLM Pareto study in the paper.

*Caveat:* the repo is license-gated. Accept the license on the model page with the
same HF account as your token, or `from_pretrained` returns 401.

### Phi-4 (14B) — the upside candidate

The model most likely to *outperform* the 7Bs after fine-tuning. Phi-4 has the best
reasoning-per-parameter among current open SLMs (trained heavily on curated synthetic
reasoning data) and is MIT-licensed. This matters because your own error analysis
found the bottleneck is **multi-table JOIN reasoning, not SQL syntax** — a reasoning
deficit, exactly what Phi-4 is strong at. Trade-offs, accepted knowingly:

| Axis | Phi-4 (14B) vs the 7–8B models |
|---|---|
| Fine-tuning efficiency | ~2x time/VRAM; needs `per_device_bs=1`, `grad_accum=16` |
| Inference cost | ~2x per query — pushes against the AEGIS cost axis |
| SQL reasoning | Best of the three on multi-step/nested reasoning |
| Context | 16K — ample (our longest prompt is ~3.7K tokens) |
| LoRA targets | Packed modules (`qkv_proj`, `gate_up_proj`), hence a different target list |

LoRA rank is halved (r=16 vs 32) at 14B to keep the adapter lean; the larger base
needs less adapter capacity for the same task shift.

### Alternatives considered and rejected

- **Gemma-2-9B** — 8K context, no code-specialized variant at 9B, and a restrictive
  license; strictly worse than Phi-4 for this use.
- **DeepSeek-Coder-7B** — strong SQL prior, but Qwen2.5-Coder supersedes it on every
  recent SQL leaderboard.
- **Starting from an already-SQL-tuned checkpoint** (CSC-SQL, SLM-SQL) — this is the
  *highest-EX* path and is discussed in §9, but it measures "can I improve an
  already-tuned model," not "what does my corpus buy," which is what this study asks.

---

## 3. Dataset construction

Full statistics, licenses, and limitations: **`data/DATA_CARD.md`**. The reasoning:

**BIRD 500 audit.** Structurally sound — 500/500 well-formed triples, zero duplicate
questions, zero duplicate records, one consistent system prompt, 100% sqlglot-parseable,
67 databases. Two real issues: 13 targets carried trailing semicolons the other 487
didn't (inconsistent target distribution → normalized away), and the file has **no
`Evidence` field** even though the AEGIS planner injects evidence at inference. The
latter is a train/inference mismatch and is the single highest-value future revision
(§ Data Card, limitation 1).

**Why the Spider subset is stratified, not random.** The BIRD 500 is join- and
aggregate-heavy but thin where it matters: HAVING 7, nested 37, GROUP BY 49, set
operations 1, window functions 0. Random Spider sampling would have inherited
Spider's own easy/medium skew and left those holes. The builder classifies every
Spider train query and fills explicit quotas (priority: set_op > nested > having >
multi_join > group_by > single_join > single_table_analytic), after paraphrase
dedupe on `(db_id, normalized query)` and with a 12-example cap per database.

Result — the merged corpus, versus BIRD alone:

| Construct | BIRD 500 | + Spider 500 | Merged 1,000 |
|---|---|---|---|
| HAVING | 7 | 74 | **81** |
| Nested subquery | 37 | 140 | **177** |
| GROUP BY | 49 | 173 | **222** |
| Set ops (UNION/INTERSECT/EXCEPT) | 1 | 50 | **51** |
| JOIN | 382 | 278 | 660 |
| Databases | 67 | 131 | **198** |

**Splits.** 80/10/10 stratified on (source × structural bucket), seed 42, so rare
constructs appear in validation and test rather than landing entirely in train.
Train/dev contamination is impossible by construction: both halves come from
*training* splits only, so BIRD-dev evaluation stays clean.

---

## 4. Every configuration parameter

Set in the notebook `parameters` cell; override per run via `configs/*.yaml`.

### LoRA / QLoRA

| Parameter | Default | What it does · why this value · effect of changing |
|---|---|---|
| `load_in_4bit` (in code) | `True` NF4 | Quantizes frozen base weights to 4-bit NF4 so a 7–14B model trains on one GPU. **Why:** ~4x VRAM reduction at ~1% quality cost. **Change:** disable only if you have ≥80 GB and want the last fraction of quality. |
| `bnb_4bit_compute_dtype` | `bfloat16` | Dtype of the dequantized matmuls. **Why:** bf16 has fp32's exponent range — no loss-scaling instability. **Change:** `float16` only on pre-Ampere GPUs (T4/V100). |
| `bnb_4bit_use_double_quant` | `True` | Quantizes the quantization constants too. **Why:** ~0.4 GB free at no measurable cost. |
| `lora_r` | 32 (16 for Phi-4) | Rank = capacity of the update. **Why:** 32 is the sweet spot for domain SFT on ~1k examples; below 8 underfits SQL dialect, above 64 overfits a small corpus. **Change:** raise with much more data; lower if val loss diverges from train loss early. |
| `lora_alpha` | 64 (32) | Scaling; effective LR multiplier is `alpha/r`. **Why:** the `alpha = 2r` convention → multiplier 2. **Change:** keep the 2:1 ratio when changing `r`, or you are silently changing the LR. |
| `lora_dropout` | 0.05 | Dropout on adapter activations. **Why:** light regularization for 1k examples × 3 epochs. **Change:** 0.1 if overfitting; 0.0 for large corpora. |
| `target_modules` | attn + MLP | Which matrices get adapters. **Why:** attention-only adapters underperform on tasks needing new *knowledge* (schema→SQL mapping); MLP inclusion is worth the extra params. **Change:** Phi-4 packs projections (`qkv_proj`, `gate_up_proj`) — using Llama's names there raises `Target modules not found`. |

### Optimization

| Parameter | Default | What it does · why this value · effect of changing |
|---|---|---|
| `learning_rate` | 1e-4 (8e-5 Phi-4) | Step size. **Why:** LoRA tolerates ~10x full-FT LRs; 1e-4 is the standard QLoRA setting; larger models prefer slightly lower. **Change:** loss spiking/NaN → 5e-5; flat loss → 2e-4. |
| `epochs` | 3 | Passes over 798 examples. **Why:** 1 underfits the output format; ≥5 memorizes at this size. **Change:** watch `eval_loss` — `load_best_model_at_end` already keeps the best epoch. |
| `per_device_bs` | 2 (1 Phi-4) | Sequences per forward pass. **Why:** at `max_seq_len=4096` this is what fits in 24–48 GB. **Change:** first knob to drop on OOM. |
| `grad_accum` | 8 (16 Phi-4) | Micro-batches per optimizer step. **Why:** effective batch = `bs × accum` = **16** for all three models — identical optimization geometry makes the three runs comparable. **Change:** raise it whenever you lower `per_device_bs`, to hold 16. |
| `lr_scheduler_type` | `cosine` | LR decay shape. **Why:** smooth decay to ~0 gives a stable final checkpoint; better than linear for short runs. |
| `warmup_ratio` | 0.03 | Fraction of steps ramping LR from 0. **Why:** prevents a large first step from wrecking freshly initialized adapters. **Change:** raise to 0.1 if the first logged losses spike. |
| `weight_decay` | 0.0 | L2 on trainable params. **Why:** LoRA is already low-rank-constrained; decay mostly fights the adapter. |
| `max_seq_len` | 4096 | Truncation length. **Why:** the longest merged prompt is ~3.7k tokens, so **nothing is truncated**; VRAM scales with this. **Change:** lowering to 2048 truncates ~3% of records (silent label loss on the longest schemas) — prefer lowering batch size instead. |
| `packing` | `False` | Concatenating examples into full-length blocks. **Why:** packing is incompatible with clean completion-only masking; correctness beats throughput here. |
| `gradient_checkpointing` | `True` | Recompute activations in backward. **Why:** ~40% VRAM saved for ~20% slowdown — the trade that makes 4096-token training fit. `use_reentrant=False` avoids known PEFT warnings/bugs. |
| `bf16` | `True` | Mixed-precision compute. **Why:** Ampere+ native; no gradient scaler needed. |
| `seed` | 42 | Seeds shuffling and init. **Why:** reproducibility — same seed + same data ⇒ same run. |

### Runtime / plumbing

| Parameter | Default | Purpose |
|---|---|---|
| `response_template` | model-specific | The exact string after which loss is computed. **Must match the tokenizer's chat template byte-for-byte** — a mismatch silently trains on nothing useful (see §10). |
| `eval_strategy` / `save_strategy` | `epoch` | Evaluate and checkpoint each epoch; with `save_total_limit=2` and `load_best_model_at_end`, disk stays bounded and the best adapter is kept. |
| `resume` | `True` | Auto-resume from the newest `checkpoint-*` — the RunPod-interruption insurance policy. |
| `push_to_hub` | `True` | Upload adapter + tokenizer + `training_config.json` + model card after training. |
| `merge_adapter` | `False` | Also write merged fp16 weights. **Why off by default:** adapters are ~200 MB vs ~15 GB. Turn on for vLLM serving. |
| `quick_eval_n` | 50 | Test records for the in-notebook proxy metric. Raise for a tighter estimate, at generation-time cost. |

---

## 5. RunPod deployment

**Pod choice.** PyTorch 2.4 / CUDA 12.4 template. A single 48 GB card (A6000/L40S)
runs all three comfortably; 24 GB (A5000/3090/4090) runs the 7–8B models and runs
Phi-4 only with `per_device_bs=1` and `max_seq_len=2048`. An 80 GB A100/H100 removes
all constraints and roughly halves wall-clock.

**Persistent volume — do this before anything else.** Attach a network volume
mounted at `/workspace` and clone *into it*. Container-local storage is wiped when
the pod stops; `/workspace` is not. Point HF caches there too, or you will re-download
15 GB of weights after every restart:

```bash
export HF_HOME=/workspace/.hf
export HF_HUB_ENABLE_HF_TRANSFER=1     # faster downloads
echo 'export HF_HOME=/workspace/.hf' >> ~/.bashrc
```

**Full setup:**

```bash
cd /workspace
git clone <your-repo-url> text2sql_aegis_experiment
cd text2sql_aegis_experiment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env && nano .env       # paste HF_TOKEN
python -c "import torch;print(torch.cuda.get_device_name(0))"
```

**Expected wall-clock** (798 examples × 3 epochs, effective batch 16 → ~150 steps):

| Model | 24 GB | 48 GB | 80 GB |
|---|---|---|---|
| Qwen2.5-Coder-7B | ~55 min | ~35 min | ~20 min |
| Llama-3.1-8B | ~65 min | ~40 min | ~25 min |
| Phi-4 (14B) | ~2.5 h (bs=1, 2048) | ~80 min | ~45 min |

---

## 6. tmux session management

SSH drops kill foreground processes; a multi-hour training run must never be one.

```bash
tmux new -s text2sql          # create + attach
tmux ls                       # list sessions
tmux attach -t text2sql       # reattach after a disconnect
tmux kill-session -t text2sql # destroy when finished
```

Inside tmux: **Ctrl-b d** detaches (training keeps running), **Ctrl-b c** opens a new
window, **Ctrl-b n/p** cycles windows, **Ctrl-b "** splits horizontally (handy for
`watch -n 1 nvidia-smi` beside the training log).

**Recovery after a disconnect.** Reconnect via SSH, `tmux attach -t text2sql`, and the
run is exactly where you left it. If the *pod itself* restarted, tmux is gone too —
that is what `resume: true` is for: re-run the same papermill command and training
picks up from the newest checkpoint in `outputs/<run>/`.

A useful monitoring layout:

```bash
tmux new -s text2sql
# pane 1: papermill ...
# Ctrl-b "  → pane 2:
watch -n 1 nvidia-smi          # VRAM + utilization
# Ctrl-b "  → pane 3:
htop                           # CPU/RAM (dataloader stalls show here)
tail -f outputs/qwen_aegis_output.ipynb   # papermill writes progressively
```

---

## 7. Papermill execution

Papermill runs a notebook headlessly, injects parameters, and writes an executed copy
with all outputs — your experiment record.

```bash
# with a config file (preferred — the config is versioned with the run)
papermill notebooks/qwen_aegis.ipynb outputs/qwen_aegis_output.ipynb -f configs/qwen.yaml

# with inline parameter overrides
papermill notebooks/qwen_aegis.ipynb outputs/qwen_smoke.ipynb \
    -p epochs 1 -p quick_eval_n 10 -p push_to_hub False

# log to a file as well as the terminal
papermill notebooks/phi4_aegis.ipynb outputs/phi4_output.ipynb \
    -f configs/phi4.yaml --log-output 2>&1 | tee outputs/phi4_run.log
```

**Always smoke-test first** (`-p epochs 1 -p quick_eval_n 10 -p push_to_hub False`):
it exercises data loading, the chat template, the collator, and the save path in a
few minutes, so a typo doesn't surface two hours into the real run.

**Parameter passing** works because the first code cell is tagged `parameters`;
papermill injects an override cell right after it. `-p` values are parsed as
YAML scalars (`-p push_to_hub False` gives a bool, not a string); use `-r` for raw
strings.

**Resuming.** Papermill itself has no resume — the *training* does. Re-issuing the
same command with `resume: true` continues from the last checkpoint. Write to a fresh
output path (`..._output_run2.ipynb`) so you keep both records.

**Running all three sequentially:**

```bash
for m in qwen llama31 phi4; do
  nb=$( [ $m = qwen ] && echo qwen_aegis || ([ $m = llama31 ] && echo llama31_aegis || echo phi4_aegis) )
  papermill notebooks/$nb.ipynb outputs/${nb}_output.ipynb -f configs/$m.yaml --log-output \
      2>&1 | tee outputs/${m}_run.log || echo "FAILED: $m"
done
```

---

## 8. Evaluation

**What the notebook reports.** Two proxies on the held-out test split: sqlglot
**parse rate** (a floor sanity check — a prompt-format regression drops it
immediately) and whitespace/case-**normalized exact match** (deliberately
conservative: it *undercounts*, since a semantically correct query written
differently scores zero).

**What it is not.** Neither is BIRD Execution Accuracy. EX executes both predicted
and gold SQL against the real database and compares result sets. Reporting normalized
EM as EX would overstate nothing but would *understate* the model and, worse, is not
the metric in your paper.

**Full BIRD evaluation** (the number that belongs in the paper):

1. Download the BIRD dev databases and `dev.json` from the official BIRD release.
2. Generate predictions with the fine-tuned model **inside the AEGIS pipeline**
   (planner-built schema context, not the raw full schema — the prompt must match
   training).
3. Run the official BIRD evaluation script for EX (and VES).
4. Report per-difficulty (simple / moderate / challenging), as your paper's Table 1 does.

Report the bare-model number and the with-harness number separately. The paper's
central claim — that the harness dominates the checkpoint — depends on that split
being visible.

---

## 9. Path to >70% EX

Honest staircase, because SFT on 1,000 examples will not get there alone:

| Stage | Expected BIRD-dev EX | Why |
|---|---|---|
| Base instruct model, naive prompt | ~30–37% | Your measured pre-optimization baseline (32.53 → 36.83) |
| **+ this SFT corpus** (bare model) | ~45–55% | Output-format lock-in, schema-reading, construct coverage |
| **+ AEGIS harness** (self-consistency, value grounding, post-processing, repair) | ~55–63% | Your paper measures the harness at ~4x the checkpoint swap (36.83 → 54.63) |
| **+ FK-key exposure & multi-hop closure** | +2–5 pts | Multi-table JOIN reasoning is the identified bottleneck |
| **+ ε-DP remote path for routed-hard queries** | **63%+ → 70%+** | Already at 63.0% with GPT-4o; the remaining gap is router tuning |

Three levers, in order of expected return:

1. **Initialize from an already-SQL-tuned checkpoint.** Fine-tuning
   `CSC-SQL-Merge-Qwen2.5-Coder-7B` (or SLM-SQL) instead of the instruct base starts
   near 54% rather than ~35%. Change one line — `model_id` in `configs/qwen.yaml`.
   This is the fastest route to a high local number, at the cost of a less clean
   "what did my corpus buy" attribution.
2. **Tune the router threshold θ.** Your paper's hybrid at θ=0.7 scored *below*
   local-only, so θ is mistuned; a sweep is cheap and directly moves the headline.
3. **Scale the corpus.** 1,000 curated examples is a deliberate starting point. If
   the multi-SLM study plateaus, add SynSQL-style synthetic data before touching the
   recipe — data, not hyperparameters, is the binding constraint at this size.

---

## 10. Troubleshooting

### CUDA out of memory

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Then, in order: `per_device_bs` → 1 (raise `grad_accum` to hold effective batch 16);
confirm `gradient_checkpointing=True`; lower `max_seq_len` to 3072 (2048 truncates
~3% of records); for Phi-4 on 24 GB use `lora_r=8`. Clear a leaked cache between
runs with `torch.cuda.empty_cache()`, and check for a zombie process holding VRAM:

```bash
nvidia-smi   # note the PID, then:
kill -9 <PID>
```

### Hugging Face authentication failures

- `401 Unauthorized` → token missing/expired. Confirm `.env` exists (not just
  `.env.example`), that `load_dotenv()` ran, and that the token has **write** scope.
- `403 Forbidden` on Llama-3.1 → gated repo. Accept the license on the model page
  with the *same account* as the token.
- Push fails after a long training run → don't retrain:
  `python scripts/push_to_hub.py outputs/qwen_aegis/adapter Daveonyango254/Qwen25-Coder7B-Aegis-Text2SQL`
- Verify identity: `python -c "from huggingface_hub import whoami; print(whoami())"`

### BitsAndBytes installation issues

`CUDA Setup failed` / `libbitsandbytes_cpu.so` means the wheel didn't match the CUDA
runtime:

```bash
python -m bitsandbytes            # prints the diagnostic
pip uninstall -y bitsandbytes && pip install bitsandbytes==0.45.0
echo $LD_LIBRARY_PATH             # should include /usr/local/cuda/lib64
```

If it still fails, the pod's CUDA differs from the wheel's — switch to a PyTorch
2.4/CUDA 12.4 RunPod template rather than fighting the install.

### Flash Attention errors

Flash-attn is **optional** here: the notebook try/excepts the import and falls back
to `sdpa`, which is fast and correct. If you want it:

```bash
pip install flash-attn --no-build-isolation   # ~10 min compile; needs Ampere+
```

`undefined symbol` after install = version mismatch with torch → uninstall it and
stay on `sdpa`. Never install it on a T4/V100 (unsupported architecture).

### Dataset formatting errors

- `KeyError: 'messages'` → you pointed at a manifest file; training files are
  `train.jsonl` / `validation.jsonl`, manifests are `*_manifest.jsonl`.
- **Loss stays flat / predictions are empty** → almost always a `response_template`
  mismatch. Verify what the template actually emits:
  ```python
  print(repr(tokenizer.apply_chat_template(ds["train"][0]["messages"], tokenize=False)))
  ```
  and make `response_template` a byte-exact substring of it. Qwen/Phi-4 and Llama use
  entirely different markers — this is the #1 silent failure in this pipeline.
- `Token indices sequence length is longer than...` → harmless warning if
  `max_seq_len` covers your data (it does at 4096); it fires during raw tokenization.
- Re-validate any regenerated data with `python scripts/inspect_dataset.py data/train.jsonl`.

### RunPod volume mount problems

`df -h /workspace` to confirm the volume is attached and has space; if it's missing,
the pod was launched without it — stop the pod, attach the volume, restart (data in
container storage is already lost). `No space left on device` mid-training is usually
checkpoints or the HF cache: `du -sh outputs/* $HF_HOME`, then drop `save_total_limit`
to 1 and delete stale `checkpoint-*` directories. Always `cd /workspace/...` before
cloning — a repo cloned into `/root` disappears on restart.

### Interrupted training recovery

Everything needed is already on disk in `outputs/<run>/checkpoint-*` (weights,
optimizer state, scheduler state, RNG state). Re-run the same papermill command:
`resume: true` finds the newest checkpoint and continues. Confirm what you have:

```bash
ls -la outputs/qwen_aegis/           # checkpoint-50, checkpoint-100, ...
cat outputs/qwen_aegis/checkpoint-100/trainer_state.json | head -20
```

To resume from a *specific* checkpoint, set `last_ckpt` explicitly in cell 6. To
start clean, delete the run directory — otherwise resume silently continues an old
run and your "fresh" experiment isn't fresh.

### Checkpoint resume procedures (summary)

| Situation | Action |
|---|---|
| SSH dropped, pod alive | `tmux attach -t text2sql` — nothing was interrupted |
| Pod restarted | Re-run the papermill command; `resume: true` handles it |
| OOM crash mid-epoch | Lower `per_device_bs`, raise `grad_accum`, re-run — resumes from last epoch checkpoint |
| Push failed post-training | `scripts/push_to_hub.py` — never retrain to retry an upload |
| Want a truly fresh run | `rm -rf outputs/<run>` first |
