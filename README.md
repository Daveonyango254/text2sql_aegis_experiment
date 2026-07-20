# text2sql_aegis_experiment

Fine-tune three Small Language Models as the **local generator of the AEGIS-SQL
hybrid system** (three-axis constrained NL2SQL: accuracy / cost / privacy), on a
curated 1,000-example BIRD+Spider corpus, with automatic Hugging Face publishing
and reproducible RunPod execution.

| Notebook | Model | Hub repo | Why it's in the study |
|---|---|---|---|
| `notebooks/qwen_aegis.ipynb` | Qwen2.5-Coder-7B-Instruct | `Daveonyango254/Qwen25-Coder7B-Aegis-Text2SQL` | Strongest open 7B SQL prior (base of CSC-SQL / SLM-SQL / SQL-R1); primary local-path candidate |
| `notebooks/llama31_aegis.ipynb` | Llama-3.1-8B-Instruct | `Daveonyango254/Llama31-8B-Aegis-Text2SQL` | Continuity baseline (predecessor system's generator); 128K ctx; biggest ecosystem |
| `notebooks/phi4_aegis.ipynb` | Phi-4 (14B) | `Daveonyango254/Phi4-14B-Aegis-Text2SQL` | Best reasoning-per-parameter; the candidate most likely to beat the 7Bs after SFT |

Full rationale, every training parameter explained, and all troubleshooting:
**[`DETAILED_GUIDE.md`](DETAILED_GUIDE.md)**. Dataset construction and statistics:
**[`data/DATA_CARD.md`](data/DATA_CARD.md)**.

## Layout

```
data/        bird_train.jsonl · spider_train.jsonl · merged_train.jsonl
             train.jsonl (798) · validation.jsonl (101) · test.jsonl (101)
             DATA_CARD.md · split_stats.json · *_manifest.jsonl · raw/ (vendored sources)
notebooks/   qwen_aegis.ipynb · llama31_aegis.ipynb · phi4_aegis.ipynb
configs/     qwen.yaml · llama31.yaml · phi4.yaml      (papermill -f overrides)
scripts/     build_spider_subset.py · clean_bird.py · merge_and_split.py
             inspect_dataset.py · quick_eval.py · push_to_hub.py · build_notebooks.py
outputs/     checkpoints, adapters, executed notebooks land here
```

## Quickstart (RunPod)

```bash
# 1. Launch a RunPod GPU pod (PyTorch 2.4 / CUDA 12.4 template, ≥48 GB VRAM
#    recommended; 24 GB works for the 7–8B models), attach a persistent volume
#    at /workspace, then SSH in.
cd /workspace
git clone <your-repo-url> text2sql_aegis_experiment
cd text2sql_aegis_experiment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # paste your HF token into .env
                              # (Llama-3.1 is gated: accept its license on the
                              #  model page with the same HF account first)

# 2. Everything long-running goes inside tmux (survives SSH drops):
tmux new -s text2sql

# 3. Train (Qwen first — the reference run), non-interactively via papermill:
papermill notebooks/qwen_aegis.ipynb outputs/qwen_aegis_output.ipynb -f configs/qwen.yaml

# detach: Ctrl-b d   ·   reattach later: tmux attach -t text2sql
# monitor in a second pane/window:  watch -n 1 nvidia-smi
```

Each notebook trains (QLoRA, completion-only loss), evaluates a quick proxy on the
held-out test split, saves the adapter under `outputs/<run>/adapter/`, and **pushes
model + tokenizer + training config + model card to the Hub automatically** when
`push_to_hub: true`.

## Rebuilding the dataset from scratch

```bash
python scripts/clean_bird.py            # raw BIRD 500 -> data/bird_train.jsonl
python scripts/build_spider_subset.py   # vendored Spider train -> 500-example subset
python scripts/merge_and_split.py       # merged + stratified 798/101/101 splits
python scripts/inspect_dataset.py data/merged_train.jsonl
```

Deterministic under the default `--seed 42`; the Spider raw inputs are vendored in
`data/raw/` so no external download is needed.

## The >70% EX target — read this before training

SFT on 1,000 examples alone will **not** reach 70% BIRD-dev EX; that number is
reachable only as **model + AEGIS harness**: execution-guided self-consistency over
result sets, database value grounding, deterministic post-processing, and the
Reviewer's repair round (the paper measures the harness at roughly 4x the
contribution of the checkpoint swap). The notebooks end with a `LocalGenerator`
adapter that plugs the fine-tuned model straight into that pipeline, and
`DETAILED_GUIDE.md` § "Path to >70%" lays out the honest staircase — including when
to switch the init from the instruct base to an already-SQL-tuned checkpoint.
