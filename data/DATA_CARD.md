# Data Card — AEGIS Text-to-SQL SFT Corpus (v1)

## Sources & licenses
| File | Source benchmark | Records | License |
|---|---|---|---|
| `bird_train.jsonl` | BIRD **train** split (curated by project owner; cleaned here) | 500 | CC BY-SA 4.0 |
| `spider_train.jsonl` | Spider **train** split (diversity-stratified subset built here) | 500 | CC BY-SA 4.0 |
| `merged_train.jsonl` | union of the above | 1,000 | CC BY-SA 4.0 |

Attribution: BIRD (Li et al., 2023, "Can LLM Already Serve as a Database Interface?");
Spider (Yu et al., 2018). Both training splits only — **no BIRD-dev or Spider-dev
contamination**, so evaluation on BIRD-dev remains clean.

## Record schema
One JSON object per line: `{"messages": [system, user, assistant]}`.
- **system** (constant): "You are an expert SQL generator. Return only executable SQL with no explanation."
- **user**: `Database: <db_id>` + blank line + `Schema:` + `Tables` block
  (`Name(\n    col,\n    col\n)` per table) + `Foreign Keys` block (`A.x -> B.y` lines)
  + `Question:\n<question>` + `Return only SQL.` — byte-identical template across both sources.
- **assistant**: bare SQL (no code fence, no trailing semicolon).

## Cleaning applied to the BIRD 500
- Verified structure (500/500 system|user|assistant), zero duplicate questions/records.
- Stripped trailing semicolons from 13 targets (487/500 already had none) for a
  consistent target distribution.
- SQL syntax-verified with sqlglot (sqlite): 500/500 parse. Three records that trip
  sqlglot's tokenizer do so only on backtick-quoted identifiers containing spaces
  (valid SQLite) — verified via masked re-parse and kept unchanged.

## Spider subset construction (why + how)
The BIRD 500 is join/aggregate heavy but thin on HAVING (7), nested subqueries (37),
set operations (1), GROUP BY (49). The Spider 500 was therefore selected by
structural bucket with explicit quotas (priority: set_op > nested > having >
multi_join > group_by > single_join > single_table_analytic), after paraphrase
dedupe on (db_id, normalized query), with a 12-example per-database cap for domain
diversity. Realized buckets: multi_join 100, nested 90, group_by 80, having 70,
single_join 60, set_op 50, single_table_analytic 50 — across 131 of Spider's 140
train databases. Reproduce exactly: `python scripts/build_spider_subset.py --seed 42`
(raw inputs vendored in `data/raw/`). Full manifest: `spider_selection_manifest.json`.

**Window functions**: Spider's grammar contains none and the BIRD 500 contains none;
the corpus therefore has **0 window-function targets**. This is a documented gap, not
an oversight — add BIRD-train window examples in a future revision if the AEGIS
error analysis shows windowed queries failing.

## Splits (`train.jsonl` 798 / `validation.jsonl` 101 / `test.jsonl` 101)
Stratified 80/10/10 on (source × structural bucket), seed 42, so rare constructs
appear in every split (any stratum with ≥3 examples contributes ≥1 val and ≥1 test
record). Per-record provenance for every split lives in the line-aligned
`*_manifest.jsonl` files; aggregate numbers in `split_stats.json`.

## Corpus statistics (merged, n=1,000)
- 100% sqlglot-parseable; 198 distinct databases (67 BIRD + 131 Spider).
- Features: join 660 · aggregate 499 · group_by 222 · order_by 195 · nested 177 ·
  having 81 · set_op 51 · window 0.
- Prompt length: p50 ≈ 1.0k chars, p90 ≈ 2.3k, max ≈ 12.8k (≈3.7k tokens) →
  `max_seq_len = 4096` covers every record; nothing is truncated.

## Known limitations & recommended future improvements
1. **No `Evidence` field.** The curated BIRD file omits BIRD's analyst evidence
   strings, while the AEGIS planner injects evidence at inference. The
   `LocalGenerator` adapter accepts an optional `evidence=` slot so inference
   still works; re-attaching evidence from the original BIRD `train.json`
   (question-keyed) is the single highest-value data revision.
2. **Schema serialization carries no column types or primary keys.** Kept
   deliberately for byte-consistency with the existing 500; types are known to
   help CAST/comparison behavior. Revising both halves together (types +
   PK markers) is a worthwhile controlled experiment.
3. **1,000 examples is deliberately small** (curated-SFT regime). If the multi-SLM
   study plateaus, scale with SynSQL-style synthetic data before touching the recipe.
