#!/usr/bin/env python3
"""Build a diversity-stratified ~500-example Spider subset in the exact BIRD chat format.

Why this script exists
----------------------
The 500-example BIRD set is join/aggregate heavy but thin on HAVING (7), nested
subqueries (37), set operations (1), and GROUP BY (49), and contains no window
functions. A naive random Spider sample would replicate Spider's own skew
(~55% easy/medium single-table). This builder therefore classifies every Spider
train query by structural features and fills explicit per-bucket quotas so the
merged 1,000-example corpus covers the SQL constructs the AEGIS local path must
learn. Spider's grammar contains no window functions; that gap is documented in
the data card rather than papered over.

Selection rules
---------------
1. Paraphrase dedupe: Spider often pairs 2 questions with an identical query;
   keep at most one per (db_id, whitespace-normalized query).
2. Bucket priority (first match wins): set_op > nested > having > multi_join
   > group_by > single_join > single_table_analytic > simple.
3. Per-database cap (default 12) for domain diversity across the 140 train DBs.
4. Deterministic under --seed.

Output records are byte-compatible with the BIRD file: {"messages":[system,user,assistant]}
with the identical schema serialization (Tables + Foreign Keys sections, no types)
and the identical instruction wrapper.
"""
import argparse, json, random, re, sys, collections

SYSTEM_PROMPT = ("You are an expert SQL generator. Return only executable SQL "
                 "with no explanation.")

BUCKET_QUOTAS = [  # (bucket, target_n) — priority order is classification order below
    ("set_op", 50),
    ("nested", 90),
    ("having", 70),
    ("multi_join", 100),
    ("group_by", 80),
    ("single_join", 60),
    ("single_table_analytic", 50),
    ("simple", 0),  # filler only if quotas above cannot be met
]

def collapse_ws_outside_quotes(sql: str) -> str:
    """Collapse runs of whitespace, but never inside single-quoted literals."""
    parts = sql.split("'")
    for i in range(0, len(parts), 2):          # even indexes are outside quotes
        parts[i] = re.sub(r"\s+", " ", parts[i])
    return "'".join(parts).strip().rstrip(";").strip()

def classify(sql: str) -> str:
    s = f" {sql.upper()} "
    joins = s.count(" JOIN ")
    if re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", s):            return "set_op"
    if s.count("SELECT") >= 2:                                    return "nested"
    if re.search(r"\bHAVING\b", s):                               return "having"
    if joins >= 2:                                                return "multi_join"
    if re.search(r"\bGROUP BY\b", s):                             return "group_by"
    if joins == 1:                                                return "single_join"
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", s) or re.search(r"\bORDER BY\b", s):
        return "single_table_analytic"
    return "simple"

def render_schema(db: dict) -> str:
    """Serialize a tables.json entry exactly like the BIRD records (names only + FK arrows)."""
    tnames = db["table_names_original"]
    cols_by_table = collections.defaultdict(list)
    for ti, cname in db["column_names_original"]:
        if ti >= 0:
            cols_by_table[ti].append(cname)
    blocks = []
    for ti, t in enumerate(tnames):
        cols = ",\n    ".join(cols_by_table[ti])
        blocks.append(f"{t}(\n    {cols}\n)")
    fk_lines = []
    for src, dst in db["foreign_keys"]:
        sti, sc = db["column_names_original"][src]
        dti, dc = db["column_names_original"][dst]
        fk_lines.append(f"{tnames[sti]}.{sc} -> {tnames[dti]}.{dc}")
    schema = "Tables\n\n" + "\n\n".join(blocks)
    if fk_lines:
        schema += "\n\nForeign Keys\n\n" + "\n".join(fk_lines)
    return schema

def to_record(db_id: str, schema_txt: str, question: str, sql: str) -> dict:
    user = (f"Database: {db_id}\n\nSchema:\n\n{schema_txt}\n\n"
            f"Question:\n{question.strip()}\n\nReturn only SQL.")
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": sql},
    ]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_json", default="data/raw/spider_train_slim.json")
    ap.add_argument("--tables_json", default="data/raw/spider_tables.json")
    ap.add_argument("--out", default="data/spider_train.jsonl")
    ap.add_argument("--manifest", default="data/spider_selection_manifest.json")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per_db_cap", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    data = json.load(open(a.spider_json))
    tables = {d["db_id"]: d for d in json.load(open(a.tables_json))}
    schema_cache = {db: render_schema(t) for db, t in tables.items()}

    # 1) normalize + paraphrase dedupe
    seen, pool = set(), []
    for ex in data:
        sql = collapse_ws_outside_quotes(ex["query"])
        key = (ex["db_id"], sql.upper())
        if key in seen:
            continue
        seen.add(key)
        pool.append({"db_id": ex["db_id"], "question": ex["question"],
                     "sql": sql, "bucket": classify(sql)})
    rng.shuffle(pool)

    windows = sum(bool(re.search(r"\bOVER\s*\(", p["sql"].upper())) for p in pool)
    assert windows == 0, "unexpected window functions in Spider train"

    # 2) quota fill with per-db cap
    by_bucket = collections.defaultdict(list)
    for p in pool:
        by_bucket[p["bucket"]].append(p)
    db_count, picked = collections.Counter(), []
    def take(bucket, k):
        got = 0
        for p in by_bucket[bucket]:
            if got >= k: break
            if db_count[p["db_id"]] >= a.per_db_cap: continue
            picked.append(p); db_count[p["db_id"]] += 1; got += 1
        return got
    filled = {b: take(b, q) for b, q in BUCKET_QUOTAS}
    # top up to n with remaining diverse examples (relaxing nothing but quotas)
    if len(picked) < a.n:
        chosen = {id(p) for p in picked}
        for p in pool:
            if len(picked) >= a.n: break
            if id(p) in chosen or db_count[p["db_id"]] >= a.per_db_cap: continue
            picked.append(p); db_count[p["db_id"]] += 1
            filled[p["bucket"]] = filled.get(p["bucket"], 0) + 1
    picked = picked[: a.n]
    rng.shuffle(picked)

    with open(a.out, "w") as f:
        for p in picked:
            f.write(json.dumps(to_record(p["db_id"], schema_cache[p["db_id"]],
                                         p["question"], p["sql"])) + "\n")
    manifest = {
        "n": len(picked), "seed": a.seed, "per_db_cap": a.per_db_cap,
        "bucket_counts": dict(collections.Counter(p["bucket"] for p in picked)),
        "distinct_dbs": len({p["db_id"] for p in picked}),
        "records": [{"i": i, "db_id": p["db_id"], "bucket": p["bucket"],
                     "question": p["question"]} for i, p in enumerate(picked)],
    }
    json.dump(manifest, open(a.manifest, "w"), indent=1)
    print(f"[build_spider_subset] wrote {len(picked)} records -> {a.out}")
    print(f"  buckets: {manifest['bucket_counts']}")
    print(f"  distinct DBs: {manifest['distinct_dbs']} (cap {a.per_db_cap}/db)")

if __name__ == "__main__":
    sys.exit(main())
