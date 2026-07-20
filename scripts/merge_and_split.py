#!/usr/bin/env python3
"""Merge BIRD + Spider chat files and produce stratified train/validation/test splits.

Split methodology (why): with only 1,000 examples a random split can starve the
val/test sets of rare constructs (HAVING, set ops). We therefore stratify on
(source, structural bucket): every stratum contributes proportionally to the
80/10/10 split, with at least one val and one test example for any stratum of
size >= 3. Deterministic under --seed. A line-aligned manifest records the
(source, bucket, db) of every merged record for later error analysis.
"""
import argparse, json, random, re, collections

def classify(sql: str) -> str:
    s = f" {sql.upper()} "
    joins = s.count(" JOIN ")
    if re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", s): return "set_op"
    if s.count("SELECT") >= 2:                        return "nested"
    if re.search(r"\bHAVING\b", s):                   return "having"
    if joins >= 2:                                    return "multi_join"
    if re.search(r"\bGROUP BY\b", s):                 return "group_by"
    if joins == 1:                                    return "single_join"
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", s) or re.search(r"\bORDER BY\b", s):
        return "single_table_analytic"
    return "simple"

def db_of(rec):
    u = rec["messages"][1]["content"]
    return u.split("\n", 1)[0].replace("Database:", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird", default="data/bird_train.jsonl")
    ap.add_argument("--spider", default="data/spider_train.jsonl")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []
    for src, path in (("bird", a.bird), ("spider", a.spider)):
        for line in open(path):
            r = json.loads(line)
            rows.append({"rec": r, "source": src,
                         "bucket": classify(r["messages"][2]["content"]),
                         "db": db_of(r)})
    rng.shuffle(rows)

    strata = collections.defaultdict(list)
    for row in rows:
        strata[(row["source"], row["bucket"])].append(row)
    train, val, test = [], [], []
    for key in sorted(strata):
        grp = strata[key]; n = len(grp)
        n_test = max(1, round(n * a.test_frac)) if n >= 3 else 0
        n_val  = max(1, round(n * a.val_frac))  if n >= 3 else 0
        test += grp[:n_test]; val += grp[n_test:n_test + n_val]; train += grp[n_test + n_val:]
    for part in (train, val, test): rng.shuffle(part)

    def dump(name, part):
        with open(f"{a.outdir}/{name}.jsonl", "w") as f:
            for row in part: f.write(json.dumps(row["rec"]) + "\n")
        with open(f"{a.outdir}/{name}_manifest.jsonl", "w") as f:
            for row in part:
                f.write(json.dumps({"source": row["source"], "bucket": row["bucket"], "db": row["db"]}) + "\n")
        return len(part)

    merged = train + val + test
    n_m = dump("merged_train", merged); n_t = dump("train", train)
    n_v = dump("validation", val);      n_te = dump("test", test)
    stats = {
        "counts": {"merged": n_m, "train": n_t, "validation": n_v, "test": n_te},
        "by_source": dict(collections.Counter(r["source"] for r in merged)),
        "by_bucket": dict(collections.Counter(r["bucket"] for r in merged)),
        "train_by_bucket": dict(collections.Counter(r["bucket"] for r in train)),
        "val_by_bucket": dict(collections.Counter(r["bucket"] for r in val)),
        "test_by_bucket": dict(collections.Counter(r["bucket"] for r in test)),
        "distinct_dbs": len({r["db"] for r in merged}), "seed": a.seed,
    }
    json.dump(stats, open(f"{a.outdir}/split_stats.json", "w"), indent=1)
    print("[merge_and_split]", json.dumps(stats["counts"]))
    print("  by_source:", stats["by_source"])
    print("  by_bucket:", stats["by_bucket"])

if __name__ == "__main__":
    main()
