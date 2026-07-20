#!/usr/bin/env python3
"""Validate + normalize the curated BIRD-500 file into data/bird_train.jsonl.

Normalizations (why): trailing semicolons appear on only 13/500 targets, so we
strip them for a consistent target distribution; whitespace is trimmed. SQL is
syntax-checked with sqlglot (sqlite); records whose only 'failure' is sqlglot's
tokenizer choking on backtick-quoted identifiers containing spaces are verified
by a backtick-masking re-parse and kept (they are valid SQLite).
"""
import json, re, sys, sqlglot, collections

def parses(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql, read="sqlite"); return True
    except Exception:
        masked = re.sub(r"`[^`]*`", "x", sql)     # backtick identifiers -> x
        try:
            sqlglot.parse_one(masked, read="sqlite"); return True
        except Exception:
            return False

def main(inp="data/raw/bird_train_500_original.jsonl", out="data/bird_train.jsonl"):
    recs = [json.loads(l) for l in open(inp)]
    bad, fixed_semis = [], 0
    for i, r in enumerate(recs):
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"], i
        a = r["messages"][2]["content"].strip()
        if a.endswith(";"):
            a = a.rstrip(";").rstrip(); fixed_semis += 1
        r["messages"][2]["content"] = a
        if not parses(a):
            bad.append(i)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[clean_bird] {len(recs)} records -> {out} | stripped ';' on {fixed_semis} | unparseable: {bad}")
    if bad: sys.exit(f"unparseable records present: {bad}")

if __name__ == "__main__":
    main(*sys.argv[1:])
