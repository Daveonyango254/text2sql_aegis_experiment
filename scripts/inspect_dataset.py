#!/usr/bin/env python3
"""Print structural statistics for any chat-format jsonl (sanity + data-card numbers)."""
import sys, json, re, statistics, collections, sqlglot

def main(path):
    recs = [json.loads(l) for l in open(path)]
    a = [r["messages"][2]["content"] for r in recs]
    u = [r["messages"][1]["content"] for r in recs]
    feats = collections.Counter(); ok = 0
    for s in a:
        try: sqlglot.parse_one(re.sub(r"`[^`]*`", "x", s), read="sqlite"); ok += 1
        except Exception: pass
        S = f" {s.upper()} "
        for name, pat in [("join", r" JOIN "), ("group_by", r"\bGROUP BY\b"),
                          ("having", r"\bHAVING\b"), ("order_by", r"\bORDER BY\b"),
                          ("agg", r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\("),
                          ("nested", None), ("set_op", r"\b(UNION|INTERSECT|EXCEPT)\b"),
                          ("window", r"\bOVER\s*\(")]:
            if name == "nested":
                feats[name] += S.count("SELECT") >= 2
            elif re.search(pat, S):
                feats[name] += 1
    ul = [len(x) for x in u]
    print(f"{path}: n={len(recs)} parse_ok={ok} "
          f"user_chars(p50/p90/max)={int(statistics.median(ul))}/{sorted(ul)[int(.9*len(ul))]}/{max(ul)} "
          f"(~max_tokens≈{int(max(ul)/3.5)})")
    print("  features:", dict(feats))
    print("  dbs:", len({x.split(chr(10),1)[0] for x in u}))

if __name__ == "__main__":
    for p in sys.argv[1:]: main(p)
