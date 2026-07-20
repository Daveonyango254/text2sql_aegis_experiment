#!/usr/bin/env python3
"""Harness-free quick metrics for a predictions file against a gold chat jsonl.

Reports (why these two): (1) sqlite parse rate via sqlglot — a floor sanity
metric catching prompt-format regressions instantly; (2) whitespace/case
normalized exact match — a conservative proxy that correlates with EX on this
corpus but never overstates it. Full BIRD Execution Accuracy requires the BIRD
databases + official evaluator; see DETAILED_GUIDE.md 'Full BIRD evaluation'.

predictions file: jsonl with {"sql": "..."} per line, aligned with the gold file.
"""
import sys, json, re, sqlglot

def norm(s):
    s = re.sub(r"\s+", " ", s.strip().rstrip(";")).strip()
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", s)
    return s.upper()

def main(gold_path, pred_path):
    gold = [json.loads(l)["messages"][2]["content"] for l in open(gold_path)]
    pred = [json.loads(l)["sql"] for l in open(pred_path)]
    assert len(gold) == len(pred), (len(gold), len(pred))
    ok = em = 0
    for g, p in zip(gold, pred):
        try: sqlglot.parse_one(re.sub(r"`[^`]*`", "x", p), read="sqlite"); ok += 1
        except Exception: pass
        em += norm(g) == norm(p)
    n = len(gold)
    print(json.dumps({"n": n, "parse_rate": round(ok/n, 4), "normalized_em": round(em/n, 4)}))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
