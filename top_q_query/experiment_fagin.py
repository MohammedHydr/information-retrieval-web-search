#!/usr/bin/env python3
"""
experiment_fagin.py
-------------------
Runs Fagin’s algorithm for every query in a plain-text benchmark file
(one query per line) and every k in {5,10,…,50}.  Stores the statistics
in results/fagin_stats.csv.

MAKE SURE the two paths below point to real files:
  • INDEX_FILE     – inverted tf-idf index (term-centric)
  • BENCHMARK_FILE – text file, one query per line
"""

import os, sys, time, csv
from typing import Dict, List, Tuple

import pandas as pd

# ---------- PATHS – edit if your files live elsewhere -----------------
INDEX_FILE = os.path.abspath("../output/inverted_index_tfidf.txt")
BENCHMARK_FILE = os.path.abspath("../output/query_benchmark.txt")
OUT_CSV = os.path.abspath("results/fagin_stats.csv")
# ---------------------------------------------------------------------

K_VALUES = list(range(5, 55, 5))  # 5,10, … 50


# ---------------------------------------------------------------------
# 1. helper: load inverted index
# ---------------------------------------------------------------------
def load_index(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Expected line format (term-centric, already sorted DESC on score):
        real  625  00a9c:3.21,17x7b:2.95,99ff2:2.70,...
    """
    if not os.path.exists(path):
        sys.exit(f"❌  INDEX_FILE not found: {path}")

    postings: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            term, _, rest = line.partition(" ")
            _, _, plist = rest.partition(" ")  # skip df
            pairs = []
            for p in plist.strip().split(","):
                if not p:
                    continue
                try:
                    did, sc = p.split(":")
                    pairs.append((did, float(sc)))
                except ValueError:
                    continue  # malformed entry
            postings[term] = pairs
    return postings


# ---------------------------------------------------------------------
# 2. helper: load queries (plain text, one per line)
# ---------------------------------------------------------------------
def load_queries(path: str) -> List[str]:
    if not os.path.exists(path):
        sys.exit(f"❌  BENCHMARK_FILE not found: {path}")
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


# ---------------------------------------------------------------------
# 3. import the algorithm
# ---------------------------------------------------------------------
try:
    from fagin import fagin_topk
except ImportError:
    sys.exit("❌  Could not import fagin_topk – is fagin.py in the "
             "same folder or on PYTHONPATH?")


# ---------------------------------------------------------------------
# 4. main experiment
# ---------------------------------------------------------------------
def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    postings = load_index(INDEX_FILE)
    queries = load_queries(BENCHMARK_FILE)

    print(f"→ Loaded {len(postings):,} terms from index")
    print(f"→ Loaded {len(queries):,} queries from benchmark\n")

    rows = []
    for qid, query in enumerate(queries, 1):
        print(f"Processing Q{qid}: {query!r}")
        terms = query.lower().split()  # stem here if your index is stemmed
        for k in K_VALUES:
            _, stats = fagin_topk(terms, k, postings)
            rows.append(dict(qid=qid, query=query, k=k, **stats))

    if not rows:
        sys.exit("❌  No statistics collected – check index / queries")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n✓  {len(rows)} rows written to {OUT_CSV}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Total experiment time:", round(time.time() - t0, 1), "s")
