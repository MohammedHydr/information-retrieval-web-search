#!/usr/bin/env python3
"""
Convert a *forward* tf-idf index (one line per document) to an *inverted*
index (one line per term, posting list sorted DESC on score).

Input format  (whitespace-separated)
------------------------------------
docID   term1:score1   term2:score2   ...

Output format  (space / comma separators – exactly what the
experiment scripts from my previous message expect)
----------------------------------------------------
term    df    docID1:score1,docID2:score2,docID3:score3,...
"""
import re
import sys, os
from collections import defaultdict

FWD_FILE = "../output/tfidf_index.txt"  # <– your file
INV_FILE = "../output/inverted_index_tfidf.txt"

NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

def main():
    if not os.path.exists(FWD_FILE):
        sys.exit(f"❌ {FWD_FILE} not found")

    postings = defaultdict(list)   # term -> list[(docID, score)]

    with open(FWD_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():               # skip blanks
                continue
            parts = line.split()
            docID = parts[0]                   # **keep as string**
            for token in parts[1:]:
                term, rest = token.split(":", 1)   # first colon
                m = NUM_RE.match(rest)            # numeric tf-idf
                if not m:
                    continue                      # malformed
                score = float(m.group())
                postings[term].append((docID, score))

    # sort posting lists by descending score
    for term in postings:
        postings[term].sort(key=lambda x: x[1], reverse=True)

    os.makedirs(os.path.dirname(INV_FILE), exist_ok=True)
    with open(INV_FILE, "w", encoding="utf-8") as out:
        for term, plist in sorted(postings.items()):
            df   = len(plist)
            line = f"{term} {df} " + ",".join(f"{d}:{s}" for d, s in plist)
            out.write(line + "\n")

    print(f"✓ inverted index written to {INV_FILE}  "
          f"({len(postings):,} terms)")

if __name__ == "__main__":
    main()