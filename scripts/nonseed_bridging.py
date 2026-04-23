#!/usr/bin/env python3
"""
nonseed_bridging.py — Non-seed CPC bridging centrality diagnostic (Table S10).

Computes CPC bridging centrality on the 443-family analytically active subset,
flags each code as seed or non-seed (by checking against itc_codebook.json),
and reports:
  Panel A: Full top-10 with seed-anchor flag
  Panel B: Non-seed-only top-10

Reference values (from manuscript §4.4 and Supplementary Table S10):
  Non-seed top-4 in overall top-10: E02D (#7), H02J (#8), Y02E (#9), E21B (#10)
  Non-seed #5–#10: Y02P, E21C, Y02A, H02S, B32B, F04B

Usage:
    python scripts/nonseed_bridging.py
"""

import csv
import json
import os
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
JSON_PATH = os.path.join(DATA_DIR, "phase2_453_families.json")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")
CODEBOOK_PATH = os.path.join(DATA_DIR, "itc_codebook.json")


def load_excluded_ids():
    """Load lens_ids excluded from the 443-family analytical layer."""
    excluded = set()
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("analysis_set_443") == "false":
                excluded.add(row["lens_id"])
    return excluded


def load_seed_prefixes():
    """Extract 4-char CPC seed prefixes from itc_codebook.json."""
    with open(CODEBOOK_PATH, encoding="utf-8") as f:
        codebook = json.load(f)
    prefixes = set()
    for domain in codebook.values():
        for anchor in domain.get("cpc_anchors", []):
            prefixes.add(anchor[:4])
    return prefixes


def brandes_betweenness(binary, nn):
    """Brandes algorithm for unweighted betweenness centrality."""
    bet = np.zeros(nn)
    for s in range(nn):
        stack = []
        pred = [[] for _ in range(nn)]
        sigma = np.zeros(nn)
        sigma[s] = 1.0
        dist = np.full(nn, -1)
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            v = queue[qi]; qi += 1
            stack.append(v)
            for w in np.nonzero(binary[v])[0]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(nn)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]
    bet /= ((nn - 1) * (nn - 2))
    return bet


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        families_all = json.load(f)

    excluded_ids = load_excluded_ids()
    seed_prefixes = load_seed_prefixes()

    families_active = [f for f in families_all if f["lens_id"] not in excluded_ids]
    print(f"Analytically active families: {len(families_active)}")
    print(f"Seed CPC prefixes ({len(seed_prefixes)}): {sorted(seed_prefixes)}")
    print()

    # Build CPC co-occurrence graph (4-char prefix level)
    all_cpc = Counter()
    family_cpc_sets = []
    for fam in families_active:
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            continue
        codes = sorted(set(
            c.strip()[:4] for c in cpc_str.split(";;") if len(c.strip()) >= 4
        ))
        family_cpc_sets.append(codes)
        for code in codes:
            all_cpc[code] += 1

    code_list = sorted(all_cpc.keys())
    cidx = {c: i for i, c in enumerate(code_list)}
    nn = len(code_list)
    print(f"Unique CPC groups (4-char): {nn}")

    adj = np.zeros((nn, nn), dtype=int)
    for codes in family_cpc_sets:
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                ci, cj = cidx[codes[i]], cidx[codes[j]]
                adj[ci, cj] += 1
                adj[cj, ci] += 1

    binary = (adj >= 1).astype(int)
    np.fill_diagonal(binary, 0)
    degree = binary.sum(axis=1) / (nn - 1)

    bet = brandes_betweenness(binary, nn)

    ranked = sorted(range(nn), key=lambda i: (-degree[i], -bet[i], code_list[i]))

    # ── Panel A: Full top-10 with seed flag ──
    print()
    print("=" * 70)
    print("Panel A: CPC bridging centrality top-10 with seed-anchor flag")
    print("         (N = 443 active families, Brandes betweenness)")
    print("=" * 70)
    print()
    print(f"{'Rank':>4}  {'CPC':>5}  {'Degree':>7}  {'Between.':>8}  {'Seed?':>5}")
    print("-" * 40)
    for rank, i in enumerate(ranked[:10]):
        code = code_list[i]
        d = round(degree[i], 3)
        b = round(bet[i], 3)
        is_seed = "Y" if code in seed_prefixes else "N"
        print(f"  {rank+1:2d}   {code:>5}   {d:>7.3f}   {b:>8.3f}   {is_seed:>5}")

    seed_count = sum(1 for i in ranked[:10] if code_list[i] in seed_prefixes)
    nonseed_count = 10 - seed_count
    print(f"\n  Seed codes in top-10: {seed_count}/10")
    print(f"  Non-seed codes in top-10: {nonseed_count}/10")

    # ── Panel B: Non-seed only top-10 ──
    print()
    print("=" * 70)
    print("Panel B: Non-seed CPC bridging centrality top-10")
    print("         (codes not used as retrieval seed anchors)")
    print("=" * 70)
    print()
    print(f"{'Rank':>4}  {'CPC':>5}  {'Degree':>7}  {'Between.':>8}  {'Overall':>7}")
    print("-" * 45)

    non_seed_ranked = [i for i in ranked if code_list[i] not in seed_prefixes]
    for rank, i in enumerate(non_seed_ranked[:10]):
        code = code_list[i]
        d = round(degree[i], 3)
        b = round(bet[i], 3)
        overall_rank = ranked.index(i) + 1
        print(f"  {rank+1:2d}   {code:>5}   {d:>7.3f}   {b:>8.3f}   #{overall_rank:>5}")

    print()
    print("Note: 'Overall' shows the code's rank in the full (seed + non-seed)")
    print("bridging centrality ranking. Non-seed codes at positions #7–#10 in")
    print("the overall ranking provide the most conservative evidence of")
    print("cross-domain connectivity, as their bridging role is not fully")
    print("independent of the CPC-anchored retrieval strategy but is not")
    print("directly seeded by it. Because the rescue pass also relied on CPC")
    print("co-classification bridging against already-admitted records, the")
    print("centrality ranks remain route-conditioned; however, excluding the")
    print("nine rescue-route families preserves the leading bridging trio and")
    print("the overall rank structure.")


if __name__ == "__main__":
    main()
