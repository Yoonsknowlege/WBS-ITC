#!/usr/bin/env python3
"""
nonseed_bridging.py — CPC bridging centrality diagnostic (Online Resource 1, Table S10).

Computes CPC bridging centrality on the 443-family analytically active subset
and reports two diagnostic flags separately, in line with Online Resource 1
Table S10:

  - Stage-1 retrieval seed?  (the 9 CPC prefixes used in the primary
    Lens.org query: B64G, C04B, B33Y, E04H, B22F, B28B, G01N, B25J, E04B;
    see lens_query.txt and Online Resource 1 Appendix S2)
  - ITC codebook anchor?     (CPC prefixes that operationalise one or
    more of the 15 ITC domains in the frozen codebook;
    see itc_codebook.json)

These two flags are not equivalent: most Stage-1 retrieval seeds also
serve as codebook anchors, but the codebook contains additional anchors
that were not Stage-1 retrieval seeds. The dual flag system makes the
CPC-conditioning of the analytical layer auditable.

Panels:
  Panel A: Full top-10 with both flags
  Panel B: Non-codebook-anchor top-10

Reference values (manuscript Section 4.3 and Online Resource 1 Table S10):
  Non-codebook-anchor top-4 in overall top-10:
    E02D (#7), H02J (#8), Y02E (#9), E21B (#10)

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

STAGE1_RETRIEVAL_SEEDS = {
    "B64G", "C04B", "B33Y", "E04H", "B22F",
    "B28B", "G01N", "B25J", "E04B",
}


def load_excluded_ids():
    excluded = set()
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("analysis_set_443") == "false":
                excluded.add(row["lens_id"])
    return excluded


def load_codebook_anchors():
    with open(CODEBOOK_PATH, encoding="utf-8") as f:
        codebook = json.load(f)
    prefixes = set()
    for domain in codebook.values():
        for anchor in domain.get("cpc_anchors", []):
            prefixes.add(anchor[:4])
    return prefixes


def brandes_betweenness(binary, nn):
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
            v = queue[qi]
            qi += 1
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
    codebook_anchors = load_codebook_anchors()

    families_active = [f for f in families_all if f["lens_id"] not in excluded_ids]
    print("Analytically active families:", len(families_active))
    print("Stage-1 retrieval seeds (%d):" % len(STAGE1_RETRIEVAL_SEEDS), sorted(STAGE1_RETRIEVAL_SEEDS))
    print("ITC codebook anchors (%d):" % len(codebook_anchors), sorted(codebook_anchors))
    print()

    all_cpc = Counter()
    family_cpc_sets = []
    for fam in families_active:
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            continue
        codes = sorted(set(c.strip()[:4] for c in cpc_str.split(";;") if len(c.strip()) >= 4))
        family_cpc_sets.append(codes)
        for code in codes:
            all_cpc[code] += 1

    code_list = sorted(all_cpc.keys())
    cidx = {c: i for i, c in enumerate(code_list)}
    nn = len(code_list)
    print("Unique CPC groups (4-char):", nn)

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

    print()
    print("=" * 80)
    print("Panel A: CPC bridging centrality top-10 with Stage-1 retrieval seed flag")
    print("         and ITC codebook anchor flag")
    print("         (N = 443 active families, Brandes betweenness)")
    print("=" * 80)
    print()
    header_a = "%4s  %5s  %7s  %8s  %14s  %17s" % ("Rank", "CPC", "Degree", "Between.", "Stage-1 seed?", "Codebook anchor?")
    print(header_a)
    print("-" * 70)
    for rank, i in enumerate(ranked[:10]):
        code = code_list[i]
        d = round(degree[i], 3)
        b = round(bet[i], 3)
        is_stage1 = "Y" if code in STAGE1_RETRIEVAL_SEEDS else "N"
        is_anchor = "Y" if code in codebook_anchors else "N"
        print("  %2d   %5s   %7.3f   %8.3f   %14s  %17s" % (rank + 1, code, d, b, is_stage1, is_anchor))

    stage1_count = sum(1 for i in ranked[:10] if code_list[i] in STAGE1_RETRIEVAL_SEEDS)
    anchor_count = sum(1 for i in ranked[:10] if code_list[i] in codebook_anchors)
    print()
    print("  Stage-1 retrieval seeds in top-10:", stage1_count, "/10")
    print("  ITC codebook anchors in top-10:   ", anchor_count, "/10")

    print()
    print("=" * 70)
    print("Panel B: Non-codebook-anchor CPC bridging centrality top-10")
    print("         (codes not operationalising any of the 15 ITC domains)")
    print("=" * 70)
    print()
    header_b = "%4s  %5s  %7s  %8s  %14s  %7s" % (
        "Rank", "CPC", "Degree", "Between.", "Stage-1 seed?", "Overall")
    print(header_b)
    print("-" * 60)

    non_anchor_ranked = [i for i in ranked if code_list[i] not in codebook_anchors]
    for rank, i in enumerate(non_anchor_ranked[:10]):
        code = code_list[i]
        d = round(degree[i], 3)
        b = round(bet[i], 3)
        is_stage1 = "Y" if code in STAGE1_RETRIEVAL_SEEDS else "N"
        overall_rank = ranked.index(i) + 1
        print("  %2d   %5s   %7.3f   %8.3f   %14s   #%5d" % (
            rank + 1, code, d, b, is_stage1, overall_rank))

    print()
    print("Note: Stage-1 retrieval seeds and ITC codebook anchors are reported")
    print("separately because they reflect different methodological decisions:")
    print("Stage-1 seeds were fixed before retrieval, while codebook anchors")
    print("were added during pilot codebook construction. Non-codebook-anchor")
    print("codes appearing in the overall top-10 provide the most conservative")
    print("evidence of cross-domain connectivity, because their bridging role")
    print("is not directly seeded by codebook operationalisation. Because the")
    print("rescue pass also relied on CPC co-classification bridging against")
    print("already-admitted records, the centrality ranks remain route-")
    print("conditioned; however, excluding the nine rescue-route families")
    print("preserves the leading bridging trio and the overall rank structure.")


if __name__ == "__main__":
    main()
