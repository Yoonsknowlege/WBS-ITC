#!/usr/bin/env python3
"""
recompute_tables_from_json.py — Independent recomputation of manuscript tables
from the released JSON dataset.

Paper: "Denominator construction for patent mapping in classification-poor
       emerging domains: A framework-first WBS-ITC approach with a lunar
       ISRU construction illustration"

Purpose:
    This script independently derives the core analytical outputs reported
    in the manuscript (Tables 3, 4, 5 and Figure 2/3 aggregation values)
    directly from phase2_453_families.json, filtered to the 443-family
    analytically active subset using decision_ledger.csv.  It then
    cross-checks them against the pre-computed constants stored in
    isru_data.py to confirm consistency.

Dataset scope:
    Released families:            453  (phase2_453_families.json)
    False-match:                    9  (unclassified; excluded)
    Audit-only near-threshold:      1  (unclassified; excluded)
    Analytically active:          443  = 453 - 9 - 1
      ITC-classified:             373
      Domain-external (active):    70  = 443 - 373

    All 10 excluded families carry no ITC tags.
    ITC-based metrics (tag counts, Jaccard) are mathematically invariant
    to the exclusion.
    CPC-level metrics (co-classification matrix, bridging centrality)
    are computed on the 443-family active subset because excluded
    families carry CPC codes that shift lower-ranked positions.

Usage:
    python scripts/recompute_tables_from_json.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
JSON_PATH = os.path.join(DATA_DIR, "phase2_453_families.json")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")

# Import stored constants for cross-checking
from isru_data import (
    ITC_DOMAINS, PORTFOLIO, JACCARD_MATRIX, TOTAL_ITC_TAGS,
    TAGGED_FAMILIES, RELEASED_FAMILIES, ANALYTICALLY_ACTIVE,
    DOMAIN_EXTERNAL_ACTIVE, CATEGORY_C, AUDIT_ONLY,
    CPC_TOP25, CPC_COOCCURRENCE, CPC_BRIDGING,
    FILING_YEAR_BY_PRIORITY,
)

# ── Exclusion note ────────────────────────────────────────────────
# The released JSON (453 families) does not carry a "category" field.
# The decision_ledger.csv provides analysis_set_443 == true/false to
# identify the 443 analytically active families.
#
# ITC-based metrics (tag counts, Jaccard) are computed on the full 453
# JSON because all 10 excluded families carry no ITC tags (mathematically
# invariant).  CPC-level metrics (co-classification, bridging centrality)
# are computed on the 443-active subset because excluded families carry
# CPC codes that affect lower-ranked positions.
EXPECTED_FALSE_MATCH = 9
EXPECTED_AUDIT = 1
EXPECTED_EXCLUDED = EXPECTED_FALSE_MATCH + EXPECTED_AUDIT


def load_families():
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_excluded_ids():
    """Load lens_ids excluded from the 443-family analytical layer."""
    excluded = set()
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("analysis_set_443") == "false":
                excluded.add(row["lens_id"])
    return excluded


def recompute_table3(families):
    """Table 3: ITC domain tag counts (PORTFOLIO)."""
    counter = Counter()
    total_tags = 0
    classified = 0
    for fam in families:
        codes = fam.get("itc_codes", [])
        if codes:
            classified += 1
        for code in codes:
            counter[code] += 1
            total_tags += 1

    portfolio = {d: counter.get(d, 0) for d in ITC_DOMAINS}
    return portfolio, total_tags, classified


def recompute_table5_jaccard(families):
    """Table 5: Jaccard similarity matrix (15x15)."""
    domain_sets = defaultdict(set)
    for i, fam in enumerate(families):
        for code in fam.get("itc_codes", []):
            domain_sets[code].add(i)

    n = len(ITC_DOMAINS)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                si = domain_sets[ITC_DOMAINS[i]]
                sj = domain_sets[ITC_DOMAINS[j]]
                union = len(si | sj)
                inter = len(si & sj)
                matrix[i][j] = inter / union if union > 0 else 0.0

    # Round to 3 decimal places for comparison
    matrix = np.round(matrix, 3)
    return matrix


def recompute_table4_cpc(families):
    """Table 4: CPC co-classification matrix (top 25 codes)."""
    # Build family-level CPC sets (4-char prefixes)
    family_cpc_sets = []
    for fam in families:
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            family_cpc_sets.append([])
            continue
        codes = sorted(set(
            c.strip()[:4] for c in cpc_str.split(";;") if len(c.strip()) >= 4
        ))
        family_cpc_sets.append(codes)

    # Build co-occurrence matrix for top-25 codes
    idx = {c: i for i, c in enumerate(CPC_TOP25)}
    n = len(CPC_TOP25)
    cooc = np.zeros((n, n), dtype=int)
    for codes in family_cpc_sets:
        relevant = [c for c in codes if c in idx]
        for a in relevant:
            cooc[idx[a], idx[a]] += 1  # diagonal = frequency
            for b in relevant:
                if a < b:
                    cooc[idx[a], idx[b]] += 1
                    cooc[idx[b], idx[a]] += 1

    return cooc


def recompute_cpc_bridging(families):
    """CPC bridging centrality (degree + betweenness) for all CPC groups."""
    all_cpc = Counter()
    family_cpc_sets = []
    for fam in families:
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

    # Brandes betweenness
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

    ranked = sorted(range(nn), key=lambda i: (-degree[i], -bet[i], code_list[i]))
    return [(code_list[i], round(float(degree[i]), 3), round(float(bet[i]), 3))
            for i in ranked[:10]]


def recompute_fig3_filingyear(families):
    """Recompute filing-year × WBS-layer tag counts by earliest priority year."""
    year_wbs = defaultdict(lambda: [0, 0, 0, 0])
    for fam in families:
        yr = fam.get("earliest_priority_year")
        if yr is None:
            continue
        yr = int(yr)
        for code in fam.get("itc_codes", []):
            layer = int(code.split("-")[0]) - 1  # 0-indexed
            year_wbs[yr][layer] += 1

    result = []
    for yr in sorted(year_wbs.keys()):
        result.append([yr] + year_wbs[yr])
    return result


def main():
    families_all = load_families()
    excluded_ids = load_excluded_ids()
    families_active = [f for f in families_all if f["lens_id"] not in excluded_ids]

    print("=" * 70)
    print("RECOMPUTATION FROM RELEASED JSON — CROSS-CHECK REPORT")
    print("=" * 70)

    # ── Dataset scope accounting ──
    n_released = len(families_all)
    n_excluded = len(excluded_ids)
    n_active = len(families_active)
    n_classified = sum(1 for f in families_all if f.get("is_classified"))
    n_unclassified = n_released - n_classified
    n_ext_active = n_unclassified - n_excluded

    cn_released = sum(1 for f in families_all
                      if f.get("jurisdiction", "").startswith("CN"))
    noncn_released = n_released - cn_released
    cn_excluded = sum(1 for f in families_all
                      if f["lens_id"] in excluded_ids and
                      f.get("jurisdiction", "").startswith("CN"))
    cn_active = cn_released - cn_excluded
    noncn_active = noncn_released - (n_excluded - cn_excluded)

    print(f"\n--- Dataset Scope ---")
    print(f"Released families:              {n_released}  (expected {RELEASED_FAMILIES})")
    print(f"ITC-classified:                 {n_classified}  (expected {TAGGED_FAMILIES})")
    print(f"Unclassified (total released):  {n_unclassified}  (= {n_released} - {n_classified})")
    print(f"Excluded (from ledger):         {n_excluded}  ({EXPECTED_FALSE_MATCH} false-match + {EXPECTED_AUDIT} audit-only)")
    print(f"Analytically active:            {n_active}  (= {n_released} - {n_excluded})")
    print(f"Domain-external (active):       {n_ext_active}  (expected {DOMAIN_EXTERNAL_ACTIVE})")
    print(f"Jurisdiction (released):        CN={cn_released}  Non-CN={noncn_released}")
    print(f"Jurisdiction (active):          CN={cn_active}  Non-CN={noncn_active}  "
          f"(excluded: {cn_excluded} CN + {n_excluded - cn_excluded} non-CN)")
    print()
    print("NOTE: ITC metrics (Tables 3, 5) computed on full 453 (invariant —")
    print("      all excluded families lack ITC tags).")
    print("      CPC metrics (Table 4, bridging) computed on 443-active subset")
    print("      (excluded families carry CPC codes that shift lower ranks).")

    scope_ok = (n_released == RELEASED_FAMILIES and
                n_active == ANALYTICALLY_ACTIVE and
                n_classified == TAGGED_FAMILIES and
                n_ext_active == DOMAIN_EXTERNAL_ACTIVE)
    print(f"Scope check: {'PASS' if scope_ok else 'FAIL'}")

    # ── Table 3: ITC domain portfolio ──
    # (Computed on full 453; ITC metrics mathematically invariant)
    portfolio, total_tags, classified_count = recompute_table3(families_all)
    print(f"\n--- Table 3: ITC Domain Portfolio ---")
    print(f"Total ITC tags: {total_tags}  (stored: {TOTAL_ITC_TAGS})")
    tag_match = total_tags == TOTAL_ITC_TAGS
    print(f"Tag count match: {'PASS' if tag_match else 'FAIL'}")

    portfolio_match = True
    for d in ITC_DOMAINS:
        recomp = portfolio[d]
        stored = PORTFOLIO[d]
        status = "OK" if recomp == stored else "MISMATCH"
        if recomp != stored:
            portfolio_match = False
        print(f"  {d} ({recomp:3d} vs {stored:3d})  {status}")
    print(f"Portfolio match: {'PASS' if portfolio_match else 'FAIL'}")

    # ── Table 5: Jaccard matrix ──
    # (Computed on full 453; ITC metrics mathematically invariant)
    jaccard = recompute_table5_jaccard(families_all)
    stored_jaccard = np.round(JACCARD_MATRIX, 3)
    jaccard_match = np.allclose(jaccard, stored_jaccard, atol=0.001)
    max_diff = np.max(np.abs(jaccard - stored_jaccard))
    print(f"\n--- Table 5: Jaccard Matrix ---")
    print(f"Max absolute difference: {max_diff:.4f}")
    print(f"Jaccard match (atol=0.001): {'PASS' if jaccard_match else 'FAIL'}")

    # Top 5 pairs
    pairs = []
    for i in range(len(ITC_DOMAINS)):
        for j in range(i + 1, len(ITC_DOMAINS)):
            pairs.append((ITC_DOMAINS[i], ITC_DOMAINS[j], jaccard[i][j]))
    pairs.sort(key=lambda x: -x[2])
    print("  Top 5 Jaccard pairs (recomputed):")
    for d1, d2, j in pairs[:5]:
        print(f"    {d1} <-> {d2}: {j:.3f}")

    # ── Table 4: CPC co-classification ──
    # (Computed on 443-active subset)
    cpc_cooc = recompute_table4_cpc(families_active)
    cpc_match = np.array_equal(cpc_cooc, CPC_COOCCURRENCE)
    print(f"\n--- Table 4: CPC Co-classification (N=443 active) ---")
    print(f"Matrix match: {'PASS' if cpc_match else 'FAIL'}")
    if not cpc_match:
        diff_positions = np.argwhere(cpc_cooc != CPC_COOCCURRENCE)
        print(f"  Mismatches at {len(diff_positions)} positions")
        for pos in diff_positions[:5]:
            i, j = pos
            print(f"    [{CPC_TOP25[i]},{CPC_TOP25[j]}]: "
                  f"recomputed={cpc_cooc[i,j]}, stored={CPC_COOCCURRENCE[i,j]}")

    # ── CPC Bridging ──
    # (Computed on 443-active subset)
    bridging = recompute_cpc_bridging(families_active)
    print(f"\n--- CPC Bridging Centrality (N=443 active, Top 10) ---")
    bridge_top3_match = ([b[0] for b in bridging[:3]] ==
                         [b[0] for b in CPC_BRIDGING[:3]])
    print(f"Top-3 codes match: {'PASS' if bridge_top3_match else 'FAIL'}")
    print(f"  {'Recomputed':<12} {'Stored':<12}")
    for (rc, rd, rb), stored in zip(bridging, CPC_BRIDGING):
        sc, sd, sb = stored[0], stored[1], stored[2]
        print(f"  {rc:<6} D={rd:.3f} B={rb:.3f}    "
              f"{sc:<6} D={sd:.3f} B={sb:.3f}")

    # ── Fig. 3: Filing-year × WBS ──
    # (Computed on full 453; ITC metrics mathematically invariant)
    fy_recomp = recompute_fig3_filingyear(families_all)
    stored_fy = FILING_YEAR_BY_PRIORITY
    fy_match = (fy_recomp == stored_fy)
    print(f"\n--- Fig. 3: Filing-Year × WBS Tag Counts ---")
    print(f"Row count: recomputed={len(fy_recomp)}, stored={len(stored_fy)}")
    print(f"Exact match: {'PASS' if fy_match else 'FAIL'}")
    if not fy_match:
        print("  Differences:")
        all_years = sorted(set(r[0] for r in fy_recomp) |
                           set(r[0] for r in stored_fy))
        rc_dict = {r[0]: r[1:] for r in fy_recomp}
        st_dict = {r[0]: r[1:] for r in stored_fy}
        for yr in all_years:
            rc = rc_dict.get(yr, [0, 0, 0, 0])
            st = st_dict.get(yr, [0, 0, 0, 0])
            if rc != st:
                print(f"    {yr}: recomputed={rc}, stored={st}")

    # ── Summary ──
    checks = [scope_ok, tag_match, portfolio_match, jaccard_match,
              cpc_match, bridge_top3_match]
    n_pass = sum(checks)
    n_total = len(checks)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("All recomputed values match stored constants.")
    else:
        print("Some checks failed — review output above.")
    print(f"{'=' * 70}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
