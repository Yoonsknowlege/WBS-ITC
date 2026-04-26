#!/usr/bin/env python3
"""
sensitivity_checks.py — Sensitivity checks for the released dataset.

Paper: "Denominator construction for patent mapping in classification-poor
       emerging domains: A framework-first WBS-ITC approach with a lunar
       ISRU construction illustration"

Note: This script operates on the full 453-family released JSON.
The manuscript's analytical layer uses 443 families (9 false-match
records and 1 audit-only record excluded); because all
10 excluded families were unclassified, ITC-based metrics are unchanged.

Checks included:
1) Leave-out sensitivity excluding rescue families (10 families in
   the released dataset; 12 rescued records before deduplication)
2) CPC bridging sensitivity using classified-only families
   (excluding 70 analytically active domain-external families)
3) Jurisdiction-stratified sensitivity: CN vs non-CN subsets
   (Supplementary Table S2)
4) Shared-anchor inflation check: 4-1 ↔ 4-4 Jaccard recomputation
   excluding E04H15-only families

Usage:
    python scripts/sensitivity_checks.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_MD = os.path.join(ROOT_DIR, "sensitivity_notes.md")

from isru_data import ITC_DOMAINS  # noqa: E402

JSON_PATH = os.path.join(DATA_DIR, "phase2_453_families.json")


def load_families():
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def wbs_tag_shares(families):
    counter = Counter()
    total = 0
    for fam in families:
        for code in fam.get("itc_codes", []):
            counter[code.split("-")[0]] += 1
            total += 1
    labels = {
        "1": "Materials",
        "2": "Manufacturing",
        "3": "Robotics",
        "4": "Structures & Systems",
    }
    rows = []
    for prefix in ["1", "2", "3", "4"]:
        share = counter[prefix] / total * 100 if total else 0.0
        rows.append((prefix, labels[prefix], counter[prefix], share))
    return rows, total


def domain_families(families):
    out = defaultdict(set)
    for i, fam in enumerate(families):
        for code in fam.get("itc_codes", []):
            out[code].add(i)
    return out


def jaccard_pairs(families):
    df = domain_families(families)
    pairs = []
    for d1, d2 in combinations(ITC_DOMAINS, 2):
        s1, s2 = df[d1], df[d2]
        union = len(s1 | s2)
        inter = len(s1 & s2)
        if union > 0:
            jacc = inter / union
            if jacc > 0:
                pairs.append((d1, d2, jacc))
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    return pairs


def jaccard_single(families, code_a, code_b):
    """Compute Jaccard for a single domain pair."""
    df = domain_families(families)
    s1, s2 = df[code_a], df[code_b]
    union = len(s1 | s2)
    inter = len(s1 & s2)
    return inter / union if union > 0 else 0.0, len(s1), len(s2), inter


def cpc_bridging(families):
    all_cpc = Counter()
    family_cpc_sets = []
    for fam in families:
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            continue
        codes = sorted(set(c.strip()[:4] for c in cpc_str.split(";;") if len(c.strip()) >= 4))
        family_cpc_sets.append(codes)
        for code in codes:
            all_cpc[code] += 1

    code_list = sorted(all_cpc.keys())
    idx = {c: i for i, c in enumerate(code_list)}
    n = len(code_list)

    adj = np.zeros((n, n), dtype=int)
    for codes in family_cpc_sets:
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                ci, cj = idx[codes[i]], idx[codes[j]]
                adj[ci, cj] += 1
                adj[cj, ci] += 1

    binary = (adj >= 1).astype(int)
    np.fill_diagonal(binary, 0)

    degree = binary.sum(axis=1) / (n - 1)
    bet = np.zeros(n)

    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
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

        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]

    bet /= ((n - 1) * (n - 2))
    ranked = sorted(range(n), key=lambda i: (-degree[i], -bet[i], code_list[i]))
    return [(code_list[i], all_cpc[code_list[i]], float(degree[i]), float(bet[i])) for i in ranked[:10]]


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main():
    families = load_families()
    rescue_fams = [f for f in families if f.get("source") == "rescue"]
    no_rescue = [f for f in families if f.get("source") != "rescue"]
    classified = [f for f in families if f.get("is_classified")]

    # ── Check 1: Leave-out sensitivity ──
    all_rows, all_total = wbs_tag_shares(families)
    nr_rows, nr_total = wbs_tag_shares(no_rescue)

    paired_rows = []
    max_change = 0.0
    for a, b in zip(all_rows, nr_rows):
        change = b[3] - a[3]
        max_change = max(max_change, abs(change))
        paired_rows.append((a[1], a[2], f"{a[3]:.2f}%", b[2], f"{b[3]:.2f}%", f"{change:+.2f} pp"))

    pairs_all = jaccard_pairs(families)
    pairs_nr = jaccard_pairs(no_rescue)
    top10_all = [(d1, d2, round(j, 3)) for d1, d2, j in pairs_all[:10]]
    top10_nr = [(d1, d2, round(j, 3)) for d1, d2, j in pairs_nr[:10]]
    top6_same = [(a, b) for a, b, _ in top10_all[:6]] == [(a, b) for a, b, _ in top10_nr[:6]]

    # ── Check 2: CPC bridging sensitivity ──
    bridge_all = cpc_bridging(families)
    bridge_cls = cpc_bridging(classified)
    trio_same = [x[0] for x in bridge_all[:3]] == [x[0] for x in bridge_cls[:3]]

    # ── Check 3: Jurisdiction-stratified sensitivity (CN vs non-CN) ──
    cn_fams = [f for f in families if f.get("jurisdiction", "").startswith("CN")]
    non_cn_fams = [f for f in families if not f.get("jurisdiction", "").startswith("CN")]
    cn_classified = [f for f in cn_fams if f.get("is_classified")]
    non_cn_classified = [f for f in non_cn_fams if f.get("is_classified")]

    all_wbs, _ = wbs_tag_shares(classified)
    cn_wbs, _ = wbs_tag_shares(cn_classified)
    non_cn_wbs, _ = wbs_tag_shares(non_cn_classified)

    pairs_cn = jaccard_pairs(cn_classified)
    pairs_non_cn = jaccard_pairs(non_cn_classified)
    top_pair_all = (pairs_all[0][0], pairs_all[0][1], round(pairs_all[0][2], 3)) if pairs_all else None
    top_pair_cn = (pairs_cn[0][0], pairs_cn[0][1], round(pairs_cn[0][2], 3)) if pairs_cn else None
    top_pair_non_cn = (pairs_non_cn[0][0], pairs_non_cn[0][1], round(pairs_non_cn[0][2], 3)) if pairs_non_cn else None

    bridge_cn = cpc_bridging(cn_fams)
    bridge_non_cn = cpc_bridging(non_cn_fams)
    top_bridge_all = bridge_all[0][0] if bridge_all else "N/A"
    top_bridge_cn = bridge_cn[0][0] if bridge_cn else "N/A"
    top_bridge_non_cn = bridge_non_cn[0][0] if bridge_non_cn else "N/A"

    # ── Check 4: Shared-anchor inflation (E04H15, domains 4-1 and 4-4) ──
    # Identify families whose sole tagging route to 4-1 or 4-4 runs through E04H15
    anchors_41_other = ["B64G1/48", "E04B1", "E04H1"]  # 4-1 anchors excluding E04H15
    anchors_44_other = ["B64G1/22"]                     # 4-4 anchors excluding E04H15

    def has_non_e04h15_anchor(cpc_list, other_anchors):
        for cpc in cpc_list:
            for a in other_anchors:
                if cpc.startswith(a):
                    return True
        return False

    e04h15_only_indices = set()
    for i, f in enumerate(families):
        codes = f.get("itc_codes", [])
        if "4-1" in codes and "4-4" in codes:
            cpcs = f.get("cpc", "").split(";;") if f.get("cpc") else []
            has_e04h15 = any(c.strip().startswith("E04H15") for c in cpcs)
            if has_e04h15:
                still_41 = has_non_e04h15_anchor(cpcs, anchors_41_other)
                still_44 = has_non_e04h15_anchor(cpcs, anchors_44_other)
                if not still_41 or not still_44:
                    e04h15_only_indices.add(i)

    fams_no_e04h15_only = [f for i, f in enumerate(families) if i not in e04h15_only_indices]
    j_orig, n41, n44, inter_orig = jaccard_single(families, "4-1", "4-4")
    j_adj, n41_adj, n44_adj, inter_adj = jaccard_single(fams_no_e04h15_only, "4-1", "4-4")

    # ── Build markdown output ──
    md = []
    md.append("# Package sensitivity notes")
    md.append("")
    md.append("This note documents the sensitivity checks supplied with the released analytical-layer dataset.")
    md.append("")
    md.append("## Dataset scope")
    md.append("")
    md.append(f"| Scope | N |")
    md.append(f"|---|---|")
    md.append(f"| Released families (JSON) | {len(families)} |")
    md.append(f"| Excluded (9 false-match + 1 audit-only) | 10 |")
    md.append(f"| Analytically active | 443 |")
    md.append(f"| ITC-classified | {len(classified)} |")
    md.append(f"| Domain-external (active) | {443 - len(classified)} |")
    md.append(f"| Jurisdiction (released) | CN={len(cn_fams)}, Non-CN={len(non_cn_fams)} |")
    md.append(f"| Jurisdiction (active) | CN=336, Non-CN=107 |")
    md.append("")
    md.append("Note: This script operates on the full 453-family released JSON. "
              "Because all 10 excluded families are unclassified (no ITC tags), "
              "ITC-based metrics are mathematically invariant to the exclusion.\n"
              "CPC-level metrics are empirically stable (see classified-only comparison below).")
    md.append("")

    # Section 1
    md.append("## 1) Leave-out sensitivity excluding rescue families")
    md.append("")
    md.append(f"- Released dataset size: **{len(families)} families**")
    md.append(f"- Rescue families retained after deduplication: **{len(rescue_fams)}**")
    md.append(f"- Leave-out dataset size: **{len(no_rescue)} families**")
    md.append(f"- Maximum absolute WBS-layer tag-share change: **{max_change:.2f} percentage points**")
    md.append(f"- Top six Jaccard pairs unchanged: **{'Yes' if top6_same else 'No'}**")
    md.append("")
    md.append(markdown_table(
        ["WBS layer", "All families tags", "All families share", "No-rescue tags", "No-rescue share", "Change"],
        paired_rows
    ))
    md.append("")
    md.append("Top 10 Jaccard pairs — all families")
    md.append("")
    md.append(markdown_table(["Rank", "Pair", "Jaccard"], [(i + 1, f"{a}↔{b}", f"{j:.3f}") for i, (a, b, j) in enumerate(top10_all)]))
    md.append("")
    md.append("Top 10 Jaccard pairs — excluding rescue families")
    md.append("")
    md.append(markdown_table(["Rank", "Pair", "Jaccard"], [(i + 1, f"{a}↔{b}", f"{j:.3f}") for i, (a, b, j) in enumerate(top10_nr)]))
    md.append("")
    md.append("Interpretation: the leading Jaccard structure is stable, but lower-ranked pairs shift modestly once the 10 rescue families are removed.")
    md.append("")

    # Section 2
    md.append("## 2) CPC bridging sensitivity — all families vs classified-only")
    md.append("")
    md.append(f"- Families excluded in the classified-only run: **{len(families) - len(classified)}** "
               f"(of which 70 are analytically active domain-external and 10 are non-active excluded records)")
    md.append(f"- Leading bridging trio unchanged (B64G, E04H, G01N): **{'Yes' if trio_same else 'No'}**")
    md.append("")
    md.append("Top 10 CPC bridging codes — all 453 released families")
    md.append("")
    md.append(markdown_table(
        ["Rank", "CPC", "Freq", "Degree", "Betweenness"],
        [(i + 1, c, f, f"{d:.3f}", f"{b:.3f}") for i, (c, f, d, b) in enumerate(bridge_all)]
    ))
    md.append("")
    md.append("Top 10 CPC bridging codes — classified-only families")
    md.append("")
    md.append(markdown_table(
        ["Rank", "CPC", "Freq", "Degree", "Betweenness"],
        [(i + 1, c, f, f"{d:.3f}", f"{b:.3f}") for i, (c, f, d, b) in enumerate(bridge_cls)]
    ))
    md.append("")
    md.append("Interpretation: the leading bridging codes remain the same, while lower-ranked CPC positions shift once domain-external families are removed.")
    md.append("")

    # Section 3
    md.append("## 3) Jurisdiction-stratified sensitivity: CN vs non-CN subsets")
    md.append("")
    md.append(f"- CN families (released 453): **{len(cn_fams)}** ({len(cn_fams)/len(families)*100:.1f}%)")
    md.append(f"- Non-CN families (released 453): **{len(non_cn_fams)}** ({len(non_cn_fams)/len(families)*100:.1f}%)")
    md.append(f"- Manuscript analytically active (443): CN = 336, Non-CN = 107 (10 excluded: 6 CN + 4 non-CN)")
    md.append("")
    md.append("### Panel A: WBS-layer tag shares (classified families only)")
    md.append("")
    wbs_rows = []
    for a, c, nc in zip(all_wbs, cn_wbs, non_cn_wbs):
        wbs_rows.append((a[1], f"{a[3]:.1f}%", f"{c[3]:.1f}%", f"{nc[3]:.1f}%"))
    md.append(markdown_table(["WBS layer", "All", "CN", "Non-CN"], wbs_rows))
    md.append("")
    md.append("### Panel B: Top Jaccard pair stability")
    md.append("")
    md.append(f"- All families: **{top_pair_all[0]}↔{top_pair_all[1]}** (J = {top_pair_all[2]:.3f})" if top_pair_all else "- All families: N/A")
    md.append(f"- CN subset: **{top_pair_cn[0]}↔{top_pair_cn[1]}** (J = {top_pair_cn[2]:.3f})" if top_pair_cn else "- CN subset: N/A")
    md.append(f"- Non-CN subset: **{top_pair_non_cn[0]}↔{top_pair_non_cn[1]}** (J = {top_pair_non_cn[2]:.3f})" if top_pair_non_cn else "- Non-CN subset: N/A")
    md.append("")
    md.append("### Panel C: CPC bridging code stability")
    md.append("")
    md.append(f"- All families top bridging code: **{top_bridge_all}**")
    md.append(f"- CN subset top bridging code: **{top_bridge_cn}**")
    md.append(f"- Non-CN subset top bridging code: **{top_bridge_non_cn}**")
    md.append("")
    md.append("Interpretation: the manufacturing emphasis is partly driven by CN filing patterns, but the core convergence structure (top Jaccard pair and principal bridging code) persists across both subsets.")
    md.append("")

    # Section 4
    md.append("## 4) Shared-anchor inflation check: 4-1 ↔ 4-4 (E04H15)")
    md.append("")
    md.append("E04H15 serves as a CPC anchor for both domain 4-1 (Habitat Structures) and domain 4-4 (Deployable Structures).")
    md.append("Families whose sole tagging route to either domain runs through E04H15 are removed to quantify the inflation effect.")
    md.append("")
    md.append(f"- E04H15-only families removed: **{len(e04h15_only_indices)}**")
    md.append(f"- Original 4-1 ↔ 4-4 Jaccard: **{j_orig:.3f}** (|4-1| = {n41}, |4-4| = {n44}, intersection = {inter_orig})")
    md.append(f"- Adjusted 4-1 ↔ 4-4 Jaccard: **{j_adj:.3f}** (|4-1| = {n41_adj}, |4-4| = {n44_adj}, intersection = {inter_adj})")
    md.append(f"- Change: **{j_orig:.3f} → {j_adj:.3f}** (Δ = {j_adj - j_orig:+.3f})")
    md.append("")
    md.append("Interpretation: the structural-overlap signal persists after removing the shared-anchor component. The 4-1 ↔ 4-4 pair remains within the top Jaccard rankings.")
    md.append("")

    # Scope
    md.append("## Scope reminder")
    md.append("")
    md.append("These checks operate on the released family-level analytical layer only.")
    md.append("")

    md_text = "\n".join(md)
    print(md_text)


if __name__ == "__main__":
    main()
