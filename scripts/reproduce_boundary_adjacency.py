#!/usr/bin/env python3
"""
reproduce_boundary_adjacency.py
===============================================================================
Reconstructs the inherited-vs-constructed field-boundary adjacency contrast
reported in the manuscript:

    Figure 5  - two panels: (A) domain-pair co-tagging Jaccard, inherited
                (CPC-only) vs constructed (assignment) boundary; (B) rank
                trajectories of the leading co-tagging pairs between layers.
    Table 9  - leading reordering domain pairs (Jaccard and rank per layer).
    Headline statistics - Spearman rho, Kendall tau, agreement rates, and the
                number of active co-tagging pairs per layer.

All quantities are recomputed from the single released input
`data/assignment_table_long.csv` (the family-domain assignment table). No
retrieval is re-run and no consensus judgment is re-litigated; the script
operates entirely on the released released assignment record.

The script first checks the released assignment table against the headline counts
reported in the manuscript and aborts if any check fails:
    795 assignment tags | 373 classified families | 677 rule-positive tags
    agreement 53.0% (union) / 42.9% (CPC) / 26.4% (keyword)
    374 assignment-only additions | 256 suppressions

Input columns expected in the assignment table (one row per family x ITC-domain cell):
    lens_id, domain, cpc_rule_hit, keyword_rule_hit, rule_union_hit,
    assignment_hit, action, status_category, analysis_set_443

Outputs (written under --outdir):
    figures/fig5_boundary_vs_convergence.png
    tables/table9_boundary_adjacency.csv

Usage:
    python reproduce_boundary_adjacency.py \
        --table data/assignment_table_long.csv \
        --outdir .

Dependencies: pandas, numpy, scipy, matplotlib  (Python 3.8+)
===============================================================================
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ITC domain labels (manuscript Table 4) -------------------------------- #
DOMAIN_NAMES = {
    "1-1": "Regolith Processing", "1-2": "Binder/Geopolymer", "1-3": "Composite/Ceramic",
    "2-1": "Extrusion AM",        "2-2": "Powder Bed",         "2-3": "Solar/Laser Sintering",
    "2-4": "Process Monitoring",  "3-1": "Autonomous Robots",  "3-2": "Teleoperation",
    "3-3": "Autonomous Construction", "4-1": "Habitat Structures", "4-2": "Shielding",
    "4-3": "Landing Pads",        "4-4": "Deployable Structures", "4-5": "Life Support/ECLSS",
}

# Pairs highlighted in Figure 5 / Table 9 (selected as the leading reordering pairs)
HEADLINE_PAIRS = [("3-3", "4-1"), ("2-1", "2-4"), ("4-2", "4-4"), ("1-3", "2-3")]

# Manuscript headline counts used for the integrity check
EXPECTED = dict(adj_tags=795, classified=373, rule_pos=677,
                additions=374, suppressions=256,
                rec_union=0.530, rec_cpc=0.429, rec_kw=0.264)


def load_active_layer(path):
    """Load the assignment table and restrict to the 443-family active analytical layer."""
    df = pd.read_csv(path)
    needed = {"lens_id", "domain", "cpc_rule_hit", "keyword_rule_hit",
              "rule_union_hit", "assignment_hit", "analysis_set_443"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: assignment table is missing columns: {sorted(missing)}")
    a = df[df["analysis_set_443"] == True].copy()  # noqa: E712  (explicit flag match)
    return a


def integrity_check(a):
    """Check the released assignment table reproduces the manuscript's headline counts."""
    adj = a[a.assignment_hit == 1]
    got = dict(
        adj_tags=int(adj.shape[0]),
        classified=int(adj.lens_id.nunique()),
        rule_pos=int((a.rule_union_hit == 1).sum()),
        additions=int(((a.assignment_hit == 1) & (a.rule_union_hit == 0)).sum()),
        suppressions=int(((a.rule_union_hit == 1) & (a.assignment_hit == 0)).sum()),
        rec_union=round((adj.rule_union_hit == 1).mean(), 3),
        rec_cpc=round((adj.cpc_rule_hit == 1).mean(), 3),
        rec_kw=round((adj.keyword_rule_hit == 1).mean(), 3),
    )
    print("Integrity check against manuscript headline counts:")
    ok = True
    for k, exp in EXPECTED.items():
        match = "OK " if got[k] == exp else "FAIL"
        if got[k] != exp:
            ok = False
        print(f"  [{match}] {k:13s} expected {exp!s:>7}  got {got[k]!s:>7}")
    if not ok:
        sys.exit("ERROR: assignment table does not reproduce the reported counts; aborting.")
    print("  -> all checks passed.\n")


def membership(a, flag, domains):
    """domain -> set of families assigned to it under the given hit flag."""
    sub = a[a[flag] == 1]
    return {d: set(sub.loc[sub.domain == d, "lens_id"]) for d in domains}


def jaccard_vector(mem, pairs):
    """Jaccard similarity for every domain pair (0 when both domains are empty)."""
    out = {}
    for i, j in pairs:
        A, B = mem.get(i, set()), mem.get(j, set())
        union = len(A | B)
        out[(i, j)] = (len(A & B) / union) if union else 0.0
    return out


def compute(a):
    """Compute adjacency vectors, statistics, and the reordering table."""
    domains = sorted(a.domain.unique())
    pairs = list(combinations(domains, 2))  # 105 for 15 domains

    cpc = jaccard_vector(membership(a, "cpc_rule_hit", domains), pairs)
    uni = jaccard_vector(membership(a, "rule_union_hit", domains), pairs)
    adj = jaccard_vector(membership(a, "assignment_hit", domains), pairs)

    x_cpc = np.array([cpc[p] for p in pairs])
    x_uni = np.array([uni[p] for p in pairs])
    y_adj = np.array([adj[p] for p in pairs])

    rho_main, _ = spearmanr(x_cpc, y_adj)
    tau_main, _ = kendalltau(x_cpc, y_adj)
    rho_uni, _ = spearmanr(x_uni, y_adj)

    # Sensitivity: restrict both layers to the 373 ITC-classified families
    classified = set(a.loc[a.assignment_hit == 1, "lens_id"])
    ac = a[a.lens_id.isin(classified)]
    cpc_c = jaccard_vector(membership(ac, "cpc_rule_hit", domains), pairs)
    adj_c = jaccard_vector(membership(ac, "assignment_hit", domains), pairs)
    rho_sens, _ = spearmanr([cpc_c[p] for p in pairs], [adj_c[p] for p in pairs])

    rank_cpc = {p: r + 1 for r, p in enumerate(sorted(pairs, key=lambda p: cpc[p], reverse=True))}
    rank_adj = {p: r + 1 for r, p in enumerate(sorted(pairs, key=lambda p: adj[p], reverse=True))}

    stats = dict(
        n_pairs=len(pairs),
        rho_main=round(rho_main, 3), tau_main=round(tau_main, 3),
        rho_union=round(rho_uni, 3), rho_sensitivity=round(rho_sens, 3),
        active_cpc=sum(1 for p in pairs if cpc[p] > 0),
        active_uni=sum(1 for p in pairs if uni[p] > 0),
        active_adj=sum(1 for p in pairs if adj[p] > 0),
    )
    return dict(domains=domains, pairs=pairs, cpc=cpc, uni=uni, adj=adj,
                rank_cpc=rank_cpc, rank_adj=rank_adj, stats=stats)


def write_table9(res, outdir):
    """Write the Table 9 reordering-pair table as CSV."""
    rows = []
    for p in HEADLINE_PAIRS:
        i, j = p
        rows.append({
            "domain_pair": f"{DOMAIN_NAMES[i]} x {DOMAIN_NAMES[j]}",
            "code_pair": f"{i} x {j}",
            "inherited_jaccard": round(res["cpc"][p], 3),
            "inherited_rank": res["rank_cpc"][p],
            "constructed_jaccard": round(res["adj"][p], 3),
            "constructed_rank": res["rank_adj"][p],
            "rank_shift": f"{res['rank_cpc'][p]} -> {res['rank_adj'][p]}",
        })
    df = pd.DataFrame(rows)
    path = os.path.join(outdir, "tables", "table9_boundary_adjacency.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path, df


def make_figure(res, outdir):
    """Reproduce Figure 5 (two panels)."""
    pairs, cpc, adj = res["pairs"], res["cpc"], res["adj"]
    rk_cpc, rk_adj = res["rank_cpc"], res["rank_adj"]
    rho = res["stats"]["rho_main"]
    idx = {p: k for k, p in enumerate(pairs)}
    cpc_v = np.array([cpc[p] for p in pairs])
    adj_v = np.array([adj[p] for p in pairs])

    def label(p):
        return f"{DOMAIN_NAMES[p[0]]} \u00d7 {DOMAIN_NAMES[p[1]]}"

    RED, BLUE = "#b2182b", "#2166ac"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 0.8, "xtick.direction": "in",
                         "ytick.direction": "in"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.9))

    # Panel A -- inherited vs constructed adjacency
    axA.scatter(cpc_v, adj_v, s=22, facecolor="none", edgecolor="0.5", linewidth=0.8, zorder=3)
    m = max(cpc_v.max(), adj_v.max()) * 1.10
    axA.plot([0, m], [0, m], ls="--", color="0.6", lw=1, zorder=1)
    axA.text(m * 0.98, m * 0.90, "y = x", color="0.5", ha="right", fontsize=9, style="italic")
    anno = {("3-3", "4-1"): (RED, "right", "top", -0.006, -0.004),
            ("2-1", "2-4"): (BLUE, "center", "bottom", 0, 0.014),
            ("4-2", "4-4"): (BLUE, "left", "center", 0.008, 0),
            ("1-3", "2-3"): (BLUE, "left", "center", 0.008, 0)}
    for p, (col, ha, va, dx, dy) in anno.items():
        k = idx[p]
        axA.scatter(cpc[p], adj[p], s=44, color=col, zorder=5)
        axA.annotate(label(p), (cpc[p], adj[p]), xytext=(cpc[p] + dx, adj[p] + dy),
                     fontsize=8, color=col, ha=ha, va=va)
    axA.set_xlabel("Inherited-classification adjacency\n(CPC-only Jaccard)")
    axA.set_ylabel("Constructed-boundary adjacency\n(final-assignment Jaccard)")
    axA.set_xlim(-0.012, m); axA.set_ylim(-0.012, m)
    axA.text(0.015, m * 0.985, f"Spearman \u03c1 = {rho:.2f}\n({len(pairs)} domain pairs)",
             fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.7))
    axA.set_title("(A) Adjacency conclusion depends on the boundary", fontsize=10.5, pad=8)

    # Panel B -- rank trajectories of leading pairs
    for p in HEADLINE_PAIRS:
        r0, r1 = rk_cpc[p], rk_adj[p]
        col = RED if p == ("3-3", "4-1") else BLUE
        lw = 2.4 if p in [("3-3", "4-1"), ("2-1", "2-4")] else 1.5
        axB.plot([0, 1], [r0, r1], color=col, lw=lw, marker="o", ms=5, zorder=3)
        axB.text(-0.04, r0, f"{label(p)}  ", ha="right", va="center", fontsize=8.2, color=col)
        axB.text(1.04, r1, f"#{r1}", ha="left", va="center", fontsize=8.6, color=col, weight="bold")
        axB.text(-0.005, r0 + 1.4, f"#{r0}", ha="right", va="top", fontsize=7.5, color=col)
    axB.set_xlim(-1.15, 1.45); axB.set_ylim(56, -3)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["Inherited\n(CPC-only)", "Constructed\n(final assignment)"])
    axB.set_ylabel("Co-tagging rank  (1 = strongest of %d pairs)" % len(pairs))
    axB.set_title("(B) Leading co-tagging pairs reorder", fontsize=10.5, pad=8)
    for spine in ["top", "right"]:
        axB.spines[spine].set_visible(False)
    axB.tick_params(axis="x", length=0)

    plt.tight_layout()
    path = os.path.join(outdir, "figures", "fig5_boundary_vs_convergence.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="data/assignment_table_long.csv",
                    help="path to the released family-domain assignment table CSV")
    ap.add_argument("--outdir", default=".",
                    help="output directory (figures/ and tables/ are created here)")
    args = ap.parse_args()

    a = load_active_layer(args.table)
    integrity_check(a)
    res = compute(a)

    fig_path = make_figure(res, args.outdir)
    tab_path, tab_df = write_table9(res, args.outdir)

    s = res["stats"]
    print("Boundary-adjacency contrast (inherited CPC-only vs constructed assignment):")
    print(f"  domain pairs                         : {s['n_pairs']}")
    print(f"  Spearman rho (inherited, constructed): {s['rho_main']:.3f}")
    print(f"  Kendall  tau (inherited, constructed): {s['tau_main']:.3f}")
    print(f"  Spearman rho (union, constructed)    : {s['rho_union']:.3f}")
    print(f"  Spearman rho, 373-classified only    : {s['rho_sensitivity']:.3f}")
    print(f"  active co-tagging pairs (CPC/union/adj): "
          f"{s['active_cpc']}/{s['active_uni']}/{s['active_adj']} of {s['n_pairs']}")
    print("\nTable 9 (leading reordering pairs):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(tab_df.to_string(index=False))
    print(f"\nWrote figure : {fig_path}")
    print(f"Wrote table  : {tab_path}")


if __name__ == "__main__":
    main()