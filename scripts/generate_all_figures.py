#!/usr/bin/env python3
"""
ISRU Construction Patent Analysis - Figure Generation Script
=============================================================
Generates the code-generated manuscript figures (body Figs. 2, 3, 4) and
the supplementary CPC heatmap (Online Resource 1, Fig. S1).

Paper: "Capability-based domain assignment for patent-based convergence analysis in classification-dispersed fields: an ISRU space-construction case"

Figure mapping:
  Fig. 2 (body)            — ITC domain portfolio bar chart
  Fig. 3 (body)            — Filing-year distribution by WBS layer (stacked bar + cumulative)
  Fig. 4 (body)            — ITC-domain Jaccard similarity matrix
  Fig. S1 (Online Resource 1) — CPC co-classification heatmap (top 25 codes)

Usage:
    cd scripts
    python generate_all_figures.py
"""
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from isru_data import (ITC_DOMAINS, ITC_LABELS, PORTFOLIO, WBS_COLORS,
                        FILING_YEAR_BY_PRIORITY, JACCARD_MATRIX,
                        CPC_TOP25, CPC_COOCCURRENCE, get_wbs)

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ============================================================
# Fig. 2: Two-panel — Panel A (15 ITC tag counts) + Panel B (single
# grey bar for 70 active domain-external families). Two separate axes,
# not a single combined chart, to keep tag counts and family counts on
# distinct scales. Matches the body Fig. 2 image inserted in the
# manuscript and its Panel A / Panel B caption.
# ============================================================
def fig2_itc_portfolio_bar():
    from matplotlib.patches import Patch

    DOMAIN_EXTERNAL_ACTIVE = 70

    fig, (axA, axB) = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 7),
        gridspec_kw={'width_ratios': [6.0, 1.2], 'wspace': 0.40},
    )

    # ─── Panel A: 15 ITC assignment tag counts (795 tags total) ───
    domains_sorted = sorted(ITC_DOMAINS, key=lambda d: PORTFOLIO[d], reverse=True)
    valsA = [PORTFOLIO[d] for d in domains_sorted]
    colorsA = [WBS_COLORS[get_wbs(d)] for d in domains_sorted]
    labelsA = [f"{d}\n{ITC_LABELS[d]}" for d in domains_sorted]

    barsA = axA.bar(range(len(domains_sorted)), valsA, color=colorsA,
                    edgecolor='white', linewidth=0.5, width=0.7)
    axA.set_xticks(range(len(domains_sorted)))
    axA.set_xticklabels(labelsA, fontsize=8, rotation=45, ha='right')
    axA.set_ylabel("Final family-domain tags")
    axA.set_title("(A) ITC domains (N = 373 ITC-classified families; 795 tags)", fontsize=11, pad=10)
    axA.set_xlabel("")
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.grid(axis='y', alpha=0.2)

    for bar, val in zip(barsA, valsA):
        axA.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    legend_items = [
        Patch(facecolor=WBS_COLORS["1"], label="WBS-1 Materials"),
        Patch(facecolor=WBS_COLORS["2"], label="WBS-2 Manufacturing"),
        Patch(facecolor=WBS_COLORS["3"], label="WBS-3 Robotics"),
        Patch(facecolor=WBS_COLORS["4"], label="WBS-4 Structures & Systems"),
    ]
    axA.legend(handles=legend_items, loc='upper right',
               frameon=True, fancybox=True, fontsize=9)

    # ─── Panel B: single grey bar for 70 active domain-external families ───
    barB = axB.bar([0], [DOMAIN_EXTERNAL_ACTIVE], color="#999999",
                   edgecolor='white', linewidth=0.5, width=0.6)
    axB.set_xticks([0])
    axB.set_xticklabels(["Active\ndomain-\nexternal"], fontsize=9)
    axB.set_xlim(-0.7, 0.7)
    axB.set_ylabel("Active families (count, not tags)")
    axB.set_title("(B) Boundary-set accounting", fontsize=11, pad=10)
    axB.set_xlabel("")
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    axB.grid(axis='y', alpha=0.2)
    axB.text(0, DOMAIN_EXTERNAL_ACTIVE + 1, str(DOMAIN_EXTERNAL_ACTIVE),
             ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.savefig(os.path.join(FIGDIR, 'fig2_itc_portfolio_bar.png'), bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print("  fig2_itc_portfolio_bar.png  (two-panel with title-based labels; A=15 ITC tags, B=single 70 bar)")


# ============================================================
# Fig. 3: Filing-Year Distribution by WBS Layer
# ============================================================
def fig3_wbs_filing_year():
    data = np.array(FILING_YEAR_BY_PRIORITY)
    years = data[:, 0].astype(int)
    wbs1 = data[:, 1]
    wbs2 = data[:, 2]
    wbs3 = data[:, 3]
    wbs4 = data[:, 4]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Stacked bar
    x = np.arange(len(years))
    bar_width = 0.7
    b1 = ax1.bar(x, wbs1, bar_width, label='WBS-1: Materials',
                 color=WBS_COLORS["1"], edgecolor='white', linewidth=0.3)
    b2 = ax1.bar(x, wbs2, bar_width, bottom=wbs1, label='WBS-2: Manufacturing',
                 color=WBS_COLORS["2"], edgecolor='white', linewidth=0.3)
    b3 = ax1.bar(x, wbs3, bar_width, bottom=wbs1+wbs2, label='WBS-3: Robotics',
                 color=WBS_COLORS["3"], edgecolor='white', linewidth=0.3)
    b4 = ax1.bar(x, wbs4, bar_width, bottom=wbs1+wbs2+wbs3,
                 label='WBS-4: Structures & Systems',
                 color=WBS_COLORS["4"], edgecolor='white', linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel("Earliest Priority Year")
    ax1.set_ylabel("Annual WBS-Layer Tag Count")
    # Title intentionally omitted to match the body Fig. 3 image
    # inserted in the manuscript (caption supplied alongside Fig. 3).
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    cumulative = np.cumsum(wbs1 + wbs2 + wbs3 + wbs4)
    ax2.plot(x, cumulative, color='#333333', linewidth=2, marker='o',
             markersize=4, label='Cumulative tag count', zorder=5)
    ax2.set_ylabel("Cumulative Tag Count")
    ax2.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               frameon=True, fancybox=True)

    fig.savefig(os.path.join(FIGDIR, 'fig3_wbs_filing_year.png'))
    plt.close(fig)
    print("  fig3_wbs_filing_year.png")


# ============================================================
# Fig. S1 (Online Resource 1): CPC Co-classification Heatmap
# ============================================================
def fig4_cpc_coclass_heatmap():
    fig, ax = plt.subplots(figsize=(14, 11))

    # Mask diagonal for cleaner view (optional: show full)
    sns.heatmap(CPC_COOCCURRENCE, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CPC_TOP25, yticklabels=CPC_TOP25,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Co-occurrence Count', 'shrink': 0.7},
                annot_kws={'size': 7}, ax=ax)
    # Title intentionally omitted to match Online Resource 1 Fig. S1
    # image (caption supplied alongside the figure).
    ax.set_xlabel("CPC Code")
    ax.set_ylabel("CPC Code")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    fig.savefig(os.path.join(FIGDIR, 'figS1_cpc_coclass_heatmap.png'))
    plt.close(fig)
    print("  figS1_cpc_coclass_heatmap.png")


# ============================================================
# Fig. 4 (body): ITC-Domain Jaccard Similarity Matrix
# ============================================================
def fig4_itc_jaccard_matrix():
    """Body Fig. 4 — ITC-domain Jaccard similarity matrix.

    The Jaccard matrix is recomputed directly from the released JSON
    (data/phase2_453_families.json) so that the displayed 2-decimal
    cells match raw computation rather than the 3-decimal rounded
    constants stored in isru_data.JACCARD_MATRIX.
    """
    import json
    json_path = os.path.join(os.path.dirname(__file__), '..',
                             'data', 'phase2_453_families.json')
    with open(json_path, encoding='utf-8') as f:
        families = json.load(f)
    classified = [fam for fam in families if fam.get('is_classified', False)]
    domain_sets = {d: set() for d in ITC_DOMAINS}
    for i, fam in enumerate(classified):
        for code in fam.get('itc_codes', []):
            if code in domain_sets:
                domain_sets[code].add(i)
    n = len(ITC_DOMAINS)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                si, sj = domain_sets[ITC_DOMAINS[i]], domain_sets[ITC_DOMAINS[j]]
                u = len(si | sj)
                matrix[i][j] = (len(si & sj) / u) if u > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(matrix.shape[0], dtype=bool)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                xticklabels=ITC_DOMAINS, yticklabels=ITC_DOMAINS,
                vmin=0, vmax=0.4, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
                annot_kws={'size': 8}, mask=mask, ax=ax)
    # Title intentionally omitted to match the body Fig. 4 image
    # inserted in the manuscript (caption supplied alongside Fig. 4;
    # diagonals omitted to suppress self-similarity).
    ylabels = [f"{d}\n{ITC_LABELS[d]}" for d in ITC_DOMAINS]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    xlabels = [f"{d}\n{ITC_LABELS[d]}" for d in ITC_DOMAINS]
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)

    fig.savefig(os.path.join(FIGDIR, 'fig4_itc_jaccard_matrix.png'))
    plt.close(fig)
    print("  fig4_itc_jaccard_matrix.png  (computed from JSON, diagonals masked, code+domain x-axis)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating ISRU Construction figures (body Figs. 2, 3, 4 and Online Resource 1 Fig. S1)...")
    fig2_itc_portfolio_bar()
    fig3_wbs_filing_year()
    fig4_cpc_coclass_heatmap()        # Online Resource 1 Fig. S1
    fig4_itc_jaccard_matrix()         # Body Fig. 4
    print(f"All figures saved to {os.path.abspath(FIGDIR)}")
