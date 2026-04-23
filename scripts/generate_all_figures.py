#!/usr/bin/env python3
"""
ISRU Construction Patent Analysis - Figure Generation Script
=============================================================
Generates the four code-generated manuscript figures (Figs. 2–5).

Paper: "Denominator construction for patent mapping in classification-poor
       emerging domains: A framework-first WBS-ITC approach with a lunar
       ISRU construction illustration"

Figure mapping:
  Fig. 2 — ITC domain portfolio bar chart
  Fig. 3 — Filing-year distribution by WBS layer (stacked bar + cumulative)
  Fig. 4 — CPC co-classification heatmap (top 25 codes)
  Fig. 5 — ITC-domain Jaccard similarity matrix

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
# Fig. 2: ITC Domain Portfolio Bar Chart
# ============================================================
def fig2_itc_portfolio_bar():
    fig, ax = plt.subplots(figsize=(12, 7))
    domains_sorted = sorted(ITC_DOMAINS, key=lambda d: PORTFOLIO[d], reverse=True)
    vals = [PORTFOLIO[d] for d in domains_sorted]
    colors = [WBS_COLORS[get_wbs(d)] for d in domains_sorted]
    labels = [f"{d}\n{ITC_LABELS[d]}" for d in domains_sorted]

    # Add domain-external gray bar (70 analytically active domain-external families)
    DOMAIN_EXTERNAL_ACTIVE = 70
    domains_sorted.append("Ext")
    vals.append(DOMAIN_EXTERNAL_ACTIVE)
    colors.append("#999999")
    labels.append("Domain-\nexternal")

    bars = ax.bar(range(len(domains_sorted)), vals, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.7)
    ax.set_xticks(range(len(domains_sorted)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel("Number of Tagged Families")
    ax.set_title("Patent Portfolio by ITC Domain (N = 443 analytically active families; 795 ITC tags)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=WBS_COLORS["1"], label="WBS-1: Materials"),
        Patch(facecolor=WBS_COLORS["2"], label="WBS-2: Manufacturing"),
        Patch(facecolor=WBS_COLORS["3"], label="WBS-3: Robotics"),
        Patch(facecolor=WBS_COLORS["4"], label="WBS-4: Structures & Systems"),
        Patch(facecolor="#999999", label="Domain-external"),
    ]
    ax.legend(handles=legend_items, loc='upper right', frameon=True, fancybox=True)

    fig.savefig(os.path.join(FIGDIR, 'fig2_itc_portfolio_bar.png'))
    plt.close(fig)
    print("  fig2_itc_portfolio_bar.png")


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
    ax1.set_title("Filing-Year Distribution by WBS Layer (Earliest Priority Year)")
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
# Fig. 4: CPC Co-classification Heatmap
# ============================================================
def fig4_cpc_coclass_heatmap():
    fig, ax = plt.subplots(figsize=(14, 11))

    # Mask diagonal for cleaner view (optional: show full)
    sns.heatmap(CPC_COOCCURRENCE, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CPC_TOP25, yticklabels=CPC_TOP25,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Co-occurrence Count', 'shrink': 0.7},
                annot_kws={'size': 7}, ax=ax)
    ax.set_title("CPC Co-classification Network (443-family analytically active layer, top 25 codes)")
    ax.set_xlabel("CPC Code")
    ax.set_ylabel("CPC Code")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    fig.savefig(os.path.join(FIGDIR, 'fig4_cpc_coclass_heatmap.png'))
    plt.close(fig)
    print("  fig4_cpc_coclass_heatmap.png")


# ============================================================
# Fig. 5: ITC-Domain Jaccard Similarity Matrix
# ============================================================
def fig5_itc_jaccard_matrix():
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(JACCARD_MATRIX, annot=True, fmt=".2f", cmap="RdBu_r",
                xticklabels=ITC_DOMAINS, yticklabels=ITC_DOMAINS,
                vmin=0, vmax=0.4, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
                annot_kws={'size': 8}, ax=ax)
    ax.set_title("ITC-Domain Jaccard Similarity Matrix (N = 373 ITC-classified families)")
    ylabels = [f"{d}\n{ITC_LABELS[d]}" for d in ITC_DOMAINS]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    ax.set_xticklabels(ITC_DOMAINS, rotation=45, ha='right', fontsize=9)

    fig.savefig(os.path.join(FIGDIR, 'fig5_itc_jaccard_matrix.png'))
    plt.close(fig)
    print("  fig5_itc_jaccard_matrix.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating ISRU Construction figures (Figs. 2–5)...")
    fig2_itc_portfolio_bar()
    fig3_wbs_filing_year()
    fig4_cpc_coclass_heatmap()
    fig5_itc_jaccard_matrix()
    print(f"All figures saved to {os.path.abspath(FIGDIR)}")
