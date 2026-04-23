#!/usr/bin/env python3
"""
wbs_layer_concordance.py — WBS-layer concordance and Krippendorff's α diagnostic.

Computes:
  1. WBS-layer-level Cohen's κ and PABAK (rule-union vs. adjudicated)
  2. Krippendorff's α (nominal) over the 15-domain binary matrix

Reference values (from manuscript §3.5 and Supplementary Table S6a):
  - WBS-1 Materials     κ = 0.257
  - WBS-2 Manufacturing κ = 0.590
  - WBS-3 Robotics      κ = 0.269
  - WBS-4 Structures    κ = 0.617
  - Krippendorff's α (nominal, 15-domain) = 0.519

Usage:
    python scripts/wbs_layer_concordance.py
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")

# 15 ITC domains grouped by WBS layer
WBS_LAYERS = {
    "WBS-1 Materials": ["1-1", "1-2", "1-3"],
    "WBS-2 Manufacturing": ["2-1", "2-2", "2-3", "2-4"],
    "WBS-3 Robotics": ["3-1", "3-2", "3-3"],
    "WBS-4 Structures & Systems": ["4-1", "4-2", "4-3", "4-4", "4-5"],
}

DOMAIN_ORDER = [
    "1-1", "1-2", "1-3",
    "2-1", "2-2", "2-3", "2-4",
    "3-1", "3-2", "3-3",
    "4-1", "4-2", "4-3", "4-4", "4-5",
]


def parse_domains(s):
    """Parse semicolon-delimited domain string into a set."""
    if not s or s.strip() == '':
        return set()
    return set(s.strip().split(';'))


def cohens_kappa(tp, fp, fn, tn):
    """Compute Cohen's κ from a 2×2 table."""
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def pabak(tp, fp, fn, tn):
    """Compute prevalence-adjusted bias-adjusted kappa."""
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    return 2 * po - 1


def krippendorff_alpha_binary_matrix(rule_matrix, adj_matrix, n_families, n_domains):
    """
    Compute Krippendorff's α (nominal) for a binary reliability matrix.

    Each family × domain cell is rated by two coders (rule-union, adjudicated)
    as 0 or 1. We treat the 15-domain vector per family as 15 separate items,
    each rated by 2 coders. This yields N_items = n_families × n_domains items,
    each with 2 ratings.

    For nominal data with 2 coders per item:
      D_o = (number of disagreements) / N_items
      D_e = 2 * p * (1 - p)  where p = proportion of 1s across all ratings
      α = 1 - D_o / D_e
    """
    n_items = n_families * n_domains
    total_ratings = n_items * 2  # 2 coders per item

    # Count disagreements and total 1s
    disagreements = 0
    total_ones = 0
    for i in range(n_families):
        for j in range(n_domains):
            r = rule_matrix[i][j]
            a = adj_matrix[i][j]
            total_ones += r + a
            if r != a:
                disagreements += 1

    p = total_ones / total_ratings
    D_o = disagreements / n_items
    D_e = 2 * p * (1 - p)

    if D_e == 0:
        return 1.0

    # Apply finite-sample correction: α = 1 - (D_o / D_e) * ((n_items * 2 - 1) / (n_items * 2 - 2))
    # For large n this is negligible, but included for correctness
    m = total_ratings  # total individual ratings
    correction = (m - 1) / (m - 2) if m > 2 else 1.0

    # Krippendorff's α with finite-sample correction
    # Using the standard formula: α = 1 - (D_o / D_e)
    # The correction factor is applied differently in the canonical form.
    # Canonical: α = 1 - ((n-1) * D_o) / D_e  where n = total coders×items
    # But for the item-level formulation with pairs:
    #   D_o = sum of within-item disagreements / (n_items * n_coders * (n_coders-1) / 2)
    #   For 2 coders: D_o = disagreements / n_items (each item has 1 pair)
    #   D_e = expected disagreement from marginal distribution
    # Let's use the exact canonical formulation.

    # Canonical Krippendorff's α (nominal, 2 coders):
    # n_u = total number of pairable values = sum over items of (n_coders_for_item * (n_coders - 1))
    #      = n_items * 2 * 1 = 2 * n_items (each item has 2 coders, choose 2 = 1 pair)
    # Actually the standard formula:
    # D_o = (1/n_items) * sum_i [ (n_ci choose 2)^{-1} * sum_{c!=c'} delta(v_ic, v_ic') ]
    # For 2 coders, (2 choose 2)^{-1} = 1, so D_o = disagreements / n_items
    #
    # D_e uses marginal frequencies:
    # n_total_values = n_items * 2
    # n_0 = total 0-ratings, n_1 = total 1-ratings
    # D_e = (n_0 * n_1) / (n_total_values * (n_total_values - 1) / 2)
    #      = 2 * n_0 * n_1 / (n_total_values * (n_total_values - 1))

    n_1 = total_ones
    n_0 = total_ratings - total_ones
    D_e_canonical = 2 * n_0 * n_1 / (total_ratings * (total_ratings - 1))

    alpha = 1.0 - D_o / D_e_canonical

    return alpha


def main():
    with open(LEDGER_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 443 analytically active families
    active = [r for r in rows if r['analysis_set_443'] == 'true']
    n_families = len(active)
    n_domains = len(DOMAIN_ORDER)

    print(f"Analytically active families: {n_families}")
    print(f"Domains: {n_domains}")
    print()

    # ── Part 1: WBS-layer-level κ and PABAK ──
    print("=" * 75)
    print("Part 1: WBS-layer concordance (rule-union vs. adjudicated)")
    print("=" * 75)
    print()
    print(f"{'WBS Layer':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'κ':>7} {'PABAK':>7}")
    print("-" * 75)

    for layer_name, domains in WBS_LAYERS.items():
        tp = fp = fn = tn = 0
        for r in active:
            itc = parse_domains(r['itc_codes'])
            rule = parse_domains(r['cpc_rule_domains']) | parse_domains(r['keyword_rule_domains'])

            # At WBS-layer level: family is tagged if ANY domain in the layer is tagged
            in_itc = any(d in itc for d in domains)
            in_rule = any(d in rule for d in domains)

            if in_itc and in_rule:
                tp += 1
            elif not in_itc and in_rule:
                fp += 1
            elif in_itc and not in_rule:
                fn += 1
            else:
                tn += 1

        k = cohens_kappa(tp, fp, fn, tn)
        p = pabak(tp, fp, fn, tn)

        print(f"{layer_name:<30} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
              f"{k:>7.3f} {p:>7.3f}")

    # ── Part 2: Krippendorff's α (nominal) over 15-domain binary matrix ──
    print()
    print("=" * 75)
    print("Part 2: Krippendorff's α (nominal) — 15-domain binary matrix")
    print("=" * 75)
    print()

    # Build binary matrices: rule_matrix[i][j] and adj_matrix[i][j]
    rule_matrix = []
    adj_matrix = []

    for r in active:
        itc = parse_domains(r['itc_codes'])
        rule = parse_domains(r['cpc_rule_domains']) | parse_domains(r['keyword_rule_domains'])

        rule_row = [1 if d in rule else 0 for d in DOMAIN_ORDER]
        adj_row = [1 if d in itc else 0 for d in DOMAIN_ORDER]

        rule_matrix.append(rule_row)
        adj_matrix.append(adj_row)

    alpha = krippendorff_alpha_binary_matrix(
        rule_matrix, adj_matrix, n_families, n_domains
    )

    # Also compute summary statistics for verification
    total_agreements = 0
    total_disagreements = 0
    total_rule_ones = 0
    total_adj_ones = 0

    for i in range(n_families):
        for j in range(n_domains):
            r_val = rule_matrix[i][j]
            a_val = adj_matrix[i][j]
            if r_val == a_val:
                total_agreements += 1
            else:
                total_disagreements += 1
            total_rule_ones += r_val
            total_adj_ones += a_val

    total_cells = n_families * n_domains
    po = total_agreements / total_cells

    print(f"Total cells: {n_families} × {n_domains} = {total_cells}")
    print(f"Agreements: {total_agreements} ({po:.3f})")
    print(f"Disagreements: {total_disagreements} ({total_disagreements/total_cells:.3f})")
    print(f"Rule-union tags (1s): {total_rule_ones}")
    print(f"Adjudicated tags (1s): {total_adj_ones}")
    print()
    print(f"Krippendorff's α (nominal) = {alpha:.3f}")
    print()

    # Cross-check with pooled κ from concordance_diagnostic
    # (should match since both use the same 2×2 pooling)
    pooled_tp = sum(
        1 for i in range(n_families) for j in range(n_domains)
        if rule_matrix[i][j] == 1 and adj_matrix[i][j] == 1
    )
    pooled_fp = sum(
        1 for i in range(n_families) for j in range(n_domains)
        if rule_matrix[i][j] == 1 and adj_matrix[i][j] == 0
    )
    pooled_fn = sum(
        1 for i in range(n_families) for j in range(n_domains)
        if rule_matrix[i][j] == 0 and adj_matrix[i][j] == 1
    )
    pooled_tn = total_cells - pooled_tp - pooled_fp - pooled_fn

    k_pooled = cohens_kappa(pooled_tp, pooled_fp, pooled_fn, pooled_tn)
    p_pooled = pabak(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

    print("Cross-check (pooled domain-level, should match concordance_diagnostic.py):")
    print(f"  Pooled κ = {k_pooled:.3f}, PABAK = {p_pooled:.3f}")
    print(f"  TP={pooled_tp}, FP={pooled_fp}, FN={pooled_fn}, TN={pooled_tn}")


if __name__ == "__main__":
    main()
