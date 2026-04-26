#!/usr/bin/env python3
"""
concordance_diagnostic.py — Rule-to-adjudicated concordance (Supplementary Table S6a).

Computes per-domain and pooled Cohen's κ and PABAK between the deterministic
rule-union layer (CPC ∪ keyword) and the adjudicated ITC layer, treating each
family × domain cell as a binary agreement unit.

Reference: the 443 analytically active families × 15 ITC domains = 6645 cells.

Usage:
    python scripts/concordance_diagnostic.py
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")

# Domain labels (ordered by WBS layer)
DOMAIN_ORDER = [
    "1-1", "1-2", "1-3",
    "2-1", "2-2", "2-3", "2-4",
    "3-1", "3-2", "3-3",
    "4-1", "4-2", "4-3", "4-4", "4-5",
]

DOMAIN_NAMES = {
    "1-1": "Regolith Processing",
    "1-2": "Binder / Geopolymer",
    "1-3": "Composite / Ceramic",
    "2-1": "Extrusion AM",
    "2-2": "Powder Bed",
    "2-3": "Solar / Laser Sintering",
    "2-4": "Process Monitoring",
    "3-1": "Autonomous Robots",
    "3-2": "Teleoperation",
    "3-3": "Autonomous Construction",
    "4-1": "Habitat Structures",
    "4-2": "Shielding",
    "4-3": "Landing Pads",
    "4-4": "Deployable Structures",
    "4-5": "Life Support / ECLSS",
}


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


def main():
    with open(LEDGER_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to 443 analytically active families (concordance uses all active,
    # including domain-external, to capture both additions and suppressions)
    active = [r for r in rows if r['analysis_set_443'] == 'true']

    print(f"Analytically active families: {len(active)}")
    print(f"Domains: {len(DOMAIN_ORDER)}")
    print(f"Total cells: {len(active)} × {len(DOMAIN_ORDER)} = {len(active) * len(DOMAIN_ORDER)}")
    print()

    # Per-domain 2×2 tables
    print("=" * 85)
    print("Supplementary Table S6a: Rule-to-adjudicated concordance diagnostic")
    print(f"(N = {len(active)} analytically active families × {len(DOMAIN_ORDER)} domains)")
    print("=" * 85)
    print()
    print(f"{'Domain':<8} {'Name':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'κ':>7} {'PABAK':>7} {'Prev':>6}")
    print("-" * 85)

    pooled_tp = pooled_fp = pooled_fn = pooled_tn = 0

    for domain in DOMAIN_ORDER:
        tp = fp = fn = tn = 0
        for r in active:
            itc = parse_domains(r['itc_codes'])
            rule = parse_domains(r['cpc_rule_domains']) | parse_domains(r['keyword_rule_domains'])
            in_itc = domain in itc
            in_rule = domain in rule
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
        prevalence = (tp + fn) / len(active) if len(active) > 0 else 0
        name = DOMAIN_NAMES.get(domain, domain)

        print(f"{domain:<8} {name:<25} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
              f"{k:>7.3f} {p:>7.3f} {prevalence:>6.3f}")

        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn
        pooled_tn += tn

    print("-" * 85)
    k_pooled = cohens_kappa(pooled_tp, pooled_fp, pooled_fn, pooled_tn)
    p_pooled = pabak(pooled_tp, pooled_fp, pooled_fn, pooled_tn)
    total_cells = pooled_tp + pooled_fp + pooled_fn + pooled_tn
    pooled_prev = (pooled_tp + pooled_fn) / total_cells if total_cells > 0 else 0
    print(f"{'Pooled':<8} {'(all domains)':<25} {pooled_tp:>4} {pooled_fp:>4} {pooled_fn:>4} {pooled_tn:>4} "
          f"{k_pooled:>7.3f} {p_pooled:>7.3f} {pooled_prev:>6.3f}")
    print()
    print("Notes:")
    print("  TP = both rule-union and adjudicated assign the domain")
    print("  FP = rule-union assigns but adjudicated does not")
    print("  FN = adjudicated assigns but rule-union does not (adjudication-only tags)")
    print("  TN = neither assigns")
    print("  κ = Cohen's kappa; PABAK = prevalence-adjusted bias-adjusted kappa")
    print("  Prev = domain prevalence in adjudicated layer")
    print()
    print(f"Pooled κ = {k_pooled:.3f}, PABAK = {p_pooled:.3f}")
    print(f"Total adjudication-only tags (FN): {pooled_fn} of {pooled_tp + pooled_fn} "
          f"({pooled_fn / (pooled_tp + pooled_fn) * 100:.1f}%)")


if __name__ == "__main__":
    main()
