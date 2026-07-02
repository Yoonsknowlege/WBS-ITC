#!/usr/bin/env python3
"""
concordance_diagnostic.py — Per-domain rule-hit / assignment agreement
(Online Resource 1, Table S6a).

For each ITC domain, compares the deterministic rule-union layer
(CPC-or-keyword) with the claim-reviewed WBS-ITC assignment over the
443 analytically active families x 15 domains = 6,645 family-domain cells,
and reports the per-domain transformation counts: rule-positive tags,
assignment tags, assignment-only additions, rule-positive tags not retained
(suppressions), and the net change.

These are internal family-domain layer-comparison counts, not
classifier-performance metrics.

Usage:
    python scripts/concordance_diagnostic.py
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TABLE_PATH = os.path.join(DATA_DIR, "assignment_table.csv")

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


def main():
    with open(TABLE_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to 443 analytically active families (uses all active, including
    # domain-external, to capture both additions and suppressions).
    active = [r for r in rows if r['analysis_set_443'] == 'true']

    print(f"Analytically active families: {len(active)}")
    print(f"Domains: {len(DOMAIN_ORDER)}")
    print(f"Total cells: {len(active)} x {len(DOMAIN_ORDER)} = {len(active) * len(DOMAIN_ORDER)}")
    print()

    print("=" * 82)
    print("Online Resource 1, Table S6a: Per-domain rule-hit / assignment agreement")
    print(f"(N = {len(active)} analytically active families x {len(DOMAIN_ORDER)} domains "
          f"= {len(active) * len(DOMAIN_ORDER)} cells)")
    print("=" * 82)
    print()
    print(f"{'Domain':<8} {'Name':<25} {'Rule':>5} {'Adj.':>5} "
          f"{'Added':>6} {'Suppr.':>7} {'Net D':>6}")
    print("-" * 82)

    pool_shared = pool_suppr = pool_added = 0

    for domain in DOMAIN_ORDER:
        shared = suppr = added = 0  # in both / rule-only / assignment-only
        for r in active:
            itc = parse_domains(r['itc_codes'])
            rule = parse_domains(r['cpc_rule_domains']) | parse_domains(r['keyword_rule_domains'])
            in_itc = domain in itc
            in_rule = domain in rule
            if in_itc and in_rule:
                shared += 1
            elif in_rule and not in_itc:
                suppr += 1
            elif in_itc and not in_rule:
                added += 1

        rule_total = shared + suppr      # rule-positive tags for this domain
        asg_total = shared + added       # assignment tags for this domain
        net = added - suppr
        name = DOMAIN_NAMES.get(domain, domain)
        print(f"{domain:<8} {name:<25} {rule_total:>5} {asg_total:>5} "
              f"{added:>6} {suppr:>7} {net:>+6}")

        pool_shared += shared
        pool_suppr += suppr
        pool_added += added

    print("-" * 82)
    rule_total = pool_shared + pool_suppr
    asg_total = pool_shared + pool_added
    net = pool_added - pool_suppr
    print(f"{'Pooled':<8} {'(all domains)':<25} {rule_total:>5} {asg_total:>5} "
          f"{pool_added:>6} {pool_suppr:>7} {net:>+6}")
    print()
    print("Notes:")
    print("  Rule   = rule-positive tags for the domain (rule-union)")
    print("  Adj.   = claim-reviewed WBS-ITC assignment tags for the domain")
    print("  Added  = assignment-only tags (assignment assigns, rule-union does not)")
    print("  Suppr. = rule-positive tags not retained after claim review")
    print("  Net D  = Added - Suppr. = change from rule layer to assignment")
    print()
    print(f"Pooled: {rule_total} rule-positive tags -> {asg_total} assignment tags "
          f"({pool_added} added, {pool_suppr} not retained, net {net:+d}).")
    print("These are internal family-domain layer-comparison counts, not")
    print("classifier-performance metrics.")


if __name__ == "__main__":
    main()
