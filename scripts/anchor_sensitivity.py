#!/usr/bin/env python3
"""
anchor_sensitivity.py — CPC anchor sensitivity (Supplementary Table S8).

For each of the top-10 Jaccard co-tagging pairs in the adjudicated layer:
  Panel A (published): Uses pre-specified domain-relevant CPC anchors
  Panel B (auto-detected): Uses the most-frequent CPC prefix in the intersection

Panel A reproduces Supplementary Table S8 of the manuscript. Panel B provides
a complementary automatic-detection comparison for full transparency.

Anchor selection in Panel A prioritizes the CPC prefix most structurally
relevant to the domain pair (theory-guided), not simply the most frequent
prefix. Both panels are printed for reviewer inspection.

Usage:
    python scripts/anchor_sensitivity.py
"""

import json
import os
from collections import Counter, defaultdict
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
JSON_PATH = os.path.join(DATA_DIR, "phase2_453_families.json")


def load_active_families():
    """Load families and filter to analytically active (443)."""
    with open(JSON_PATH, encoding="utf-8") as f:
        all_fams = json.load(f)
    # Filter to ITC-classified families (373) for Jaccard computation
    classified = [f for f in all_fams if f.get("is_classified", False)]
    return classified


def get_domain_sets(families):
    """Build domain -> set of family indices mapping from adjudicated ITC codes."""
    domain_sets = defaultdict(set)
    for i, fam in enumerate(families):
        for code in fam.get("itc_codes", []):
            domain_sets[code].add(i)
    return domain_sets


def jaccard(set_a, set_b):
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def find_dominant_cpc_prefix(families, intersection_indices):
    """Find the most frequent CPC 4-character prefix among intersection families."""
    prefix_counter = Counter()
    for idx in intersection_indices:
        fam = families[idx]
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            continue
        cpcs = [c.strip() for c in cpc_str.split(";;") if c.strip()]
        seen_prefixes = set()
        for cpc in cpcs:
            prefix = cpc[:4]
            if prefix not in seen_prefixes:
                prefix_counter[prefix] += 1
                seen_prefixes.add(prefix)
    if not prefix_counter:
        return None, 0
    most_common = prefix_counter.most_common(1)[0]
    return most_common[0], most_common[1]


def count_families_with_prefix(families, indices, prefix):
    """Count how many families in the given index set carry the CPC prefix."""
    count = 0
    for idx in indices:
        fam = families[idx]
        cpc_str = fam.get("cpc", "")
        if not cpc_str:
            continue
        cpcs = [c.strip() for c in cpc_str.split(";;") if c.strip()]
        if any(c.startswith(prefix) for c in cpcs):
            count += 1
    return count


# Published anchor assignments from Supplementary Table S8.
# Anchor selection prioritizes the CPC prefix most structurally relevant
# to the domain pair (domain-relevant, not purely most frequent).
PUBLISHED_ANCHORS = {
    ('2-1', '2-4'): 'B33Y',
    ('4-1', '4-4'): 'E04H',
    ('4-2', '4-4'): 'B64G',
    ('2-1', '2-2'): 'B33Y',
    ('1-3', '2-1'): 'B33Y',
    ('1-3', '2-3'): 'C04B',
    ('1-1', '2-4'): 'G01N',
    ('1-3', '4-1'): 'E04B',
    ('4-2', '4-5'): 'B64G',
    ('4-1', '4-2'): 'E04H',
}

CPC_DESCRIPTORS = {
    'B33Y': 'additive mfg',
    'E04H': 'structures',
    'E04B': 'construction',
    'B64G': 'space vehicles',
    'G01N': 'testing/analysis',
    'C04B': 'ceramics/cement',
    'B22F': 'powder metallurgy',
    'B23K': 'soldering/welding',
    'E01C': 'roads/pads',
    'F24S': 'solar heating',
    'B28B': 'shaping clay',
}


def compute_removal(families, domain_sets, d1, d2, j_raw, prefix):
    """Compute anchor-removal results for a given pair and CPC prefix."""
    intersection = domain_sets[d1] & domain_sets[d2]

    if prefix is None:
        return None, 0, j_raw, 0.0

    families_to_remove = set()
    for idx in intersection:
        fam = families[idx]
        cpc_str = fam.get("cpc", "")
        cpcs = [c.strip() for c in cpc_str.split(";;") if c.strip()]
        if any(c.startswith(prefix) for c in cpcs):
            families_to_remove.add(idx)

    n_removed = len(families_to_remove)
    adj_d1 = domain_sets[d1] - families_to_remove
    adj_d2 = domain_sets[d2] - families_to_remove
    j_adj = jaccard(adj_d1, adj_d2)
    delta = j_adj - j_raw
    return prefix, n_removed, j_adj, delta


def interpret(delta, j_adj):
    if abs(delta) < 0.03:
        return "Robust"
    elif j_adj > 0.10:
        return "Attenuated; residual robust"
    elif abs(delta) > 0.10:
        return "Strongly conditioned"
    else:
        return "Moderately conditioned"


def print_panel(title, families, domain_sets, sorted_pairs, use_published):
    print()
    print("=" * 95)
    print(title)
    print(f"(N = {len(families)} ITC-classified families)")
    print("=" * 95)
    print()
    print(f"{'#':<3} {'Pair':<12} {'J':>6} {'Anchor mode':<8} {'CPC anchor tested':<22} "
          f"{'N rem.':>6} {'J_adj':>6} {'Δ':>8} {'Reading'}")
    print("-" * 95)

    mode_label = "theory" if use_published else "auto"

    for rank, ((d1, d2), j_raw) in enumerate(sorted_pairs, 1):
        intersection = domain_sets[d1] & domain_sets[d2]

        if use_published:
            pair_key = (d1, d2) if (d1, d2) in PUBLISHED_ANCHORS else (d2, d1)
            if pair_key in PUBLISHED_ANCHORS:
                prefix = PUBLISHED_ANCHORS[pair_key]
            else:
                prefix, _ = find_dominant_cpc_prefix(families, intersection)
        else:
            prefix, _ = find_dominant_cpc_prefix(families, intersection)

        prefix, n_removed, j_adj, delta = compute_removal(
            families, domain_sets, d1, d2, j_raw, prefix)

        if prefix is None:
            print(f"{rank:<3} {d1}↔{d2:<7} {j_raw:>6.3f} {mode_label:<8} {'(none)':22} "
                  f"{'0':>6} {j_raw:>6.3f} {'0.000':>8} No shared anchor")
            continue

        interp = interpret(delta, j_adj)
        desc = CPC_DESCRIPTORS.get(prefix, '')
        anchor_label = f"{prefix} ({desc})" if desc else prefix

        print(f"{rank:<3} {d1}↔{d2:<7} {j_raw:>6.3f} {mode_label:<8} {anchor_label:<22} "
              f"{n_removed:>6} {j_adj:>6.3f} {delta:>+8.3f} {interp}")

    print("-" * 95)


def main():
    families = load_active_families()
    domain_sets = get_domain_sets(families)

    # Compute all pairwise Jaccard values
    all_domains = sorted(domain_sets.keys())
    all_pairs = {}
    for d1, d2 in combinations(all_domains, 2):
        j = jaccard(domain_sets[d1], domain_sets[d2])
        all_pairs[(d1, d2)] = j

    # Get top 10 pairs by Jaccard
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: -x[1])[:10]

    # Panel A: Pre-specified domain-relevant anchors (reproduces Table S8)
    print_panel(
        "Panel A: Pre-specified domain-relevant anchors (Supplementary Table S8)",
        families, domain_sets, sorted_pairs, use_published=True)

    # Panel B: Auto-detected most-frequent anchors (complementary comparison)
    print_panel(
        "Panel B: Auto-detected most-frequent CPC prefix (complementary)",
        families, domain_sets, sorted_pairs, use_published=False)

    print()
    print("Notes:")
    print("  Panel A uses pre-specified domain-relevant CPC anchors (theory-guided selection).")
    print("  Panel B uses the most frequent CPC 4-char prefix in the intersection (automatic).")
    print("  J = raw Jaccard similarity on adjudicated ITC tags")
    print("  N rem. = intersection families carrying that prefix (removed from both domain sets)")
    print("  J_adj = Jaccard after removal; Δ = J_adj − J")
    print("  'Strongly conditioned': most overlap attributable to shared CPC infrastructure")
    print("  'Attenuated; residual robust': signal weakens but non-trivial residual remains")
    print("  'Robust': overlap largely independent of tested anchor")


if __name__ == "__main__":
    main()
