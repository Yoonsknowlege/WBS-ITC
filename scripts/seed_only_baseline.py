"""
Seed-CPC-only and rule-union baseline comparison (Online Resource 1, Table S14).

Computes a side-by-side comparison of three delineation layers on the same
443-family analytically active layer:
  - CPC-rule only (seed-CPC matching alone)
  - CPC-or-keyword (rule union, deterministic first-pass)
  - Adjudicated (released ITC mapping layer)

For each layer, the script reports:
  - Number of families with at least one assigned ITC tag
  - Total ITC tag count
  - WBS-layer tag-share distribution (Materials / Manufacturing / Robotics / Structures)
  - Top-6 Jaccard co-tagging pairs

This is a rule-vs-adjudicated portfolio diagnostic, NOT an external classifier
validation. It illustrates how the framework-first WBS-ITC adjudicated layer
differs in WBS distribution and adjacency profile from a pure CPC-rule or
rule-union delineation on the same admitted analytical layer.
"""

import csv
import os
from collections import Counter, defaultdict
from itertools import combinations

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

ITC_DOMAINS = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '2-4',
               '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '4-4', '4-5']

WBS_MAP = {
    '1-1': 'WBS-1 Materials', '1-2': 'WBS-1 Materials', '1-3': 'WBS-1 Materials',
    '2-1': 'WBS-2 Manufacturing', '2-2': 'WBS-2 Manufacturing',
    '2-3': 'WBS-2 Manufacturing', '2-4': 'WBS-2 Manufacturing',
    '3-1': 'WBS-3 Robotics', '3-2': 'WBS-3 Robotics', '3-3': 'WBS-3 Robotics',
    '4-1': 'WBS-4 Structures', '4-2': 'WBS-4 Structures',
    '4-3': 'WBS-4 Structures', '4-4': 'WBS-4 Structures', '4-5': 'WBS-4 Structures',
}


def parse_doms(s):
    return set(s.strip().split(';')) if s and s.strip() else set()


def compute_layer(assigns, name):
    """Compute portfolio + WBS share + top Jaccard pairs for one delineation layer."""
    tag_count = Counter()
    n_classified = 0
    for doms in assigns:
        if doms:
            n_classified += 1
        for d in doms:
            tag_count[d] += 1
    total_tags = sum(tag_count.values())

    wbs_count = Counter()
    for d, c in tag_count.items():
        wbs_count[WBS_MAP.get(d, '?')] += c

    domain_families = defaultdict(set)
    for i, doms in enumerate(assigns):
        for d in doms:
            domain_families[d].add(i)

    jaccard_pairs = []
    for d1, d2 in combinations(ITC_DOMAINS, 2):
        s1, s2 = domain_families[d1], domain_families[d2]
        if not s1 or not s2:
            continue
        inter, union = len(s1 & s2), len(s1 | s2)
        if union > 0:
            jaccard_pairs.append((d1, d2, inter / union, inter))
    jaccard_pairs.sort(key=lambda x: -x[2])

    return {
        'name': name,
        'n_classified': n_classified,
        'total_tags': total_tags,
        'wbs_count': dict(wbs_count),
        'wbs_share_pct': {k: round(v / total_tags * 100, 1) if total_tags > 0 else 0
                          for k, v in wbs_count.items()},
        'top_jaccard': jaccard_pairs[:6],
    }


def main():
    ledger = os.path.join(DATA_DIR, 'decision_ledger.csv')
    with open(ledger) as f:
        rows = list(csv.DictReader(f))

    active = [r for r in rows if r['analysis_set_443'] == 'true']

    cpc_only = [parse_doms(r['cpc_rule_domains']) for r in active]
    rule_union = [parse_doms(r['cpc_rule_domains']) | parse_doms(r['keyword_rule_domains'])
                  for r in active]
    adjudicated = [parse_doms(r['itc_codes']) for r in active]

    layers = [
        compute_layer(cpc_only, 'CPC-rule only'),
        compute_layer(rule_union, 'CPC-or-keyword (rule union)'),
        compute_layer(adjudicated, 'Adjudicated (released)'),
    ]

    print('=' * 90)
    print('Online Resource 1, Table S14: Within-layer comparison of deterministic')
    print('first-pass and adjudicated mapping outputs on the 443-family active layer')
    print('(N = %d analytically active families)' % len(active))
    print('=' * 90)
    print()
    print('--- Panel A. Portfolio coverage and WBS-layer tag-share distribution ---')
    print()
    header = "%-32s %13s %11s %9s %9s %9s %9s" % (
        'Layer', 'N classified', 'Total tags', 'WBS-1 %', 'WBS-2 %', 'WBS-3 %', 'WBS-4 %')
    print(header)
    print('-' * 95)
    wbs_order = ['WBS-1 Materials', 'WBS-2 Manufacturing',
                 'WBS-3 Robotics', 'WBS-4 Structures']
    for L in layers:
        shares = [L['wbs_share_pct'].get(w, 0) for w in wbs_order]
        line = "%-32s %13d %11d %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (
            L['name'], L['n_classified'], L['total_tags'],
            shares[0], shares[1], shares[2], shares[3])
        print(line)
    print()

    print('--- Panel B. Top-6 Jaccard co-tagging pairs by layer ---')
    print()
    for L in layers:
        print('  %s:' % L['name'])
        for d1, d2, j, n in L['top_jaccard']:
            print('    %s <-> %s: J=%.3f (intersection n=%d)' % (d1, d2, j, n))
        print()

    print('--- Reading ---')
    print('Adjudication produces a materially different WBS profile from rule-only')
    print('delineation: Manufacturing share contracts (39.8% -> 31.8%), Robotics')
    print('contracts further (9.7% -> 4.7%), and Structures expands (26.8% -> 35.5%).')
    print('The leading manufacturing-monitoring Jaccard value (2-1 <-> 2-4) is 2.75x')
    print('larger in the adjudicated layer (J = 0.124 -> 0.341), and the')
    print('habitat-envelope Jaccard value (4-1 <-> 4-4) is 2.31x larger in the')
    print('adjudicated layer (J = 0.127 -> 0.294). The CPC-rule-only top pair')
    print('3-3 <-> 4-1 (J = 0.162) does not appear in the adjudicated top-6,')
    print('illustrating mechanism-level suppression of broad autonomous-construction')
    print('rule hits at the deterministic first-pass.')
    print()
    print('Note. This table reports a within-layer delineation-strategy comparison')
    print('on the same 443-family analytically active layer, not external classifier')
    print('validation. The adjudicated layer is the operational reference layer for')
    print('this study, as bounded in the claim-boundary and discussion sections of the main text;')
    print('this comparison illustrates what adjudication adds and removes relative to')
    print('the deterministic first-pass, not comparative superiority over alternative')
    print('delineation methods generally.')


if __name__ == '__main__':
    main()
