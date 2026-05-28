#!/usr/bin/env python3
"""
expanded_baseline.py — Expanded within-layer baseline comparison
(Online Resource 1, Table S16).

Paper: "Auditable denominator design for patent-based field delineation
       in classification-dispersed emerging technologies"

Extends the within-layer comparison of Table S14 to include three
additional bounded delineation variants on the same 443-family active
layer:

  - CPC-rule only           (already in Table S14)
  - Keyword-rule only       (NEW; compares against rule-recovery
                             baseline reported in Table S9)
  - CPC-or-keyword (union)   (already in Table S14)
  - Explicit-lunar/regolith subset
                            (NEW; restricts the active layer to families
                             whose title or abstract contains an
                             explicit-lunar/regolith keyword, applies
                             rule-union to that subset; tests
                             "explicit-lunar-only" delineation)
  - Adjudicated (released)  (already in Table S14)

For each variant, reports:
  - N classified families
  - Total tags
  - Tag-recovery vs released adjudicated layer (TP / 795)
  - Rule-positive retention (TP / variant total tags)
  - WBS-layer tag shares
  - Top-6 Jaccard overlap with the released adjudicated top-6

This is a *bounded operational benchmarking* exercise on the same
admitted analytical layer; it does NOT claim comparative superiority
over external delineation methods generally.

Usage:
    python scripts/expanded_baseline.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
LEDGER_PATH = os.path.join(DATA_DIR, 'decision_ledger.csv')
JSON_PATH = os.path.join(DATA_DIR, 'phase2_453_families.json')

ITC_DOMAINS = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '2-4',
               '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '4-4', '4-5']

WBS_MAP = {
    '1-1': '1', '1-2': '1', '1-3': '1',
    '2-1': '2', '2-2': '2', '2-3': '2', '2-4': '2',
    '3-1': '3', '3-2': '3', '3-3': '3',
    '4-1': '4', '4-2': '4', '4-3': '4', '4-4': '4', '4-5': '4',
}

# Explicit lunar/regolith keywords (restricted retrieval surrogate)
EXPLICIT_LUNAR_TERMS = [
    'lunar', 'regolith', 'moon', 'mars', 'martian', 'in-situ resource',
    'planetary surface', 'extraterrestrial', 'space construction'
]
_lunar_re = re.compile(r'\b(' + '|'.join(EXPLICIT_LUNAR_TERMS) + r')\b', re.I)


def parse_doms(s):
    return set(s.strip().split(';')) if s and s.strip() else set()


def has_explicit_lunar_text(text):
    return bool(_lunar_re.search(text or ''))


def compute_layer(assigns, label, ref_tag_set=None):
    """Compute classified, total tags, WBS shares, Top-6 Jaccard pairs for a layer.
    If ref_tag_set is supplied, also compute tag-recovery (TP/795)."""
    tag_count = Counter()
    n_class = 0
    domain_fams = defaultdict(set)
    family_tag_set = []
    for i, doms in enumerate(assigns):
        if doms:
            n_class += 1
        family_tag_set.append(doms)
        for d in doms:
            tag_count[d] += 1
            domain_fams[d].add(i)
    total_tags = sum(tag_count.values())

    # WBS shares
    wbs = Counter()
    for d, c in tag_count.items():
        wbs[WBS_MAP.get(d, '?')] += c
    wbs_share = {k: round(v / total_tags * 100, 1) if total_tags else 0.0
                 for k, v in wbs.items()}

    # Top-6 Jaccard pairs
    pairs = []
    for d1, d2 in combinations(ITC_DOMAINS, 2):
        s1, s2 = domain_fams[d1], domain_fams[d2]
        if not s1 or not s2:
            continue
        inter, union = len(s1 & s2), len(s1 | s2)
        if union:
            pairs.append((d1, d2, inter / union))
    pairs.sort(key=lambda x: -x[2])
    top6 = pairs[:6]
    top6_set = {(p[0], p[1]) for p in top6}

    # Tag recovery against reference adjudicated layer (TP / 795 reference total)
    if ref_tag_set is not None:
        ref_set = ref_tag_set
        var_set = set()
        for i, doms in enumerate(family_tag_set):
            for d in doms:
                var_set.add((i, d))
        tp = len(var_set & ref_set)
        ref_n = len(ref_set)
        var_n = len(var_set)
        tag_recovery = tp / ref_n * 100 if ref_n else 0.0
        retention = tp / var_n * 100 if var_n else 0.0
    else:
        tp = None
        tag_recovery = None
        retention = None

    return {
        'label': label,
        'n_classified': n_class,
        'total_tags': total_tags,
        'wbs_share': wbs_share,
        'top6': top6,
        'top6_set': top6_set,
        'tp_against_ref': tp,
        'tag_recovery_pct': tag_recovery,
        'rule_pos_retention_pct': retention,
    }


def main():
    # Load ledger
    with open(LEDGER_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    active = [r for r in rows if r['analysis_set_443'] == 'true']
    print(f'Active families: {len(active)}')

    # Load JSON for title+abstract (explicit-lunar filter)
    with open(JSON_PATH, encoding='utf-8') as f:
        fams_json = json.load(f)
    fam_by_id = {fam['lens_id']: fam for fam in fams_json}

    # Build per-family rule-only / keyword-only / rule-union / adjudicated tag sets
    cpc_only = []
    kw_only = []
    rule_union = []
    adjudicated = []
    lunar_mask = []

    for r in active:
        cpc = parse_doms(r['cpc_rule_domains'])
        kw = parse_doms(r['keyword_rule_domains'])
        adj = parse_doms(r['itc_codes'])
        cpc_only.append(cpc)
        kw_only.append(kw)
        rule_union.append(cpc | kw)
        adjudicated.append(adj)

        # Lunar mask from JSON title+abstract
        fam = fam_by_id.get(r['lens_id'], {})
        title = fam.get('title', '') or ''
        abst = fam.get('abstract', '') or ''
        if isinstance(abst, list):
            abst = ' '.join(str(a) for a in abst)
        elif isinstance(abst, dict):
            abst = str(abst)
        lunar_mask.append(has_explicit_lunar_text(title + ' ' + abst))

    # Build ref_tag_set from adjudicated for tag-recovery computation
    ref_tag_set = set()
    for i, doms in enumerate(adjudicated):
        for d in doms:
            ref_tag_set.add((i, d))

    # Explicit-lunar-restricted layer: rule_union AND (lunar mask)
    explicit_lunar_layer = [
        (rule_union[i] if lunar_mask[i] else set())
        for i in range(len(active))
    ]
    n_lunar = sum(1 for m in lunar_mask if m)
    print(f'Active families with explicit-lunar/regolith terms in title/abstract: {n_lunar}')

    layers = [
        compute_layer(cpc_only,           'CPC-rule only',           ref_tag_set),
        compute_layer(kw_only,            'Keyword-rule only',       ref_tag_set),
        compute_layer(rule_union,         'CPC-or-keyword (union)',   ref_tag_set),
        compute_layer(explicit_lunar_layer,
                                          'Explicit-lunar subset (rule-union restricted)',
                                          ref_tag_set),
        compute_layer(adjudicated,        'Adjudicated (released)',  ref_tag_set),
    ]

    # Released reference top-6 set
    released_top6_set = layers[-1]['top6_set']

    print()
    print('=' * 105)
    print('Online Resource 1, Table S16: Bounded operational benchmarking')
    print('on the 443-family active layer. Adjudicated layer is the reference.')
    print('=' * 105)
    print()
    header = ('%-44s %5s %6s %9s %10s %6s %6s %6s %6s %9s' % (
        'Layer', 'N cl.', 'Tags', 'Tag rec.', 'Rule ret.',
        'WBS-1', 'WBS-2', 'WBS-3', 'WBS-4', 'Top-6 ov.'))
    print(header)
    print('-' * 105)
    for L in layers:
        if L['label'].startswith('Adjudicated'):
            tag_rec_str = 'ref'
            retention_str = 'ref'
            top6_ov = f"{len(L['top6_set'] & released_top6_set)}/6 (ref)"
        else:
            tag_rec_str = f"{L['tag_recovery_pct']:.1f}%"
            retention_str = f"{L['rule_pos_retention_pct']:.1f}%"
            top6_ov = f"{len(L['top6_set'] & released_top6_set)}/6"
        w = L['wbs_share']
        print('%-44s %5d %6d %9s %10s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %9s' % (
            L['label'], L['n_classified'], L['total_tags'],
            tag_rec_str, retention_str,
            w.get('1', 0), w.get('2', 0), w.get('3', 0), w.get('4', 0),
            top6_ov))
    print('-' * 105)
    print()
    print('Note. "Tag rec." = TP / 795 against the adjudicated reference layer.')
    print('"Rule ret." = TP / variant total tags (variant-positive retention).')
    print('"Top-6 ov." = number of the adjudicated top-6 Jaccard pairs that')
    print('also appear in the variant top-6.')
    print('This is bounded operational benchmarking against deterministic and')
    print('restrictive delineation variants on the same admitted analytical')
    print('layer; it is not a claim of comparative superiority over alternative')
    print('delineation methods generally.')

    print()
    print('Top-6 Jaccard pairs per variant:')
    for L in layers:
        print(f'  {L["label"]}:')
        for d1, d2, j in L['top6']:
            mark = ' [ref]' if (d1, d2) in released_top6_set else ''
            print(f'    {d1} <-> {d2}: J={j:.3f}{mark}')
        print()


if __name__ == '__main__':
    main()
