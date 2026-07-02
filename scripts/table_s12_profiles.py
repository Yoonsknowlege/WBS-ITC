#!/usr/bin/env python3
"""
table_s12_profiles.py — Numeric components of Online Resource 1, Table S12.

Paper: "Capability-based domain assignment for patent-based convergence analysis in classification-dispersed fields: an ISRU space-construction case"

This script regenerates the **numeric components** of Online Resource 1,
Table S12 (Descriptive profiles of selected high-overlap intersection sets):

  - Adjacency pair (ITC domain pair)
  - Jaccard similarity
  - Intersection size (number of co-tagged families)
  - Dominant CPC top-3 prefixes within the intersection
  - 2020-or-later share (filing-year priority)

The interpretive keyword summaries reported in Table S12 ("core technology
keywords" column) are documented in the manuscript but are not regenerated
by this script.

The pairs reported here are those exhibiting the highest Jaccard values
within the 373-family ITC-classified subset, restricted to the within-
codebook adjacencies discussed in the results and discussion sections of the main
text.
"""

import csv
import json
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Table S12 priority pairs (high-overlap zones discussed in body Section 4 / 5.3)
TARGET_PAIRS = [
    ('2-1', '2-4'),  # Manufacturing - Process Monitoring (highest J)
    ('4-1', '4-4'),  # Habitat - Deployable
    ('4-2', '4-4'),  # Shielding - Deployable
    ('2-1', '2-2'),  # Manufacturing variants overlap
    ('1-3', '2-1'),  # Composite/Ceramic - Manufacturing
    ('1-3', '2-3'),  # Composite/Ceramic - Solar/Thermal Process
]


def parse_doms(s):
    return set(s.strip().split(';')) if s and s.strip() else set()


def main():
    table_path = os.path.join(DATA_DIR, 'assignment_table.csv')
    with open(table_path, 'r') as f:
        rows = list(csv.DictReader(f))

    # ITC-classified subset
    classified = [r for r in rows if r['is_classified'] == 'True']

    # Build per-domain family sets and per-family CPC + year (best effort)
    domain_families = {}
    for r in classified:
        for d in parse_doms(r['itc_codes']):
            domain_families.setdefault(d, set()).add(r['lens_id'])

    # Try to load family-level JSON for CPC and priority-year
    json_path = os.path.join(DATA_DIR, 'phase2_453_families.json')
    family_meta = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # data may be a list of family dicts or {lens_id: {...}}
        if isinstance(data, list):
            iterator = data
        elif isinstance(data, dict):
            iterator = data.values()
        else:
            iterator = []

        for fam in iterator:
            try:
                lens_id = (fam.get('lens_id')
                           or fam.get('id')
                           or fam.get('family_id')
                           or '')
                # Earliest priority year
                year = None
                for k in ('earliest_priority_year', 'priority_year',
                          'earliest_priority', 'date_published'):
                    val = fam.get(k)
                    if val:
                        try:
                            year = int(str(val)[:4])
                            break
                        except (ValueError, TypeError):
                            continue
                # CPC prefixes (4-char)
                # Keep both raw occurrence list AND family-level unique
                # set: ESM Table S12 Dominant CPC reports CPC assignment
                # occurrence counts across the intersection set, while a
                # family-level binary count uses the unique set.
                cpc_codes_raw = []
                for k in ('cpc_codes', 'cpc', 'cpc_prefixes'):
                    if k in fam and fam[k]:
                        codes = fam[k]
                        if isinstance(codes, str):
                            # Lens 'cpc' field uses ';;' as the
                            # multi-class delimiter
                            codes = codes.split(';;') if ';;' in codes else codes.split(';')
                        for c in codes:
                            c = str(c).strip()
                            if c:
                                cpc_codes_raw.append(c[:4])
                        break
                family_meta[lens_id] = {
                    'year': year,
                    'cpc_prefixes_raw': cpc_codes_raw,
                    'cpc_prefixes_unique': set(cpc_codes_raw),
                }
            except Exception:
                continue

    print('=' * 95)
    print('Online Resource 1, Table S12: Numeric components of high-overlap')
    print('intersection sets (within the 373-family ITC-classified subset).')
    print('=' * 95)
    print()
    print(f"{'Pair':<14} {'J':>6} {'|∩|':>5} {'2020+ %':>9}  {'Top-3 CPC prefixes':<35}")
    print('-' * 95)

    for d1, d2 in TARGET_PAIRS:
        s1 = domain_families.get(d1, set())
        s2 = domain_families.get(d2, set())
        inter = s1 & s2
        union = s1 | s2
        if not union:
            continue
        j = len(inter) / len(union)

        # 2020-or-later share
        years = [family_meta.get(lid, {}).get('year') for lid in inter]
        years_known = [y for y in years if isinstance(y, int)]
        if years_known:
            share_2020plus = (sum(1 for y in years_known if y >= 2020)
                              / len(years_known) * 100)
            share_str = f'{share_2020plus:.0f}%'
        else:
            share_str = 'n/a'

        # Dominant CPC top-3 — counted by CPC assignment occurrence
        # within the intersection set, matching ESM Table S12. The
        # cooperative-classification tagging scheme prefixes (Y02, Y04,
        # Y10) are tag-like, not primary technical subclasses; they are
        # excluded from the dominant-CPC ranking so the displayed top-3
        # reflects substantive technical classes only.
        Y_TAG_SCHEME = {'Y02', 'Y04', 'Y10'}
        cpc_counter = Counter()
        for lid in inter:
            for prefix in family_meta.get(lid, {}).get('cpc_prefixes_raw', []):
                if prefix[:3] in Y_TAG_SCHEME:
                    continue
                cpc_counter[prefix] += 1
        top3 = ', '.join(p for p, _ in cpc_counter.most_common(3)) or 'n/a'

        print(f"{f'{d1} <-> {d2}':<14} {j:>6.3f} {len(inter):>5} {share_str:>9}  {top3:<35}")

    print()
    print('Note. Interpretive keyword summaries ("core technology keywords"')
    print('column of Table S12) are documented in the manuscript but are not')
    print('script-generated; they reflect author reading of the families that')
    print('appear in each intersection set.')


if __name__ == '__main__':
    main()
