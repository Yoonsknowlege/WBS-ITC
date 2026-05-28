#!/usr/bin/env python3
"""
build_audit_sample.py — Generate a stratified reviewer-executable blind
audit sample for the released WBS-ITC operational layer (Table S15
protocol).

Paper: "Auditable denominator design for patent-based field delineation
       in classification-dispersed emerging technologies"

Builds a stratified random sample (target ~80 patent families) spanning:
  - Rule-retained positives  (rule-hit retained after adjudication; no suppression)
  - Adjudication additions   (family has at least one adjudication-only tag)
  - Rule suppressions        (family has at least one rule-positive suppression)
  - Domain-external active   (active but no ITC assignment)
  - False / audit-only       (excluded records)

Strata may overlap at the family level (e.g., a family can have both
retained and suppressed cells); families are drawn without replacement
across strata in priority order.

Output:
  data/audit_sample.csv      — blind sheet (no released ITC codes, no stratum)
  data/audit_sample_key.csv  — author-only key for later scoring

Usage:
    python scripts/build_audit_sample.py
"""
import csv
import json
import os
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
LEDGER_PATH = os.path.join(DATA_DIR, 'decision_ledger.csv')
JSON_PATH = os.path.join(DATA_DIR, 'phase2_453_families.json')
OUT_BLIND = os.path.join(DATA_DIR, 'audit_sample.csv')
OUT_KEY = os.path.join(DATA_DIR, 'audit_sample_key.csv')

STRATA = [
    # (name, target n, priority)
    ('rule_retained_positive', 18),
    ('adjudication_addition',  22),
    ('rule_suppression',       18),
    ('domain_external_active', 15),
    ('false_or_audit_only',     7),
]
RNG_SEED = 20260513


def parse_set(s):
    return set(d.strip() for d in s.split(';')) if s and s.strip() else set()


def classify_family(row):
    """Return list of strata the family is eligible for."""
    out = []
    sc = row.get('screening_category', '')
    if sc in ('false_match', 'audit_only'):
        out.append('false_or_audit_only')
        return out
    if sc == 'domain_external_active':
        out.append('domain_external_active')
        return out
    if sc != 'included_classified':
        return out
    itc = parse_set(row.get('itc_codes', ''))
    cpc = parse_set(row.get('cpc_rule_domains', ''))
    kw = parse_set(row.get('keyword_rule_domains', ''))
    rule_union = cpc | kw
    rule_retained_overlap = bool(rule_union & itc)
    only_adjudication = bool(itc - rule_union)
    only_rule = bool(rule_union - itc)
    if rule_retained_overlap and not only_rule:
        out.append('rule_retained_positive')
    if only_adjudication:
        out.append('adjudication_addition')
    if only_rule:
        out.append('rule_suppression')
    return out


def main():
    with open(LEDGER_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with open(JSON_PATH, encoding='utf-8') as f:
        fams = {fam['lens_id']: fam for fam in json.load(f)}

    pool = defaultdict(list)
    for r in rows:
        for s in classify_family(r):
            pool[s].append(r)

    print('Eligible family counts per stratum:')
    for name, n in STRATA:
        print(f'  {name}: {len(pool[name])} eligible (target {n})')

    random.seed(RNG_SEED)
    chosen = []
    chosen_ids = set()
    for stratum, n_target in STRATA:
        items = [r for r in pool[stratum] if r['lens_id'] not in chosen_ids]
        if len(items) <= n_target:
            picked = items
        else:
            picked = random.sample(items, n_target)
        for r in picked:
            r2 = dict(r)
            r2['stratum'] = stratum
            chosen.append(r2)
            chosen_ids.add(r['lens_id'])
    print(f'\nTotal sample: {len(chosen)} families (target 80)')

    # Blind sheet: NO stratum column (stratum would leak family type to the
    # coder; e.g., 'rule_suppression' or 'domain_external_active' would tip
    # the coder that empty/changed coding is "expected"). Stratum is kept in
    # the author-only key file and joined back at scoring time.
    blind_fields = ['lens_id', 'jurisdiction', 'publication_year',
                    'title', 'abstract', 'cpc',
                    'coder_itc_codes', 'coder_notes']
    with open(OUT_BLIND, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=blind_fields)
        w.writeheader()
        for r in chosen:
            fam = fams.get(r['lens_id'], {})
            title = fam.get('title', '') or ''
            abst = fam.get('abstract', '') or ''
            if isinstance(abst, list):
                abst = ' '.join(str(a) for a in abst)
            elif isinstance(abst, dict):
                abst = str(abst)
            cpc = fam.get('cpc', '')
            if isinstance(cpc, list):
                cpc = ';;'.join(str(c) for c in cpc)
            w.writerow({
                'lens_id': r['lens_id'],
                'jurisdiction': r.get('jurisdiction', fam.get('jurisdiction', '')),
                'publication_year': fam.get('publication_year', ''),
                'title': title[:500],
                'abstract': abst[:2000],
                'cpc': cpc,
                'coder_itc_codes': '',
                'coder_notes': '',
            })
    print(f'\nBlind sheet written to: {OUT_BLIND}')

    key_fields = ['lens_id', 'stratum',
                  'released_itc_codes',
                  'released_screening_category',
                  'released_adjudication_action']
    with open(OUT_KEY, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=key_fields)
        w.writeheader()
        for r in chosen:
            w.writerow({
                'lens_id': r['lens_id'],
                'stratum': r['stratum'],
                'released_itc_codes': r.get('itc_codes', ''),
                'released_screening_category': r.get('screening_category', ''),
                'released_adjudication_action': r.get('adjudication_action', ''),
            })
    print(f'Author-only key written to: {OUT_KEY}')
    print()
    print('Workflow:')
    print('  1. Hand audit_sample.csv to an external coder.')
    print('  2. Coder fills coder_itc_codes (semicolon-separated, e.g. "1-1;2-3";')
    print('     leave empty for domain-external-active and false/audit-only).')
    print('  3. Run scripts/score_audit.py to compute bounded reliability metrics.')


if __name__ == '__main__':
    main()
