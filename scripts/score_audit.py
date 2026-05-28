#!/usr/bin/env python3
"""
score_audit.py — Score a completed independent-coder audit sheet.

Paper: "Auditable denominator design for patent-based field delineation
       in classification-dispersed emerging technologies"

Reads `data/audit_sample.csv` (with the coder's filled
`coder_itc_codes` column) and `data/audit_sample_key.csv` (released
ITC truth), and computes bounded reliability/agreement metrics.

Reports:
  - Exact family-level tag-set agreement
  - Family-level Jaccard similarity (average across families)
  - Family-domain binary-cell agreement: precision, recall, F1,
    Cohen's kappa over the n_families x 15 matrix
  - Stratum-level breakdown

These are bounded reliability checks of the released operational layer,
NOT validation against an external gold standard.

Usage:
    python scripts/score_audit.py
"""
import csv
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
BLIND_PATH = os.path.join(DATA_DIR, 'audit_sample.csv')
KEY_PATH = os.path.join(DATA_DIR, 'audit_sample_key.csv')

ITC_DOMAINS = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '2-4',
               '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '4-4', '4-5']


def parse_codes(s):
    if not s or not s.strip():
        return frozenset()
    return frozenset(c.strip() for c in s.replace(',', ';').split(';') if c.strip())


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cohens_kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_yes_a = (tp + fp) / n
    p_yes_b = (tp + fn) / n
    pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0


def main():
    if not os.path.exists(BLIND_PATH):
        print(f'Audit sample not found at: {BLIND_PATH}')
        print('Run scripts/build_audit_sample.py first.')
        return

    blind = {r['lens_id']: r for r in csv.DictReader(open(BLIND_PATH, encoding='utf-8'))}
    key = {r['lens_id']: r for r in csv.DictReader(open(KEY_PATH, encoding='utf-8'))}

    # Stratum lives in the author-only key file (it would leak family type
    # if exposed on the blind sheet). Join it back at scoring time.
    def stratum_of(lens_id, row):
        return key.get(lens_id, {}).get('stratum', row.get('stratum', 'unknown'))

    n_total = len(blind)
    n_filled = sum(1 for lens_id, r in blind.items()
                   if r.get('coder_itc_codes', '').strip() or stratum_of(lens_id, r) in ('domain_external_active', 'false_or_audit_only'))

    print('Independent-coder audit scoring')
    print('=' * 70)
    print(f'Sample size: {n_total}')
    print(f'Coder-completed rows: {n_filled}')
    print()
    if n_filled < n_total:
        print('Warning: not all rows have coder_itc_codes filled.')
        print('Domain-external and false/audit-only rows may legitimately be empty.')
        print()

    # Family-level metrics
    exact_match = 0
    jaccard_sum = 0.0
    by_stratum = defaultdict(lambda: {'n': 0, 'exact': 0, 'jaccard_sum': 0.0,
                                       'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    total_cells = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    for lens_id, row in blind.items():
        coder = parse_codes(row.get('coder_itc_codes', ''))
        truth = parse_codes(key[lens_id]['released_itc_codes'])
        stratum = stratum_of(lens_id, row)
        s = by_stratum[stratum]
        s['n'] += 1
        if coder == truth:
            exact_match += 1
            s['exact'] += 1
        j = jaccard(coder, truth)
        jaccard_sum += j
        s['jaccard_sum'] += j
        for d in ITC_DOMAINS:
            cy = d in coder
            ty = d in truth
            if cy and ty:
                total_cells['tp'] += 1
                s['tp'] += 1
            elif cy and not ty:
                total_cells['fp'] += 1
                s['fp'] += 1
            elif not cy and ty:
                total_cells['fn'] += 1
                s['fn'] += 1
            else:
                total_cells['tn'] += 1
                s['tn'] += 1

    print('Overall metrics')
    print('-' * 70)
    print(f'  Exact family tag-set agreement: {exact_match}/{n_total} = {exact_match/n_total*100:.1f}%')
    print(f'  Mean family-level Jaccard:      {jaccard_sum/n_total:.3f}')
    tp, fp, fn, tn = total_cells['tp'], total_cells['fp'], total_cells['fn'], total_cells['tn']
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    kappa = cohens_kappa(tp, fp, fn, tn)
    print(f'  Cell-level TP/FP/FN/TN:         {tp}/{fp}/{fn}/{tn}')
    print(f'  Precision (cell):               {precision:.3f}')
    print(f'  Recall (cell):                  {recall:.3f}')
    print(f'  F1 (cell):                      {f1:.3f}')
    print(f"  Cohen's kappa (cell):           {kappa:.3f}")
    print()

    print('Stratum-level breakdown')
    print('-' * 70)
    print(f'{"Stratum":<28} {"n":>4} {"Exact":>7} {"Jaccard":>9} {"F1":>6}')
    for s, d in sorted(by_stratum.items()):
        if d['n'] == 0:
            continue
        p = d['tp'] / (d['tp'] + d['fp']) if (d['tp'] + d['fp']) else 0
        r = d['tp'] / (d['tp'] + d['fn']) if (d['tp'] + d['fn']) else 0
        f1s = 2 * p * r / (p + r) if (p + r) else 0
        print(f'{s:<28} {d["n"]:>4} {d["exact"]/d["n"]*100:>6.1f}% '
              f'{d["jaccard_sum"]/d["n"]:>9.3f} {f1s:>6.3f}')

    print()
    print('Note. This is a bounded independent-coder reliability check of the')
    print('released operational layer, not external gold-standard validation.')
    print('Disagreements may reflect adjudication ambiguity rather than coding error.')


if __name__ == '__main__':
    main()
