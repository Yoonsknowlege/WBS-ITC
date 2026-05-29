"""
Rule-layer recovery diagnostic (Online Resource 1, Table S9).

Computes domain-tag-level rule-positive retention and adjudicated-tag recovery for:
  - CPC-rule only
  - Keyword-rule only
  - CPC-or-keyword (rule union)
  - Adjudication value-added

Operational reference layer: adjudicated ITC assignments (795 tags across
373 families). These are rule-to-adjudicated recovery diagnostics against
the authors' adjudicated operational standard, not external classifier
validation.

Column glossary (printed below):
  Rule-positive retention  = TP / (TP + FP)
      share of rule-positive tags retained after adjudication
  Adjudicated-tag recovery = TP / total adjudicated tags
      share of adjudicated tags recovered by the rule layer
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def parse_domains(s):
    """Parse semicolon-delimited domain string into a set."""
    if not s or s.strip() == '':
        return set()
    return set(s.strip().split(';'))


def main():
    ledger_path = os.path.join(DATA_DIR, 'decision_ledger.csv')

    with open(ledger_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to 443 analytically active families
    active = [r for r in rows if r['analysis_set_443'] == 'true']
    classified = [r for r in active if r['is_classified'] == 'True']

    print(f"Analytically active families: {len(active)}")
    print(f"ITC-classified families: {len(classified)}")
    print()

    # Domain-tag-level rule-positive retention and adjudicated-tag recovery
    cpc_tp = cpc_fp = cpc_fn = 0
    kw_tp = kw_fp = kw_fn = 0
    union_tp = union_fp = union_fn = 0

    for r in classified:
        itc = parse_domains(r['itc_codes'])
        cpc = parse_domains(r['cpc_rule_domains'])
        kw = parse_domains(r['keyword_rule_domains'])
        union = cpc | kw

        cpc_tp += len(itc & cpc)
        cpc_fp += len(cpc - itc)
        cpc_fn += len(itc - cpc)

        kw_tp += len(itc & kw)
        kw_fp += len(kw - itc)
        kw_fn += len(itc - kw)

        union_tp += len(itc & union)
        union_fp += len(union - itc)
        union_fn += len(itc - union)

    total_itc_tags = sum(len(parse_domains(r['itc_codes'])) for r in classified)
    adj_only = union_fn  # tags in ITC but not in rule union

    print("=" * 90)
    print("Online Resource 1, Table S9: Rule-layer recovery diagnostic")
    print(f"(N = {len(classified)} ITC-classified families, {total_itc_tags} adjudicated tags)")
    print("=" * 90)
    print()
    header = (f"{'Layer':<30} {'TP':>5} {'FP':>5} "
              f"{'Rule-pos retention':>20} {'Adj-tag recovery':>18} {'Adj-only':>10}")
    print(header)
    print("-" * 90)

    def fmt_row(name, tp, fp, fn, show_adj=False):
        retention = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recovery = tp / total_itc_tags * 100
        adj_str = f"{fn} ({fn/total_itc_tags*100:.1f}%)" if show_adj else "—"
        line = (f"{name:<30} {tp:>5} {fp:>5} "
                f"{retention:>19.1f}% {recovery:>17.1f}% {adj_str:>10}")
        print(line)

    fmt_row("CPC-rule only", cpc_tp, cpc_fp, cpc_fn)
    fmt_row("Keyword-rule only", kw_tp, kw_fp, kw_fn)
    fmt_row("CPC-or-keyword (rule union)", union_tp, union_fp, union_fn, show_adj=True)
    fmt_row("Adjudicated ITC (reference)", total_itc_tags, 0, 0)

    print()
    print(f"Key finding: {adj_only} of {total_itc_tags} adjudicated tags "
          f"({adj_only/total_itc_tags*100:.1f}%)")
    print("are assigned only through expert adjudication and are not recovered")
    print("by either deterministic rule layer.")
    print()
    print("Note: Rule-positive retention and adjudicated-tag recovery are")
    print("rule-to-adjudicated recovery diagnostics against the authors'")
    print("adjudicated operational standard, not external classifier validation.")


if __name__ == '__main__':
    main()
